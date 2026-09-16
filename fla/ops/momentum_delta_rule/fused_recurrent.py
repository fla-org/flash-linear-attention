# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


import torch
import triton
import triton.language as tl

from fla.ops.momentum_delta_rule.naive import recurrent_momentum_delta_rule_ref
from fla.ops.utils.op import exp
from fla.utils import input_guard


@triton.heuristics({
    'USE_INITIAL_S': lambda args: args['s0'] is not None,
    'STORE_FINAL_S': lambda args: args['st'] is not None,
    'USE_INITIAL_M': lambda args: args['m0'] is not None,
    'STORE_FINAL_M': lambda args: args['mt'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_momentum_delta_rule_fwd_kernel(
    q,
    k,
    v,
    p,
    log_alpha,
    log_mu,
    beta,
    eta,
    o,
    s0,
    m0,
    st,
    mt,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_S: tl.constexpr,
    STORE_FINAL_S: tl.constexpr,
    USE_INITIAL_M: tl.constexpr,
    STORE_FINAL_M: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_P_TIMES_ALPHA: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_p = p + (bos * H + i_h) * K + o_k
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
        p_eta = eta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
        p_eta = eta + bos * HV + i_hv

    p_log_alpha = log_alpha + bos * HV + i_hv
    p_log_mu = log_mu + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_S = tl.zeros([BK, BV], dtype=tl.float32)
    b_M = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_S:
        p_s0 = s0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_S += tl.load(p_s0, mask=mask_h, other=0).to(tl.float32)
    if USE_INITIAL_M:
        p_m0 = m0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_M += tl.load(p_m0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_p = tl.load(p_p, mask=mask_k, other=0).to(tl.float32)

        b_log_alpha = tl.load(p_log_alpha).to(tl.float32)
        b_mu = tl.load(p_log_mu).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)
            b_p = b_p / (tl.sqrt(tl.sum(b_p * b_p)) + 1e-6)

        if USE_P_TIMES_ALPHA:
            b_p = b_p * exp(b_log_alpha)

        b_q = b_q * scale
        # [BV]
        b_v = b_v - tl.sum(b_S * b_p[:, None], 0)  # delta rule
        # [BK, BV]
        b_M *= exp(b_mu)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
            b_eta = tl.load(p_eta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
            b_eta = tl.load(p_eta).to(tl.float32)

        b_v *= b_eta
        # [BK, BV]
        b_M = b_M - b_k[:, None] * b_v[None, :]                # momentum rule

        b_S = b_S * exp(b_log_alpha) - b_beta * b_M            # state update

        # [BV]
        b_o = tl.sum(b_S * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_p += H * K
        p_o += HV * V

        p_log_alpha += HV
        p_log_mu += HV
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)
        p_eta += HV * (V if IS_BETA_HEADWISE else 1)

    if STORE_FINAL_S:
        p_ht = st + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_S.to(p_ht.dtype.element_ty), mask=mask_h)

    if STORE_FINAL_M:
        p_ht = mt + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_M.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_momentum_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    log_alpha: torch.Tensor,
    log_mu: torch.Tensor,
    beta: torch.Tensor,
    eta: torch.Tensor,
    scale: float,
    initial_S: torch.Tensor,
    initial_M: torch.Tensor,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
    use_p_times_alpha: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    o = q.new_empty(NK, *v.shape)
    if output_final_state:
        final_S = q.new_empty(N, HV, K, V, dtype=torch.float32)
        final_M = q.new_empty(N, HV, K, V, dtype=torch.float32)
    else:
        final_S = None
        final_M = None

    grid = (NK, NV, N * HV)
    fused_recurrent_momentum_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        p=p,
        log_alpha=log_alpha,
        log_mu=log_mu,
        beta=beta,
        eta=eta,
        o=o,
        s0=initial_S,
        m0=initial_M,
        st=final_S,
        mt=final_M,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_P_TIMES_ALPHA=use_p_times_alpha,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_S, final_M


class FusedRecurrentMomentumDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        log_alpha: torch.Tensor,
        log_mu: torch.Tensor,
        beta: torch.Tensor,
        eta: torch.Tensor,
        scale: float,
        initial_S: torch.Tensor,
        initial_M: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        use_p_times_alpha: bool = False,
    ):
        o, final_S, final_M = fused_recurrent_momentum_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            p=p,
            log_alpha=log_alpha,
            log_mu=log_mu,
            beta=beta,
            eta=eta,
            scale=scale,
            initial_S=initial_S,
            initial_M=initial_M,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_p_times_alpha=use_p_times_alpha,
            cu_seqlens=cu_seqlens
        )

        ctx.save_for_backward(q, k, v, p, log_alpha, log_mu, beta, eta,
                              initial_S, initial_M, cu_seqlens)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_p_times_alpha = use_p_times_alpha

        return o, final_S, final_M

    @staticmethod
    @input_guard
    def backward(ctx, do, dst, dmt):
        q, k, v, p, log_alpha, log_mu, beta, eta, initial_S, initial_M, cu_seqlens = ctx.saved_tensors
        with torch.enable_grad():
            q_r = q.detach().requires_grad_(True)
            k_r = k.detach().requires_grad_(True)
            v_r = v.detach().requires_grad_(True)
            p_r = p.detach().requires_grad_(True)
            log_alpha_r = log_alpha.detach().requires_grad_(True)
            log_mu_r = log_mu.detach().requires_grad_(True)
            beta_r = beta.detach().requires_grad_(True)
            eta_r = eta.detach().requires_grad_(True) if eta is not None else None
            k_eta = k_r if eta_r is None else k_r * eta_r.unsqueeze(-1)
            p_eff = p_r if not ctx.use_p_times_alpha else p_r * log_alpha_r.exp().unsqueeze(-1)
            initial_S_r = initial_S.detach().requires_grad_(True) if initial_S is not None else None
            initial_M_r = initial_M.detach().requires_grad_(True) if initial_M is not None else None
            o, final_state = recurrent_momentum_delta_rule_ref(
                q=q_r, k=k_eta, v=v_r, p=p_eff, log_alpha=log_alpha_r, log_mu=log_mu_r,
                beta=beta_r, eta=torch.ones_like(beta_r), scale=ctx.scale,
                initial_S=initial_S_r, initial_M=initial_M_r, output_final_state=True,
            )
            inputs = (q_r, k_r, v_r, p_r, log_alpha_r, log_mu_r, beta_r)
            if eta_r is not None:
                inputs += (eta_r,)
            if initial_S_r is not None:
                inputs += (initial_S_r, initial_M_r)
            grads = torch.autograd.grad(
                (o, final_state[0], final_state[1]), inputs,
                grad_outputs=(
                    do,
                    torch.zeros_like(final_state[0]) if dst is None else dst,
                    torch.zeros_like(final_state[1]) if dmt is None else dmt,
                ),
                allow_unused=True,
            )
        dq, dk, dv, dp, dlog_alpha, dlog_mu, dbeta = grads[:7]
        deta = grads[7] if eta is not None else None
        state_offset = 8 if eta is not None else 7
        ds0, dm0 = grads[state_offset:state_offset + 2] if initial_S is not None else (None, None)
        return dq, dk, dv, dp, dlog_alpha, dlog_mu, dbeta, deta, None, ds0, dm0, None, None, None, None


def fused_recurrent_momentum_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    log_mu: torch.Tensor,
    p: torch.Tensor = None,
    beta: torch.Tensor = None,
    eta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_p_times_alpha: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        log_alpha (torch.Tensor):
            Log-scale coefficients of shape `[B, T, H]`.
        log_mu (torch.Tensor):
            Log-decay coefficients of shape `[B, T, H]`.
        p (torch.Tensor, optional):
            Auxiliary keys of shape `[B, T, H, K]`. Defaults to `k` when None.
        beta (torch.Tensor, optional):
            Forget gate coefficients of shape `[B, T, H]`. Defaults to ones when None.
        eta (torch.Tensor, optional):
            Per-token scaling factors of shape `[B, T, H]`. Defaults to ones when None.
        scale (Optional[float]):
            Scale factor for attention scores. If not provided, it defaults to `1 / sqrt(K)`.
        initial_state (Optional[torch.Tensor]):
            Initial state tensor of shape `[2, N, H, K, V]`, where the first element is `S`
            and the second element is `M`. For equal-length inputs, `N` equals batch size `B`.
        output_final_state (bool, optional):
            Whether to return the final state. Default: `False`.
        cu_seqlens (Optional[torch.LongTensor]):
            Cumulative sequence lengths of shape `[N+1]` for variable-length training.
        use_qk_l2norm_in_kernel (bool, optional):
            Whether to apply L2 normalization to q, k, and p before the kernel call.
        use_p_times_alpha (bool, optional):
            Whether to scale `p` by `exp(log_alpha)` internally.
            If set to 'True', the input 'p' = `alpha * Norm(k)`,
            where `alpha` is a learnable scalar and `Norm(k)` is the L2 normalization of `k`.

    Returns:
        o (torch.Tensor):
            Output tensor of shape `[B, T, H, V]`.
        final_state (Optional[torch.Tensor]):
            Final state tensor of shape `[2, N, H, K, V]` when `output_final_state=True`, otherwise `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> log_alpha = -torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda')
        >>> log_mu = -torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> eta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda')
        >>> h0 = torch.randn(2, B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = fused_recurrent_momentum_delta_rule(
        ...     q, k, v, log_alpha, log_mu,
        ...     beta=beta,
        ...     eta=eta,
        ...     initial_state=h0,
        ...     output_final_state=True
        ... )
    """

    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    if eta is None:
        eta = torch.ones_like(q[..., 0])
    if p is None:
        p = k
    if initial_state is not None:
        initial_S, initial_M = initial_state[0], initial_state[1]
    else:
        initial_S, initial_M = None, None

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_S is not None and initial_S.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_S.shape[0]}."
            )

    o, final_S, final_M = FusedRecurrentMomentumDeltaRuleFunction.apply(
        q,
        k,
        v,
        p,
        log_alpha,
        log_mu,
        beta,
        eta,
        scale,
        initial_S,
        initial_M,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        use_p_times_alpha
    )
    if output_final_state:
        final_state = torch.stack([final_S, final_M], dim=0)
    else:
        final_state = None

    return o, final_state
