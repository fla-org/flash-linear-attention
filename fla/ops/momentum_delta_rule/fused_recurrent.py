# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch
import triton
import triton.language as tl

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.delta_rule.fused_recurrent import (
    fused_recurrent_delta_rule_bwd_kernel,
    fused_recurrent_delta_rule_fwd_kernel,
)
from fla.ops.utils.op import exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

# Degenerate (mu->0) uses DeltaRule kernels.


def fused_recurrent_mode_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
):
    """Degenerate fused forward (mu->0)."""
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 1
    num_warps = 1

    o = q.new_empty(NK, *v.shape)
    if output_final_state:
        final_state = q.new_empty(N, H, K, V, dtype=torch.float32)
    else:
        final_state = None

    u = torch.empty_like(v)

    grid = (NV * NK * N * H,)
    fused_recurrent_delta_rule_fwd_kernel[grid](
        q, k, v, u, beta, o, initial_state, final_state, cu_seqlens,
        scale=1.0 / math.sqrt(K), T=T, B=B, H=H, K=K, V=V, BK=BK, BV=BV,
        IS_BETA_HEADWISE=beta.ndim == v.ndim, num_warps=num_warps, num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, u, final_state


def fused_recurrent_mode_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    dht: torch.Tensor,
    do: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 1
    num_warps = 2
    beta_vector = beta.ndim == v.ndim
    dq = q.new_empty(NV, *q.shape)
    dk = q.new_empty(NV, *k.shape)
    dv = q.new_empty(NK, *v.shape)
    if beta_vector:
        db = q.new_empty(NV, NK, B, T, H, V)
    else:
        db = q.new_empty(NV, B, T, H)
    grid = (NV * NK * N * H,)
    if initial_state is not None and initial_state.requires_grad:
        dh0 = torch.empty_like(initial_state, dtype=torch.float32)
    else:
        dh0 = None
    fused_recurrent_delta_rule_bwd_kernel[grid](
        q, k, v, beta, initial_state, dh0, dht, do, dq, dk, dv, db, cu_seqlens,
        scale=1.0 / math.sqrt(K), T=T, B=B, H=H, K=K, V=V, BK=BK, BV=BV, NK=NK,
        IS_BETA_HEADWISE=beta_vector, num_warps=num_warps, num_stages=num_stages,
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    db = db.sum((0, 1)) if beta_vector else db.sum(0)
    return dq, dk, dv, db, dh0


class FusedRecurrentModeRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None
        o, u, final_state = fused_recurrent_mode_rule_fwd(
            q=q, k=k, v=v, beta=beta, initial_state=initial_state,
            output_final_state=output_final_state, cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, beta, initial_state, u)
        ctx.scale = 1.0 / math.sqrt(q.shape[-1])
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cu_seqlens = cu_seqlens
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        q, q_rstd, k, k_rstd, v, beta, initial_state, u = ctx.saved_tensors
        dq, dk, dv, db, dh0 = fused_recurrent_mode_rule_bwd(
            q=q, k=k, v=u, beta=beta, dht=dht, do=do, initial_state=initial_state, cu_seqlens=ctx.cu_seqlens,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), dh0, None, None, None


@torch.compiler.disable
def fused_recurrent_mode_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
):
    r"""Degenerate fused recurrent (mu->0). Use fused_recurrent_momentum_delta_rule for full."""
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "FusedRecurrentModeRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape (batch size, seq len, num heads)."
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.")
    o, final_state = FusedRecurrentModeRuleFunction.apply(
        q, k, v, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens,
    )
    return o, final_state


# Full momentum uses Triton kernels.

@triton.heuristics({
    'USE_INITIAL_S': lambda args: args['s0'] is not None,
    'STORE_FINAL_S': lambda args: args['st'] is not None,
    'USE_INITIAL_M': lambda args: args['m0'] is not None,
    'STORE_FINAL_M': lambda args: args['mt'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_mode_rule_fwd_kernel(
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
        p_s0 = s0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_S += tl.load(p_s0, mask=mask_h, other=0).to(tl.float32)
    if USE_INITIAL_M:
        p_m0 = m0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
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
        b_v = b_v - tl.sum(b_S * b_p[:, None], 0)
        b_M *= exp(b_mu)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
            b_eta = tl.load(p_eta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
            b_eta = tl.load(p_eta).to(tl.float32)

        b_v *= b_eta
        b_M = b_M - b_k[:, None] * b_v[None, :]

        b_S = b_S * exp(b_log_alpha) - b_beta * b_M

        b_o = tl.sum(b_S * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H*K
        p_k += H*K
        p_v += HV*V
        p_p += H*K
        p_o += HV*V

        p_log_alpha += HV
        p_log_mu += HV
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)
        p_eta += HV * (V if IS_BETA_HEADWISE else 1)

    if STORE_FINAL_S:
        p_ht = st + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_S.to(p_ht.dtype.element_ty), mask=mask_h)

    if STORE_FINAL_M:
        p_ht = mt + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
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
    fused_recurrent_mode_rule_fwd_kernel[grid](
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

        return o, final_S, final_M

    @staticmethod
    @input_guard
    def backward(ctx, do, dst, dmt):
        raise NotImplementedError(
            "Backward pass is not implemented yet and we do not have plans to implement it "
            "because we haven't figured out how to compute dg without materializing the full "
            "hidden states for all time steps."
        )


@torch.compiler.disable
def fused_recurrent_momentum_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    log_mu: torch.Tensor,
    p: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    eta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
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
