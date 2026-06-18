# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# This kernel is modified from the Decode kernel of the vllm gdn/kda model.

import math

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus
from fla.utils import input_guard


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "USE_INITIAL_ATK": lambda args: args["a0"] is not None,
        "STORE_FINAL_ATK": lambda args: args["at"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_precond_kda_fwd_kernel(
    q,
    k,
    v,
    g,
    g_atk,
    beta_atk,
    beta,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    a0,
    at,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    lower_bound,
    log_atk_scale,
    scale: tl.constexpr,
    logx,
    eps,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_INITIAL_ATK: tl.constexpr,
    STORE_FINAL_ATK: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    NK = tl.cdiv(K, BK)
    i_k = pid % NK
    pid_rest = pid // NK

    i_v = pid_rest % NV
    i_nh = pid_rest // NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    if T == 0:
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv

    p_g = g + (bos * HV + i_hv) * K + o_k
    # ATK pointers (per-HV-head: g_atk, beta_atk, log_atk_scale are shape-HV like v/beta/g)
    p_g_atk = g_atk + bos * HV + i_hv
    p_beta_atk = beta_atk + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v

    center = tl.load(log_atk_scale + i_hv).to(tl.float32)

    mask_k = o_k < K
    mask_v = o_v < V
    if TRANSPOSE_STATE:
        mask_h = mask_v[:, None] & mask_k[None, :]
    else:
        mask_h = mask_k[:, None] & mask_v[None, :]

    if TRANSPOSE_STATE:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            p_h0 = (
                h0
                + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                    tl.int64
                )
                * stride_init_state_token
            )
            if TRANSPOSE_STATE:
                p_h0 = p_h0 + i_hv * K * V + o_v[:, None] * K + o_k[None, :]
            else:
                p_h0 = p_h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        else:
            if TRANSPOSE_STATE:
                p_h0 = h0 + (i_n * HV + i_hv) * K * V + o_v[:, None] * K + o_k[None, :]
            else:
                p_h0 = h0 + (i_n * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    b_a = tl.zeros([BK], dtype=tl.float32)
    if USE_INITIAL_ATK:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            p_a0 = (
                a0
                + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                    tl.int64
                )
                * stride_init_state_token
            )
            p_a0 = p_a0 + i_hv * K + o_k
        else:
            p_a0 = a0 + (i_n * HV + i_hv) * K + o_k
        b_a += tl.load(p_a0, mask=mask_k, other=0).to(tl.float32)

    for i_t in tl.range(0, T, num_stages=num_stages):
        b_q = tl.load(p_q, mask=mask_k, other=0, eviction_policy='evict_last').to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0, eviction_policy='evict_last').to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0, eviction_policy='evict_first').to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_g = tl.load(p_g, eviction_policy='evict_last').to(tl.float32)

        if USE_GATE_IN_KERNEL:
            b_A = tl.load(A_log + i_hv).to(tl.float32)

            if HAS_DT_BIAS:
                b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0).to(tl.float32)
                b_g = b_g + b_bias

            if USE_LOWER_BOUND:
                b_gk = lower_bound * tl.sigmoid(exp(b_A) * b_g)
            else:
                b_gk = -exp(b_A) * softplus(b_g)
        else:
            b_gk = b_g

        b_g_atk = tl.load(p_g_atk).to(tl.float32)
        b_beta_atk = tl.load(p_beta_atk).to(tl.float32)

        # ATK: A_t = exp(g_atk) * A_{t-1} + beta_atk * k^2
        b_a = exp(b_g_atk) * b_a + b_beta_atk * (b_k * b_k)

        b_ell = tl.log(b_a + eps)
        b_r = b_ell - center
        b_s = b_r / (1.0 + tl.abs(b_r))
        b_M = tl.exp(-logx * b_s)
        b_k_precond = b_k * b_M

        if TRANSPOSE_STATE:
            b_h *= exp(b_gk[None, :])
        else:
            b_h *= exp(b_gk[:, None])

        if TRANSPOSE_STATE:
            b_v -= tl.sum(b_h * b_k[None, :], 1)
        else:
            b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0, eviction_policy='evict_first').to(tl.float32)
        else:
            b_beta = tl.load(p_beta, eviction_policy='evict_last').to(tl.float32)
        b_v *= b_beta
        if TRANSPOSE_STATE:
            b_h += b_v[:, None] * b_k_precond[None, :]
            b_o = tl.sum(b_h * b_q[None, :], 1)
        else:
            b_h += b_k_precond[:, None] * b_v[None, :]
            b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v, eviction_policy='evict_first')

        if IS_CONTINUOUS_BATCHING:
            if INPLACE_FINAL_STATE:
                p_ht = (
                    ht
                    + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                        tl.int64
                    )
                    * stride_final_state_token
                )
            else:
                p_ht = ht + (bos + i_t) * stride_final_state_token
            if TRANSPOSE_STATE:
                p_ht = p_ht + i_hv * K * V + o_v[:, None] * K + o_k[None, :]
            else:
                p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

            # Store ATK state per timestep during continuous batching (only from first v-block and k-block)
            if i_v == 0 and i_k == 0:
                if INPLACE_FINAL_STATE:
                    p_at_cb = (
                        at
                        + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                            tl.int64
                        )
                        * stride_final_state_token
                    )
                else:
                    p_at_cb = at + (bos + i_t) * stride_final_state_token
                p_at_cb = p_at_cb + i_hv * K + o_k
                tl.store(p_at_cb, b_a.to(p_at_cb.dtype.element_ty), mask=mask_k)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_g += HV * K
        p_g_atk += HV
        p_beta_atk += HV
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)

    if not IS_CONTINUOUS_BATCHING:
        if STORE_FINAL_STATE:
            if TRANSPOSE_STATE:
                p_ht = ht + (i_n * HV + i_hv) * K * V + o_v[:, None] * K + o_k[None, :]
            else:
                p_ht = ht + (i_n * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        if STORE_FINAL_ATK:
            if i_v == 0 and i_k == 0:
                p_at = at + (i_n * HV + i_hv) * K + o_k
                tl.store(p_at, b_a.to(p_at.dtype.element_ty), mask=mask_k)


@torch.compiler.disable
def fused_recurrent_precond_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_atk: torch.Tensor,
    beta_atk: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    initial_A_state: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    out: torch.Tensor | None = None,
    transpose_state_layout: bool = False,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if scale is None:
        scale = k.shape[-1] ** -0.5

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = 32

    if log_atk_scale is None:
        log_atk_scale = torch.full((HV,), -0.2, device=k.device, dtype=torch.float32)
    log_atk_scale = log_atk_scale.contiguous()

    logx = math.log(x) if x > 0 else 0.0

    if out is None:
        out = torch.zeros_like(v)
    else:
        assert out.shape == v.shape
    if inplace_final_state:
        assert initial_state is not None
        final_state = initial_state
        final_A_state = initial_A_state
    elif output_final_state:
        if transpose_state_layout:
            final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
        final_A_state = q.new_empty(N, HV, K, dtype=torch.float32)
    else:
        final_state = None
        final_A_state = None

    stride_init_state_token = initial_state.stride(0) if initial_state is not None else 1
    stride_final_state_token = final_state.stride(0) if final_state is not None else 1

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    NK = triton.cdiv(K, BK)
    grid = (triton.cdiv(V, BV) * NK * N * HV, )
    fused_recurrent_precond_kda_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        g_atk=g_atk,
        beta_atk=beta_atk,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=final_state,
        a0=initial_A_state,
        at=final_A_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        lower_bound=lower_bound,
        log_atk_scale=log_atk_scale,
        scale=scale,
        logx=logx,
        eps=eps,
        N=N,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_GATE_IN_KERNEL=use_gate_in_kernel,
        TRANSPOSE_STATE=transpose_state_layout,
        num_warps=4,
        num_stages=2,
    )

    return out, final_state, final_A_state


@input_guard
def fused_recurrent_precond_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_atk: torch.Tensor,
    beta_atk: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor = None,
    initial_A_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV, K]`.
        g_atk (torch.Tensor):
            scalar gate for ATK in log space of shape `[B, T, H]`.
        beta_atk (torch.Tensor):
            betas for ATK of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        A_log (Optional[torch.Tensor]):
            Per-head log decay parameter of shape `[H]`. Required when `use_gate_in_kernel=True`.
            Default: `None`.
        dt_bias (Optional[torch.Tensor]):
            Per-head-key bias of shape `[H * K]`. Used when `use_gate_in_kernel=True`.
            Default: `None`.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        initial_A_state (Optional[torch.Tensor]):
            Initial ATK diagonal state of shape `[N, H, K]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use L2 normalization in the kernel. Default: `False`.
        use_gate_in_kernel (Optional[bool]):
            Whether to compute the log-space KDA decay internally.
            If `True`, `g` acts as the raw input for `-exp(A_log) * softplus(g + dt_bias)`.
            `A_log` must be provided. Default: `False`.
        lower_bound (Optional[float]):
            If provided, uses `lower_bound * sigmoid(exp(A_log) * g)` instead of
            `-exp(A_log) * softplus(g)` for gate computation. Only used when
            `use_gate_in_kernel=True`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        transpose_state_layout (bool):
            Whether to use transposed state layout `[V, K]` instead of `[K, V]`. Default: `False`.
        x (float):
            Squash range parameter. Default: `1.5`.
        eps (float):
            Epsilon for numerical stability. Default: `1e-6`.
        log_atk_scale (Optional[torch.Tensor]):
            Per-head log-space center of shape `[H]`. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.
        A_state (torch.Tensor):
            Final ATK diagonal state of shape `[N, H, K]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.precond_kda import fused_recurrent_precond_kda
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, K, device='cuda'))
        >>> g_atk = F.logsigmoid(torch.rand(B, T, H, device='cuda'))
        >>> beta_atk = torch.rand(B, T, H, device='cuda').sigmoid()
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht, at = fused_recurrent_precond_kda(
            q, k, v, g, g_atk, beta_atk, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g))
        >>> g_atk, beta_atk, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (g_atk, beta_atk, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var, at_var = fused_recurrent_precond_kda(
            q, k, v, g, g_atk, beta_atk, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state, final_A_state = fused_recurrent_precond_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        g_atk=g_atk,
        beta_atk=beta_atk,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        initial_A_state=initial_A_state,
        inplace_final_state=False,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        transpose_state_layout=transpose_state_layout,
        x=x,
        eps=eps,
        log_atk_scale=log_atk_scale,
    )
    return o, final_state, final_A_state
