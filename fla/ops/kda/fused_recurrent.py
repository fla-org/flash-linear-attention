# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# This kernel is modified from the Decode kernel of the vllm gdn/kda model.

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
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_kda_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    lower_bound,
    scale: tl.constexpr,
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
    HAS_DT_BIAS: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
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
    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

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
            p_h0 = p_h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        else:
            # For non-continuous batching: h0 has shape [N, HV, K, V]
            # where N = B (batch size) for equal-length sequences
            p_h0 = h0 + (i_n * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in tl.range(0, T, num_stages=num_stages):
        b_q = tl.load(p_q, mask=mask_k, other=0, eviction_policy='evict_last').to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0, eviction_policy='evict_last').to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0, eviction_policy='evict_first').to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        # [BK, BV]
        b_g = tl.load(p_g, eviction_policy='evict_last').to(tl.float32)

        # Compute gate if USE_GATE_IN_KERNEL is True
        if USE_GATE_IN_KERNEL:
            # Load A_log for this head
            b_A = tl.load(A_log + i_h).to(tl.float32)

            # Apply dt_bias if exists
            if HAS_DT_BIAS:
                b_bias = tl.load(dt_bias + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
                b_g = b_g + b_bias

            # Compute gate based on lower_bound
            if USE_LOWER_BOUND:
                # lower_bound * sigmoid(exp(A_log) * g)
                b_gk = lower_bound * tl.sigmoid(exp(b_A) * b_g)
            else:
                # -exp(A_log) * softplus(g)
                b_gk = -exp(b_A) * softplus(b_g)
        else:
            # Use g directly as gate
            b_gk = b_g

        b_h *= exp(b_gk[:, None])

        # [BV]
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0, eviction_policy='evict_first').to(tl.float32)
        else:
            b_beta = tl.load(p_beta, eviction_policy='evict_last').to(tl.float32)
        b_v *= b_beta
        # [BK, BV]
        b_h += b_k[:, None] * b_v[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v, eviction_policy='evict_first')

        if IS_CONTINUOUS_BATCHING:
            # keep the states for multi-query tokens
            # only save per step in vllm
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
            p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_g += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)

    if not IS_CONTINUOUS_BATCHING:
        # Non-vLLM mode: only save final state at the end of loop (standard behavior)
        if STORE_FINAL_STATE:
            # Use unified indexing: (i_n * HV + i_hv) for [N, HV, K, V] layout
            # Both initial_state and final_state use this formula for consistency
            p_ht = ht + (i_n * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@torch.compiler.disable
def fused_recurrent_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
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
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = k.shape[-1] ** -0.5

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = 32

    if out is None:
        out = torch.zeros_like(v)
    else:
        assert out.shape == v.shape
    if inplace_final_state:
        assert initial_state is not None
        final_state = initial_state
    elif output_final_state:
        final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
    else:
        final_state = None

    stride_init_state_token = initial_state.stride(0) if initial_state is not None else 1
    stride_final_state_token = final_state.stride(0) if final_state is not None else 1

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (triton.cdiv(V, BV) * N * HV, )
    fused_recurrent_kda_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        lower_bound=lower_bound,
        scale=scale,
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
        num_warps=4,
        num_stages=2,
    )

    return out, final_state


@input_guard
def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use L2 normalization in the kernel. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.kda import fused_recurrent_kda
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, K, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_recurrent_kda(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_recurrent_kda(
            q, k, v, g, beta,
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

    o, final_state = fused_recurrent_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=False,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state
