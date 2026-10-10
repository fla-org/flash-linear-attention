# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import autotune_cache_kwargs, input_guard


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
    ],
    key=['BK', 'BV', 'USE_Q_L2NORM', 'USE_K_L2NORM'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['B', 'T'])
def fused_recurrent_oja2_fwd_kernel(
    q,
    k,
    v,
    gv,
    b,
    c,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BKF: tl.constexpr,
    USE_Q_L2NORM: tl.constexpr,
    USE_K_L2NORM: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # each program owns BK rows of the [K, V] state with every slot: the erase read reduces over the slots of a row,
    # so rows are independent, and only the output, a reduction over K, is left as one partial sum per row block
    pid = tl.program_id(0).to(tl.int64)
    NK = tl.cdiv(K, BK)
    i_k, i_nh = pid % NK, pid // NK
    i_n, i_h = i_nh // H, i_nh % H
    all = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    o_k = i_k * BK + tl.arange(0, BK)
    o_kf = tl.arange(0, BKF)
    o_v = tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_qf = q + (bos * H + i_h) * K + o_kf
    p_kf = k + (bos * H + i_h) * K + o_kf
    p_c = c + (bos * H + i_h) * K + o_k
    p_v = v + (bos * H + i_h) * V + o_v
    p_gv = gv + (bos * H + i_h) * V + o_v
    p_b = b + (bos * H + i_h) * V + o_v
    p_o = o + ((i_k * all + bos) * H + i_h) * V + o_v

    mask_k = o_k < K
    mask_kf = o_kf < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        # the norms run over the whole head dimension, not just this block of rows
        if USE_Q_L2NORM:
            b_qf = tl.load(p_qf, mask=mask_kf, other=0).to(tl.float32)
            b_q = b_q / tl.sqrt(tl.sum(b_qf * b_qf) + 1e-6)
        if USE_K_L2NORM:
            b_kf = tl.load(p_kf, mask=mask_kf, other=0).to(tl.float32)
            b_k = b_k / tl.sqrt(tl.sum(b_kf * b_kf) + 1e-6)
        b_q = b_q * scale
        b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=mask_v, other=0).to(tl.float32)
        b_c = tl.load(p_c, mask=mask_k, other=0).to(tl.float32)

        # [BK, BV]
        b_h *= exp(b_gv[None, :])

        # read the state along the gated erase direction, then write (c * k) minus what was read
        # [BK]
        b_erase = tl.sum(b_h * (b_b * b_v)[None, :], 1)
        b_write = b_c * b_k - b_erase
        b_h += b_write[:, None] * b_v[None, :]

        # [BV], the contribution of this block of rows
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H*K
        p_k += H*K
        p_qf += H*K
        p_kf += H*K
        p_c += H*K
        p_v += H*V
        p_gv += H*V
        p_b += H*V
        p_o += H*V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_oja2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_q_l2norm: bool = False,
    use_k_l2norm: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    if V > 128:
        raise ValueError(f"The number of slots must be at most 128 for the fused recurrent kernel, got {V}.")
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    # every program keeps all slots of its rows, so the state tile is [BK, BV] with BV covering the whole slot axis
    BKF, BV = triton.next_power_of_2(K), triton.next_power_of_2(V)
    BK = min(BKF, 32)
    NK = triton.cdiv(K, BK)

    o = q.new_empty(NK, *v.shape, dtype=torch.float32)
    final_state = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    grid = (NK * N * H,)
    fused_recurrent_oja2_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        gv=gv,
        b=b,
        c=c,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        BKF=BKF,
        USE_Q_L2NORM=use_q_l2norm,
        USE_K_L2NORM=use_k_l2norm,
    )
    o = o.sum(0).to(v.dtype)
    return o, final_state


class FusedRecurrentOja2Function(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gv: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        use_q_l2norm: bool,
        use_k_l2norm: bool,
        cu_seqlens: torch.LongTensor | None,
    ):
        o, final_state = fused_recurrent_oja2_fwd(
            q=q,
            k=k,
            v=v,
            gv=gv,
            b=b,
            c=c,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_q_l2norm=use_q_l2norm,
            use_k_l2norm=use_k_l2norm,
            cu_seqlens=cu_seqlens,
        )

        return o, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Backward pass is not implemented for the fused recurrent Oja2 kernel. "
            "Use `chunk_gated_oja_rule2` for training."
        )


def fused_recurrent_gated_oja_rule2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_q_l2norm: bool = False,
    use_k_l2norm: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Token-by-token Oja2, the inference counterpart of `chunk_gated_oja_rule2`. Forward only.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            slot code of shape `[B, T, H, M]`, with `M <= 128`.
        gv (torch.Tensor):
            log-decay of shape `[B, T, H, M]`.
        b (torch.Tensor):
            per-slot erase gate of shape `[B, T, H, M]`.
        c (torch.Tensor):
            per-channel write gate of shape `[B, T, H, K]`.
        scale (float, Optional):
            Scale factor for the attention scores. If not provided, it defaults to `1 / sqrt(K)`.
        initial_state (torch.Tensor, Optional):
            Initial state of shape `[N, H, K, M]` for `N` input sequences. Default: `None`.
        output_final_state (bool, Optional):
            Whether to output the final state of shape `[N, H, K, M]`. Default: `False`.
        use_q_l2norm (bool, Optional):
            Whether to L2-normalize `q` inside the kernel. Default: `False`.
        use_k_l2norm (bool, Optional):
            Whether to L2-normalize `k` inside the kernel. Default: `False`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length inputs.
            Requires `B = 1`. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, M]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, M]` if `output_final_state=True` else `None`.
    """
    if 'use_qk_l2norm_in_kernel' in kwargs and (not use_q_l2norm and not use_k_l2norm):
        use_q_l2norm = True
        use_k_l2norm = True

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    assert q.shape == k.shape, "q and k must have the same shape."
    assert v.shape == gv.shape == b.shape, "v, gv and b must have the same shape."
    assert c.shape == k.shape, "c and k must have the same shape."
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state = FusedRecurrentOja2Function.apply(
        q,
        k,
        v,
        gv,
        b,
        c,
        scale,
        initial_state,
        output_final_state,
        use_q_l2norm,
        use_k_l2norm,
        cu_seqlens,
    )
    return o, final_state
