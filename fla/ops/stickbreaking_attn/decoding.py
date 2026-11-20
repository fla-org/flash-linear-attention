# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.stickbreaking_attn.parallel import _stickbreaking_attn_weights
from fla.ops.utils.op import exp2
from fla.utils import autocast_custom_fwd, autotune_cache_kwargs, check_shared_mem, input_guard


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens_k'] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'HQ', 'G', 'K', 'V', 'BS', 'BK', 'BV', 'ATTEND_CURRENT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['TQ', 'T'])
def stickbreaking_attn_decoding_fwd_kernel(
    q,
    k,
    v,
    o,
    rem,
    scale,
    cu_seqlens_q,
    cu_seqlens_k,
    sequence_ids,
    TQ,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    ATTEND_CURRENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_th, i_v = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_t, i_hq = i_th // HQ, i_th % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n = tl.load(sequence_ids + i_t).to(tl.int64)
        eos_q = tl.load(cu_seqlens_q + i_n + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens_k + i_n).to(tl.int64), tl.load(cu_seqlens_k + i_n + 1).to(tl.int64)
    else:
        i_n = i_t // TQ
        eos_q = i_n * TQ + TQ
        bos, eos = i_n * T, i_n * T + T
    # the queries are the last tokens of their sequence: each sees every key before it, and its own with ATTEND_CURRENT
    TK = eos - bos - (eos_q - i_t)
    if ATTEND_CURRENT:
        TK += 1
    RCP_LN2: tl.constexpr = 1.4426950216

    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    # [BK]
    b_q = tl.load(q + i_th * K + o_d, mask=o_d < K, other=0.).to(tl.float32)
    # [BV]
    b_o = tl.zeros([BV], dtype=tl.float32)
    b_log_rem = tl.zeros([], dtype=tl.float32)

    # visit nearer blocks first to carry their remaining stick into earlier blocks
    for i_s in range(0, TK, BS):
        # [BS]
        o_k = TK - BS - i_s + tl.arange(0, BS)
        m_k = o_k >= 0
        p_k = k + (bos * H + i_h) * K + o_k[:, None] * (H*K) + o_d[None, :]
        p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]
        # [BS, BK]
        b_k = tl.load(p_k, mask=m_k[:, None] & (o_d[None, :] < K), other=0.)
        # [BS, BV]
        b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.)

        # [BS]
        b_s = tl.sum(b_q[None, :] * b_k.to(tl.float32), 1) * scale * RCP_LN2
        b_log_om_beta, b_p = _stickbreaking_attn_weights(b_s=b_s, b_log_rem=b_log_rem, m_s=m_k, AXIS=0)
        # the weights are rounded to the value dtype, as for the dot in the parallel kernel
        b_o += tl.sum(b_p.to(b_v.dtype).to(tl.float32)[:, None] * b_v.to(tl.float32), 0)
        b_log_rem += tl.sum(b_log_om_beta, 0)

    tl.store(o + i_th * V + o_v, b_o.to(o.dtype.element_ty), mask=o_v < V)
    if i_v == 0:
        tl.store(rem + i_th, exp2(b_log_rem).to(rem.dtype.element_ty))


class StickBreakingAttentionDecodingFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, scale, attend_current, cu_seqlens_q, cu_seqlens_k):
        T, H, K, V = *k.shape[1:], v.shape[-1]
        HQ = q.shape[2]
        BS = 64 if check_shared_mem('hopper', q.device.index) else 32
        BK = max(16, triton.next_power_of_2(K))
        BV = min(128, max(16, triton.next_power_of_2(V)))
        NV = triton.cdiv(V, BV)

        sequence_ids = None
        if cu_seqlens_k is not None:
            # the sequence of each query token, read from the query offsets without a device sync
            o_t = torch.arange(q.shape[1], dtype=cu_seqlens_q.dtype, device=q.device)
            sequence_ids = torch.searchsorted(cu_seqlens_q[1:], o_t, right=True)
        o = q.new_empty(*q.shape[:-1], V)
        rem = q.new_empty(q.shape[:-1])
        # query tokens and heads go on the first grid axis, which is not limited to 65535 programs
        grid = (q.shape[0] * q.shape[1] * HQ, NV)
        stickbreaking_attn_decoding_fwd_kernel[grid](
            q=q,
            k=k,
            v=v,
            o=o,
            rem=rem,
            scale=scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            sequence_ids=sequence_ids,
            TQ=q.shape[1],
            T=T,
            H=H,
            HQ=HQ,
            G=HQ // H,
            K=K,
            V=V,
            BS=BS,
            BK=BK,
            BV=BV,
            ATTEND_CURRENT=attend_current,
        )
        return o, rem

    @staticmethod
    @input_guard
    def backward(ctx, do, drem):
        raise NotImplementedError("Stick-breaking attention decoding is forward only.")


def stickbreaking_attn_decoding(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    attend_current: bool = False,
    cu_seqlens_q: torch.LongTensor | None = None,
    cu_seqlens_k: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Stick-breaking attention for new queries over cached keys and values.

    The queries of each sequence are its last tokens, whose keys and values are already in the cache,
    so the result matches the same rows of `parallel_stickbreaking_attn` over the whole sequence.
    Query and key/value lengths are independent, and with `cu_seqlens_q` and `cu_seqlens_k` they can differ per sequence.
    This is a forward-only inference API: backward raises `NotImplementedError`.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, TQ, HQ, K]`, or `[1, TQ, HQ, K]` packed with `cu_seqlens_q`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]` with `TQ <= T`, or `[1, T, H, K]` packed with `cu_seqlens_k`.
            GQA is applied if HQ is divisible by H.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`, or `[1, T, H, V]` packed with `cu_seqlens_k`.
        scale (float, Optional):
            Scale factor for attention scores. Default: `1 / sqrt(K)`.
        attend_current (bool, Optional):
            Whether a query also attends to the key of its own token. Default: `False`.
        cu_seqlens_q (torch.LongTensor, Optional):
            Cumulative query lengths of shape `[N+1]`, given together with `cu_seqlens_k`.
            Each sequence must have at most as many queries as keys. Default: `None`.
        cu_seqlens_k (torch.LongTensor, Optional):
            Cumulative key/value lengths of shape `[N+1]`, given together with `cu_seqlens_q`. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, TQ, HQ, V]`.
        rem (torch.Tensor):
            The stick each query leaves unused, of shape `[B, TQ, HQ]`.
    """
    if cu_seqlens_q is None and cu_seqlens_k is None:
        valid = q.ndim == 4 and k.ndim == 4 and q.shape[0] == k.shape[0] and q.shape[1] <= k.shape[1]
    else:
        valid = (
            cu_seqlens_q is not None and cu_seqlens_k is not None and len(cu_seqlens_q) == len(cu_seqlens_k)
            and q.ndim == 4 and k.ndim == 4 and q.shape[0] == k.shape[0] == 1
        )
    if not valid or v.ndim != 4 or k.shape[:3] != v.shape[:3] or q.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"Expected q, k and v of shapes [B, TQ, HQ, K], [B, T, H, K] and [B, T, H, V] with TQ <= T, "
            f"or [1, TQ, HQ, K], [1, T, H, K] and [1, T, H, V] with both cu_seqlens_q and cu_seqlens_k of N+1 offsets, "
            f"got {tuple(q.shape)}, {tuple(k.shape)} and {tuple(v.shape)}.",
        )
    HQ, H = q.shape[2], k.shape[2]
    if H == 0 or HQ % H != 0:
        raise ValueError(f"The number of query heads ({HQ}) must be divisible by the number of key/value heads ({H}).")
    if q.shape[-1] > 256:
        raise ValueError(f"The key dimension ({q.shape[-1]}) can not be larger than 256.")
    if v.shape[-1] > 256:
        raise ValueError(f"The value dimension ({v.shape[-1]}) can not be larger than 256.")
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, rem = StickBreakingAttentionDecodingFunction.apply(q, k, v, scale, attend_current, cu_seqlens_q, cu_seqlens_k)
    return o, rem
