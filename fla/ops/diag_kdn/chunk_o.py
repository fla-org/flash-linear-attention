# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import IS_NVIDIA_HOPPER, IS_TF32_SUPPORTED, autotune_cache_kwargs

if IS_TF32_SUPPORTED:
    DOT_PRECISION = tl.constexpr('tf32x3')
    DOT_PRECISION_FAST = tl.constexpr('tf32')
else:
    DOT_PRECISION = tl.constexpr('ieee')
    DOT_PRECISION_FAST = tl.constexpr('ieee')

NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_diag_kdn_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    q += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h) * K*V
    A += (bos * H + i_h) * BT

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    m_t = o_t < T
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]

    # the readout runs at tf32 on fp32 operands, both across and within chunks
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        b_q = tl.load(q + o_t[:, None] * (H*K) + o_k[None, :], mask=m_tk, other=0.).to(tl.float32)
        b_g = tl.load(g + o_t[:, None] * (H*K) + o_k[None, :], mask=m_tk, other=0.).to(tl.float32)
        b_h = tl.load(h + o_k[:, None] * V + o_v[None, :], mask=m_k[:, None] & m_v[None, :], other=0.).to(tl.float32)
        b_o += tl.dot(b_q * exp2(b_g), b_h, input_precision=DOT_PRECISION_FAST)
    b_o *= scale

    b_v = tl.load(v + o_t[:, None] * (H*V) + o_v[None, :], mask=m_tv, other=0.).to(tl.float32)
    b_A = tl.load(A + o_t[:, None] * (H*BT) + o_i[None, :], mask=m_t[:, None], other=0.).to(tl.float32)
    b_A = tl.where(o_i[:, None] >= o_i[None, :], b_A, 0.)
    b_o += tl.dot(b_A, b_v, input_precision=DOT_PRECISION_FAST)
    tl.store(o + o_t[:, None] * (H*V) + o_v[None, :], b_o.to(o.dtype.element_ty), mask=m_tv)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'V', 'BT', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_diag_kdn_bwd_kernel_dAv(
    v,
    A,
    do,
    dv,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    A += (bos * H + i_h) * BT
    dA += (bos * H + i_h) * BT

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    m_t = o_t < T
    # [BT, BT], the transposed causal scores
    b_A = tl.load(A + o_t[None, :] * (H*BT) + o_i[:, None], mask=m_t[None, :], other=0.).to(tl.float32)
    b_A = tl.where((o_i[:, None] <= o_i[None, :]) & (m_t[:, None] & m_t[None, :]), b_A, 0.)
    b_A = b_A.to(do.dtype.element_ty)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        # [BV, BT]
        b_v = tl.load(v + o_v[:, None] + o_t[None, :] * (H*V), mask=m_v[:, None] & m_t[None, :], other=0.).to(tl.float32)
        # [BT, BV]
        b_do = tl.load(do + o_t[:, None] * (H*V) + o_v[None, :], mask=m_tv, other=0.)
        # the score gradient runs at tf32x3 on fp32 operands, the value gradient on operands in the input dtype
        b_dA += tl.dot(b_do.to(tl.float32), b_v, input_precision=DOT_PRECISION)
        b_dv = tl.dot(b_A, b_do)
        tl.store(dv + o_t[:, None] * (H*V) + o_v[None, :], b_dv.to(dv.dtype.element_ty), mask=m_tv)
    b_dA = tl.where(o_i[:, None] >= o_i[None, :], b_dA * scale, 0.)
    tl.store(dA + o_t[:, None] * (H*BT) + o_i[None, :], b_dA.to(dA.dtype.element_ty), mask=m_t[:, None])


def chunk_diag_kdn_fwd_o(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 32,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.empty_like(v, dtype=torch.float)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_diag_kdn_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return o


def chunk_diag_kdn_bwd_dAv(
    v: torch.Tensor,
    do: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 32,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, V = v.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BV = min(max(triton.next_power_of_2(V), 16), 128)

    dA = v.new_empty(B, T, H, BT, dtype=torch.float)
    dv = torch.empty_like(do, dtype=torch.float)
    chunk_diag_kdn_bwd_kernel_dAv[(NT, B * H)](
        v=v,
        A=A,
        do=do,
        dv=dv,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        V=V,
        BT=BT,
        BV=BV,
    )
    return dA, dv
