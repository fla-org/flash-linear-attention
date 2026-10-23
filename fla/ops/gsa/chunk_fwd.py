# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_h import chunk_fwd_h
from fla.ops.gla.chunk import chunk_gla_fwd
from fla.ops.utils import prepare_chunk_indices, softmax_fwd
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2, unflatten_program_id
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gsa_fwd_kernel_inter(
    q,
    k,
    h,
    g,
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
    i_v, i_t = unflatten_program_id(tl.cdiv(V, BV))
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV).to(tl.int64)
    m_t = o_t < T
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK).to(tl.int64)
        m_k = o_k < K
        m_qk = m_t[:, None] & m_k[None, :]
        m_kt = m_k[:, None] & m_t[None, :]
        m_kv = m_k[:, None] & m_v[None, :]
        p_q = q + (bos * H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_k = k + (bos * H + i_h) * K + o_k[:, None] + o_t[None, :] * (H*K)
        p_h = h + (i_tg * H + i_h) * K*V + o_k[:, None] * V + o_v[None, :]

        # [BT, BK]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        b_q = (b_q * scale).to(b_q.dtype)
        # [BK, BT]
        b_k = tl.load(p_k, mask=m_kt, other=0.0)
        # [BK, BV]
        b_h = tl.load(p_h, mask=m_kv, other=0.0)
        # [BT, BV]
        b_o = tl.dot(b_q, b_h, b_o)
        # [BT, BT]
        b_A = tl.dot(b_q, b_k, b_A)
    m_A = m_t[:, None] & (o_i[None, :] < BT)
    p_g = g + (bos * H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
    p_o = o + (bos * H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
    p_A = A + (bos * H + i_h) * BT + o_t[:, None] * (H*BT) + o_i[None, :]
    # [BT, BV]
    b_g = tl.load(p_g, mask=m_tv, other=0.0)
    b_o = b_o * exp2(b_g)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_tv)

    # [BT, BT]
    b_A = tl.where(m_s, b_A, 0.)
    if i_v == 0:
        tl.store(p_A, b_A.to(p_A.dtype.element_ty), mask=m_A)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_gsa_fwd_kernel_intra(
    v,
    g,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    NV = tl.cdiv(V, BV)
    i_v, i_c, i_bh = tl.program_id(0).to(tl.int64) % NV, tl.program_id(0).to(tl.int64) // NV, tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = (i_c // NC).to(tl.int64), i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BC)
    o_v = i_v * BV + tl.arange(0, BV).to(tl.int64)
    m_v = o_v < V

    if i_t * BT + i_i * BC >= T:
        return

    o_c = i_t * BT + i_i * BC + tl.arange(0, BC)
    m_c = o_c < T
    m_cv = m_c[:, None] & m_v[None, :]
    p_g = g + (bos * H + i_h) * V + o_c[:, None] * (H*V) + o_v[None, :]
    p_gn = g + (bos + min(i_t * BT + i_i * BC, T)) * H*V + i_h * V + o_v
    # [BV,]
    b_gn = tl.load(p_gn, mask=m_v, other=0)
    # [BC, BV]
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(0, i_i):
        o_j = i_t * BT + i_j * BC + tl.arange(0, BC)
        o_jA = i_j * BC + tl.arange(0, BC)
        m_jv = (o_j[:, None] < T) & m_v[None, :]
        m_A = m_c[:, None] & (o_jA[None, :] < BT)
        p_A = A + (bos*H+i_h) * BT + o_c[:, None] * (H*BT) + o_jA[None, :]
        p_v = v + (bos*H+i_h) * V + o_j[:, None] * (H*V) + o_v[None, :]
        p_gv = g + (bos*H+i_h) * V + o_j[:, None] * (H*V) + o_v[None, :]
        # [BC, BV]
        b_v = tl.load(p_v, mask=m_jv, other=0.0)
        b_gv = tl.load(p_gv, mask=m_jv, other=0.0)
        b_vg = (b_v * exp2(b_gn[None, :] - b_gv)).to(b_v.dtype)
        # [BC, BC]
        b_A = tl.load(p_A, mask=m_A, other=0.0)
        b_o = tl.dot(b_A, b_vg, b_o)
    # [BC, BV]
    b_g = tl.load(p_g, mask=m_cv, other=0.0)
    b_o *= exp2(b_g - b_gn[None, :])

    o_A = (bos + i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BT + i_h * BT + i_i * BC
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        p_v = v + (bos + i_t * BT + i_i * BC + j) * H*V + i_h * V + o_v
        p_gv = g + (bos + i_t * BT + i_i * BC + j) * H*V + i_h * V + o_v
        # [BC,]
        b_A = tl.load(A + o_A + j, mask=m_A, other=0)
        # [BV,]
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
        # [BC, BV]
        b_vg = b_v[None, :] * exp2(b_g - b_gv[None, :])
        # mask noncausal products before accumulating to avoid 0 * inf = NaN
        b_o += tl.where(o_i[:, None] >= j, b_A[:, None] * b_vg, 0.)
    p_o = o + (bos*H + i_h) * V + o_c[:, None] * (H*V) + o_v[None, :]
    b_o += tl.load(p_o, mask=m_cv, other=0.0)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_cv)


def chunk_gsa_fwd_k(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor | None = None,
    output_final_state: bool = False,
    scale: float = 1.,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BC = min(16, BT)
    BV = min(64, triton.next_power_of_2(V))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)

    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=None,
        gk=None,
        gv=g,
        h0=h0,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        states_in_fp32=False,
        chunk_indices=chunk_indices,
    )
    o = v.new_empty(B, T, H, V)
    A = q.new_empty(B, T, H, BT)
    def grid(meta): return (triton.cdiv(V, meta['BV']) * NT, B * H)
    chunk_gsa_fwd_kernel_inter[grid](
        q=q,
        k=k,
        h=h,
        g=g,
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

    def grid(meta): return (triton.cdiv(V, meta['BV']) * NT * NC, B * H)
    chunk_gsa_fwd_kernel_intra[grid](
        v=v,
        g=g,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        V=V,
        BT=BT,
        BC=BC,
        BV=BV,
        NC=NC,
        num_warps=4,
        num_stages=2,
    )
    return h, ht, o


def chunk_gsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    scale: float = 1.,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    hk, hkt, ok = chunk_gsa_fwd_k(
        q=q,
        k=k,
        v=s,
        g=g,
        h0=hk0,
        output_final_state=output_final_state,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    # p is kept in fp32 for safe softmax backward
    p = softmax_fwd(ok, dtype=torch.float)

    qv = p.to(q.dtype)
    _, Av, hv, hvt, ov = chunk_gla_fwd(
        q=qv,
        k=s,
        v=v,
        g=None,
        g_cumsum=g,
        scale=1.,
        initial_state=hv0,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    return hk, hkt, ok, p, Av, hv, hvt, ov
