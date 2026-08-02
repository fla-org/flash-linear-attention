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
from fla.ops.utils.op import exp
from fla.utils import check_shared_mem, is_nvidia_hopper

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT']
)
@triton.jit(do_not_specialize=['T'])
def chunk_oja_fwd_inter(
    q,
    k,
    h,
    gv,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    o_t = i_t * BT + tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < T
    m_tv = m_t[:, None] & (o_v[None, :] < V)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k[None, :] < K)
        m_kt = (o_k[:, None] < K) & m_t[None, :]
        m_kv = (o_k[:, None] < K) & (o_v[None, :] < V)
        p_q = q + (bos * HQ + i_hq) * K + o_t[:, None] * (HQ*K) + o_k[None, :]
        p_k = k + (bos * H + i_h) * K + o_k[:, None] + o_t[None, :] * (H*K)
        p_h = h + (i_tg * H + i_h) * K*V + o_k[:, None] * V + o_v[None, :]

        # [BT, BK]
        b_q = tl.load(p_q, mask=m_tk, other=0.0)
        b_q = (b_q * scale).to(b_q.dtype)
        # [BK, BT]
        b_k = tl.load(p_k, mask=m_kt, other=0.0)
        # [BK, BV]
        b_h = tl.load(p_h, mask=m_kv, other=0.0)
        # [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BT]
        b_A += tl.dot(b_q, b_k)
    p_g = gv + (bos * H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
    p_o = o + (bos * HQ + i_hq) * V + o_t[:, None] * (HQ*V) + o_v[None, :]
    o_A = tl.arange(0, BT)
    m_AT = m_t[:, None] & (o_A[None, :] < BT)
    p_A = A + (bos * HQ + i_hq) * BT + o_t[:, None] * (HQ*BT) + o_A[None, :]
    # [BT, BV]
    b_g = tl.load(p_g, mask=m_tv, other=0.0)
    b_o = b_o * exp(b_g)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_tv)

    # [BT, BT]
    b_A = tl.where(m_s, b_A, 0.)
    if i_v == 0:
        tl.store(p_A, b_A.to(p_A.dtype.element_ty), mask=m_AT)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def chunk_oja_fwd_intra(
    v,
    gv,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    i_t, i_i = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if i_t * BT + i_i * BC >= T:
        return

    o_r = i_t * BT + i_i * BC + tl.arange(0, BC)
    m_rv = (o_r[:, None] < T) & (o_v[None, :] < V)
    p_g = gv + (bos * H + i_h) * V + o_r[:, None] * (H*V) + o_v[None, :]
    p_gn = gv + (bos + min(i_t * BT + i_i * BC, T)) * H*V + i_h * V + o_v
    # [BV,]
    b_gn = tl.load(p_gn, mask=m_v, other=0)
    # [BC, BV]
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(0, i_i):
        o_j = i_t * BT + i_j * BC + tl.arange(0, BC)
        o_Aj = i_j * BC + tl.arange(0, BC)
        m_jv = (o_j[:, None] < T) & (o_v[None, :] < V)
        m_A = (o_r[:, None] < T) & (o_Aj[None, :] < BT)
        p_A = A + (bos*HQ+i_hq) * BT + o_r[:, None] * (HQ*BT) + o_Aj[None, :]
        p_v = v + (bos*H+i_h) * V + o_j[:, None] * (H*V) + o_v[None, :]
        p_gv = gv + (bos*H+i_h) * V + o_j[:, None] * (H*V) + o_v[None, :]
        # [BC, BV]
        b_v = tl.load(p_v, mask=m_jv, other=0.0)
        b_gv = tl.load(p_gv, mask=m_jv, other=0.0)
        b_vg = (b_v * exp(b_gn[None, :] - b_gv)).to(b_v.dtype)
        # [BC, BC]
        b_A = tl.load(p_A, mask=m_A, other=0.0)
        b_o += tl.dot(b_A, b_vg)
    # [BC, BV]
    b_g = tl.load(p_g, mask=m_rv, other=0.0)
    b_o *= exp(b_g - b_gn[None, :])

    o_i = tl.arange(0, BC)
    o_A = (bos + i_t * BT + i_i * BC + tl.arange(0, BC)) * HQ*BT + i_hq * BT + i_i * BC
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        p_vj = v + (bos + i_t * BT + i_i * BC + j) * H*V + i_h * V + o_v
        p_gvj = gv + (bos + i_t * BT + i_i * BC + j) * H*V + i_h * V + o_v
        # [BC,]
        b_A = tl.load(A + o_A + j, mask=m_A, other=0)
        # [BV,]
        b_v = tl.load(p_vj, mask=m_v, other=0).to(tl.float32)
        b_gv = tl.load(p_gvj, mask=m_v, other=0).to(tl.float32)
        # [BC, BV]
        b_vg = b_v[None, :] * exp(b_g - b_gv[None, :])
        # avoid 0 * inf = inf
        b_o += tl.where(o_i[:, None] >= j, b_A[:, None] * b_vg, 0.)
    p_o = o + (bos*HQ + i_hq) * V + o_r[:, None] * (HQ*V) + o_v[None, :]
    b_o += tl.load(p_o, mask=m_rv, other=0.0)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_rv)


def chunk_oja_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    h: torch.Tensor,
    scale: float = 1.,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)
    BV = min(64, triton.next_power_of_2(V))
    HQ = q.shape[2]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NG = HQ // H

    o = v.new_empty(B, T, HQ, V)
    A = q.new_empty(B, T, HQ, BT)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * HQ)
    chunk_oja_fwd_inter[grid](
        q,
        k,
        h,
        gv,
        o,
        A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        NG=NG,
    )

    def grid(meta): return (triton.cdiv(V, meta['BV']), NT * NC, B * HQ)
    chunk_oja_fwd_intra[grid](
        v,
        gv,
        o,
        A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        HQ=HQ,
        H=H,
        V=V,
        BT=BT,
        BC=BC,
        BV=BV,
        NC=NC,
        NG=NG,
        num_warps=4,
        num_stages=2
    )
    return A, o


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
    ],
    key=["BT"]
)
@triton.jit(do_not_specialize=['T'])
def chunk_oja_bwd_kernel_dA(
    v,
    gv,
    do,
    dA,
    chunk_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i, i_j = i_c // (NC * NC), (i_c % (NC * NC)) // NC, (i_c % (NC * NC)) % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if i_t * BT + i_i * BC >= T:
        return

    # [BC, BC]
    b_dA = tl.zeros([BC, BC], dtype=tl.float32)
    if i_i > i_j:
        o_r = i_t * BT + i_i * BC + tl.arange(0, BC)
        o_kj = i_t * BT + i_j * BC + tl.arange(0, BC)
        m_rv = (o_r[:, None] < T) & (o_v[None, :] < V)
        m_vj = (o_v[:, None] < V) & (o_kj[None, :] < T)
        p_v = v + (bos*H+i_h) * V + o_v[:, None] + o_kj[None, :] * (H*V)
        p_gv = gv + (bos*H+i_h) * V + o_v[:, None] + o_kj[None, :] * (H*V)
        p_gn = gv + (bos + i_t*BT + i_i*BC) * H*V + i_h * V + o_v
        p_g = gv + (bos*H+i_h) * V + o_r[:, None] * (H*V) + o_v[None, :]
        p_do = do + (bos*H+i_h) * V + o_r[:, None] * (H*V) + o_v[None, :]
        # [BV,]
        b_gn = tl.load(p_gn, mask=m_v, other=0.)
        # [BC, BV]
        b_g = tl.load(p_g, mask=m_rv, other=0.0)
        b_do = tl.load(p_do, mask=m_rv, other=0.0)
        b_do = (b_do * exp(b_g - b_gn[None, :]) * scale).to(b_do.dtype)
        # [BV, BC]
        b_v = tl.load(p_v, mask=m_vj, other=0.0)
        b_gv = tl.load(p_gv, mask=m_vj, other=0.0)
        b_vg = (b_v * exp(b_gn[:, None] - b_gv)).to(b_v.dtype)
        # [BC, BC]
        b_dA = tl.dot(b_do, b_vg)
    elif i_i == i_j:
        o_r = i_t * BT + i_i * BC + tl.arange(0, BC)
        m_rv = (o_r[:, None] < T) & (o_v[None, :] < V)
        p_g = gv + (bos*H + i_h) * V + o_r[:, None] * (H*V) + o_v[None, :]
        p_do = do + (bos*H + i_h) * V + o_r[:, None] * (H*V) + o_v[None, :]
        p_vj = v + (bos + i_t*BT + i_j*BC) * H*V + i_h * V + o_v
        p_gvj = gv + (bos + i_t*BT + i_j*BC) * H*V + i_h * V + o_v
        # [BC, BV]
        b_g = tl.load(p_g, mask=m_rv, other=0.0)
        b_do = tl.load(p_do, mask=m_rv, other=0.0) * scale
        m_v = o_v < V

        o_i = tl.arange(0, BC)
        # [BC, BC]
        m_dA = o_i[:, None] >= o_i[None, :]
        for j in range(0, min(BC, T - i_t * BT - i_j * BC)):
            # [BV,]
            b_v = tl.load(p_vj, mask=m_v, other=0).to(tl.float32)
            b_gv = tl.load(p_gvj, mask=m_v, other=0).to(tl.float32)
            # [BC,]
            b_dAj = tl.sum(b_do * b_v[None, :] * exp(b_g - b_gv[None, :]), 1)
            b_dA = tl.where((o_i == j)[None, :], b_dAj[:, None], b_dA)

            p_vj += H*V
            p_gvj += H*V
        b_dA = tl.where(m_dA, b_dA, 0.)

    o_r = i_t * BT + i_i * BC + tl.arange(0, BC)
    o_Aj = i_j * BC + tl.arange(0, BC)
    p_dA = dA+((i_v*all+bos)*H+i_h)*BT + o_r[:, None] * (H*BT) + o_Aj[None, :]
    tl.store(p_dA, b_dA.to(dA.dtype.element_ty), mask=(o_r[:, None] < T) & (o_Aj[None, :] < BT))


def chunk_oja_bwd_dA(
    v: torch.Tensor,
    gv: torch.Tensor,
    do: torch.Tensor,
    scale: float = 1.,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64
):
    B, T, H, V = v.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)
    BV = min(64, triton.next_power_of_2(V))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NV = triton.cdiv(V, BV)

    dA = v.new_empty(NV, B, T, H, BT)
    # 计算dA
    grid = (NV, NT * NC * NC, B * H)
    chunk_oja_bwd_kernel_dA[grid](
        v,
        gv,
        do,
        dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        V=V,
        BT=BT,
        BC=BC,
        BV=BV,
        NC=NC,
    )
    dA = dA.sum(0, dtype=dA.dtype)

    return dA


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['BT']
)
@triton.jit(do_not_specialize=['T'])
def chunk_oja_bwd_kernel_dqk(
    q,
    k,
    h,
    gv,
    A,
    dq,
    dk,
    dA,
    do,
    scale,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
        all = tl.cast(B * T, tl.int64)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    m_t = o_t < T
    m_tk = m_t[:, None] & (o_k[None, :] < K)
    o_A = tl.arange(0, BT)
    m_AT = m_t[:, None] & (o_A[None, :] < BT)
    # [B, T, H, BT]
    p_q = q + (bos*H+i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
    p_k = k + (bos*H+i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
    p_A = A + ((i_k*all+bos)*H+i_h)*BT + o_t[:, None] * (H*BT) + o_A[None, :]
    b_q = tl.load(p_q, mask=m_tk, other=0.0)
    b_k = tl.load(p_k, mask=m_tk, other=0.0)

    b_A = tl.dot((b_q * scale).to(b_q.dtype), tl.trans(b_k))
    b_A = tl.where(m_s, b_A, 0.)
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), mask=m_AT)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)

    # 先计算do对应的dq
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_h = (o_v[:, None] < V) & (o_k[None, :] < K)
        m_tv = m_t[:, None] & (o_v[None, :] < V)
        p_h = h + (i_tg * H + i_h) * K*V + o_v[:, None] + o_k[None, :] * V
        p_do = do + (bos*H+i_h)*V + o_t[:, None] * (H*V) + o_v[None, :]
        p_gv = gv + (bos*H+i_h)*V + o_t[:, None] * (H*V) + o_v[None, :]
        b_h = tl.load(p_h, mask=m_h, other=0.0)
        b_do = tl.load(p_do, mask=m_tv, other=0.0)
        b_gv = tl.load(p_gv, mask=m_tv, other=0.0)
        b_do = (b_do * exp(b_gv) * scale).to(b_do.dtype)
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))

    # 接着计算dA对应的dq, dk
    p_dA = dA + (bos*H + i_h) * BT + o_t[:, None] * (H*BT) + o_A[None, :]
    p_dq = dq + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
    p_dk = dk + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
    # [BT, BT]
    b_dA = tl.load(p_dA, mask=m_AT, other=0.0)
    # [BT, BK]
    b_dq += tl.dot(b_dA.to(b_q.dtype), b_k)
    b_dk = tl.dot(tl.trans(b_dA).to(b_q.dtype), b_q)

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_tk)
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_tk)


def chunk_oja_bwd_dqk(
    q: torch.Tensor,
    k: torch.Tensor,
    h: torch.Tensor,
    gv: torch.Tensor,
    dA: torch.Tensor,
    do: torch.Tensor,
    scale: float = 1.,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64
):
    B, T, H, K, V = *q.shape, gv.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BK = min(64, triton.next_power_of_2(K))
    BV = min(64, triton.next_power_of_2(V))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NK = triton.cdiv(K, BK)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    A = dA.new_empty(NK, B, T, H, BT)
    # 计算dA
    grid = (NK, NT, B * H)
    chunk_oja_bwd_kernel_dqk[grid](
        q,
        k,
        h,
        gv,
        A,
        dq,
        dk,
        dA,
        do,
        scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV
    )

    A = A.sum(0, dtype=A.dtype)

    return A, dq, dk


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def chunk_oja_bwd_kernel_dv_o(
    v,
    g,
    o,
    A,
    do,
    dv,
    dv2,
    dg,
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
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if i_t * BT + i_i * BC >= T:
        return

    o_r = i_t * BT + i_i * BC + tl.arange(0, BC)
    m_rv = (o_r[:, None] < T) & (o_v[None, :] < V)
    p_gv = g + (bos*H+i_h)*V + o_r[:, None] * (H*V) + o_v[None, :]
    p_gn = g + (bos + min(i_t * BT + i_i * BC + BC, T)-1)*H*V + i_h*V + o_v
    # [BV,]
    b_gn = tl.load(p_gn, mask=m_v, other=0)
    # [BC, BV]
    b_gv = tl.load(p_gv, mask=m_rv, other=0.0)
    b_dvg = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        o_j = i_t * BT + i_j * BC + tl.arange(0, BC)
        o_Ai = i_i * BC + tl.arange(0, BC)
        m_jv = (o_j[:, None] < T) & (o_v[None, :] < V)
        m_A = (o_Ai[:, None] < BT) & (o_j[None, :] < T)
        p_g = g + (bos*H+i_h) * V + o_j[:, None] * (H*V) + o_v[None, :]
        p_A = A + (bos*H+i_h) * BT + o_Ai[:, None] + o_j[None, :] * (H*BT)
        p_do = do + (bos*H+i_h) * V + o_j[:, None] * (H*V) + o_v[None, :]
        # [BC, BV]
        b_g = tl.load(p_g, mask=m_jv, other=0.0)
        b_do = tl.load(p_do, mask=m_jv, other=0.0) * exp(b_g - b_gn[None, :])
        # [BC, BC]
        b_A = tl.load(p_A, mask=m_A, other=0.0)
        # [BC, BV]
        b_dvg += tl.dot(b_A, b_do.to(b_A.dtype))
    b_dv = b_dvg * exp(b_gn[None, :] - b_gv)

    o_i = tl.arange(0, BC)
    o_c = i_i * BC + tl.arange(0, BC)

    p_g = g + (bos + i_t * BT + i_i * BC) * H*V + i_h * V + o_v
    p_A = A + (bos + i_t*BT + i_i*BC) * H*BT + i_h * BT + o_c
    p_do = do + (bos + i_t*BT + i_i*BC) * H*V + i_h * V + o_v
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_A = tl.load(p_A)
        # [BV,]
        b_g = tl.load(p_g, mask=m_v, other=0)
        b_do = tl.load(p_do, mask=m_v, other=0)
        # [BC, BV]
        m_i = o_i[:, None] <= j
        b_dv += tl.where(m_i, exp(b_g[None, :] - b_gv) * b_A[:, None] * b_do[None, :], 0.)

        p_g += H * V
        p_A += H * BT
        p_do += H * V
    p_o = o + (bos*H+i_h)*V + o_r[:, None] * (H*V) + o_v[None, :]
    p_v = v + (bos*H+i_h)*V + o_r[:, None] * (H*V) + o_v[None, :]
    p_do = do + (bos*H+i_h)*V + o_r[:, None] * (H*V) + o_v[None, :]
    p_dv = dv + (bos*H+i_h)*V + o_r[:, None] * (H*V) + o_v[None, :]
    p_dv2 = dv2 + (bos*H+i_h)*V + o_r[:, None] * (H*V) + o_v[None, :]
    p_dg = dg + (bos*H+i_h)*V + o_r[:, None] * (H*V) + o_v[None, :]

    b_o = tl.load(p_o, mask=m_rv, other=0.0).to(tl.float32)
    b_v = tl.load(p_v, mask=m_rv, other=0.0).to(tl.float32)
    b_do = tl.load(p_do, mask=m_rv, other=0.0).to(tl.float32)
    b_dv = b_dv + tl.load(p_dv, mask=m_rv, other=0.0).to(tl.float32)
    b_dg = b_o * b_do - b_v * b_dv
    tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), mask=m_rv)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_rv)


def chunk_oja_bwd_dv_o(
    v: torch.Tensor,
    gv: torch.Tensor,
    o: torch.Tensor,
    A: torch.Tensor,
    dv: torch.Tensor,
    do: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64
):
    B, T, H, V = v.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)
    BV = min(64, triton.next_power_of_2(V))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)

    dv2 = torch.empty_like(v, dtype=torch.float)
    dgv = torch.empty_like(gv)
    # 计算dA
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT * NC, B * H)
    chunk_oja_bwd_kernel_dv_o[grid](
        v=v,
        g=gv,
        o=o,
        A=A,
        do=do,
        dv=dv,
        dv2=dv2,
        dg=dgv,
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
        num_stages=2
    )
    return dv2, dgv
