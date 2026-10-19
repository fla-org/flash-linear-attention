# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla.ops.gla.chunk import chunk_gla_bwd, chunk_gla_fwd
from fla.ops.gsa.gate import fused_additive_gate
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices, softmax_bwd, softmax_fwd
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp2, unflatten_program_id
from fla.utils import autotune_cache_kwargs, input_guard


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


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
    ],
    key=['H', 'V', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gsa_bwd_kernel_dA(
    v,
    g,
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
    NV = tl.cdiv(V, BV)
    i_v, i_c, i_bh = tl.program_id(0).to(tl.int64) % NV, tl.program_id(0).to(tl.int64) // NV, tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i, i_j = (i_c // (NC * NC)).to(tl.int64), (i_c % (NC * NC)) // NC, (i_c % (NC * NC)) % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    o_v = i_v * BV + tl.arange(0, BV).to(tl.int64)
    m_v = o_v < V

    if i_t * BT + i_i * BC >= T:
        return

    o_c = i_t * BT + i_i * BC + tl.arange(0, BC)
    o_jA = i_j * BC + tl.arange(0, BC)
    m_c = o_c < T
    m_A = m_c[:, None] & (o_jA[None, :] < BT)
    m_cv = m_c[:, None] & m_v[None, :]
    p_dA = dA+((i_v*all+bos)*H+i_h)*BT + o_c[:, None] * (H*BT) + o_jA[None, :]

    # [BC, BC]
    b_dA = tl.zeros([BC, BC], dtype=tl.float32)
    if i_i > i_j:
        o_j = i_t * BT + i_j * BC + tl.arange(0, BC)
        m_vj = m_v[:, None] & (o_j[None, :] < T)
        p_v = v + (bos*H+i_h) * V + o_v[:, None] + o_j[None, :] * (H*V)
        p_gv = g + (bos*H+i_h) * V + o_v[:, None] + o_j[None, :] * (H*V)
        p_gn = g + (bos + i_t*BT + i_i*BC) * H*V + i_h * V + o_v
        p_g = g + (bos*H+i_h) * V + o_c[:, None] * (H*V) + o_v[None, :]
        p_do = do + (bos*H+i_h) * V + o_c[:, None] * (H*V) + o_v[None, :]
        # [BV,]
        b_gn = tl.load(p_gn, mask=m_v, other=0.)
        # [BC, BV]
        b_g = tl.load(p_g, mask=m_cv, other=0.0)
        b_do = tl.load(p_do, mask=m_cv, other=0.0)
        b_do = (b_do * exp2(b_g - b_gn[None, :])).to(b_do.dtype)
        # [BV, BC]
        b_v = tl.load(p_v, mask=m_vj, other=0.0)
        b_gv = tl.load(p_gv, mask=m_vj, other=0.0)
        b_vg = (b_v * exp2(b_gn[:, None] - b_gv)).to(b_v.dtype)
        # [BC, BC]
        b_dA = tl.dot(b_do, b_vg) * scale
    elif i_i == i_j:
        p_g = g + (bos*H + i_h) * V + o_c[:, None] * (H*V) + o_v[None, :]
        p_do = do + (bos*H + i_h) * V + o_c[:, None] * (H*V) + o_v[None, :]
        p_v = v + (bos + i_t*BT + i_j*BC) * H*V + i_h * V + o_v
        p_gv = g + (bos + i_t*BT + i_j*BC) * H*V + i_h * V + o_v
        # [BC, BV]
        b_g = tl.load(p_g, mask=m_cv, other=0.0)
        b_do = tl.load(p_do, mask=m_cv, other=0.0) * scale
        m_v = o_v < V

        o_i = tl.arange(0, BC)
        # [BC, BC]
        m_dA = o_i[:, None] >= o_i[None, :]
        for j in range(0, min(BC, T - i_t * BT - i_j * BC)):
            # [BV,]
            b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            # [BC,]
            b_dAj = tl.sum(b_do * b_v[None, :] * exp2(b_g - b_gv[None, :]), 1)
            b_dA = tl.where((o_i == j)[None, :], b_dAj[:, None], b_dA)

            p_v += H*V
            p_gv += H*V
        b_dA = tl.where(m_dA, b_dA, 0.)
    tl.store(p_dA, b_dA.to(dA.dtype.element_ty), mask=m_A)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gsa_bwd_kernel_dqkvg(
    q,
    k,
    v,
    h,
    g,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dg,
    dgv,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t = unflatten_program_id(tl.cdiv(K, BK))
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    o_i = tl.arange(0, BT)
    o_t = min(i_t * BT + BT, T)
    m_s = o_i[:, None] >= o_i[None, :]

    o_q = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK).to(tl.int64)
    m_q = o_q < T
    m_k = o_k < K
    m_qk = m_q[:, None] & m_k[None, :]
    m_A = m_q[:, None] & (o_i[None, :] < BT)
    p_q = q + (bos*H+i_h) * K + o_q[:, None] * (H*K) + o_k[None, :]
    p_k = k + (bos*H+i_h) * K + o_q[:, None] * (H*K) + o_k[None, :]
    p_A = A + ((i_k*all+bos)*H+i_h)*BT + o_q[:, None] * (H*BT) + o_i[None, :]

    # [BT, BK]
    b_q = tl.load(p_q, mask=m_qk, other=0.0)
    b_k = tl.load(p_k, mask=m_qk, other=0.0)
    # [BT, BT]
    b_A = tl.dot((b_q * scale).to(b_q.dtype), tl.trans(b_k))
    b_A = tl.where(m_s, b_A, 0.)
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), mask=m_A)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV).to(tl.int64)
        m_v = o_v < V
        m_qv = m_q[:, None] & m_v[None, :]
        m_vk = m_v[:, None] & m_k[None, :]
        m_kv = m_k[:, None] & m_v[None, :]
        p_v = v + (bos*H+i_h)*V + o_q[:, None] * (H*V) + o_v[None, :]
        p_g = g + (bos*H+i_h)*V + o_q[:, None] * (H*V) + o_v[None, :]
        p_gn = g + (bos + o_t - 1) * H*V + i_h * V + o_v
        p_do = do + (bos*H+i_h)*V + o_q[:, None] * (H*V) + o_v[None, :]
        p_dv = dv + ((i_k*all+bos)*H+i_h)*V + o_q[:, None] * (H*V) + o_v[None, :]
        p_dg = dg + (bos*H+i_h)*V + o_q[:, None] * (H*V) + o_v[None, :]
        p_dgv = dgv+((i_k*all+bos)*H+i_h)*V + o_q[:, None] * (H*V) + o_v[None, :]
        p_h = h + (i_tg * H + i_h) * K*V + o_v[:, None] + o_k[None, :] * V
        p_dh = dh + (i_tg * H + i_h) * K*V + o_k[:, None] * V + o_v[None, :]

        # [BV,]
        b_gn = tl.load(p_gn, mask=m_v, other=0)
        # [BT, BV]
        b_v = tl.load(p_v, mask=m_qv, other=0.0)
        b_g = tl.load(p_g, mask=m_qv, other=0.0)
        b_gv = exp2(b_gn[None, :] - b_g)
        # [BV, BK]
        b_h = tl.load(p_h, mask=m_vk, other=0.0)
        # [BT, BV]
        b_do = tl.load(p_do, mask=m_qv, other=0.0)
        b_do = (b_do * exp2(b_g)).to(b_do.dtype)
        # [BK, BV]
        b_dh = tl.load(p_dh, mask=m_kv, other=0.0)
        # [BV]
        b_dg = tl.sum(tl.trans(b_h) * b_dh, 0) * exp2(b_gn)

        b_dh = b_dh.to(b_k.dtype)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_k.dtype)) * scale
        b_dk = tl.dot((b_v * b_gv).to(b_v.dtype), tl.trans(b_dh), b_dk)
        # [BT, BV]
        b_dv = tl.dot(b_k, b_dh) * b_gv
        # [BV]
        b_dg += tl.sum(b_dv * b_v, 0)

        if i_k == 0:
            b_dgv = tl.load(p_dg, mask=m_qv, other=0.0) + b_dg[None, :]
        else:
            b_dgv = tl.zeros([BT, BV], dtype=tl.float32) + b_dg[None, :]

        tl.store(p_dgv, b_dgv.to(p_dgv.dtype.element_ty), mask=m_qv)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_qv)
    p_dA = dA + (bos*H + i_h) * BT + o_q[:, None] * (H*BT) + o_i[None, :]
    p_dq = dq + (bos*H + i_h) * K + o_q[:, None] * (H*K) + o_k[None, :]
    p_dk = dk + (bos*H + i_h) * K + o_q[:, None] * (H*K) + o_k[None, :]
    # [BT, BT]
    b_dA = tl.load(p_dA, mask=m_A, other=0.0)
    # [BT, BK]
    b_dq = tl.dot(b_dA, b_k, b_dq)
    b_dk = tl.dot(tl.trans(b_dA).to(b_k.dtype), b_q, b_dk)

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_qk)
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_qk)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_gsa_bwd_kernel_intra_dvg(
    v,
    g,
    o,
    A,
    do,
    dv,
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
    p_gv = g + (bos*H+i_h)*V + o_c[:, None] * (H*V) + o_v[None, :]
    p_gn = g + (bos + min(i_t * BT + i_i * BC + BC, T)-1)*H*V + i_h*V + o_v
    # [BV,]
    b_gn = tl.load(p_gn, mask=m_v, other=0)
    # [BC, BV]
    b_gv = tl.load(p_gv, mask=m_cv, other=0.0)
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        o_j = i_t * BT + i_j * BC + tl.arange(0, BC)
        o_iA = i_i * BC + tl.arange(0, BC)
        m_j = (i_t * BT + i_j * BC + o_i) < T
        m_jv = m_j[:, None] & m_v[None, :]
        m_A = (o_iA[:, None] < BT) & m_j[None, :]
        p_g = g + (bos*H+i_h) * V + o_j[:, None] * (H*V) + o_v[None, :]
        p_A = A + (bos*H+i_h) * BT + o_iA[:, None] + o_j[None, :] * (H*BT)
        p_do = do + (bos*H+i_h) * V + o_j[:, None] * (H*V) + o_v[None, :]
        # [BC, BV]
        b_g = tl.load(p_g, mask=m_jv, other=0.0)
        b_do = tl.load(p_do, mask=m_jv, other=0.0) * tl.where(m_j[:, None], exp2(b_g - b_gn[None, :]), 0)
        # [BC, BC]
        b_A = tl.load(p_A, mask=m_A, other=0.0)
        # [BC, BV]
        b_dv = tl.dot(b_A, b_do.to(b_A.dtype), b_dv)
    b_dv *= exp2(b_gn[None, :] - b_gv)

    p_g = g + (bos + i_t * BT + i_i * BC) * H*V + i_h * V + o_v
    p_A = A + (bos + i_t*BT + i_i*BC) * H*BT + i_h * BT + i_i * BC + o_i
    p_do = do + (bos + i_t*BT + i_i*BC) * H*V + i_h * V + o_v
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_A = tl.load(p_A)
        # [BV,]
        b_g = tl.load(p_g, mask=m_v, other=0)
        b_do = tl.load(p_do, mask=m_v, other=0)
        # [BC, BV]
        m_i = o_i[:, None] <= j
        b_dv += tl.where(m_i, exp2(b_g[None, :] - b_gv) * b_A[:, None] * b_do[None, :], 0.)

        p_g += H * V
        p_A += H * BT
        p_do += H * V
    p_o = o + (bos*H+i_h)*V + o_c[:, None] * (H*V) + o_v[None, :]
    p_v = v + (bos*H+i_h)*V + o_c[:, None] * (H*V) + o_v[None, :]
    p_do = do + (bos*H+i_h)*V + o_c[:, None] * (H*V) + o_v[None, :]
    p_dv = dv + (bos*H+i_h)*V + o_c[:, None] * (H*V) + o_v[None, :]
    p_dg = dg + (bos*H+i_h)*V + o_c[:, None] * (H*V) + o_v[None, :]

    b_o = tl.load(p_o, mask=m_cv, other=0.0).to(tl.float32)
    b_v = tl.load(p_v, mask=m_cv, other=0.0).to(tl.float32)
    b_do = tl.load(p_do, mask=m_cv, other=0.0).to(tl.float32)
    b_dv = b_dv + tl.load(p_dv, mask=m_cv, other=0.0).to(tl.float32)
    b_dg = b_o * b_do - b_v * b_dv
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_cv)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_cv)


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


def chunk_gsa_bwd_k(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    h0: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    dg: torch.Tensor,
    scale: float = 1.,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))
    BV = min(64, triton.next_power_of_2(V))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    if h is None:
        h, _ = chunk_fwd_h(
            k=k,
            v=v,
            g=None,
            gk=None,
            gv=g,
            h0=h0,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            states_in_fp32=False,
        )
    dh, dh0 = chunk_bwd_dh(
        q=q,
        k=k,
        v=v,
        g=None,
        gk=None,
        gv=g,
        do=do,
        h0=h0,
        dht=dht,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        states_in_fp32=True,
    )
    dA = q.new_empty(NV, B, T, H, BT)
    grid = (NV * NT * NC * NC, B * H)
    chunk_gsa_bwd_kernel_dA[grid](
        v=v,
        g=g,
        do=do,
        dA=dA,
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
    dA = dA.sum(0, dtype=dA.dtype) if NV > 1 else dA.squeeze(0)

    A = do.new_empty(NK, B, T, H, BT)
    dq = torch.empty_like(q)
    dk = k.new_empty(B, T, H, K)
    dv = v.new_empty(NK, B, T, H, V)
    dgv = g.new_empty(NK, B, T, H, V, dtype=torch.float)
    grid = (NK * NT, B * H)
    chunk_gsa_bwd_kernel_dqkvg[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        A=A,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        dgv=dgv,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    A = A.sum(0, dtype=A.dtype) if NK > 1 else A.squeeze(0)
    dv = dv.sum(0, dtype=dv.dtype) if NK > 1 else dv.squeeze(0)
    dgv = dgv.sum(0, dtype=dgv.dtype) if NK > 1 else dgv.squeeze(0)

    def grid(meta): return (triton.cdiv(V, meta['BV']) * NT * NC, B * H)
    chunk_gsa_bwd_kernel_intra_dvg[grid](
        v=v,
        g=g,
        o=o,
        A=A,
        do=do,
        dv=dv,
        dg=dg,
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
    dg = dgv.add_(
        chunk_local_cumsum(
            dg,
            chunk_size=BT,
            reverse=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
    )

    return dq, dk, dv, dg, dh0


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


def chunk_gsa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    ok: torch.Tensor,
    p: torch.Tensor,
    A: torch.Tensor,
    h: tuple[torch.Tensor, torch.Tensor],
    initial_state: tuple[torch.Tensor, torch.Tensor] | None,
    scale: float,
    do: torch.Tensor,
    dht: tuple[torch.Tensor, torch.Tensor],
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state

    hk, hv = h
    dhkt, dhvt = dht

    qv = p.to(q.dtype)
    dqv, dsv, dv, dg, dhv0 = chunk_gla_bwd(
        q=qv,
        k=s,
        v=v,
        g=None,
        g_cumsum=g,
        scale=1.,
        initial_state=hv0,
        h=hv,
        A=A,
        do=do,
        dht=dhvt,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    dok = softmax_bwd(p, dqv, dtype=ok.dtype)

    dq, dk, dsk, dg, dhk0 = chunk_gsa_bwd_k(
        q=q,
        k=k,
        v=s,
        g=g,
        h0=hk0,
        h=hk,
        o=ok,
        do=dok,
        dht=dhkt,
        dg=dg,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    ds = dsv.add_(dsk)
    dg = dg.to(s.dtype)
    return dq, dk, dv, ds, dg, dhk0, dhv0


class ChunkGSAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        s: torch.Tensor,
        g: torch.Tensor,
        scale: float,
        hk0: torch.Tensor | None,
        hv0: torch.Tensor | None,
        output_final_state: bool,
        checkpoint_level: int,
        cu_seqlens: torch.LongTensor | None,
        cu_seqlens_cpu: torch.LongTensor | None,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if chunk_size is None:
            chunk_size = 64

        if cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(
                cu_seqlens=cu_seqlens,
                chunk_size=chunk_size,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )
        else:
            chunk_indices = None

        # pre-scale gates so both stages use exp2 with the same cumulative values.
        g_org = g
        g = chunk_local_cumsum(
            g=g,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        hk, hkt, ok, p, Av, hv, hvt, ov = chunk_gsa_fwd(
            q=q,
            k=k,
            v=v,
            s=s,
            g=g,
            initial_state=(hk0, hv0),
            output_final_state=output_final_state,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )

        if checkpoint_level >= 1:
            g = g_org
        if checkpoint_level > 1:
            hk, hv = None, None

        ctx.save_for_backward(
            q,
            k,
            v,
            s,
            g,
            ok,
            p,
            Av,
            hk0,
            hv0,
            hk,
            hv,
            chunk_indices,
        )
        ctx.checkpoint_level = checkpoint_level
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_size = chunk_size
        return ov, hkt, hvt

    @staticmethod
    @input_guard
    def backward(ctx, dov, dhkt=None, dhvt=None):
        (
            q,
            k,
            v,
            s,
            g,
            ok,
            p,
            Av,
            hk0,
            hv0,
            hk,
            hv,
            chunk_indices,
        ) = ctx.saved_tensors
        scale = ctx.scale
        cu_seqlens = ctx.cu_seqlens
        chunk_size = ctx.chunk_size

        if ctx.checkpoint_level >= 1:
            g = chunk_local_cumsum(
                g=g,
                chunk_size=chunk_size,
                scale=RCP_LN2,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        dq, dk, dv, ds, dg, dhk0, dhv0 = chunk_gsa_bwd(
            q=q,
            k=k,
            v=v,
            s=s,
            g=g,
            ok=ok,
            p=p,
            A=Av,
            h=(hk, hv),
            initial_state=(hk0, hv0),
            scale=scale,
            do=dov,
            dht=(dhkt, dhvt),
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        return dq, dk, dv, ds, dg, None, dhk0, dhv0, None, None, None, None, None


@torch.compiler.disable
def chunk_gsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    checkpoint_level: int = 2,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
            GQA is performed if `H` is not equal to `HQ`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        s (torch.Tensor):
            slot representations of shape `[B, T, H, M]`.
        g (Optional[torch.Tensor]):
            Forget gates of shape `[B, T, H, M]` applied along the slot dimension.
            If not provided, this function is equivalent to vanilla ABC. Default: `None`.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[Tuple[torch.Tensor]]):
            Initial state tuple having tensors of shape `[N, H, K, M]` and `[N, H, M, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state tuple, having tensors of shape `[N, H, K, M]` and `[N, H, M, V]`.
            Default: `False`.
        checkpoint_level (Optional[int]):
            Checkpointing level; higher values will save more memory and do more recomputations during backward.
            Default: `2`.
            - Level `0`: no memory saved, no recomputation.
            - Level `1`: recompute the fp32 cumulative values during backward.
            - Level `2`: recompute cumulative gates and chunk states during backward.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        cu_seqlens_cpu (torch.LongTensor):
            CPU copy of `cu_seqlens` to avoid unnecessary device synchronization. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]`.
        final_state (Tuple[torch.Tensor]):
            Final state tuple having tensors of shape `[N, H, K, M]` and `[N, H, M, V]` if `output_final_state=True`.
            Both entries are `None` otherwise.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gsa import chunk_gsa
        # inputs with equal lengths
        >>> B, T, H, K, V, M = 4, 2048, 4, 512, 512, 64
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> s = torch.randn(B, T, H, M, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, M, device='cuda'))
        >>> h0 = (torch.randn(B, H, K, M, device='cuda'), torch.randn(B, H, M, V, device='cuda'))
        >>> o, (hk, hv) = chunk_gsa(
            q, k, v, s, g,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, s, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, s, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, (hk_var, hv_var) = chunk_gsa(
            q, k, v, s, g,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert hk.allclose(hk_var)
        >>> assert hv.allclose(hv_var)
    """
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    chunk_size = kwargs.pop('chunk_size', None)
    if chunk_size is not None and (chunk_size <= 0 or chunk_size & (chunk_size - 1)):
        raise ValueError(f"`chunk_size` must be a power of 2, got {chunk_size}.")
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state[0].shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state[0].shape[0]}.",
            )
    assert checkpoint_level in [0, 1, 2]
    if g is None:
        s, g, _ = fused_additive_gate(
            x=s,
            cu_seqlens=cu_seqlens,
            output_dtype=k.dtype,
            gate_dtype=torch.float32,
        )
    if scale is None:
        scale = q.shape[-1] ** -0.5

    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    H, HQ = k.shape[2], q.shape[2]
    if HQ < H or HQ % H:
        raise ValueError(f"The number of query heads ({HQ}) must be a multiple of the number of key heads ({H}).")
    NG = HQ // H
    if NG > 1:
        k, v, s, g = (x.repeat_interleave(NG, dim=2) for x in (k, v, s, g))
        hk0, hv0 = (x.repeat_interleave(NG, dim=1) if x is not None else None for x in (hk0, hv0))
    o, *final_state = ChunkGSAFunction.apply(
        q,
        k,
        v,
        s,
        g,
        scale,
        hk0,
        hv0,
        output_final_state,
        checkpoint_level,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_size,
    )
    if NG > 1 and output_final_state:
        final_state = [x[:, ::NG].contiguous() for x in final_state]
    return o, final_state
