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
from fla.ops.utils.op import exp2, unflatten_program_id
from fla.utils import (
    IS_NPU,
    IS_NVIDIA_BLACKWELL,
    IS_NVIDIA_HOPPER,
    TRITON_ABOVE_3_4_0,
    TRITON_ABOVE_3_7_1,
    TRITON_ABOVE_3_8_0,
    ascend_compile_kwargs,
    autotune_cache_kwargs,
    check_shared_mem,
)

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]
BLACKWELL_8_WARP_RACE = IS_NVIDIA_BLACKWELL and not TRITON_ABOVE_3_8_0
CHUNK_BWD_DV_LOCAL_NUM_WARPS = [2, 4] if BLACKWELL_8_WARP_RACE else NUM_WARPS


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    num_householder: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t = unflatten_program_id(tl.cdiv(V, BV))
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

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

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * num_householder * H + i_h) * K
    v += (bos * num_householder * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K*V

    o_t = i_t * BT + tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < T
    m_v = m_t[:, None] & (o_v[None, :] < V)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_q = m_t[:, None] & (o_k[None, :] < K)
        m_h = (o_k[:, None] < K) & (o_v[None, :] < V)
        p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
        p_h = h + o_k[:, None] * V + o_v[None, :]
        # [BT, BK]
        b_q = tl.load(p_q, mask=m_q, other=0.0)
        # [BK, BV]
        b_h = tl.load(p_h, mask=m_h, other=0.0)
        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o = tl.dot(b_q, b_h, b_o)

    if USE_G:
        g += bos * H + i_h
        p_g = g + o_t * H
        b_g = tl.load(p_g, mask=m_t, other=0.0)
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        b_m = tl.where(m_A, exp2(b_g[:, None] - b_g[None, :]), 0)
        b_o = b_o * exp2(b_g)[:, None]
    else:
        b_m = ((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)).to(tl.float32)

    for i_dp in range(num_householder):
        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_q = m_t[:, None] & (o_k[None, :] < K)
            m_k = (o_k[:, None] < K) & m_t[None, :]
            p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
            p_k = k+i_dp*H*K + o_k[:, None] + o_t[None, :] * (num_householder*H*K)
            # [BT, BK]
            b_q = tl.load(p_q, mask=m_q, other=0.0)
            # [BK, BT]
            b_k = tl.load(p_k, mask=m_k, other=0.0)
            # [BT, BK] @ [BK, BT] -> [BT, BT]
            b_A = tl.dot(b_q, b_k, b_A)
        b_A = b_A * b_m
        p_v = v+i_dp*H*V + o_t[:, None] * (H*V*num_householder) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_o = tl.dot(b_A.to(b_v.dtype), b_v, b_o)
    b_o = b_o * scale
    p_o = o + o_t[:, None] * (H*V) + o_v[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)


def chunk_gated_delta_product_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,  # cumsum of log decay
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    num_householder: int = 1,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    assert q.shape[1] * num_householder == k.shape[1], "q.shape[1] * num_householder must be equal to k.shape[1]"
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    o = v.new_empty(B, T, H, V).fill_(-float('inf'))
    def grid(meta): return (triton.cdiv(V, meta['BV']) * NT, B * H)
    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        num_householder=num_householder,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return o


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in ([4] if IS_NPU else CHUNK_BWD_DV_LOCAL_NUM_WARPS)
        for num_stages in ([1] if IS_NPU else ([2, 3, 4] if check_shared_mem('ada') else [1, 2, 3]))
    ],
    key=['H', 'K', 'V', 'BT', 'BQ', 'BK', 'BV', 'USE_G'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_product_bwd_kernel_dv_local(
    q,
    k,
    g,
    do,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    num_householder: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_tok = eos - bos
    else:
        T_tok = T.to(tl.int64)
        bos = i_b * T_tok
    bos_dp = bos * num_householder
    T_dp = T_tok * num_householder

    q += (bos * H + i_h) * K
    k += (bos_dp * H + i_h) * K
    do += (bos * H + i_h) * V
    dv += (bos_dp * H + i_h) * V

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T_dp
    q_start = (i_t * BT) // num_householder
    q_end = tl.minimum((i_t + 1) * BT, T_dp) // num_householder
    o_q = q_start + tl.arange(0, BQ)
    m_q = o_q < q_end
    o_q_dp = o_q * num_householder + (num_householder - 1)

    b_A = tl.zeros([BT, BQ], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_k = k + o_t[:, None] * (H * K) + o_k[None, :]
        p_q = q + o_k[:, None] + o_q[None, :] * (H * K)
        b_k = tl.load(p_k, mask=m_t[:, None] & m_k[None, :], other=0.0)
        b_q = tl.load(p_q, mask=m_k[:, None] & m_q[None, :], other=0.0)
        b_A = tl.dot(b_k, b_q, b_A)

    m_A = (o_t[:, None] <= o_q_dp[None, :]) & (m_t[:, None] & m_q[None, :])
    if USE_G:
        g += bos_dp * H + i_h
        b_g_k = tl.load(g + o_t * H, mask=m_t, other=0.0).to(tl.float32)
        b_g_q = tl.load(g + o_q_dp * H, mask=m_q, other=0.0).to(tl.float32)
        b_A = b_A * exp2(tl.where(m_A, b_g_q[None, :] - b_g_k[:, None], 0.0))
    b_A = tl.where(m_A, b_A * scale, 0.0).to(do.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        p_do = do + o_q[:, None] * (H * V) + o_v[None, :]
        p_dv = dv + o_t[:, None] * (H * V) + o_v[None, :]
        b_do = tl.load(p_do, mask=m_q[:, None] & m_v[None, :], other=0.0)
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_DW': lambda args: args['dw'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in ([4] if IS_NPU else NUM_WARPS)
        for num_stages in ([1] if IS_NPU else ([2, 3, 4] if check_shared_mem('ada') else [1, 2, 3]))
    ],
    key=['H', 'K', 'V', 'BT', 'BQ', 'BK', 'BV', 'USE_G', 'USE_DW'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_product_bwd_kernel_dqkwg(
    q,
    k,
    v,
    g,
    h,
    do,
    dh,
    dq,
    dk,
    dw,
    dv,
    dg,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    num_householder: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_DW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t = unflatten_program_id(tl.cdiv(K, BK))
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    all_dp = T.to(tl.int64) * (B * num_householder)
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_tok = eos - bos
    else:
        T_tok = T.to(tl.int64)
        NT = tl.cdiv(T_tok * num_householder, BT)
        i_tg = i_b * NT + i_t
        bos = i_b * T_tok
    bos_dp = bos * num_householder
    T_dp = T_tok * num_householder

    q += (bos * H + i_h) * K
    dq += (bos * H + i_h) * K
    do += (bos * H + i_h) * V
    k += (bos_dp * H + i_h) * K
    dk += (bos_dp * H + i_h) * K
    v += (bos_dp * H + i_h) * V
    h += (i_tg * H + i_h) * K * V
    dh += (i_tg * H + i_h) * K * V

    if USE_DW:
        dw += (bos_dp * H + i_h) * K
        dv += (bos_dp * H + i_h) * V

    if USE_G:
        dg += i_k * all_dp * H + bos_dp * H + i_h
        b_dg_last = tl.zeros([1], dtype=tl.float32)

    b_dq = tl.zeros([BQ, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BQ, BT], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T_dp
    q_start = (i_t * BT) // num_householder
    q_end = tl.minimum((i_t + 1) * BT, T_dp) // num_householder
    o_q = q_start + tl.arange(0, BQ)
    m_q = o_q < q_end
    o_q_dp = o_q * num_householder + (num_householder - 1)

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    m_qk = m_q[:, None] & m_k[None, :]
    m_tk = m_t[:, None] & m_k[None, :]

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        m_qv = m_q[:, None] & m_v[None, :]
        m_h = m_v[:, None] & m_k[None, :]

        p_v = v + o_t[:, None] * (H * V) + o_v[None, :]
        p_do = do + o_q[:, None] * (H * V) + o_v[None, :]
        p_h = h + o_v[:, None] + o_k[None, :] * V
        p_dh = dh + o_v[:, None] + o_k[None, :] * V

        b_v = tl.load(p_v, mask=m_tv, other=0.0)
        b_do = tl.load(p_do, mask=m_qv, other=0.0)
        b_h = tl.load(p_h, mask=m_h, other=0.0)
        b_dh = tl.load(p_dh, mask=m_h, other=0.0)

        if USE_G:
            b_dg_last += tl.sum(b_h * b_dh)
        b_ds = tl.dot(b_do.to(b_v.dtype), tl.trans(b_v), b_ds)
        b_dq = tl.dot(b_do, b_h.to(b_do.dtype), b_dq)
        b_dk = tl.dot(b_v, b_dh.to(b_v.dtype), b_dk)
        if USE_DW:
            p_dv = dv + o_t[:, None] * (H * V) + o_v[None, :]
            b_dv = tl.load(p_dv, mask=m_tv, other=0.0)
            b_dw = tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype), b_dw)

    if USE_DW:
        p_dw = dw + o_t[:, None] * (H * K) + o_k[None, :]
        tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), mask=m_tk)

    tl.debug_barrier()
    p_q = q + o_q[:, None] * (H * K) + o_k[None, :]
    p_k = k + o_t[:, None] * (H * K) + o_k[None, :]
    b_q = tl.load(p_q, mask=m_qk, other=0.0)
    b_k = tl.load(p_k, mask=m_tk, other=0.0)

    p_dq = dq + o_q[:, None] * (H * K) + o_k[None, :]
    p_dk = dk + o_t[:, None] * (H * K) + o_k[None, :]

    m_A = (o_q_dp[:, None] >= o_t[None, :]) & (m_q[:, None] & m_t[None, :])
    if USE_G:
        g += bos_dp * H + i_h
        last_idx = tl.minimum(i_t * BT + BT, T_dp) - 1
        b_g_k = tl.load(g + o_t * H, mask=m_t, other=0.0).to(tl.float32)
        b_g_q = tl.load(g + o_q_dp * H, mask=m_q, other=0.0).to(tl.float32)
        b_g_last = tl.load(g + last_idx * H).to(tl.float32)

        b_dg_last *= exp2(b_g_last)
        b_dq = b_dq * exp2(b_g_q)[:, None] * scale
        b_dk = b_dk * tl.where(m_t, exp2(tl.where(m_t, -b_g_k + b_g_last, 0.0)), 0.0)[:, None]
        b_dg_last += tl.sum(b_dk * b_k)

        b_ds = tl.where(m_A, b_ds * exp2(tl.where(m_A, b_g_q[:, None] - b_g_k[None, :], 0.0)), 0.0) * scale
        b_ds = b_ds.to(b_k.dtype)
        b_dq = tl.dot(b_ds, b_k, b_dq)
        b_dk = tl.dot(tl.trans(b_ds), b_q.to(b_k.dtype), b_dk)

        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_qk)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_tk)

        b_dg_q = tl.sum(b_dq * b_q, axis=1)
        b_dg = -tl.sum(b_dk * b_k, axis=1)
        if num_householder == 1:
            b_dg += b_dg_q
        else:
            m_q_pos = (o_t[:, None] == o_q_dp[None, :]) & m_q[None, :]
            b_dg += tl.sum(tl.where(m_q_pos, b_dg_q[None, :], 0.0), axis=1)
        b_dg = tl.where(o_t < last_idx, b_dg, b_dg + b_dg_last)
        tl.store(dg + o_t * H, b_dg.to(dg.dtype.element_ty), mask=m_t)
    else:
        b_ds = tl.where(m_A, b_ds, 0.0).to(b_k.dtype)
        b_dq = tl.dot(b_ds, b_k, b_dq) * scale
        b_dk += tl.dot(tl.trans(b_ds), b_q.to(b_k.dtype)) * scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_qk)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_tk)


def _get_bwd_tiling(device_idx: int, tensors: tuple[torch.Tensor | None, ...], bq: int = 64) -> int:
    if IS_NPU:
        return 16 if bq >= 64 else 32
    if check_shared_mem('hopper', device_idx):
        return 128
    if check_shared_mem('ampere', device_idx):
        is_16bit = all(x.element_size() == 2 for x in tensors if x is not None)
        return 128 if is_16bit else 64
    if check_shared_mem('ada', device_idx):
        return 64
    return 32


def chunk_gated_delta_product_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    num_householder: int = 1,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V = *q.shape, do.shape[-1]
    T_dp = T * num_householder
    BT = chunk_size
    BQ = max(16, triton.next_power_of_2(triton.cdiv(BT, num_householder)))
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens * num_householder, BT)
    CONST_TILING = _get_bwd_tiling(k.device.index, (q, k, do), bq=BQ)
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T_dp, BT) if cu_seqlens is None else len(chunk_indices)

    dv = do.new_empty(B, T_dp, H, V)
    grid = (NT, B * H)
    chunk_gated_delta_product_bwd_kernel_dv_local[grid](
        q=q,
        k=k,
        g=g,
        do=do,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        num_householder=num_householder,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BQ=BQ,
        BK=BK,
        BV=BV,
        **ascend_compile_kwargs(),
    )
    return dv


def chunk_gated_delta_product_bwd_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    scale: float | None = None,
    num_householder: int = 1,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if g is not None and IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0 and not TRITON_ABOVE_3_7_1:
        raise RuntimeError(
            "Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect results for "
            "gated chunk_gated_delta_product_bwd_dqkwg (see #640). Please upgrade Triton to >= 3.7.1.",
        )

    B, T, H, K, V = *q.shape, v.shape[-1]
    T_dp = T * num_householder
    BT = chunk_size
    BQ = max(16, triton.next_power_of_2(triton.cdiv(BT, num_householder)))
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens * num_householder, BT)
    NT = triton.cdiv(T_dp, BT) if cu_seqlens is None else len(chunk_indices)

    CONST_TILING = _get_bwd_tiling(k.device.index, (q, k, v, do, h, dh, w, dv), bq=BQ)
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NK = triton.cdiv(K, BK)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dw = torch.empty_like(w) if w is not None else None
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device) if g is not None else None

    grid = (NK * NT, B * H)
    chunk_gated_delta_product_bwd_kernel_dqkwg[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dw=dw,
        dv=dv,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        B=B,
        T=T,
        num_householder=num_householder,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BQ=BQ,
        BK=BK,
        BV=BV,
        **ascend_compile_kwargs(),
    )
    if dg is not None:
        dg = dg.sum(0)
    return dq, dk, dw, dg
