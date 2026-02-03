from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from fla.ops.cp.comm import all_gather_into_tensor
from fla.ops.utils.op import exp, exp2
from fla.utils import USE_CUDA_GRAPH, autotune_cache_kwargs, check_shared_mem

if TYPE_CHECKING:
    from fla.ops.cp.context import FLACPContext


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT', 'USE_EXP2', "STAGE"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def pre_process_fwd_kernel_stage1(
    k,
    v,
    w,
    g,
    gk,
    hm,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_h = tl.program_id(0), tl.program_id(1)
    i_n = 0
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    # calculate offset
    hm += i_h * K * (K + V)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * H + i_h) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    stride_v = H*V
    stride_k = H*K

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)
    # main recurrence
    for i_t in range(NT):
        p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h).to(tl.float32)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            if USE_EXP2:
                b_h1 *= exp2(b_gk_last1)[:, None]
            else:
                b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                if USE_EXP2:
                    b_h2 *= exp2(b_gk_last2)[:, None]
                else:
                    b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                if USE_EXP2:
                    b_h3 *= exp2(b_gk_last3)[:, None]
                else:
                    b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                if USE_EXP2:
                    b_h4 *= exp2(b_gk_last4)[:, None]
                else:
                    b_h4 *= exp(b_gk_last4)[:, None]

        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)

    p_h1 = tl.make_block_ptr(hm, (K, V), (K+V, 1), (0, i_v * BV), (64, BV), (1, 0))
    tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
    if K > 64:
        p_h2 = tl.make_block_ptr(hm, (K, V), (K+V, 1), (64, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
    if K > 128:
        p_h3 = tl.make_block_ptr(hm, (K, V), (K+V, 1), (128, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
    if K > 192:
        p_h4 = tl.make_block_ptr(hm, (K, V), (K+V, 1), (192, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK2': BK2}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BK2 in [32]
    ],
    key=['H', 'BT', 'USE_EXP2', 'FORWARD'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def pre_process_fwd_bwd_kernel_stage2(
    k,
    w,
    g,
    gk,
    hm,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BK1: tl.constexpr,
    BK2: tl.constexpr,
    FORWARD: tl.constexpr = True,
):
    i_k_col, i_h = tl.program_id(0), tl.program_id(1)
    i_n = 0
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)

    # calculate offset
    hm += i_h * K * (K + V)
    k += ((bos * H + i_h) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    stride_k = H*K

    row = tl.arange(0, BK1)
    col = tl.arange(0, BK2) + i_k_col * BK2

    b_m = tl.where(row[:, None] == col[None, :],  1.0,  0.0)
    for _i_t in range(NT):
        if FORWARD:
            i_t = _i_t
        else:
            i_t = NT - 1 - _i_t
        p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h).to(tl.float32)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_k = b_k * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_diag = tl.where(row[:, None] == row[None, :], b_g_last,  0.0)
        elif USE_GK:
            b_gk_last = tl.load(gk + (bos + last_idx) * H*K + i_h * K + row, mask=(row < K), other=0.).to(tl.float32)
            if USE_EXP2:
                b_gk_last = exp2(b_gk_last)
            else:
                b_gk_last = exp(b_gk_last)
            b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
        else:
            b_diag = tl.where(row[:, None] == row[None, :], 1., 0.0)
        if FORWARD:
            b_kw = tl.dot(tl.trans(b_k.to(b_w.dtype)), b_w)
        else:
            b_kw = tl.dot(tl.trans(b_w), b_k.to(b_w.dtype))
        b_m_i = b_diag - b_kw
        b_m = tl.dot(b_m_i.to(b_w.dtype), b_m.to(b_w.dtype))
    p_m = tl.make_block_ptr(hm + V, (K, K), (K+V, 1), (0, i_k_col * BK2), (BK1, BK2), (1, 0))
    tl.store(p_m, b_m.to(p_m.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'USE_EXP2'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def pre_process_fwd_kernel_merged(
    k,
    v,
    w,
    g,
    gk,
    hm,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_col, i_h = tl.program_id(0), tl.program_id(1)
    i_n = 0
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    # Determine if this block handles h (V part) or m (K part)
    # i_col is in range [0, cdiv(V + K, BLOCK_SIZE))
    # Columns [0, V) are for h, columns [V, V+K) are for m
    is_h_part = i_col * BLOCK_SIZE < V

    # Calculate offsets
    hm += i_h * K * (K + V)
    k += ((bos * H + i_h) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    stride_k = H * K

    if is_h_part:
        # ====== Stage 1: Compute h (K x V) ======
        v += ((bos * H + i_h) * V).to(tl.int64)
        stride_v = H * V
        i_v = i_col

        # Initialize h accumulators
        b_h1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        # Main recurrence for h
        for i_t in range(NT):
            # Compute decayed v
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_decay = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > 64:
                p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v_decay += tl.dot(b_w, b_h2.to(b_w.dtype))
            if K > 128:
                p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v_decay += tl.dot(b_w, b_h3.to(b_w.dtype))
            if K > 192:
                p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v_decay += tl.dot(b_w, b_h4.to(b_w.dtype))

            p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v_decay

            last_idx = min((i_t + 1) * BT, T) - 1

            # Apply g decay
            if USE_G:
                m_t = (i_t * BT + tl.arange(0, BT)) < T
                b_g_last = tl.load(g + bos * H + last_idx * H + i_h).to(tl.float32)
                p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                if USE_EXP2:
                    b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                    b_g_last = exp2(b_g_last)
                else:
                    b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                    b_g_last = exp(b_g_last)
                b_h1 *= b_g_last
                if K > 64:
                    b_h2 *= b_g_last
                if K > 128:
                    b_h3 *= b_g_last
                if K > 192:
                    b_h4 *= b_g_last

            # Apply gk decay
            if USE_GK:
                o_k1 = tl.arange(0, 64)
                b_gk_last1 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[:, None]
                else:
                    b_h1 *= exp(b_gk_last1)[:, None]
                if K > 64:
                    o_k2 = 64 + o_k1
                    b_gk_last2 = tl.load(gk + (bos + last_idx) * H * K + i_h * K +
                                         o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[:, None]
                    else:
                        b_h2 *= exp(b_gk_last2)[:, None]
                if K > 128:
                    o_k3 = 128 + o_k1
                    b_gk_last3 = tl.load(gk + (bos + last_idx) * H * K + i_h * K +
                                         o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[:, None]
                    else:
                        b_h3 *= exp(b_gk_last3)[:, None]
                if K > 192:
                    o_k4 = 192 + o_k1
                    b_gk_last4 = tl.load(gk + (bos + last_idx) * H * K + i_h * K +
                                         o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[:, None]
                    else:
                        b_h4 *= exp(b_gk_last4)[:, None]

            b_v = b_v.to(k.dtype.element_ty)

            # Update h: h += k^T @ v
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h1 += tl.dot(b_k, b_v)
            if K > 64:
                p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h2 += tl.dot(b_k, b_v)
            if K > 128:
                p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h3 += tl.dot(b_k, b_v)
            if K > 192:
                p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h4 += tl.dot(b_k, b_v)

        # Store h results
        p_h1 = tl.make_block_ptr(hm, (K, V), (K + V, 1), (0, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(hm, (K, V), (K + V, 1), (64, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(hm, (K, V), (K + V, 1), (128, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(hm, (K, V), (K + V, 1), (192, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))
    else:
        # ====== Stage 2: Compute m (K x K) ======
        # i_col is for m part, map to K dimension
        # m starts at column V, so offset = i_col * BLOCK_SIZE - V
        # Use tl.cdiv to correctly compute the number of blocks for V dimension
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)

        # Following stage2 kernel design:
        # - BK1 is the full K dimension (next_power_of_2(K))
        # - BLOCK_SIZE is the column block size (like BK2=32 in stage2)
        # Each block computes a (BK1, BLOCK_SIZE) sub-matrix of m
        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        # Initialize as identity matrix: M_0 = I
        b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for i_t in range(NT):
            # Load k and w with full BK1 rows
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                m_t = (i_t * BT + tl.arange(0, BT)) < T
                b_g_last = tl.load(g + bos * H + last_idx * H + i_h).to(tl.float32)
                p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                if USE_EXP2:
                    b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                    b_g_last = exp2(b_g_last)
                else:
                    b_k = b_k * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                    b_g_last = exp(b_g_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_g_last, 0.0)
            elif USE_GK:
                b_gk_last = tl.load(gk + (bos + last_idx) * H * K + i_h * K + row, mask=(row < K), other=0.).to(tl.float32)
                if USE_EXP2:
                    b_gk_last = exp2(b_gk_last)
                else:
                    b_gk_last = exp(b_gk_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
            else:
                b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)

            # Compute m update: m = (diag - k^T @ w) @ m
            b_kw = tl.dot(tl.trans(b_k.to(b_w.dtype)), b_w)
            b_m_i = b_diag - b_kw
            b_m = tl.dot(b_m_i.to(b_w.dtype), b_m.to(b_w.dtype))

        # Store m result
        p_m = tl.make_block_ptr(hm + V, (K, K), (K + V, 1), (0, i_k_col * BLOCK_SIZE), (BK1, BLOCK_SIZE), (1, 0))
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT', 'USE_EXP2'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['pre_or_post_num_ranks', 'rank'])
def merge_fwd_bwd_kernel(
    h,
    ag_hm,
    pre_or_post_num_ranks,
    rank,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    FORWARD: tl.constexpr = True
):
    i_v, i_h = tl.program_id(0), tl.program_id(1)
    num_ranks = pre_or_post_num_ranks.to(tl.int32)
    h += i_h * K * V
    ag_hm += i_h * K * (K + V)
    stride = H * K * (K + V)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    for idx in range(num_ranks):
        if FORWARD:
            cur_rank = rank - num_ranks + idx
        else:
            cur_rank = rank + num_ranks - idx
        p_ag_h = tl.make_block_ptr(ag_hm + cur_rank * stride, (K, V), (K + V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        b_ag_h = tl.load(p_ag_h, boundary_check=(0, 1))
        p_ag_m = tl.make_block_ptr(ag_hm + cur_rank * stride + V, (K, K), (K + V, 1), (0, 0), (BK, BK), (1, 0))
        b_ag_m = tl.load(p_ag_m, boundary_check=(0, 1))
        b_h = tl.dot(b_ag_m, b_h.to(b_ag_m.dtype)) + b_ag_h
    p_h = tl.make_block_ptr(h, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([4, 3, 2] if check_shared_mem('ampere') else [1])
        for BV in [64, 32]
    ],
    key=['H', 'K', 'V', 'BT', 'BV', 'USE_G'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def pre_process_bwd_kernel_stage1(
    q,
    k,
    w,
    g,
    gk,
    do,
    dhm,
    dv,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_h = tl.program_id(0), tl.program_id(1)
    i_n = 0
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    # [BK, BV]
    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    q += ((bos * H + i_h) * K).to(tl.int64)
    k += ((bos * H + i_h) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    do += ((bos * H + i_h) * V).to(tl.int64)
    dv += ((bos * H + i_h) * V).to(tl.int64)
    dhm += i_h * K * (V + K)

    stride_v = H*V
    stride_k = H*K

    for i_t in range(NT - 1, -1, -1):
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            bg_last = tl.load(g + (bos + last_idx) * H + i_h).to(tl.float32)
            bg_last_exp = exp(bg_last)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            b_g_exp = exp(b_g)

        p_dv = tl.make_block_ptr(dv, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        # Update dv
        p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + last_idx * H*K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
        b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + last_idx * H*K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
            b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        if K > 128:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + last_idx * H*K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
            b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        if K > 192:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + last_idx * H*K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
            b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_dv *= tl.where(m_t, exp(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, boundary_check=(0, 1))

        # Update dh
        p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        if USE_GK:
            b_dh1 *= exp(b_gk_last1[:, None])
        b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 64:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                b_dh2 *= exp(b_gk_last2[:, None])
            b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                b_dh3 *= exp(b_gk_last3[:, None])
            b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                b_dh4 *= exp(b_gk_last4[:, None])
            b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

    p_dh1 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (0, i_v * BV), (64, BV), (1, 0))
    tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
    if K > 64:
        p_dh2 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (64, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
    if K > 128:
        p_dh3 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (128, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
    if K > 192:
        p_dh4 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (192, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([4, 3, 2] if check_shared_mem('ampere') else [1])
    ],
    key=['H', 'K', 'V', 'BT', 'USE_EXP2'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def pre_process_bwd_kernel_merged(
    q,
    k,
    w,
    g,
    gk,
    do,
    dhm,
    dv,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Merged backward kernel that computes both dh (K x V) and dm (K x K) in a single kernel.

    Similar to pre_process_fwd_kernel_merged, this kernel uses a unified grid where:
    - Columns [0, V) are for computing dh (stage 1)
    - Columns [V, V+K) are for computing dm (stage 2)
    """
    i_col, i_h = tl.program_id(0), tl.program_id(1)
    i_n = 0
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    # Determine if this block handles dh (V part) or dm (K part)
    is_dh_part = i_col * BLOCK_SIZE < V

    # Calculate offsets
    q += ((bos * H + i_h) * K).to(tl.int64)
    k += ((bos * H + i_h) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    dhm += i_h * K * (V + K)
    stride_k = H * K

    if is_dh_part:
        # ====== Stage 1: Compute dh (K x V) ======
        do += ((bos * H + i_h) * V).to(tl.int64)
        dv += ((bos * H + i_h) * V).to(tl.int64)
        stride_v = H * V
        i_v = i_col

        # Initialize dh accumulators
        b_dh1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        # Main recurrence for dh (reverse order)
        for i_t in range(NT - 1, -1, -1):
            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                # Note: pre_process_bwd_kernel_stage1 always uses exp for USE_G,
                # regardless of USE_EXP2. This is for consistency with the original design.
                bg_last = tl.load(g + (bos + last_idx) * H + i_h).to(tl.float32)
                bg_last_exp = exp(bg_last)
                p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                b_g_exp = exp(b_g)

            p_dv = tl.make_block_ptr(dv, (T, V), (stride_v, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0))
            b_do = tl.load(p_do, boundary_check=(0, 1))

            # Update dv
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k1 = tl.arange(0, 64)
                if USE_EXP2:
                    b_gk_last1 = tl.load(gk + last_idx * H * K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
                else:
                    b_gk_last1 = tl.load(gk + last_idx * H * K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

            if K > 64:
                p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    o_k2 = 64 + o_k1
                    if USE_EXP2:
                        b_gk_last2 = tl.load(gk + last_idx * H * K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                    else:
                        b_gk_last2 = tl.load(gk + last_idx * H * K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

            if K > 128:
                p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    o_k3 = 128 + o_k1
                    if USE_EXP2:
                        b_gk_last3 = tl.load(gk + last_idx * H * K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                    else:
                        b_gk_last3 = tl.load(gk + last_idx * H * K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

            if K > 192:
                p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    o_k4 = 192 + o_k1
                    if USE_EXP2:
                        b_gk_last4 = tl.load(gk + last_idx * H * K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                    else:
                        b_gk_last4 = tl.load(gk + last_idx * H * K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

            if USE_G:
                m_t = (i_t * BT + tl.arange(0, BT)) < T
                # Note: pre_process_bwd_kernel_stage1 always uses exp for USE_G
                b_dv *= tl.where(m_t, exp(bg_last - b_g), 0)[:, None]
            b_dv += tl.load(p_dv, boundary_check=(0, 1))

            # Update dh
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            if USE_G:
                b_dh1 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if USE_EXP2:
                    b_dh1 *= exp2(b_gk_last1[:, None])
                else:
                    b_dh1 *= exp(b_gk_last1[:, None])
            b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 64:
                p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh2 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    if USE_EXP2:
                        b_dh2 *= exp2(b_gk_last2[:, None])
                    else:
                        b_dh2 *= exp(b_gk_last2[:, None])
                b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 128:
                p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh3 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    if USE_EXP2:
                        b_dh3 *= exp2(b_gk_last3[:, None])
                    else:
                        b_dh3 *= exp(b_gk_last3[:, None])
                b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 192:
                p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh4 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    if USE_EXP2:
                        b_dh4 *= exp2(b_gk_last4[:, None])
                    else:
                        b_dh4 *= exp(b_gk_last4[:, None])
                b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

        # Store dh results
        p_dh1 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (0, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh2 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (64, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh3 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (128, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh4 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (192, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))
    else:
        # ====== Stage 2: Compute dm (K x K) ======
        # i_col is for dm part, map to K dimension
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)

        # Following stage2 kernel design for backward (FORWARD=False)
        # - BK1 is the full K dimension (next_power_of_2(K))
        # - BLOCK_SIZE is the column block size
        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        # Initialize as identity matrix: M_0 = I
        b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for _i_t in range(NT):
            # Reverse order for backward
            i_t = NT - 1 - _i_t

            # Load k and w with full BK1 rows
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                m_t = (i_t * BT + tl.arange(0, BT)) < T
                b_g_last = tl.load(g + bos * H + last_idx * H + i_h).to(tl.float32)
                p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                if USE_EXP2:
                    b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                    b_g_last = exp2(b_g_last)
                else:
                    b_k = b_k * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                    b_g_last = exp(b_g_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_g_last, 0.0)
            elif USE_GK:
                b_gk_last = tl.load(gk + (bos + last_idx) * H * K + i_h * K + row, mask=(row < K), other=0.).to(tl.float32)
                if USE_EXP2:
                    b_gk_last = exp2(b_gk_last)
                else:
                    b_gk_last = exp(b_gk_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
            else:
                b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)

            # Compute dm update for backward: m = (diag - w^T @ k) @ m
            # Note: FORWARD=False uses tl.trans(b_w) @ b_k instead of tl.trans(b_k) @ b_w
            b_kw = tl.dot(tl.trans(b_w), b_k.to(b_w.dtype))
            b_m_i = b_diag - b_kw
            b_m = tl.dot(b_m_i.to(b_w.dtype), b_m.to(b_w.dtype))

        # Store dm result
        p_m = tl.make_block_ptr(dhm + V, (K, K), (V + K, 1), (0, i_k_col * BLOCK_SIZE), (BK1, BLOCK_SIZE), (1, 0))
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), boundary_check=(0, 1))


DTYPE = torch.float32


def chunk_gated_delta_rule_fwd_h_pre_process(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    cu_seqlens: torch.LongTensor | None = None,
    use_exp2: bool = False,
    initial_state: torch.Tensor | None = None,
    context: FLACPContext = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context is None or context.group is None:
        return initial_state
    assert initial_state is None, "When enable CP, the provided initial_state must be None."
    rank = dist.get_rank(group=context.group)

    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size
    BK = triton.next_power_of_2(K)

    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    hm = k.new_zeros(H, K, (V + K), dtype=DTYPE)
    initial_state = k.new_zeros(N, H, K, V, dtype=DTYPE)
    if not context.is_last_rank:
        BLOCK_SIZE = 32 if K <= 64 else 64
        grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), H)
        pre_process_fwd_kernel_merged[grid](
            k=k,
            v=u,
            w=w,
            g=g,
            gk=gk,
            hm=hm,
            cu_seqlens=cu_seqlens[-2:],
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK1=BK,
            USE_EXP2=use_exp2,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    ag_hm, _ = all_gather_into_tensor(hm, group=context.group)
    if not context.is_first_rank:
        def grid(meta): return (triton.cdiv(V, meta['BV']), H)
        merge_fwd_bwd_kernel[grid](
            initial_state[0],
            ag_hm,
            context.pre_num_ranks,
            rank,
            H=H,
            K=K,
            V=V,
            BK=BK,
        )
    return initial_state


def chunk_gated_delta_rule_bwd_dhu_pre_process(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    use_exp2: bool = False,
    dht: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    context: FLACPContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if context is None or context.group is None:
        return dht, initial_state
    assert dht is None, "When enable CP, the provided dht must be None."
    rank = dist.get_rank(context.group)

    B, T, H, K, V = *q.shape, do.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = 64
    assert K <= 256, "current kernel does not support head dimension being larger than 256."
    BK = triton.next_power_of_2(K)

    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1

    dhm = q.new_zeros(H, K, V + K, dtype=DTYPE)
    dht = q.new_zeros(N, H, K, V, dtype=DTYPE)
    if not context.is_first_rank:
        BLOCK_SIZE = 32 if K <= 64 else 64
        grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), H)
        pre_process_bwd_kernel_merged[grid](
            q=q,
            k=k,
            w=w,
            g=g,
            gk=gk,
            do=do,
            dhm=dhm,
            dv=dv,
            cu_seqlens=cu_seqlens[:2],
            scale=scale,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK1=BK,
            USE_EXP2=use_exp2,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)
    if not context.is_last_rank:
        def grid(meta): return (triton.cdiv(V, meta['BV']), H)
        merge_fwd_bwd_kernel[grid](
            dht[-1],
            ag_dhm,
            context.post_num_ranks,
            rank,
            H=H,
            K=K,
            V=V,
            BK=BK,
            FORWARD=False
        )
    # initial_state is None in the CP mode
    # We only need to compute dht of current rank and pass it to the backward kernel
    return dht, None


def compress_h0(h0: torch.Tensor, context: FLACPContext):
    if h0 is None or len(context.cu_seqlens) == 2:
        return h0
    # Here must use clone op or the full tensor will be saved for backward
    return h0[:1].clone()


def expand_h0(h0: torch.Tensor, context: FLACPContext):
    if h0 is None or len(context.cu_seqlens) == 2:
        return h0
    B = len(context.cu_seqlens) - 1
    expand_h0 = h0.new_zeros(B, *h0.shape[1:])
    expand_h0[:1] = h0
    return expand_h0
