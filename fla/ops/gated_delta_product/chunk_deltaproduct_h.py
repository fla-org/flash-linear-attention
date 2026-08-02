# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp2
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs

NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8, 16]


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT', 'USE_G'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_product_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    num_householder: tl.constexpr,  # number of delta products
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * tl.cdiv(T // num_householder, BT)

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h) * K*V
    v += (bos * H + i_h) * V
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h) * V
    stride_v = H*V
    stride_h = H*K*V
    stride_k = H*K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    o_k1 = tl.arange(0, 64)
    o_k2 = 64 + tl.arange(0, 64)
    o_k3 = 128 + tl.arange(0, 64)
    o_k4 = 192 + tl.arange(0, 64)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    m_h1 = (o_k1[:, None] < K) & m_v[None, :]
    m_h2 = (o_k2[:, None] < K) & m_v[None, :]
    m_h3 = (o_k3[:, None] < K) & m_v[None, :]
    m_h4 = (o_k4[:, None] < K) & m_v[None, :]

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = h0 + o_k1[:, None] * V + o_v[None, :]
        b_h1 += tl.load(p_h0_1, mask=m_h1, other=0.0).to(tl.float32)
        if K > 64:
            p_h0_2 = h0 + o_k2[:, None] * V + o_v[None, :]
            b_h2 += tl.load(p_h0_2, mask=m_h2, other=0.0).to(tl.float32)
        if K > 128:
            p_h0_3 = h0 + o_k3[:, None] * V + o_v[None, :]
            b_h3 += tl.load(p_h0_3, mask=m_h3, other=0.0).to(tl.float32)
        if K > 192:
            p_h0_4 = h0 + o_k4[:, None] * V + o_v[None, :]
            b_h4 += tl.load(p_h0_4, mask=m_h4, other=0.0).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        if i_t % num_householder == 0:
            i_t_true = i_t_int64 // num_householder
            p_h1 = h + i_t_true * stride_h + o_k1[:, None] * V + o_v[None, :]
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
            if K > 64:
                p_h2 = h + i_t_true * stride_h + o_k2[:, None] * V + o_v[None, :]
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
            if K > 128:
                p_h3 = h + i_t_true * stride_h + o_k3[:, None] * V + o_v[None, :]
                tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
            if K > 192:
                p_h4 = h + i_t_true * stride_h + o_k4[:, None] * V + o_v[None, :]
                tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)

        o_t = i_t_int64 * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_vv = m_t[:, None] & m_v[None, :]
        p_v = v + o_t[:, None] * stride_v + o_v[None, :]
        p_v_new = v_new + o_t[:, None] * stride_v + o_v[None, :] if SAVE_NEW_VALUE else None
        b_v_new = tl.zeros([BT, BV], dtype=tl.float32)
        p_w = w + o_t[:, None] * stride_k + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & (o_k1[None, :] < K), other=0.0)
        b_v_new += tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = w + o_t[:, None] * stride_k + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (o_k2[None, :] < K), other=0.0)
            b_v_new += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = w + o_t[:, None] * stride_k + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (o_k3[None, :] < K), other=0.0)
            b_v_new += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = w + o_t[:, None] * stride_k + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (o_k4[None, :] < K), other=0.0)
            b_v_new += tl.dot(b_w, b_h4.to(b_w.dtype))
        b_v_new = -b_v_new + tl.load(p_v, mask=m_vv, other=0.0)

        if SAVE_NEW_VALUE:
            p_v_new = v_new + o_t[:, None] * stride_v + o_v[None, :]
            tl.store(p_v_new, b_v_new.to(p_v_new.dtype.element_ty), mask=m_vv)

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            last_idx = min((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos * H + i_h + o_t * H
            b_g = tl.load(p_g, mask=m_t, other=0.0)
            b_v_new = b_v_new * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = exp2(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last
        b_v_new = b_v_new.to(k.dtype.element_ty)
        p_k = k + o_k1[:, None] + o_t[None, :] * stride_k
        b_k = tl.load(p_k, mask=(o_k1[:, None] < K) & m_t[None, :], other=0.0)
        b_h1 += tl.dot(b_k, b_v_new)
        if K > 64:
            p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=(o_k2[:, None] < K) & m_t[None, :], other=0.0)
            b_h2 += tl.dot(b_k, b_v_new)
        if K > 128:
            p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=(o_k3[:, None] < K) & m_t[None, :], other=0.0)
            b_h3 += tl.dot(b_k, b_v_new)
        if K > 192:
            p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=(o_k4[:, None] < K) & m_t[None, :], other=0.0)
            b_h4 += tl.dot(b_k, b_v_new)
    # epilogue
    if STORE_FINAL_STATE:
        p_ht = ht + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_ht = ht + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_ht = ht + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_ht = ht + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_h4)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [4, 3, 2]
        for BV in [64, 32]
    ],
    key=['H', 'K', 'V', 'BT', 'BV', 'USE_G'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_product_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    dh += (boh * H + i_h) * K*V
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    do += (bos * H + i_h) * V
    stride_v = H*V
    stride_h = H*K*V
    stride_k = H*K
    if USE_INITIAL_STATE:
        dh0 += i_nh * K*V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K*V

    o_k1 = tl.arange(0, 64)
    o_k2 = 64 + tl.arange(0, 64)
    o_k3 = 128 + tl.arange(0, 64)
    o_k4 = 192 + tl.arange(0, 64)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    m_h1 = (o_k1[:, None] < K) & m_v[None, :]
    m_h2 = (o_k2[:, None] < K) & m_v[None, :]
    m_h3 = (o_k3[:, None] < K) & m_v[None, :]
    m_h4 = (o_k4[:, None] < K) & m_v[None, :]

    if USE_FINAL_STATE_GRADIENT:
        p_dht1 = dht + o_k1[:, None] * V + o_v[None, :]
        b_dh1 += tl.load(p_dht1, mask=m_h1, other=0.0)
        if K > 64:
            p_dht2 = dht + o_k2[:, None] * V + o_v[None, :]
            b_dh2 += tl.load(p_dht2, mask=m_h2, other=0.0)
        if K > 128:
            p_dht3 = dht + o_k3[:, None] * V + o_v[None, :]
            b_dh3 += tl.load(p_dht3, mask=m_h3, other=0.0)
        if K > 192:
            p_dht4 = dht + o_k4[:, None] * V + o_v[None, :]
            b_dh4 += tl.load(p_dht4, mask=m_h4, other=0.0)

    for i_t in range(NT - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t_int64 * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_vv = m_t[:, None] & m_v[None, :]
        p_dh1 = dh + i_t_int64*stride_h + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_dh2 = dh + i_t_int64*stride_h + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_dh3 = dh + i_t_int64*stride_h + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_dh4 = dh + i_t_int64*stride_h + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), mask=m_h4)

        if USE_G:
            last_idx = min((i_t + 1) * BT, T) - 1
            bg_last = tl.load(g + (bos + last_idx) * H + i_h)
            bg_last_exp = exp2(bg_last)
            p_g = g + bos * H + i_h + o_t * H
            b_g = tl.load(p_g, mask=m_t, other=0.0)
            b_g_exp = exp2(b_g)
        else:
            bg_last = None
            last_idx = None
            b_g = None
            b_g_exp = None

        p_dv = dv + o_t[:, None] * stride_v + o_v[None, :]
        p_wo = do + o_t[:, None] * stride_v + o_v[None, :]
        p_dv2 = dv2 + o_t[:, None] * stride_v + o_v[None, :]

        b_wo = tl.load(p_wo, mask=m_vv, other=0.0)
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)

        # Update dv
        p_k = k + o_t[:, None] * stride_k + o_k1[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & (o_k1[None, :] < K), other=0.0)
        b_dv += tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            p_k = k + o_t[:, None] * stride_k + o_k2[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (o_k2[None, :] < K), other=0.0)
            b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        if K > 128:
            p_k = k + o_t[:, None] * stride_k + o_k3[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (o_k3[None, :] < K), other=0.0)
            b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        if K > 192:
            p_k = k + o_t[:, None] * stride_k + o_k4[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (o_k4[None, :] < K), other=0.0)
            b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, mask=m_vv, other=0.0)

        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), mask=m_vv)
        # Update dh
        p_w = w + o_k1[:, None] + o_t[None, :] * stride_k
        p_q = q + o_k1[:, None] + o_t[None, :] * stride_k
        b_w = tl.load(p_w, mask=(o_k1[:, None] < K) & m_t[None, :], other=0.0)
        b_q = tl.load(p_q, mask=(o_k1[:, None] < K) & m_t[None, :], other=0.0)
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        b_q = (b_q * scale).to(b_q.dtype)
        b_dh1 += tl.dot(b_q, b_wo.to(b_q.dtype))-tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 64:
            p_q = q + o_k2[:, None] + o_t[None, :] * stride_k
            p_w = w + o_k2[:, None] + o_t[None, :] * stride_k
            b_q = tl.load(p_q, mask=(o_k2[:, None] < K) & m_t[None, :], other=0.0)
            b_w = tl.load(p_w, mask=(o_k2[:, None] < K) & m_t[None, :], other=0.0)
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh2 += tl.dot(b_q, b_wo.to(b_q.dtype))-tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            p_q = q + o_k3[:, None] + o_t[None, :] * stride_k
            p_w = w + o_k3[:, None] + o_t[None, :] * stride_k
            b_q = tl.load(p_q, mask=(o_k3[:, None] < K) & m_t[None, :], other=0.0)
            b_w = tl.load(p_w, mask=(o_k3[:, None] < K) & m_t[None, :], other=0.0)
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh3 += tl.dot(b_q, b_wo.to(b_q.dtype))-tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            p_q = q + o_k4[:, None] + o_t[None, :] * stride_k
            p_w = w + o_k4[:, None] + o_t[None, :] * stride_k
            b_q = tl.load(p_q, mask=(o_k4[:, None] < K) & m_t[None, :], other=0.0)
            b_w = tl.load(p_w, mask=(o_k4[:, None] < K) & m_t[None, :], other=0.0)
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh4 += tl.dot(b_q, b_wo.to(b_q.dtype))-tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_dh1 = dh0 + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_dh2 = dh0 + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_dh3 = dh0 + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), mask=m_h4)


def chunk_gated_delta_product_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    num_householder: int = 1,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, u.shape[-1]
    assert T % num_householder == 0, "T must be divisible by num_householder"
    T_true = T // num_householder
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens // num_householder, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T_true, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - \
            1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens // num_householder, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    chunk_gated_delta_product_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        num_householder=num_householder,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return h, v_new, final_state


def chunk_gated_delta_product_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    dht: torch.Tensor | None,
    do: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *q.shape, do.shape[-1]

    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = 64
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    chunk_gated_delta_product_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return dh, dh0, dv2
