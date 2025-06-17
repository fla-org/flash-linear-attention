# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang, Yuqi Pan

import warnings
from typing import Optional

import torch
import triton
import triton.language as tl

from fla.modules.layernorm import group_norm
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard, is_nvidia_hopper

NUM_WARPS = [1, 2] if is_nvidia_hopper else [1, 2, 4, 8]


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_INITIAL_STATE_B': lambda args: args['hb0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def fused_chunk_ttt_linear_fwd_kernel(
    q,
    k,
    v,
    eta,
    w,
    b,
    o,
    scale,
    eps,
    h0,
    hb0,
    ht,
    hbt,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_STATE_B: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)

    o_i = tl.arange(0, BT)
    v_i = tl.arange(0, BV)
    m_A = o_i[:, None] >= o_i[None, :]
    b_w = tl.load(w + i_h * V + v_i, mask=v_i < V, other=0.)
    b_b = tl.load(b + i_h * V + v_i, mask=v_i < V, other=0.)

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    # [BV]
    b_hb = tl.zeros([BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    if USE_INITIAL_STATE_B:
        p_hb0 = tl.make_block_ptr(hb0 + i_nh * V, (V,), (1,), (0,), (BV,), (0,))
        b_hb = tl.load(p_hb0, boundary_check=(0,), padding_option="zero").to(tl.float32)

    for i_t in range(NT):
        p_q = tl.make_block_ptr(q+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT, 0), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (K, T), (1, H*K), (0, i_t*BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_e = tl.make_block_ptr(eta+(bos*H+i_h), (T,), (H,), (i_t*BT,), (BT,), (0,))
        p_e_last = eta+bos*H+i_h + (T-1)*H if i_t == NT-1 else eta+bos*H+i_h + (i_t*BT+BT-1)*H
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")

        # [BT, BV]
        b_kh = tl.dot(tl.trans(b_k), b_h.to(b_k.dtype), allow_tf32=False).to(tl.float32) + b_hb[None, :]
        b_kh = tl.where((v_i < V)[None, :], b_kh, 0.)
        mean = tl.sum(b_kh, axis=1, keep_dims=True) / V
        xbar = tl.where((v_i < V)[None, :], b_kh - mean, 0.)
        var = tl.sum(xbar * xbar, axis=1, keep_dims=True) / V
        rstd = 1 / tl.sqrt(var.to(tl.float32) + eps)
        b_kh_hat = (b_kh - mean) * rstd

        b_v = b_kh_hat.to(b_k.dtype) * b_w[None, :].to(b_k.dtype) + \
            b_b[None, :].to(b_k.dtype) - b_v.to(b_k.dtype) + tl.trans(b_k)
        b_v = tl.where((v_i < V)[None, :], b_v * b_w[None, :].to(b_k.dtype), 0.)
        b_v2 = rstd * (V * b_v - tl.sum(b_v, axis=1, keep_dims=True) - b_kh_hat.to(b_k.dtype)
                       * tl.sum(b_v * b_kh_hat.to(b_k.dtype), axis=1, keep_dims=True)) / V

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        # [BT]
        b_e = tl.load(p_e, boundary_check=(0,), padding_option="zero")
        b_q = (b_q * scale).to(b_k.dtype)

        # [BT, BT]
        b_A = tl.dot(b_q, b_k, allow_tf32=False)
        b_A = tl.where(m_A, b_A, 0)
        b_Ae = tl.where(m_A, b_e[:, None], 0.0)

        b_o = - tl.dot(b_e[:, None] * b_A.to(b_v2.dtype), b_v2, allow_tf32=False)
        b_o += b_hb[None, :] - tl.dot(b_Ae.to(b_v2.dtype), b_v2, allow_tf32=False)
        b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
        b_e_last = tl.load(p_e_last)
        b_h = b_h - tl.dot(b_e_last * b_k, b_v2.to(b_k.dtype), allow_tf32=False)
        b_hb = b_hb - tl.sum(b_e_last * b_v2.to(b_k.dtype), axis=0)
        b_h = tl.where((v_i < V)[None, :], b_h, 0.)
        b_hb = tl.where((v_i < V), b_hb, 0.)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0))
        p_hbt = tl.make_block_ptr(hbt + i_nh * V, (V,), (1,), (0,), (BV,), (0,))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_hbt, b_hb.to(p_hbt.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_INITIAL_STATE_B': lambda args: args['hb0'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def fused_chunk_ttt_linear_bwd_kernel_h(
    k,
    v,
    v2,
    x,
    y,
    r,
    w,
    b,
    eta,
    h0,
    hb0,
    h,
    do,
    dq,
    scale,
    eps,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_STATE_B: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H
    bos, _ = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT

    o_i = tl.arange(0, BT)
    v_i = tl.arange(0, BV)
    m_A = o_i[:, None] >= o_i[None, :]
    b_w = tl.load(w + i_h * V + v_i, mask=v_i < V, other=0.)
    b_b = tl.load(b + i_h * V + v_i, mask=v_i < V, other=0.)

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    # [BV]
    b_hb = tl.zeros([BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    if USE_INITIAL_STATE_B:
        p_hb0 = tl.make_block_ptr(hb0 + i_nh * V, (V,), (1,), (0,), (BV,), (0,))
        b_hb = tl.load(p_hb0, boundary_check=(0,), padding_option="zero").to(tl.float32)

    for i_t in range(NT):
        p_h = tl.make_block_ptr(h+((boh+i_t)*H+i_h)*K*V, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0))
        p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (K, T), (1, H*K), (0, i_t*BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_v2 = tl.make_block_ptr(v2+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_x = tl.make_block_ptr(x+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_y = tl.make_block_ptr(y+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_r = tl.make_block_ptr(r+bos*H+i_h, (T, 1), (H, 1), (i_t*BT, 0), (BT, 1), (1, 0))
        p_e = tl.make_block_ptr(eta+(bos*H+i_h), (T,), (H,), (i_t*BT,), (BT,), (0,))
        p_dq = tl.make_block_ptr(dq+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT, 0), (BT, BK), (1, 0))
        p_do = tl.make_block_ptr(do+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_e_last = eta+bos*H+i_h + (T-1)*H if i_t == NT-1 else eta+bos*H+i_h + (i_t*BT+BT-1)*H
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")

        b_kh = tl.dot(tl.trans(b_k), b_h.to(b_k.dtype), allow_tf32=False).to(tl.float32) + b_hb[None, :]
        b_kh = tl.where((v_i < V)[None, :], b_kh, 0.)
        mean = tl.sum(b_kh, axis=1, keep_dims=True) / V
        xbar = tl.where((v_i < V)[None, :], b_kh - mean, 0.)
        var = tl.sum(xbar * xbar, axis=1, keep_dims=True) / V
        rstd = 1 / tl.sqrt(var.to(tl.float32) + eps)
        b_kh_hat = (b_kh - mean) * rstd

        b_v = b_kh_hat.to(b_k.dtype) * b_w[None, :].to(b_k.dtype) + \
            b_b[None, :].to(b_k.dtype) - b_v.to(b_k.dtype) + tl.trans(b_k)
        b_v = tl.where((v_i < V)[None, :], b_v * b_w[None, :].to(b_k.dtype), 0.)
        b_v2 = rstd * (V * b_v - tl.sum(b_v, axis=1, keep_dims=True) - b_kh_hat.to(b_k.dtype)
                       * tl.sum(b_v * b_kh_hat.to(b_k.dtype), axis=1, keep_dims=True)) / V
        tl.store(p_x, b_kh_hat.to(p_x.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_y, b_v.to(p_y.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_r, rstd.to(p_r.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_v2, b_v2.to(p_v2.dtype.element_ty), boundary_check=(0, 1))

        b_e = tl.load(p_e, boundary_check=(0,), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")

        b_v2 = tl.where((v_i < V)[None, :], b_v2, 0.)
        b_ds = tl.dot(b_do, tl.trans(b_v2).to(b_do.dtype))
        b_ds = tl.where(m_A, b_ds, 0)
        b_ds = b_ds.to(b_k.dtype)
        b_dq = tl.dot(b_do, tl.trans(b_h).to(b_do.dtype))
        b_dq -= tl.dot(b_ds, tl.trans(b_k)) * b_e[:, None]
        b_dq *= scale

        b_e_last = tl.load(p_e_last)
        b_h = b_h - tl.dot(b_e_last * b_k, b_v2.to(b_k.dtype), allow_tf32=False)
        b_hb = b_hb - tl.sum(b_e_last * b_v2.to(b_k.dtype), axis=0)
        b_h = tl.where((v_i < V)[None, :], b_h, 0.)
        b_hb = tl.where((v_i < V), b_hb, 0.)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_INITIAL_STATE_B': lambda args: args['dhb0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_FINAL_STATE_GRADIENT_B': lambda args: args['dhbt'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def fused_chunk_ttt_linear_bwd_kernel_dh(
    q,
    k,
    v,
    v2,
    x,
    y,
    r,
    w,
    b,
    eta,
    h,
    dht,
    dhbt,
    dh0,
    dhb0,
    do,
    dk,
    dv,
    de,
    dw,
    db,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_STATE_B: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT_B: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H
    bos, _ = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    # [BV]
    b_dhb = tl.zeros([BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_nh * K*V, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1), padding_option="zero")
    if USE_FINAL_STATE_GRADIENT_B:
        p_dhbt = tl.make_block_ptr(dhbt + i_nh * V, (V,), (1,), (0,), (BV,), (0,))
        b_dhb += tl.load(p_dhbt, boundary_check=(0,), padding_option="zero")

    # [BV]
    o_i = tl.arange(0, BT)
    v_i = tl.arange(0, BV)
    m_A = o_i[:, None] >= o_i[None, :]
    m_A_t = o_i[:, None] <= o_i[None, :]
    b_w = tl.load(w + i_h * V + v_i, mask=v_i < V, other=0.)
    b_b = tl.load(b + i_h * V + v_i, mask=v_i < V, other=0.)
    b_dw = tl.zeros([BV,], dtype=b_w.dtype)
    b_db = tl.zeros([BV,], dtype=b_b.dtype)
    p_dw = tl.make_block_ptr(dw + i_nh * V, (V,), (1,), (0,), (BV,), (0,))
    p_db = tl.make_block_ptr(db + i_nh * V, (V,), (1,), (0,), (BV,), (0,))

    for i_t in range(NT - 1, -1, -1):
        p_h = tl.make_block_ptr(h+((boh+i_t)*H+i_h)*K*V, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
        p_q = tl.make_block_ptr(q+(bos*H+i_h)*K, (K, T), (1, H*K), (0, i_t*BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT, 0), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_v2 = tl.make_block_ptr(v2+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_x = tl.make_block_ptr(x+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_y = tl.make_block_ptr(y+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_r = tl.make_block_ptr(r+bos*H+i_h, (T, 1), (H, 1), (i_t*BT, 0), (BT, 1), (1, 0))
        p_e = tl.make_block_ptr(eta+(bos*H+i_h), (T,), (H,), (i_t*BT,), (BT,), (0,))
        p_dv = tl.make_block_ptr(dv+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT, 0), (BT, BK), (1, 0))
        p_do = tl.make_block_ptr(do+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_de = tl.make_block_ptr(de+(bos*H+i_h), (T,), (H,), (i_t*BT,), (BT,), (0,))
        p_e_last = eta+bos*H+i_h + (T-1)*H if i_t == NT-1 else eta+bos*H+i_h + (i_t*BT+BT-1)*H
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_e = tl.load(p_e, boundary_check=(0,), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        b_e_last = tl.load(p_e_last)
        b_A = tl.dot(b_k, b_q)
        b_A = - tl.where(m_A_t, b_A * scale * b_e[None, :], 0).to(do.dtype.element_ty)
        b_Ae = - tl.where(m_A_t, b_e[None, :], 0).to(do.dtype.element_ty)
        b_dv_new = tl.dot(b_A.to(b_do.dtype), b_do) + tl.dot(b_Ae.to(b_do.dtype), b_do)
        b_dv_new -= tl.dot(b_e_last * b_k, b_dh.to(b_k.dtype))
        b_dv_new -= b_e_last * b_dhb.to(b_k.dtype)[None, :]

        b_v2 = tl.load(p_v2, boundary_check=(0, 1), padding_option="zero").to(b_k.dtype)
        b_x = tl.load(p_x, boundary_check=(0, 1), padding_option="zero").to(b_k.dtype)
        b_y = tl.load(p_y, boundary_check=(0, 1), padding_option="zero").to(b_k.dtype)
        b_rstd = tl.load(p_r, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_dy = b_rstd * (b_dv_new * V - tl.sum(b_dv_new, axis=1, keep_dims=True) -
                         b_x * tl.sum(b_dv_new * b_x, axis=1, keep_dims=True)) / V
        b_dx = -b_rstd * (b_dv_new * tl.sum(b_x * b_y, axis=1, keep_dims=True) +
                          b_y * tl.sum(b_dv_new * b_x, axis=1, keep_dims=True)) / V
        b_drstd = tl.sum(b_dv_new.to(b_rstd.dtype) * b_v2.to(b_rstd.dtype) / b_rstd, axis=1, keep_dims=True)

        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_w = b_w.to(b_k.dtype)
        b_b = b_b.to(b_k.dtype)
        b_dv = -b_w * b_dy.to(b_k.dtype)
        b_dk = b_w * b_dy.to(b_k.dtype)
        b_dw += tl.sum(2 * b_w * b_x * b_dy.to(b_k.dtype) +
                       (b_b - b_v.to(b_k.dtype) + b_k) * b_dy.to(b_k.dtype), axis=0).to(b_dw.dtype)
        b_db += tl.sum(b_w * b_dy.to(b_k.dtype), axis=0).to(b_db.dtype)
        b_dx = b_dx.to(b_k.dtype) + b_w * b_w * b_dy.to(b_k.dtype)

        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_q = (b_q * scale).to(b_q.dtype)
        b_dkh = b_rstd * (V * b_dx - tl.sum(b_dx, axis=1, keep_dims=True) -
                          b_x * tl.sum(b_x * b_dx, axis=1, keep_dims=True)) / V
        b_dkh -= b_rstd * b_rstd * b_drstd * b_x / V
        b_dkh = tl.where((v_i < V)[None, :] * (o_i < T-i_t*BT)[:, None], b_dkh, 0.)
        b_dk += tl.dot(b_dkh, b_h.to(b_dkh.dtype)).to(b_k.dtype)

        b_ds = tl.dot(b_do, tl.trans(b_v2))
        b_ds = tl.where(m_A, b_ds, 0)
        b_ds = b_ds.to(b_k.dtype)
        i_last = (BT-1) if (i_t*BT+BT) <= T else (T % BT-1)
        mask = (o_i == i_last)
        b_dk -= b_e_last * tl.dot(b_v2, tl.trans(b_dh).to(b_v2.dtype))
        b_dk -= tl.dot(tl.trans(b_ds), tl.trans(b_q) * b_e[:, None])
        b_de = mask * tl.sum(- b_dh * tl.trans(tl.dot(tl.trans(b_v2), b_k))).to(b_k.dtype)
        b_de -= mask * tl.sum(b_dhb * tl.sum(b_v2, axis=0)).to(b_k.dtype)
        b_de -= tl.sum(tl.dot(b_ds, b_k) * tl.trans(b_q).to(b_k.dtype), axis=1)
        b_de -= tl.sum(b_ds, axis=1)
        b_dh += tl.dot(b_q, b_do.to(b_q.dtype)) + tl.dot(tl.trans(b_k).to(b_dkh.dtype), b_dkh)
        b_dhb += tl.sum(b_do + b_dkh, axis=0)
        b_dh = tl.where((v_i < V)[None, :], b_dh, 0.)
        b_dhb = tl.where((v_i < V), b_dhb, 0.)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_de, b_de.to(p_de.dtype.element_ty), boundary_check=(0,))
    tl.store(p_dw, b_dw.to(p_dw.dtype.element_ty), boundary_check=(0,))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0+i_nh*K*V, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
    if USE_INITIAL_STATE_B:
        p_dhb0 = tl.make_block_ptr(dhb0+i_nh*V, (V,), (1,), (0,), (BV,), (0,))
        tl.store(p_dhb0, b_dhb.to(p_dhb0.dtype.element_ty), boundary_check=(0,))


def fused_chunk_ttt_linear_bwd_h(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: float,
    eps: float,
    do: torch.Tensor,
    BT: int = 16,
    initial_state: torch.Tensor = None,
    initial_state_bias: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    assert cu_seqlens is None, "bwd of varlen is not implemented yet."
    B, T, H, K, V = *k.shape, v.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    N, NT = B, triton.cdiv(T, BT)
    BK, BV = triton.next_power_of_2(K), triton.next_power_of_2(V)
    assert max(BK, BV) <= 128, "current kernel does not support head dimension larger than 128."

    h = k.new_empty(B, NT, H, K, V)
    r = v.new_empty(B, T, H, 1, dtype=torch.float32)
    v2 = torch.empty_like(v)
    x = torch.empty_like(v)
    y = torch.empty_like(v)
    dq = torch.empty_like(q)

    grid = (N * H,)
    fused_chunk_ttt_linear_bwd_kernel_h[grid](
        k=k,
        v=v,
        v2=v2,
        x=x,
        y=y,
        r=r,
        w=w,
        b=b,
        eta=eta,
        h0=initial_state,
        hb0=initial_state_bias,
        h=h,
        do=do,
        dq=dq,
        scale=scale,
        eps=eps,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dq, h, v2, x, y, r


def fused_chunk_ttt_linear_bwd_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v2: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: float,
    h: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    dhbt: torch.Tensor,
    BT: int = 16,
    initial_state: torch.Tensor = None,
    initial_state_bias: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    assert cu_seqlens is None, "bwd of varlen is not implemented yet."
    B, T, H, K, V = *k.shape, v.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    N = B
    BK, BV = triton.next_power_of_2(K), triton.next_power_of_2(V)
    assert max(BK, BV) <= 128, "current kernel does not support head dimension larger than 128."

    dh0 = torch.empty_like(initial_state, dtype=torch.float32) if initial_state is not None else None
    dhb0 = torch.empty_like(initial_state_bias, dtype=torch.float32) if initial_state_bias is not None else None
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    de = torch.empty_like(eta)
    dw = w.new_empty(B, H, V)
    db = b.new_empty(B, H, V)

    grid = (N * H,)
    fused_chunk_ttt_linear_bwd_kernel_dh[grid](
        q=q,
        k=k,
        v=v,
        v2=v2,
        x=x,
        y=y,
        r=r,
        w=w,
        b=b,
        eta=eta,
        h=h,
        dht=dht,
        dhbt=dhbt,
        dh0=dh0,
        dhb0=dhb0,
        do=do,
        dk=dk,
        dv=dv,
        de=de,
        dw=dw,
        db=db,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    dw = dw.sum(dim=0)
    db = db.sum(dim=0)
    return dk, dv, de, dw, db, dh0, dhb0


def fused_chunk_ttt_linear_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: float,
    eps: float,
    initial_state: torch.Tensor,
    initial_state_bias: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
    BT: int = 16
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), triton.next_power_of_2(V)
    assert max(BK, BV) <= 128, "current kernel does not support head dimension larger than 128."
    o = torch.empty_like(v)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    final_state_bias = k.new_empty(N, H, 1, V, dtype=torch.float32) if output_final_state else None

    grid = (N * H,)
    fused_chunk_ttt_linear_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        eta=eta,
        w=w,
        b=b,
        o=o,
        scale=scale,
        eps=eps,
        h0=initial_state,
        hb0=initial_state_bias,
        ht=final_state,
        hbt=final_state_bias,
        cu_seqlens=cu_seqlens,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return o, final_state, final_state_bias


def fused_chunk_ttt_linear_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: float,
    eps: float,
    do: torch.Tensor,
    dht: torch.Tensor,
    dhbt: torch.Tensor,
    BT: int = 16,
    initial_state: torch.Tensor = None,
    initial_state_bias: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    assert cu_seqlens is None, "bwd of varlen is not implemented yet."
    dq, h, v2, x, y, rstd = fused_chunk_ttt_linear_bwd_h(
        q=q,
        k=k,
        v=v,
        w=w,
        b=b,
        eta=eta,
        scale=scale,
        eps=eps,
        do=do,
        BT=BT,
        initial_state=initial_state,
        initial_state_bias=initial_state_bias,
        cu_seqlens=cu_seqlens,
    )
    dk, dv, de, dw, db, dh0, dhb0 = fused_chunk_ttt_linear_bwd_dh(
        q=q,
        k=k,
        v=v,
        v2=v2,
        x=x,
        y=y,
        r=rstd,
        w=w,
        b=b,
        eta=eta,
        scale=scale,
        h=h,
        do=do,
        dht=dht,
        dhbt=dhbt,
        BT=BT,
        initial_state=initial_state,
        initial_state_bias=initial_state_bias,
        cu_seqlens=cu_seqlens,
    )
    return dq, dk, dv, de, dw, db, dh0, dhb0


class FusedChunkTTTLinearFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, w, b, BT, eta, scale, eps, initial_state,
                initial_state_bias, output_final_state, cu_seqlens):
        o, final_state, final_state_bias = fused_chunk_ttt_linear_fwd(
            q=q,
            k=k,
            v=v,
            w=w,
            b=b,
            eta=eta,
            scale=scale,
            eps=eps,
            BT=BT,
            initial_state=initial_state,
            initial_state_bias=initial_state_bias,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, k, v, eta, w, b, initial_state, initial_state_bias)
        ctx.BT = BT
        ctx.scale = scale
        ctx.eps = eps
        ctx.cu_seqlens = cu_seqlens
        return o.to(q.dtype), final_state, final_state_bias

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht, dhbt):
        q, k, v, eta, w, b, initial_state, initial_state_bias = ctx.saved_tensors
        dq, dk, dv, de, dw, db, dh0, dhb0 = fused_chunk_ttt_linear_bwd(
            q=q,
            k=k,
            v=v,
            w=w,
            b=b,
            eta=eta,
            scale=ctx.scale,
            eps=ctx.eps,
            do=do,
            dht=dht,
            dhbt=dhbt,
            BT=ctx.BT,
            initial_state=initial_state,
            initial_state_bias=initial_state_bias,
            cu_seqlens=ctx.cu_seqlens,
        )
        return dq.to(q), dk.to(k), dv.to(v), dw.to(w), db.to(b), None, de.to(eta), None, None, dh0, dhb0, None, None


def norm_residual(x, weight, bias, eps):
    # GroupNorm and Residual
    B, T, H, D = x.shape
    x += group_norm(
        x.reshape(B, T, -1).clone(),
        weight=weight.reshape(-1).clone(),
        bias=bias.reshape(-1).clone(),
        eps=eps,
        num_groups=H,
    ).reshape(x.shape)
    return x


def fused_chunk_ttt_linear(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: float = None,
    eps: float = 1e-6,
    chunk_size: int = 16,
    initial_state: torch.Tensor = None,
    initial_state_bias: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `(B, H, T, K)`
        k (torch.Tensor):
            keys of shape `(B, H, T, K)`
        v (torch.Tensor):
            values of shape `(B, H, T, V)`
        w (torch.Tensor):
            layer norm weight of shape `(H, V)`
        b (torch.Tensor):
            layer norm bias of shape `(H, V)`
        eta (torch.Tensor):
            Learning rate for hidden state, of shape `(B, H, T, 1)`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        chunk_size (int):
            chunk size. Default: `16`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `(B, H, K, V)`. Default: `None`.
        initial_state_bias (Optional[torch.Tensor]):
            Initial state bias of shape `(B, H, 1, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `(B, H, K, V)`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]`
        final_state (torch.Tensor):
            Final state of shape `[B, H, K, V]` if `output_final_state=True` else `None`.
        final_state_bias (torch.Tensor):
            Final state bias of shape `[B, H, 1, V]` if `output_final_state=True` else `None`.
    """
    assert q.dtype == k.dtype == v.dtype
    assert k.shape[-1] == v.shape[-1], "DK must equal to DV."
    if isinstance(eta, float):
        eta = torch.full_like(q[:, :, :, :1], eta)
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
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
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "Scale must be positive."
    o, final_state, final_state_bias = FusedChunkTTTLinearFunction.apply(
        q,
        k,
        v,
        w,
        b,
        chunk_size,
        eta,
        scale,
        eps,
        initial_state,
        initial_state_bias,
        output_final_state,
        cu_seqlens,
    )
    o = norm_residual(o, w, b, eps)
    return o, final_state, final_state_bias
