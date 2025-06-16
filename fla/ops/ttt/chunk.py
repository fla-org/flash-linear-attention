# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang, Yuqi Pan

import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.modules.layernorm import group_norm
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_INITIAL_STATE_B': lambda args: args['hb0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BT', 'BK', 'BV']
)
@triton.jit(do_not_specialize=['T'])
def chunk_ttt_linear_fwd_kernel_h(
    k,
    v,
    v_new,
    eta,
    w,
    b,
    eps,
    h,
    hb,
    h0,
    hb0,
    ht,
    hbt,
    cu_seqlens,
    chunk_offsets,
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
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    # [BV]
    b_hb = tl.zeros([BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    if USE_INITIAL_STATE_B:
        p_hb0 = tl.make_block_ptr(hb0 + i_nh * V, (V,), (1,), (i_v * BV,), (BV,), (0,))
        b_hb = tl.load(p_hb0, boundary_check=(0,), padding_option="zero").to(tl.float32)

    offs = tl.arange(0, BV)
    b_w = tl.load(w + i_h * V + offs, mask=offs < V, other=0.)
    b_b = tl.load(b + i_h * V + offs, mask=offs < V, other=0.)

    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + ((boh + i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_hb = tl.make_block_ptr(hb + ((boh + i_t) * H + i_h) * V, (V,), (1,), (i_v * BV,), (BV,), (0,))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_hb, b_hb.to(p_hb.dtype.element_ty), boundary_check=(0,))
        p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v_new = tl.make_block_ptr(v_new+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, i_v * BV), (BT, BV), (1, 0))
        p_eta_last = eta+bos*H+i_h + (T-1)*H if i_t == NT-1 else eta+bos*H+i_h + (i_t*BT+BT-1)*H
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")

        b_kh = tl.dot(tl.trans(b_k), b_h.to(b_k.dtype), allow_tf32=False).to(tl.float32) + b_hb[None, :]
        b_kh = tl.where((offs < V)[None, :], b_kh, 0.)
        mean = tl.sum(b_kh, axis=1, keep_dims=True) / V
        xbar = tl.where((offs < V)[None, :], b_kh - mean, 0.)
        var = tl.sum(xbar * xbar, axis=1, keep_dims=True) / V
        rstd = 1 / tl.sqrt(var.to(tl.float32) + eps)
        b_kh_hat = (b_kh - mean) * rstd

        b_v = b_kh_hat.to(b_k.dtype) * b_w[None, :].to(b_k.dtype) + \
            b_b[None, :].to(b_k.dtype) - b_v.to(b_k.dtype) + tl.trans(b_k)
        b_v = tl.where((offs < V)[None, :], b_v * b_w[None, :].to(b_k.dtype), 0.)
        b_v2 = rstd * (V * b_v - tl.sum(b_v, axis=1, keep_dims=True) - b_kh_hat.to(b_k.dtype)
                       * tl.sum(b_v * b_kh_hat.to(b_k.dtype), axis=1, keep_dims=True)) / V
        tl.store(p_v_new, b_v2.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))
        b_eta_last = tl.load(p_eta_last)
        b_h = b_h - tl.dot(b_eta_last * b_k, b_v2.to(b_k.dtype), allow_tf32=False)
        b_hb = b_hb - tl.sum(b_eta_last * b_v2.to(b_k.dtype), axis=0)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_hbt = tl.make_block_ptr(hbt + i_nh * V, (V,), (1,), (i_v * BV,), (BV,), (0,))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_hbt, b_hb.to(p_hbt.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=['BT'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_ttt_linear_fwd_kernel_o(
    q,
    k,
    v,
    eta,
    h,
    hb,
    o,
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
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    eta += bos * H + i_h
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h) * K * V
    hb += (i_tg * H + i_h) * V
    stride_qk = H*K
    stride_vo = H*V
    stride_eta = H

    p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (K, T), (1, stride_qk), (0, i_t * BT), (BK, BT), (0, 1))
    p_eta = tl.make_block_ptr(eta, (T,), (stride_eta,), (i_t * BT,), (BT,), (0,))
    p_h = tl.make_block_ptr(h, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
    p_hb = tl.make_block_ptr(hb, (V,), (1,), (i_v * BV,), (BV,), (0,))
    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    # [BK, BT]
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    # [BT, 1]
    b_eta = tl.load(p_eta, boundary_check=(0,), padding_option="zero")
    # [BK, BV]
    b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
    # [BV]
    b_hb = tl.load(p_hb, boundary_check=(0,), padding_option="zero")
    # [BT, BK] @ [BK, BV] -> [BT, BV]
    b_o = tl.dot(b_q, b_h, allow_tf32=False)
    # [BT, BK] @ [BK, BT] -> [BT, BT]
    b_A = tl.dot(b_q, b_k, allow_tf32=False)

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_A = tl.where(m_A, b_A, 0)
    b_Ae = tl.where(m_A, b_eta[:, None], 0.0)

    p_v = tl.make_block_ptr(v, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_o = (b_o - tl.dot(b_eta[:, None] * b_A.to(b_v.dtype), b_v, allow_tf32=False)) * scale
    b_o += b_hb[None, :] - tl.dot(b_Ae.to(b_v.dtype), b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_INITIAL_STATE_B': lambda args: args['hb0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_ttt_linear_bwd_kernel_h(
    k,
    v,
    v_new,
    eta,
    w,
    b,
    eps,
    h,
    h0,
    hb0,
    x,
    y,
    r,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_STATE_B: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    # [BV]
    b_hb = tl.zeros([BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    if USE_INITIAL_STATE_B:
        p_hb0 = tl.make_block_ptr(hb0 + i_nh * V, (V,), (1,), (i_v * BV,), (BV,), (0,))
        b_hb = tl.load(p_hb0, boundary_check=(0,), padding_option="zero").to(tl.float32)

    offs = tl.arange(0, BV)
    b_w = tl.load(w + i_h * V + offs, mask=offs < V, other=0.)
    b_b = tl.load(b + i_h * V + offs, mask=offs < V, other=0.)

    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + ((boh + i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v_new = tl.make_block_ptr(v_new+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, i_v * BV), (BT, BV), (1, 0))
        p_x = tl.make_block_ptr(x+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, i_v * BV), (BT, BV), (1, 0))
        p_y = tl.make_block_ptr(y+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, i_v * BV), (BT, BV), (1, 0))
        p_r = tl.make_block_ptr(r+bos*H+i_h, (T, 1), (H, 1), (i_t*BT, 0), (BT, 1), (1, 0))
        p_eta_last = eta+bos*H+i_h + (T-1)*H if i_t == NT-1 else eta+bos*H+i_h + (i_t*BT+BT-1)*H
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")

        b_kh = tl.dot(tl.trans(b_k), b_h.to(b_k.dtype), allow_tf32=False).to(tl.float32) + b_hb[None, :]
        b_kh = tl.where((offs < V)[None, :], b_kh, 0.)
        mean = tl.sum(b_kh, axis=1, keep_dims=True) / V
        xbar = tl.where((offs < V)[None, :], b_kh - mean, 0.)
        var = tl.sum(xbar * xbar, axis=1, keep_dims=True) / V
        rstd = 1 / tl.sqrt(var.to(tl.float32) + eps)
        b_kh_hat = (b_kh - mean) * rstd

        b_v = b_kh_hat.to(b_k.dtype) * b_w[None, :].to(b_k.dtype) + \
            b_b[None, :].to(b_k.dtype) - b_v.to(b_k.dtype) + tl.trans(b_k)
        b_v = tl.where((offs < V)[None, :], b_v * b_w[None, :].to(b_k.dtype), 0.)
        b_v2 = rstd * (V * b_v - tl.sum(b_v, axis=1, keep_dims=True) - b_kh_hat.to(b_k.dtype)
                       * tl.sum(b_v * b_kh_hat.to(b_k.dtype), axis=1, keep_dims=True)) / V
        tl.store(p_x, b_kh_hat.to(p_x.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_y, b_v.to(p_y.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_r, rstd.to(p_r.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_v_new, b_v2.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))
        b_eta_last = tl.load(p_eta_last)
        b_h = b_h - tl.dot(b_eta_last * b_k, b_v2.to(b_k.dtype), allow_tf32=False)
        b_hb = b_hb - tl.sum(b_eta_last * b_v2.to(b_k.dtype), axis=0)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [4]
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_ttt_linear_bwd_kernel_dv_local(
    q,
    k,
    eta,
    do,
    dv,
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
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    eta += bos * H + i_h
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    stride_qk = H*K
    stride_vo = H*V
    stride_eta = H

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_q = tl.make_block_ptr(q, (K, T), (1, stride_qk), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, b_q)

    p_eta = tl.make_block_ptr(eta, (T,), (stride_eta,), (i_t * BT,), (BT,), (0,))
    b_eta = tl.load(p_eta, boundary_check=(0,))
    mask = (tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :])
    b_A = - tl.where(mask, b_A * scale * b_eta[None, :], 0).to(do.dtype.element_ty)
    b_Ae = - tl.where(mask, b_eta[None, :], 0).to(do.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do) + tl.dot(b_Ae.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_FINAL_STATE_GRADIENT_B': lambda args: args['dhbt'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_INITIAL_STATE_B': lambda args: args['dhb0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8, 16]
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_ttt_linear_bwd_kernel_norm(
    q,
    k,
    v,
    v_new,
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
    dh,
    dhb,
    dv,
    dv_new,
    dk,
    dw,
    db,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT_B: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_STATE_B: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    # [BV]
    b_dhb = tl.zeros([BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1), padding_option="zero")
    if USE_FINAL_STATE_GRADIENT_B:
        p_dhbt = tl.make_block_ptr(dhbt + i_nh * V, (V,), (1,), (i_v * BV,), (BV,), (0,))
        b_dhb += tl.load(p_dhbt, boundary_check=(0,), padding_option="zero")

    # [BV]
    offs_v = tl.arange(0, BV)
    offs_t = tl.arange(0, BT)
    b_w = tl.load(w + i_h * V + offs_v, mask=offs_v < V, other=0.)
    b_b = tl.load(b + i_h * V + offs_v, mask=offs_v < V, other=0.)
    b_dw = tl.zeros([BV,], dtype=b_w.dtype)
    b_db = tl.zeros([BV,], dtype=b_b.dtype)
    p_dw = tl.make_block_ptr(dw + i_nh * V, (V,), (1,), (i_v * BV,), (BV,), (0,))
    p_db = tl.make_block_ptr(db + i_nh * V, (V,), (1,), (i_v * BV,), (BV,), (0,))

    for i_t in range(NT - 1, -1, -1):
        p_h = tl.make_block_ptr(h + ((boh+i_t) * H + i_h) * K*V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh + ((boh+i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dhb = tl.make_block_ptr(dhb + ((boh+i_t) * H + i_h) * V, (V,), (1,), (i_v * BV,), (BV,), (0,))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dhb, b_dhb.to(p_dhb.dtype.element_ty), boundary_check=(0,))
        p_q = tl.make_block_ptr(q+(bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v_new = tl.make_block_ptr(v_new+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_x = tl.make_block_ptr(x+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_y = tl.make_block_ptr(y+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv_new = tl.make_block_ptr(dv_new+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT, i_k * BK), (BT, BK), (1, 0))
        p_do = tl.make_block_ptr(do+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, i_v * BV), (BT, BV), (1, 0))
        p_r = tl.make_block_ptr(r+bos*H+i_h, (T, 1), (H, 1), (i_t*BT, 0), (BT, 1), (1, 0))
        p_eta_last = eta+bos*H+i_h + (T-1)*H if i_t == NT-1 else eta+bos*H+i_h + (i_t*BT+BT-1)*H
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_dv_new = tl.load(p_dv_new, boundary_check=(0, 1), padding_option="zero").to(b_k.dtype)
        b_eta_last = tl.load(p_eta_last)
        b_dv_new -= tl.dot(b_eta_last * b_k, b_dh.to(b_k.dtype))
        b_dv_new -= b_eta_last * b_dhb.to(b_k.dtype)[None, :]

        b_v_new = tl.load(p_v_new, boundary_check=(0, 1), padding_option="zero")
        b_x = tl.load(p_x, boundary_check=(0, 1), padding_option="zero").to(b_k.dtype)
        b_y = tl.load(p_y, boundary_check=(0, 1), padding_option="zero").to(b_k.dtype)
        b_rstd = tl.load(p_r, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_dy = b_rstd * (b_dv_new * V - tl.sum(b_dv_new, axis=1, keep_dims=True) -
                         b_x * tl.sum(b_dv_new * b_x, axis=1, keep_dims=True)) / V
        b_dx = -b_rstd * (b_dv_new * tl.sum(b_x * b_y, axis=1, keep_dims=True) +
                          b_y * tl.sum(b_dv_new * b_x, axis=1, keep_dims=True)) / V
        b_drstd = tl.sum(b_dv_new.to(b_rstd.dtype) * b_v_new.to(b_rstd.dtype) / b_rstd, axis=1, keep_dims=True)

        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_w = b_w.to(b_k.dtype)
        b_b = b_b.to(b_k.dtype)
        b_dv = -b_w * b_dy.to(b_k.dtype)
        b_dk = b_w * b_dy.to(b_k.dtype)
        b_dw += tl.sum(2 * b_w * b_x * b_dy.to(b_k.dtype) +
                       (b_b - b_v.to(b_k.dtype) + b_k) * b_dy.to(b_k.dtype), axis=0).to(b_dw.dtype)
        b_db += tl.sum(b_w * b_dy.to(b_k.dtype), axis=0).to(b_db.dtype)
        b_dx = b_dx.to(b_k.dtype) + b_w * b_w * b_dy.to(b_k.dtype)

        # d_rstd, dx --> dkh --> dk, dh
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        b_q = (b_q * scale).to(b_q.dtype)
        b_dkh = b_rstd * (V * b_dx - tl.sum(b_dx, axis=1, keep_dims=True) -
                          b_x * tl.sum(b_x * b_dx, axis=1, keep_dims=True)) / V
        b_dkh -= b_rstd * b_rstd * b_drstd * b_x / V
        b_dkh = tl.where((offs_v < V)[None, :] * (offs_t < T-i_t*BT)[:, None], b_dkh, 0.)
        b_dk += tl.dot(b_dkh, b_h.to(b_dkh.dtype)).to(b_k.dtype)
        b_dh += tl.dot(b_q, b_do.to(b_q.dtype)) + tl.dot(tl.trans(b_k).to(b_dkh.dtype), b_dkh)
        b_dhb += tl.sum(b_do + b_dkh, axis=0)
        b_dh = tl.where((offs_v < V)[None, :], b_dh, 0.)
        b_dhb = tl.where((offs_v < V), b_dhb, 0.)

        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dw, b_dw.to(p_dw.dtype.element_ty), boundary_check=(0,))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
    if USE_INITIAL_STATE_B:
        p_dhb0 = tl.make_block_ptr(dhb0+i_nh*V, (V,), (1,), (i_v * BV,), (BV,), (0,))
        tl.store(p_dhb0, b_dhb.to(p_dhb0.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dqke(
    q,
    k,
    v,
    e,
    h,
    do,
    dh,
    dhb,
    dq,
    dk,
    de,
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
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h += (i_tg * H + i_h) * K * V
    dh += (i_tg * H + i_h) * K * V
    dhb += (i_tg * H + i_h) * V
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    e += bos * H + i_h
    de += bos * H + i_h
    stride_qk = H*K
    stride_vo = H*V
    stride_e = H

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_de = tl.zeros([BT,], dtype=tl.float32)

    p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    p_e_last = (e + (i_t*BT+BT-1)*stride_e) if (i_t*BT+BT) <= T else (e + (T-1)*stride_e)
    i_last = (BT-1) if (i_t*BT+BT) <= T else (T % BT-1)
    mask = (tl.arange(0, BT) == i_last)
    b_e_last = tl.load(p_e_last)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dhb = tl.make_block_ptr(dhb, (V,), (1,), (i_v * BV,), (BV,), (0,))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        # [BV]
        b_dhb = tl.load(p_dhb, boundary_check=(0,))
        # [BT, BV] @ [BV, BT] -> [BT, BT]
        b_ds += tl.dot(b_do, tl.trans(b_v))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dk -= b_e_last * tl.dot(b_v, b_dh.to(b_v.dtype))
        b_de -= mask * tl.sum(tl.trans(b_dh) * tl.dot(tl.trans(b_k), b_v.to(b_k.dtype)))
        b_de -= mask * tl.sum(b_dhb * tl.sum(b_v, axis=0).to(b_k.dtype))

    o_i = tl.arange(0, BT)
    p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_e = tl.make_block_ptr(e, (T,), (stride_e,), (i_t * BT,), (BT,), (0,))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_e = tl.load(p_e, boundary_check=(0,))

    p_dq = tl.make_block_ptr(dq, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_de = tl.make_block_ptr(de, (T,), (stride_e,), (i_t * BT,), (BT,), (0,))

    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds, 0)
    b_ds = b_ds.to(b_k.dtype)
    b_dq -= tl.dot(b_ds, b_k) * b_e[:, None]
    b_dk -= tl.dot(tl.trans(b_ds), b_q * b_e[:, None]) * scale
    b_de -= tl.sum(scale * tl.dot(b_ds, b_k) * b_q, axis=1)
    b_de -= tl.sum(b_ds, axis=1)
    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_de, b_de.to(p_de.dtype.element_ty), boundary_check=(0,))


def chunk_ttt_linear_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    eps: float,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_bias: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    assert max(BK, BV) <= 128, "current kernel does not support head dimension larger than 128."
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'
    assert NV == 1, 'NV > 1 is not supported by TTT update rule.'

    h = k.new_empty(B, NT, H, K, V)
    hb = k.new_empty(B, NT, H, 1, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    final_state_bias = k.new_empty(N, H, 1, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(v)
    grid = (NK, NV, N * H)

    chunk_ttt_linear_fwd_kernel_h[grid](
        k=k,
        v=v,
        v_new=v_new,
        eta=eta,
        w=w,
        b=b,
        eps=eps,
        h=h,
        hb=hb,
        h0=initial_state,
        hb0=initial_state_bias,
        ht=final_state,
        hbt=final_state_bias,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return h, hb, v_new, final_state, final_state_bias


def chunk_ttt_linear_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eta: torch.Tensor,
    h: torch.Tensor,
    hb: torch.Tensor,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64
) -> torch.Tensor:
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = k.shape[-1] ** -0.5
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'
    assert NV == 1, 'NV > 1 is not supported by TTT update rule.'

    o = torch.empty_like(v)

    grid = (NV, NT, B * H)
    chunk_ttt_linear_fwd_kernel_o[grid](
        q,
        k,
        v,
        eta,
        h,
        hb,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return o


def chunk_ttt_linear_bwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    eps: float,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    assert max(BK, BV) <= 128, "current kernel does not support head dimension larger than 128."
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'
    assert NV == 1, 'NV > 1 is not supported by TTT update rule.'

    h = k.new_empty(B, NT, H, K, V)
    rstd = v.new_empty(B, T, H, 1, dtype=torch.float32)
    x = torch.empty_like(v)
    y = torch.empty_like(v)

    v_new = torch.empty_like(v)
    grid = (NK, NV, N * H)

    chunk_ttt_linear_bwd_kernel_h[grid](
        k=k,
        v=v,
        v_new=v_new,
        eta=eta,
        w=w,
        b=b,
        eps=eps,
        h=h,
        h0=initial_state,
        hb0=initial_state_bias,
        x=x,
        y=y,
        r=rstd,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NT=NT,
    )
    return h, v_new, x, y, rstd


def chunk_ttt_linear_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    eta: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 16
) -> torch.Tensor:
    B, T, H, K, V = *k.shape, do.shape[-1]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(triton.next_power_of_2(K), 128)
    BV = min(triton.next_power_of_2(V), 128)

    dv = torch.empty_like(do)
    grid = (NT, B * H)
    chunk_ttt_linear_bwd_kernel_dv_local[grid](
        q,
        k,
        eta,
        do,
        dv,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dv


def chunk_ttt_linear_bwd_norm(
    q: torch.Tensor,  # [B, H, L, D]
    k: torch.Tensor,  # [B, H, L, D]
    v: torch.Tensor,  # [B, H, L, D]
    v_new: torch.Tensor,  # [B, H, L, D]
    x: torch.Tensor,  # [B, H, L, D]
    y: torch.Tensor,  # [B, H, L, D]
    rstd: torch.Tensor,  # [B, H, L, 1]
    w: torch.Tensor,  # [H, D]
    b: torch.Tensor,  # [H, D]
    eta: torch.Tensor,  # [B, H, L, 1]
    h0: torch.Tensor,  # [B, H, D, D]
    hb0: torch.Tensor,  # [B, H, 1, D]
    h: torch.Tensor,  # [B, H, NT, D, D]
    dht: Optional[torch.Tensor],  # [B, H, D, D]
    dhbt: Optional[torch.Tensor],  # [B, H, 1, D]
    dv_new: Optional[torch.Tensor],  # [B, H, L, D]
    do: torch.Tensor,  # [B, H, L, D]
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # torch implementation of `dkh, dw, db, dk, dv` for LN^2
    assert cu_seqlens is None, "bwd of varlen is not implemented yet."
    B, T, H, K, V = *q.shape, do.shape[-1]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported by TTT.'
    assert NV == 1, 'NV > 1 is not supported by TTT.'

    dh = q.new_empty(B, NT, H, K, V)
    dhb = q.new_empty(B, NT, H, 1, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dhb0 = torch.empty_like(hb0, dtype=torch.float32) if hb0 is not None else None
    dv = torch.empty_like(v)
    dk = torch.empty_like(k)
    dw = w.new_empty(B, H, V)
    db = b.new_empty(B, H, V)

    grid = (NK, NV, N * H)
    chunk_ttt_linear_bwd_kernel_norm[grid](
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        x=x,
        y=y,
        r=rstd,
        w=w,
        b=b,
        eta=eta,
        h=h,
        dht=dht,
        dhbt=dhbt,
        dh0=dh0,
        dhb0=dhb0,
        do=do,
        dh=dh,
        dhb=dhb,
        dv=dv,
        dv_new=dv_new,
        dk=dk,
        dw=dw,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
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
    return dh, dhb, dh0, dhb0, dv, dk, dw, db


def chunk_ttt_linear_bwd_norm_ref(
    q: torch.Tensor,  # [B, H, L, D]
    k: torch.Tensor,  # [B, H, L, D]
    v: torch.Tensor,  # [B, H, L, D]
    v_new: torch.Tensor,  # [B, H, L, D]
    kh: torch.Tensor,  # [B, H, L, D]
    y: torch.Tensor,  # [B, H, L, D]
    w: torch.Tensor,  # [H, D]
    b: torch.Tensor,  # [H, D]
    eta: torch.Tensor,  # [B, H, L, 1]
    h0: torch.Tensor,  # [B, H, D, D]
    h: torch.Tensor,  # [B, H, NT, D, D]
    dht: Optional[torch.Tensor],  # [B, H, D, D]
    dv_new: Optional[torch.Tensor],  # [B, H, L, D]
    do: torch.Tensor,  # [B, H, L, D]
    scale: float,
    eps: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # torch implementation of `dkh, dw, db, dk, dv` for LN^2
    assert cu_seqlens is None, "bwd of varlen is not implemented yet."
    B, T, H, K, V = *q.shape, do.shape[-1]
    # [B, L, H, D] -> [B, H, L, D]
    q, k, v, v_new, kh, y, h, eta, dv_new, do = [
        x.transpose(1, 2) for x in
        [q, k, v, v_new, kh, y, h, eta, dv_new, do]
    ]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        q, k, v, v_new, kh, y, eta, dv_new, do = [
            F.pad(x, (0, 0, 0, pad_len)) for x in
            [q, k, v, v_new, kh, y, eta, dv_new, do]
        ]
        eta[:, :, -1, :] = eta[:, :, -(pad_len+1), :]
    # [NT, B, H, BT, D]
    q, k, v, v_new, kh, y, eta, dv_new, do = [
        x.reshape(B, H, NT, BT, -1).permute(2, 0, 1, 3, 4) for x in
        [q, k, v, v_new, kh, y, eta, dv_new, do]
    ]
    h = h.permute(2, 0, 1, 3, 4)

    # allocate
    dh = q.new_zeros(NT, B, H, K, V)
    dv = torch.zeros_like(v)
    dk = torch.zeros_like(k)
    dw = torch.zeros_like(w)
    db = torch.zeros_like(b)
    # recurrent state
    b_dh = dht if dht is not None else torch.zeros_like(dh[0])
    b_dh = b_dh.to(torch.float32)

    # [H, 1, D]
    _w = w.reshape(H, 1, V).to(torch.float32)
    _b = b.reshape(H, 1, V).to(torch.float32)

    # d_state passing
    for i_t in range(NT - 1, -1, -1):
        dh[i_t] = b_dh.to(dh.dtype)
        # [B, H, BT, D]
        _q, _k, _v, _v_new, _kh, _y, _h, _eta, _dv_new, _do = [
            x[i_t].to(torch.float32) for x in
            (q, k, v, v_new, kh, y, h, eta, dv_new, do)
        ]
        _dv_new -= (_eta[:, :, -1, :, None] * _k) @ b_dh

        mean = _kh.mean(dim=-1, keepdim=True)
        var = _kh.var(dim=-1, unbiased=False, keepdim=True).to(torch.float32)
        rstd = 1 / torch.sqrt(var + eps).to(torch.float32)
        x = (_kh - mean) * rstd
        # [B, H, BT, D]
        dy = rstd * (_dv_new*V - _dv_new.sum(dim=-1, keepdim=True) - x*(x*_dv_new).sum(dim=-1, keepdim=True)) / V
        dx = -rstd * (_dv_new*(x*_y).sum(dim=-1, keepdim=True) + _y*(x*_dv_new).sum(dim=-1, keepdim=True)) / V
        d_rstd = (_dv_new * _v_new / rstd).sum(dim=-1, keepdim=True)

        dv[i_t] = (-_w*dy).to(dv.dtype)
        dk[i_t] += (_w*dy).to(dk.dtype)
        dw += (2*_w*x*dy+(_b-_v+_k)*dy).sum(dim=(0, 2)).to(dw.dtype)
        db += (_w*dy).sum(dim=(0, 2)).to(db.dtype)
        dx += _w*_w*dy

        # d_rstd, dx --> dkh --> dk, dh
        dkh = rstd * (V * dx - dx.sum(dim=-1, keepdim=True) - x * (x * dx).sum(dim=-1, keepdim=True)) / V
        dkh -= rstd**2 * d_rstd * x / V
        dk[i_t] += (dkh @ _h.transpose(-2, -1)).to(dk.dtype)
        b_dh += (_q.transpose(-2, -1) * scale) @ _do + _k.transpose(-2, -1) @ dkh
    dh0 = b_dh.to(torch.float32) if h0 is not None else None

    # [NT, B, H, BT, D] -> [B, H, T, D]
    dv = dv.permute(1, 2, 0, 3, 4).reshape(B, H, -1, V)[:, :, :T, :]
    dk = dk.permute(1, 2, 0, 3, 4).reshape(B, H, -1, K)[:, :, :T, :]
    # [B, H, NT, D, D]
    dh = dh.permute(1, 2, 0, 3, 4)
    dv, dk, dh = [x.transpose(1, 2) for x in (dv, dk, dh)]
    dh, dv, dk, dw, db = [x.contiguous() for x in (dh, dv, dk, dw, db)]
    dh0 = dh0.contiguous() if h0 is not None else None
    return dh, dh0, dv, dk, dw, db


def chunk_ttt_linear_bwd_dqke(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eta: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dhb: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 64)
    NK = triton.cdiv(K, BK)
    assert NK == 1, "NK > 1 is not supported."

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    de = torch.empty_like(eta)
    grid = (NK, NT, B * H)

    chunk_bwd_kernel_dqke[grid](
        q=q,
        k=k,
        v=v,
        e=eta,
        h=h,
        do=do,
        dh=dh,
        dhb=dhb,
        dq=dq,
        dk=dk,
        de=de,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dq, dk, de


def chunk_ttt_linear_fwd(
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
    chunk_size: int = 16
):
    BT = chunk_size
    h, hb, v_new, final_state, final_state_bias = chunk_ttt_linear_fwd_h(
        k=k,
        v=v,
        w=w,
        b=b,
        eta=eta,
        eps=eps,
        initial_state=initial_state,
        initial_state_bias=initial_state_bias,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    o = chunk_ttt_linear_fwd_o(
        q=q,
        k=k,
        v=v_new,
        eta=eta,
        h=h,
        hb=hb,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    return o, final_state, final_state_bias


def chunk_ttt_linear_bwd(
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
    chunk_size: int = 16,
    initial_state: torch.Tensor = None,
    initial_state_bias: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    BT = chunk_size
    h, v_new, x, y, rstd = chunk_ttt_linear_bwd_h(
        k=k,
        v=v,
        w=w,
        b=b,
        eta=eta,
        eps=eps,
        initial_state=initial_state,
        initial_state_bias=initial_state_bias,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    dv_new = chunk_ttt_linear_bwd_dv_local(
        q=q,
        k=k,
        eta=eta,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    dh, dhb, dh0, dhb0, dv, dk, dw, db = chunk_ttt_linear_bwd_norm(
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        x=x,
        y=y,
        rstd=rstd,
        w=w,
        b=b,
        eta=eta,
        h0=initial_state,
        hb0=initial_state_bias,
        h=h,
        dht=dht,
        dhbt=dhbt,
        dv_new=dv_new,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    dq, dk2, de = chunk_ttt_linear_bwd_dqke(
        q=q,
        k=k,
        v=v_new,
        eta=eta,
        h=h,
        do=do,
        dh=dh,
        dhb=dhb,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    dk.add_(dk2)
    return dq, dk, dv, de, dw, db, dh0, dhb0


class ChunkTTTLinearFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
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
        cu_seqlens
    ):
        o, final_state, final_state_bias = chunk_ttt_linear_fwd(
            q=q,
            k=k,
            v=v,
            w=w,
            b=b,
            eta=eta,
            scale=scale,
            eps=eps,
            chunk_size=chunk_size,
            initial_state=initial_state,
            initial_state_bias=initial_state_bias,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, k, v, eta, w, b, initial_state, initial_state_bias)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.eps = eps
        ctx.cu_seqlens = cu_seqlens
        return o.to(q.dtype), final_state, final_state_bias

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht, dhbt):
        q, k, v, eta, w, b, initial_state, initial_state_bias = ctx.saved_tensors
        dq, dk, dv, de, dw, db, dh0, dhb0 = chunk_ttt_linear_bwd(
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
            chunk_size=ctx.chunk_size,
            initial_state=initial_state,
            initial_state_bias=initial_state_bias,
            cu_seqlens=ctx.cu_seqlens,
        )
        return dq.to(q), dk.to(k), dv.to(v), dw.to(w), db.to(b), None, de.to(eta), None, None, dh0, dhb0, None, None, None


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


def chunk_ttt_linear(
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
            Final state of shape `[B, H, K, V]` if `output_final_state=True` else `None`
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
    o, final_state, final_state_bias = ChunkTTTLinearFunction.apply(
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
