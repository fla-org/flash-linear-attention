# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=["BK", "BV"]
)
@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(
    q,  # query [B, H, T, K]/[B, T, H, K]
    k,  # key [B, H, T, K]/[B, T, H, K]
    v,  # value [B, H, T, V]/[B, T, H, V]
    w,  # log gate [B, H, T]/[B, T, H] or None
    u,  # bonus [B, H, K]
    o,  # output [NK, B, H, T, V]/[NK, B, T, H, V]
    h0,  # initial hidden state [B, H, K, V]
    ht,  # final hidden state [B, H, K, V]
    offsets,
    scale,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,  # whether to reverse the recurrence
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    if HEAD_FIRST:
        p_q = q + i_nh * T*K + ((T-1) * K if REVERSE else 0) + o_k
        p_k = k + i_nh * T*K + ((T-1) * K if REVERSE else 0) + o_k
        p_v = v + i_nh * T*V + ((T-1) * V if REVERSE else 0) + o_v
        p_w = w + i_nh * T*K + ((T-1) * K if REVERSE else 0) + o_k
        p_o = o + (i_k * B*H + i_nh) * T*V + ((T-1) * V if REVERSE else 0) + o_v
    else:
        p_q = q + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
        p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
        p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
        p_w = w + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
        p_o = o + ((i_k * all + bos) + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_u = u + i_h * K + o_k

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_u = tl.load(p_u, mask=mask_k, other=0).to(tl.float32)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_kv = b_k[:, None] * b_v[None, :]
        b_o = tl.sum((b_h + b_kv * b_u[:, None]) * b_q[:, None], 0)
        b_h = b_h * tl.exp(b_w)[:, None] + b_kv
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_k += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_v += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V
        p_w += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_o += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=["BK", "BV"]
)
@triton.jit
def fused_recurrent_rwkv6_bwd_kernel_dq(
    k,  # key [B, H, T, V]/[B, T, H, V]
    v,  # value [B, H, T, V]/[B, T, H, V]
    w,  # log gate [B, H, T]/[B, T, H]
    u,  # bonus [B, H, K]
    do,  # gradient of output [B, H, T, V]/[B, T, H, V]
    dq,  # gradient of query [NV, B, H, T, K]/[NV, B, T, H, K]
    dq1,  # gradient of query_aux [NV, B, H, T, K]/[NV, B, T, H, K]
    h0,
    offsets,
    scale,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    if HEAD_FIRST:
        p_k = k + i_nh * T*K + ((T-1) * K if REVERSE else 0) + o_k
        p_v = v + i_nh * T*V + ((T-1) * V if REVERSE else 0) + o_v
        p_w = w + i_nh * T*K + ((T-1) * K if REVERSE else 0) + o_k
        p_do = do + i_nh * T*V + ((T-1) * V if REVERSE else 0) + o_v
        p_dq = dq + (i_v * B*H + i_nh) * T*K + ((T-1) * K if REVERSE else 0) + o_k
        p_dq1 = dq1 + (i_v * B*H + i_nh) * T*K + ((T-1) * K if REVERSE else 0) + o_k
    else:
        p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
        p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
        p_w = w + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
        p_do = do + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
        p_dq = dq + ((i_v * all + bos) + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
        p_dq1 = dq1 + ((i_v * all + bos) + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_u = u + i_h * K + o_k

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_u = tl.load(p_u, mask=mask_k, other=0).to(tl.float32)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_kv = b_k[:, None] * b_v[None, :]

        b_hq = b_h * b_do[None, :]
        b_dq = tl.sum(b_hq + b_kv * b_u[:, None] * b_do[None, :], 1) * scale
        b_dq1 = tl.sum(b_hq, 1)
        b_h = b_h * tl.exp(b_w)[:, None]
        b_h += b_kv
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)
        tl.store(p_dq1, b_dq1.to(p_dq1.dtype.element_ty), mask=mask_k)

        p_k += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_v += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V
        p_w += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_do += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V
        p_dq += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_dq1 += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=["BK", "BV"]
)
@triton.jit
def fused_recurrent_rwkv6_bwd_kernel_dkv(
    q,  # query [B, H, T, K]/[B, T, H, K]
    k,  # key [B, H, T, V]/[B, T, H, V]
    v,  # value [B, H, T, V]/[B, T, H, V]
    w,  # log gate [B, H, T]/[B, T, H]
    u,  # bonus [B, H, K]
    do,  # gradient of output [B, H, T, V]/[B, T, H, V]
    dk,  # gradient of key [NV, B, H, T, K]/[NK, B, T, H, K]
    dk1,  # gradient of key_aux [NV, B, H, T, K]/[NK, B, T, H, K]
    dv,  # gradient of value [NK, B, H, T, V]/[NV, B, T, H, V]
    dh0,  # gradient of initial hidden state [N, H, K, V]
    offsets,
    scale,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    if HEAD_FIRST:
        p_q = q + i_nh * T*K + ((T-1) * K if not REVERSE else 0) + o_k
        p_k = k + i_nh * T*K + ((T-1) * K if not REVERSE else 0) + o_k
        p_v = v + i_nh * T*V + ((T-1) * V if not REVERSE else 0) + o_v
        p_w = w + i_nh * T*K + ((T-1) * K if not REVERSE else 0) + o_k
        p_do = do + i_nh * T*V + ((T-1) * V if not REVERSE else 0) + o_v
        p_dk = dk + (i_v * B*H + i_nh) * T*K + ((T-1) * K if not REVERSE else 0) + o_k
        p_dk1 = dk1 + (i_v * B*H + i_nh) * T*K + ((T-1) * K if not REVERSE else 0) + o_k
        p_dv = dv + (i_k * B*H + i_nh) * T*V + ((T-1) * V if not REVERSE else 0) + o_v
    else:
        p_q = q + (bos + ((T-1) if not REVERSE else 0)) * H*K + i_h * K + o_k
        p_k = k + (bos + ((T-1) if not REVERSE else 0)) * H*K + i_h * K + o_k
        p_v = v + (bos + ((T-1) if not REVERSE else 0)) * H*V + i_h * V + o_v
        p_w = w + (bos + ((T-1) if not REVERSE else 0)) * H*K + i_h * K + o_k
        p_do = do + (bos + ((T-1) if not REVERSE else 0)) * H*V + i_h * V + o_v
        p_dk = dk + ((i_v * all + bos) + ((T-1) if not REVERSE else 0)) * H*K + i_h * K + o_k
        p_dk1 = dk1 + ((i_v * all + bos) + ((T-1) if not REVERSE else 0)) * H*K + i_h * K + o_k
        p_dv = dv + ((i_k * all + bos) + ((T-1) if not REVERSE else 0)) * H*V + i_h * V + o_v
    p_u = u + i_h * K + o_k

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_u = tl.load(p_u, mask=mask_k, other=0).to(tl.float32)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for _ in range(T - 1, -1, -1):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_dkv = b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], 1)
        tl.store(p_dk1, b_dk.to(p_dk1.dtype.element_ty), mask=mask_k)
        b_dk += tl.sum(b_dkv * b_u[:, None] * b_v[None, :], 1)
        b_dv = tl.sum((b_dh + (b_dkv * b_u[:, None])) * b_k[:, None], 0)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)
        b_dh *= tl.exp(b_w)[:, None]
        b_dh += b_dkv

        p_q += (-1 if not REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_k += (-1 if not REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_v += (-1 if not REVERSE else 1) * (1 if HEAD_FIRST else H) * V
        p_w += (-1 if not REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_do += (-1 if not REVERSE else 1) * (1 if HEAD_FIRST else H) * V
        p_dk += (-1 if not REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_dk1 += (-1 if not REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_dv += (-1 if not REVERSE else 1) * (1 if HEAD_FIRST else H) * V

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_h)


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT, 'BK': BK}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['K']
)
@triton.jit
def fused_recurrent_rwkv6_bwd_kernel_dw(
    q,
    k,
    dq,
    dk,
    dw,
    offsets,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    REVERSE: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_k, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos
    NT = tl.cdiv(T, BT)

    o_i = tl.arange(0, BT)
    m_i = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.) if not REVERSE else tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BK], dtype=tl.float32)

    i_t = 0 if not REVERSE else NT - 1
    for _ in range(NT):
        if HEAD_FIRST:
            p_q = tl.make_block_ptr(q + i_nh * T*K, (T, K), (K, 1), (i_t * BT + 1, i_k * BK), (BT, BK), (1, 0))
            p_dq = tl.make_block_ptr(dq + i_nh * T*K, (T, K), (K, 1), (i_t * BT + 1, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k + i_nh * T*K, (T-1, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + i_nh * T*K, (T-1, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dw = tl.make_block_ptr(dw + i_nh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_q = tl.make_block_ptr(q + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + 1, i_k * BK), (BT, BK), (1, 0))
            p_dq = tl.make_block_ptr(dq + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + 1, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T-1, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + (bos*H + i_h) * K, (T-1, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dw = tl.make_block_ptr(dw + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_dq = tl.load(p_dq, boundary_check=(0, 1)).to(tl.float32)
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_dk = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
        b_dw = (b_q * b_dq * scale) - b_k * b_dk
        b_c = b_z[None, :] + tl.dot(m_i, b_dw, allow_tf32=False)
        tl.store(p_dw, b_c.to(p_dw.dtype.element_ty), boundary_check=(0, 1))
        if i_t >= 0:
            b_z += tl.sum(b_dw, 0)

        i_t += (1 if not REVERSE else -1)


def fused_recurrent_rwkv6_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True
):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if offsets is None else len(offsets) - 1
    BK, BV = min(triton.next_power_of_2(K), 32), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    h0 = initial_state
    ht = q.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    o = q.new_empty(NK, *v.shape, dtype=torch.float)

    grid = (NV, NK, N * H)
    fused_recurrent_rwkv6_fwd_kernel[grid](
        q,
        k,
        v,
        w,
        u,
        o,
        h0,
        ht,
        offsets,
        scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        REVERSE=reverse,
        HEAD_FIRST=head_first
    )
    o = o.sum(0)
    return o, ht


def fused_recurrent_rwkv6_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    do: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    reverse: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True
):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if offsets is None else len(offsets) - 1

    BK, BV = min(triton.next_power_of_2(K), 16), min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    dq = q.new_empty(NV, *q.shape, dtype=torch.float)
    dq1 = torch.empty_like(dq)

    grid = (NV, NK, N * H)
    fused_recurrent_rwkv6_bwd_kernel_dq[grid](
        k,
        v,
        w,
        u,
        do,
        dq,
        dq1,
        initial_state,
        offsets,
        scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        REVERSE=reverse,
        HEAD_FIRST=head_first
    )
    dq = dq.sum(0)
    dq1 = dq1.sum(0)

    BK, BV = min(triton.next_power_of_2(K), 32), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    dk = q.new_empty(NV, *k.shape, dtype=torch.float)
    dk1 = q.new_empty(NV, *k.shape, dtype=torch.float)
    dv = q.new_empty(NK, *v.shape, dtype=torch.float)

    dh0 = torch.empty_like(initial_state) if initial_state is not None else None
    grid = (NV, NK, N * H)
    fused_recurrent_rwkv6_bwd_kernel_dkv[grid](
        q,
        k,
        v,
        w,
        u,
        do,
        dk,
        dk1,
        dv,
        dh0,
        offsets,
        scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        REVERSE=reverse,
        HEAD_FIRST=head_first
    )
    dk = dk.sum(0)
    dk1 = dk1.sum(0)
    dv = dv.sum(0)

    dw = torch.empty_like(w)
    def grid(meta): return (triton.cdiv(meta['K'], meta['BK']), N * H)
    fused_recurrent_rwkv6_bwd_kernel_dw[grid](
        q,
        k,
        dq1,
        dk1,
        dw,
        offsets,
        scale,
        T=T,
        H=H,
        K=K,
        REVERSE=not reverse,
        HEAD_FIRST=head_first
    )
    du = (do.float() * v).sum(-1, True, dtype=torch.float) * q * k * scale
    du = du.sum((0, 2)) if head_first else du.sum((0, 1))
    return dq, dk, dv, dw, du, dh0


class FusedRecurrentRWKV6Function(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        scale: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        reverse: bool = False,
        offsets: Optional[torch.LongTensor] = None,
        head_first: bool = True
    ):
        o, ht = fused_recurrent_rwkv6_fwd(
            q=q,
            k=k,
            v=v,
            w=w,
            u=u,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            reverse=reverse,
            offsets=offsets,
            head_first=head_first
        )
        ctx.save_for_backward(q, k, v, w, u, initial_state)
        ctx.scale = scale
        ctx.reverse = reverse
        ctx.offsets = offsets
        ctx.head_first = head_first
        return o.to(v), ht

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, w, u, initial_state = ctx.saved_tensors

        dq, dk, dv, dw, du, dh0 = fused_recurrent_rwkv6_bwd(
            q=q,
            k=k,
            v=v,
            w=w,
            u=u,
            do=do,
            scale=ctx.scale,
            initial_state=initial_state,
            reverse=ctx.reverse,
            offsets=ctx.offsets,
            head_first=ctx.head_first
        )
        dh0 = dh0.to(initial_state) if dh0 is not None else dh0
        return dq.to(q), dk.to(k), dv.to(v), dw.to(w), du.to(u), None, dh0, None, None, None, None


def fused_recurrent_rwkv6(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: Optional[int] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        r (torch.Tensor):
            reception of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
            Alias: q, query in linear attention.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        w (torch.Tensor):
            data-dependent decays of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]` in log space! Alias: g.
        u (torch.Tensor):
            bonus of shape `[H, K]`
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (Optional[torch.Tensor]):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.rwkv6 import fused_recurrent_rwkv6
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, K, device='cuda'))
        >>> u = torch.randn(H, K, device='cuda')
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = fused_recurrent_rwkv6(q, k, v, g, u,
                                          initial_state=h0,
                                          output_final_state=True,
                                          head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_recurrent_rwkv6(q, k, v, g, u,
                                                  initial_state=h0,
                                                  output_final_state=True,
                                                  cu_seqlens=cu_seqlens,
                                                  head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    if cu_seqlens is not None:
        if r.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {r.shape[0]} when using `cu_seqlens`."
                             f"Please flatten variable-length inputs before processing.")
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f"The number of initial states is expected to be equal to the number of input sequences, "
                             f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = FusedRecurrentRWKV6Function.apply(
        r,
        k,
        v,
        w,
        u,
        scale,
        initial_state,
        output_final_state,
        reverse,
        cu_seqlens,
        head_first
    )
    return o, final_state
