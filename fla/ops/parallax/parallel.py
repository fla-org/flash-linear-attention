# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl
from einops import reduce

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import IS_NVIDIA_BLACKWELL, autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, contiguous


def _block_size(head_dim: int, device_index: int) -> int:
    # A single square tile size shared by all kernels so one `chunk_indices`
    # (built host-side for varlen) matches every grid. Kept modest to bound the
    # fp32 accumulator footprint Parallax carries (barv/Rv/grad accumulators).
    if check_shared_mem('hopper', device_index) and not IS_NVIDIA_BLACKWELL and head_dim <= 64:
        return 128
    return 64


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_parallax_fwd_kernel(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    row_offset = i_t * BT
    row_indices = row_offset + tl.arange(0, BT)
    row_mask = row_indices[:, None] < T
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(T, row_offset + BT), BS)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, T) // BS

    # SWA col-block boundaries. WINDOW_SIZE_LEFT < 0 disables SWA.
    if WINDOW_SIZE_LEFT >= 0:
        leftmost_valid = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = leftmost_valid // BS
        # Phase A is unmasked, so the safe zone must clear the window's left edge for
        # the tile's LAST row (row_offset + BT - 1), not its first.
        safe_left_valid = tl.maximum(0, row_offset + BT - WINDOW_SIZE_LEFT)
        SAFE_LEFT_START = (safe_left_valid + BS - 1) // BS
    else:
        FIRST_COL_BLOCK = 0
        SAFE_LEFT_START = 0
    LEFT_BORDER_END = tl.minimum(SAFE_LEFT_START, NUM_SAFE_BLOCKS)
    SAFE_MIDDLE_START = tl.maximum(FIRST_COL_BLOCK, SAFE_LEFT_START)
    RIGHT_BORDER_START = tl.maximum(FIRST_COL_BLOCK, NUM_SAFE_BLOCKS)

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_r = tl.make_block_ptr(r + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (FIRST_COL_BLOCK * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * K, (T, K), (H * K, 1), (FIRST_COL_BLOCK * BS, 0), (BS, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_barv = tl.make_block_ptr(barv + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_d1 = tl.make_block_ptr(d1 + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_bart = tl.make_block_ptr(bart + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_m = tl.make_block_ptr(m + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
    m_acc = tl.zeros((BT, 1), dtype=tl.float32) - float("inf")
    d1_acc = tl.zeros((BT, 1), dtype=tl.float32)
    d2_acc = tl.zeros((BT, 1), dtype=tl.float32)
    barv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    Rv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    # Phase 0: left-border blocks (SWA only). Window mask only.
    for col_block_id in range(FIRST_COL_BLOCK, LEFT_BORDER_END):
        col_indices = col_block_id * BS + tl.arange(0, BS)
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < T)
        )
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        Rv_acc = alpha * Rv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    # Phase A: safe blocks (no mask).
    for _safe in range(SAFE_MIDDLE_START, NUM_SAFE_BLOCKS):
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        Rv_acc = alpha * Rv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    # Phase B: right-border blocks (causal + boundary + window mask).
    for col_block_id in range(RIGHT_BORDER_START, NUM_TOTAL_BLOCKS):
        col_indices = col_block_id * BS + tl.arange(0, BS)
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & row_mask
                & (col_indices[None, :] < T)
            )
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        Rv_acc = alpha * Rv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    inv_d1 = tl.where(row_mask, 1.0 / d1_acc, 0.0)
    b_barv = barv_acc * inv_d1
    b_bart = d2_acc * inv_d1
    b_o = b_barv + b_bart * b_barv - Rv_acc * inv_d1

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_barv, b_barv.to(p_barv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_d1, d1_acc, boundary_check=(0, 1))
    tl.store(p_bart, b_bart, boundary_check=(0, 1))
    tl.store(p_m, m_acc, boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_parallax_bwd_kernel_preprocess(
    grad_o,
    o,
    barv,
    delta_t,
    delta_b,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)

    row_offset = i_t * BT
    p_grad_o = tl.make_block_ptr(grad_o + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_barv = tl.make_block_ptr(barv + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_t = tl.make_block_ptr(delta_t + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_b = tl.make_block_ptr(delta_b + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))

    b_grad_o = tl.load(p_grad_o, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    b_o = tl.load(p_o, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    b_barv = tl.load(p_barv, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    b_t = tl.sum(b_grad_o * b_o, axis=1, keep_dims=True)
    b_b = tl.sum(b_grad_o * b_barv, axis=1, keep_dims=True)

    tl.store(p_t, b_t, boundary_check=(0, 1))
    tl.store(p_b, b_b, boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_parallax_bwd_kernel_dqr(
    q,
    r,
    k,
    v,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_q,
    grad_r,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    row_offset = i_t * BT
    row_indices = row_offset + tl.arange(0, BT)
    row_mask = row_indices[:, None] < T
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(T, row_offset + BT), BS)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, T) // BS

    if WINDOW_SIZE_LEFT >= 0:
        leftmost_valid = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = leftmost_valid // BS
        # Phase A is unmasked, so the safe zone must clear the window's left edge for
        # the tile's LAST row (row_offset + BT - 1), not its first.
        safe_left_valid = tl.maximum(0, row_offset + BT - WINDOW_SIZE_LEFT)
        SAFE_LEFT_START = (safe_left_valid + BS - 1) // BS
    else:
        FIRST_COL_BLOCK = 0
        SAFE_LEFT_START = 0
    LEFT_BORDER_END = tl.minimum(SAFE_LEFT_START, NUM_SAFE_BLOCKS)
    SAFE_MIDDLE_START = tl.maximum(FIRST_COL_BLOCK, SAFE_LEFT_START)
    RIGHT_BORDER_START = tl.maximum(FIRST_COL_BLOCK, NUM_SAFE_BLOCKS)

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_r = tl.make_block_ptr(r + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (FIRST_COL_BLOCK * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * K, (T, K), (H * K, 1), (FIRST_COL_BLOCK * BS, 0), (BS, BK), (1, 0))
    p_d1 = tl.make_block_ptr(d1 + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_bart = tl.make_block_ptr(bart + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_m = tl.make_block_ptr(m + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_t = tl.make_block_ptr(delta_t + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_b = tl.make_block_ptr(delta_b + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_grad_o = tl.make_block_ptr(grad_o + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_grad_q = tl.make_block_ptr(grad_q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_grad_r = tl.make_block_ptr(grad_r + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
    b_d1 = tl.load(p_d1, boundary_check=(0, 1), padding_option="zero")
    b_bart = tl.load(p_bart, boundary_check=(0, 1), padding_option="zero")
    b_m = tl.load(p_m, boundary_check=(0, 1), padding_option="zero")
    b_t = tl.load(p_t, boundary_check=(0, 1), padding_option="zero")
    b_b = tl.load(p_b, boundary_check=(0, 1), padding_option="zero")
    grad_o_tile = tl.load(p_grad_o, boundary_check=(0, 1), padding_option="zero")
    grad_q_acc = tl.zeros((BT, BK), dtype=tl.float32)
    grad_r_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)

    # Phase 0: left-border blocks (SWA only).
    for col_block_id in range(FIRST_COL_BLOCK, LEFT_BORDER_END):
        col_indices = col_block_id * BS + tl.arange(0, BS)
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < T)
        )
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        p = w * inv_d1
        bart_minus_rk = b_bart - rk
        delta = a - b_b
        gl = p * (a - b_t + bart_minus_rk * delta)
        gu = -p * delta
        grad_q_acc = tl.dot(gl.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_q_acc)
        grad_r_acc = tl.dot(gu.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_r_acc)
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    # Phase A: safe blocks (no mask).
    for _ in range(SAFE_MIDDLE_START, NUM_SAFE_BLOCKS):
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        w = exp2(qk - b_m)
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        p = w * inv_d1
        bart_minus_rk = b_bart - rk
        delta = a - b_b
        gl = p * (a - b_t + bart_minus_rk * delta)
        gu = -p * delta
        grad_q_acc = tl.dot(gl.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_q_acc)
        grad_r_acc = tl.dot(gu.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_r_acc)
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    # Phase B: right-border blocks (causal + boundary + window mask).
    for col_block_id in range(RIGHT_BORDER_START, NUM_TOTAL_BLOCKS):
        col_indices = col_block_id * BS + tl.arange(0, BS)
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & row_mask
                & (col_indices[None, :] < T)
            )
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        p = w * inv_d1
        bart_minus_rk = b_bart - rk
        delta = a - b_b
        gl = p * (a - b_t + bart_minus_rk * delta)
        gu = -p * delta
        grad_q_acc = tl.dot(gl.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_q_acc)
        grad_r_acc = tl.dot(gu.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_r_acc)
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    grad_q_acc = scale * grad_q_acc

    tl.store(p_grad_q, grad_q_acc.to(p_grad_q.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_grad_r, grad_r_acc.to(p_grad_r.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_parallax_bwd_kernel_dkv(
    q,
    r,
    k,
    v,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_k,
    grad_v,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    col_offset = i_t * BS
    col_indices = col_offset + tl.arange(0, BS)

    start_row_block = col_offset // BT
    start_row_offset = start_row_block * BT

    num_row_blocks_qbound = tl.cdiv(T, BT)
    if WINDOW_SIZE_LEFT >= 0:
        last_row_window = tl.cdiv(col_offset + BS + WINDOW_SIZE_LEFT - 1, BT)
        num_row_blocks = tl.minimum(num_row_blocks_qbound, last_row_window)
        WINDOW_SAFE_END = (col_offset + WINDOW_SIZE_LEFT) // BT
    else:
        num_row_blocks = num_row_blocks_qbound
        WINDOW_SAFE_END = num_row_blocks

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (start_row_offset, 0), (BT, BK), (1, 0))
    p_r = tl.make_block_ptr(r + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (start_row_offset, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (col_offset, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * K, (T, K), (H * K, 1), (col_offset, 0), (BS, BK), (1, 0))
    p_d1 = tl.make_block_ptr(d1 + bos * HQ + i_hq, (T, 1), (HQ, 1), (start_row_offset, 0), (BT, 1), (1, 0))
    p_bart = tl.make_block_ptr(bart + bos * HQ + i_hq, (T, 1), (HQ, 1), (start_row_offset, 0), (BT, 1), (1, 0))
    p_m = tl.make_block_ptr(m + bos * HQ + i_hq, (T, 1), (HQ, 1), (start_row_offset, 0), (BT, 1), (1, 0))
    p_t = tl.make_block_ptr(delta_t + bos * HQ + i_hq, (T, 1), (HQ, 1), (start_row_offset, 0), (BT, 1), (1, 0))
    p_b = tl.make_block_ptr(delta_b + bos * HQ + i_hq, (T, 1), (HQ, 1), (start_row_offset, 0), (BT, 1), (1, 0))
    p_grad_o = tl.make_block_ptr(grad_o + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (start_row_offset, 0), (BT, BK), (1, 0))
    p_grad_k = tl.make_block_ptr(grad_k + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (col_offset, 0), (BS, BK), (1, 0))
    p_grad_v = tl.make_block_ptr(grad_v + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (col_offset, 0), (BS, BK), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    grad_k_acc = tl.zeros((BS, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BS, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    first_safe_row_block = tl.cdiv(col_offset + BS, BT)
    SAFE_MIDDLE_END = tl.minimum(WINDOW_SAFE_END, num_row_blocks)
    WINDOW_BORDER_START = tl.maximum(first_safe_row_block, WINDOW_SAFE_END)

    # Phase A: causal-border row blocks.
    causal_end = tl.minimum(first_safe_row_block, num_row_blocks)
    for row_block_id in range(start_row_block, causal_end):
        row_offset = row_block_id * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
        b_d1 = tl.load(p_d1, boundary_check=(0, 1), padding_option="zero")
        b_bart = tl.load(p_bart, boundary_check=(0, 1), padding_option="zero")
        b_m = tl.load(p_m, boundary_check=(0, 1), padding_option="zero")
        b_t = tl.load(p_t, boundary_check=(0, 1), padding_option="zero")
        b_b = tl.load(p_b, boundary_check=(0, 1), padding_option="zero")
        grad_o_tile = tl.load(p_grad_o, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & row_mask
                & (col_indices[None, :] < T)
            )
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        p = w * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        gl = p * (a - b_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)

        p_q = tl.advance(p_q, (BT, 0))
        p_r = tl.advance(p_r, (BT, 0))
        p_d1 = tl.advance(p_d1, (BT, 0))
        p_bart = tl.advance(p_bart, (BT, 0))
        p_m = tl.advance(p_m, (BT, 0))
        p_t = tl.advance(p_t, (BT, 0))
        p_b = tl.advance(p_b, (BT, 0))
        p_grad_o = tl.advance(p_grad_o, (BT, 0))

    # Phase B: safe row blocks (no causal/col/window mask).
    safe_b_start = tl.maximum(first_safe_row_block, start_row_block)
    for row_block_id in range(safe_b_start, SAFE_MIDDLE_END):
        row_offset = row_block_id * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
        b_d1 = tl.load(p_d1, boundary_check=(0, 1), padding_option="zero")
        b_bart = tl.load(p_bart, boundary_check=(0, 1), padding_option="zero")
        b_m = tl.load(p_m, boundary_check=(0, 1), padding_option="zero")
        b_t = tl.load(p_t, boundary_check=(0, 1), padding_option="zero")
        b_b = tl.load(p_b, boundary_check=(0, 1), padding_option="zero")
        grad_o_tile = tl.load(p_grad_o, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        w = exp2(qk - b_m)
        p = w * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        gl = p * (a - b_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)

        p_q = tl.advance(p_q, (BT, 0))
        p_r = tl.advance(p_r, (BT, 0))
        p_d1 = tl.advance(p_d1, (BT, 0))
        p_bart = tl.advance(p_bart, (BT, 0))
        p_m = tl.advance(p_m, (BT, 0))
        p_t = tl.advance(p_t, (BT, 0))
        p_b = tl.advance(p_b, (BT, 0))
        p_grad_o = tl.advance(p_grad_o, (BT, 0))

    # Phase C: window-border row blocks (SWA only).
    window_border_start = tl.maximum(WINDOW_BORDER_START, start_row_block)
    for row_block_id in range(window_border_start, num_row_blocks):
        row_offset = row_block_id * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
        b_d1 = tl.load(p_d1, boundary_check=(0, 1), padding_option="zero")
        b_bart = tl.load(p_bart, boundary_check=(0, 1), padding_option="zero")
        b_m = tl.load(p_m, boundary_check=(0, 1), padding_option="zero")
        b_t = tl.load(p_t, boundary_check=(0, 1), padding_option="zero")
        b_b = tl.load(p_b, boundary_check=(0, 1), padding_option="zero")
        grad_o_tile = tl.load(p_grad_o, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < T)
        )
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        p = w * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        gl = p * (a - b_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)

        p_q = tl.advance(p_q, (BT, 0))
        p_r = tl.advance(p_r, (BT, 0))
        p_d1 = tl.advance(p_d1, (BT, 0))
        p_bart = tl.advance(p_bart, (BT, 0))
        p_m = tl.advance(p_m, (BT, 0))
        p_t = tl.advance(p_t, (BT, 0))
        p_b = tl.advance(p_b, (BT, 0))
        p_grad_o = tl.advance(p_grad_o, (BT, 0))

    tl.store(p_grad_k, grad_k_acc.to(p_grad_k.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_grad_v, grad_v_acc.to(p_grad_v.dtype.element_ty), boundary_check=(0, 1))


def parallel_parallax_fwd(q, r, k, v, scale, cu_seqlens=None, chunk_indices=None, window_size_left=-1):
    """Parallax forward (Triton). `(B, T, HQ, D)` / packed `(1, T_total, HQ, D)` inputs.

    Returns `(o, barv, d1, bart, m)`: `o`/`barv` in the input dtype and layout;
    `d1`/`bart`/`m` are fp32 per-(position, query-head) scalars `(B, T, HQ)`.
    """
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    BK = triton.next_power_of_2(K)
    BT = _block_size(K, q.device.index)
    o = torch.empty_like(q)
    barv = torch.empty_like(q)
    d1 = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)
    bart = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)
    m = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)

    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    grid = (NT, B * HQ)
    parallel_parallax_fwd_kernel[grid](
        q, r, k, v, o, barv, d1, bart, m,
        scale, cu_seqlens, chunk_indices, T,
        HQ=HQ, H=H, G=G, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left, BT=BT, BS=BT,
        num_warps=8, num_stages=2,
    )
    return o, barv, d1, bart, m


def parallel_parallax_bwd(q, r, k, v, o, barv, d1, bart, m, grad_o, scale, cu_seqlens=None, chunk_indices=None, window_size_left=-1):
    """Parallax backward (Triton). Returns grads matching `q, r, k, v`."""
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    BK = triton.next_power_of_2(K)
    BT = _block_size(K, q.device.index)

    grad_q = torch.empty_like(q)
    grad_r = torch.empty_like(r)
    # dK/dV are written per q-head, then folded back to the kv-head axis.
    grad_k_buf = torch.empty((B, T, HQ, K), device=q.device, dtype=q.dtype)
    grad_v_buf = torch.empty((B, T, HQ, K), device=q.device, dtype=q.dtype)
    delta_t = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)
    delta_b = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)

    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    grid = (NT, B * HQ)

    parallel_parallax_bwd_kernel_preprocess[grid](
        grad_o, o, barv, delta_t, delta_b,
        cu_seqlens, chunk_indices, T,
        HQ=HQ, K=K, BK=BK, BT=BT,
        num_warps=4, num_stages=2,
    )
    parallel_parallax_bwd_kernel_dqr[grid](
        q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_q, grad_r,
        scale, cu_seqlens, chunk_indices, T,
        HQ=HQ, H=H, G=G, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left, BT=BT, BS=BT,
        num_warps=8, num_stages=2,
    )
    parallel_parallax_bwd_kernel_dkv[grid](
        q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_k_buf, grad_v_buf,
        scale, cu_seqlens, chunk_indices, T,
        HQ=HQ, H=H, G=G, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left, BT=BT, BS=BT,
        num_warps=8, num_stages=2,
    )

    if G == 1:
        grad_k = grad_k_buf
        grad_v = grad_v_buf
    else:
        grad_k = reduce(grad_k_buf, 'b t (h g) k -> b t h k', g=G, reduction='sum')
        grad_v = reduce(grad_v_buf, 'b t (h g) k -> b t h k', g=G, reduction='sum')
    return grad_q, grad_r, grad_k, grad_v


class ParallaxFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, r, k, v, scale, window_size_left, cu_seqlens):
        chunk_indices = prepare_chunk_indices(cu_seqlens, _block_size(q.shape[-1], q.device.index)) \
            if cu_seqlens is not None else None
        o, barv, d1, bart, m = parallel_parallax_fwd(q, r, k, v, scale, cu_seqlens, chunk_indices, window_size_left)
        ctx.save_for_backward(q, r, k, v, o, barv, d1, bart, m)
        ctx.scale = scale
        ctx.window_size_left = window_size_left
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        return o

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, r, k, v, o, barv, d1, bart, m = ctx.saved_tensors
        gq, gr, gk, gv = parallel_parallax_bwd(
            q, r, k, v, o, barv, d1, bart, m, do,
            ctx.scale, ctx.cu_seqlens, ctx.chunk_indices, ctx.window_size_left,
        )
        return gq.to(q), gr.to(r), gk.to(k), gv.to(v), None, None, None


def parallel_parallax(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    r"""
    Causal Parallax (parameterized local linear attention) with autograd,
    backed by Triton kernels. See `fla.ops.parallax.naive.naive_parallax` for
    the reference math.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, D]`.
        r (torch.Tensor):
            secondary queries of shape `[B, T, HQ, D]` (same shape as `q`). NOTE:
            `r` is *not* scaled by `scale`; pass it un-pre-scaled.
        k (torch.Tensor):
            keys of shape `[B, T, H, D]`. GQA is applied when `HQ` is divisible by `H`.
        v (torch.Tensor):
            values of shape `[B, T, H, D]`.
        scale (float, Optional):
            Scale applied to the `q @ k^T` logits only. If `None`, defaults to `1 / sqrt(D)`.
            Default: `None`.
        window_size (int, Optional):
            Sliding-window length. If provided, each query at position `i` only attends to
            keys in `[i - window_size + 1, i]`. If `None`, full causal attention is used.
            Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length training
            (FlashAttention convention). The batch size must be 1 when packing. Default: `None`.

    Returns:
        o (torch.Tensor):
            output of shape `[B, T, HQ, D]`.
    """
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"parallel_parallax requires bf16 or fp16 inputs, got q.dtype={q.dtype}")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
            f"Please flatten variable-length inputs before processing.",
        )
    # The kernel keeps cols [i - W + 1, i] (W keys total, diagonal included),
    # matching FLA's `window_size=W` semantics exactly (no off-by-one).
    window_size_left = -1 if window_size is None else window_size
    return ParallaxFunction.apply(q, r, k, v, float(scale), window_size_left, cu_seqlens)
