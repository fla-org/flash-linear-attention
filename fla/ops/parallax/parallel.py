# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, contiguous

# A curated config set. The upstream Parallax kernels swept 36 configs
# (3x3 tiles x 2 warps x 2 stages) for production tuning; that makes the
# per-shape autotune compile time explode in CI. These few cover the useful
# corner of the space for D <= 128 on Hopper/Ampere.
_CONFIGS = [
    triton.Config({"ROW_TILE_SIZE": 128, "COL_TILE_SIZE": 64}, num_warps=8, num_stages=2),
    triton.Config({"ROW_TILE_SIZE": 128, "COL_TILE_SIZE": 128}, num_warps=8, num_stages=2),
    triton.Config({"ROW_TILE_SIZE": 64, "COL_TILE_SIZE": 64}, num_warps=4, num_stages=2),
    triton.Config({"ROW_TILE_SIZE": 64, "COL_TILE_SIZE": 128}, num_warps=8, num_stages=3),
]
_PREPROCESS_CONFIGS = [
    triton.Config({"ROW_TILE_SIZE": 128}, num_warps=4, num_stages=2),
    triton.Config({"ROW_TILE_SIZE": 256}, num_warps=8, num_stages=2),
]


def _prune_bwd_configs_by_head_dim(configs, named_args, **kwargs):
    # At HEAD_DIM >= 256, ROW_TILE_SIZE=128 spills registers.
    head_dim = named_args.get("HEAD_DIM", 0)
    if head_dim >= 256:
        pruned = [c for c in configs if c.kwargs["ROW_TILE_SIZE"] <= 64]
        return pruned if pruned else configs[:1]
    return configs


@triton.autotune(
    configs=_CONFIGS,
    key=["HEAD_DIM", "N_REP", "WINDOW_SIZE_LEFT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["N_QUERIES", "N_KEYVALS"])
def parallel_parallax_attn_fwd_kernel(
    q_ptr,
    r_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    barv_ptr,
    d1_ptr,
    bart_ptr,
    m_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_rb, stride_rq, stride_rd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_barv_b, stride_barv_q, stride_barv_d,
    stride_d1_b, stride_d1_q,
    stride_bart_b, stride_bart_q,
    stride_m_b, stride_m_q,
    qk_scale,
    N_QUERIES,
    N_KEYVALS,
    HEAD_DIM: tl.constexpr,
    BD: tl.constexpr,
    N_REP: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    ROW_TILE_SIZE: tl.constexpr,
    COL_TILE_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(1).to(tl.int64)
    kv_batch_idx = pid_batch // N_REP
    pid_row = tl.program_id(0)
    row_offset = pid_row * ROW_TILE_SIZE
    row_indices = row_offset + tl.arange(0, ROW_TILE_SIZE)
    row_mask = row_indices[:, None] < N_QUERIES
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(N_KEYVALS, row_offset + ROW_TILE_SIZE), COL_TILE_SIZE)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, N_KEYVALS) // COL_TILE_SIZE

    # SWA col-block boundaries. WINDOW_SIZE_LEFT < 0 disables SWA.
    if WINDOW_SIZE_LEFT >= 0:
        leftmost_valid = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = leftmost_valid // COL_TILE_SIZE
        SAFE_LEFT_START = (leftmost_valid + COL_TILE_SIZE - 1) // COL_TILE_SIZE
    else:
        FIRST_COL_BLOCK = 0
        SAFE_LEFT_START = 0
    LEFT_BORDER_END = tl.minimum(SAFE_LEFT_START, NUM_SAFE_BLOCKS)
    SAFE_MIDDLE_START = tl.maximum(FIRST_COL_BLOCK, SAFE_LEFT_START)
    RIGHT_BORDER_START = tl.maximum(FIRST_COL_BLOCK, NUM_SAFE_BLOCKS)

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + pid_batch * stride_qb,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_qq, stride_qd),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, BD), order=(1, 0),
    )
    r_block_ptr = tl.make_block_ptr(
        base=r_ptr + pid_batch * stride_rb,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_rq, stride_rd),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, BD), order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + kv_batch_idx * stride_kb,
        shape=(N_KEYVALS, HEAD_DIM), strides=(stride_kk, stride_kd),
        offsets=(FIRST_COL_BLOCK * COL_TILE_SIZE, 0), block_shape=(COL_TILE_SIZE, BD), order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + kv_batch_idx * stride_vb,
        shape=(N_KEYVALS, HEAD_DIM), strides=(stride_vk, stride_vd),
        offsets=(FIRST_COL_BLOCK * COL_TILE_SIZE, 0), block_shape=(COL_TILE_SIZE, BD), order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + pid_batch * stride_ob,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_oq, stride_od),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, BD), order=(1, 0),
    )
    barv_block_ptr = tl.make_block_ptr(
        base=barv_ptr + pid_batch * stride_barv_b,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_barv_q, stride_barv_d),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, BD), order=(1, 0),
    )
    d1_block_ptr = tl.make_block_ptr(
        base=d1_ptr + pid_batch * stride_d1_b,
        shape=(N_QUERIES, 1), strides=(stride_d1_q, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    bart_block_ptr = tl.make_block_ptr(
        base=bart_ptr + pid_batch * stride_bart_b,
        shape=(N_QUERIES, 1), strides=(stride_bart_q, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    m_block_ptr = tl.make_block_ptr(
        base=m_ptr + pid_batch * stride_m_b,
        shape=(N_QUERIES, 1), strides=(stride_m_q, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )

    Q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    R = tl.load(r_block_ptr, boundary_check=(0, 1), padding_option="zero")
    m_acc = tl.zeros((ROW_TILE_SIZE, 1), dtype=tl.float32) - float("inf")
    d1_acc = tl.zeros((ROW_TILE_SIZE, 1), dtype=tl.float32)
    d2_acc = tl.zeros((ROW_TILE_SIZE, 1), dtype=tl.float32)
    barv_acc = tl.zeros((ROW_TILE_SIZE, BD), dtype=tl.float32)
    Rv_acc = tl.zeros((ROW_TILE_SIZE, BD), dtype=tl.float32)
    RCP_LN2: tl.constexpr = 1.4426950216
    qk_scale_log2 = qk_scale * RCP_LN2

    # Phase 0: left-border blocks (SWA only). Window mask only — below the
    # causal diagonal but straddling the left window edge.
    for col_block_id in range(FIRST_COL_BLOCK, LEFT_BORDER_END):
        col_offset = col_block_id * COL_TILE_SIZE
        col_indices = col_offset + tl.arange(0, COL_TILE_SIZE)
        K = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < N_KEYVALS)
        )
        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.max(qk, axis=1, keep_dims=True)
        m_new = tl.maximum(m_acc, m_new)
        # Rows whose window has not started yet stay at m_new == -inf;
        # force alpha=0, w=0 to avoid exp2(-inf - -inf) = NaN.
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        Rv_acc = alpha * Rv_acc
        barv_acc = tl.dot(w.to(V.dtype), V, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(V.dtype), V, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (COL_TILE_SIZE, 0))

    # Phase A: safe blocks (no mask).
    for col_block_id in range(SAFE_MIDDLE_START, NUM_SAFE_BLOCKS):
        K = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        m_new = tl.max(qk, axis=1, keep_dims=True)
        m_new = tl.maximum(m_acc, m_new)
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        Rv_acc = alpha * Rv_acc
        barv_acc = tl.dot(w.to(V.dtype), V, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(V.dtype), V, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (COL_TILE_SIZE, 0))

    # Phase B: right-border blocks (causal + boundary + window mask).
    for col_block_id in range(RIGHT_BORDER_START, NUM_TOTAL_BLOCKS):
        col_offset = col_block_id * COL_TILE_SIZE
        col_indices = col_offset + tl.arange(0, COL_TILE_SIZE)
        K = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < N_KEYVALS)
            )
        else:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & row_mask
                & (col_indices[None, :] < N_KEYVALS)
            )
        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.max(qk, axis=1, keep_dims=True)
        m_new = tl.maximum(m_acc, m_new)
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        Rv_acc = alpha * Rv_acc
        barv_acc = tl.dot(w.to(V.dtype), V, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(V.dtype), V, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (COL_TILE_SIZE, 0))

    inv_d1 = tl.where(row_mask, 1.0 / d1_acc, 0.0)
    barv = barv_acc * inv_d1
    bart = d2_acc * inv_d1
    o = barv + bart * barv - Rv_acc * inv_d1

    tl.store(o_block_ptr, o.to(Q.dtype), boundary_check=(0, 1))
    tl.store(barv_block_ptr, barv.to(Q.dtype), boundary_check=(0, 1))
    tl.store(d1_block_ptr, d1_acc, boundary_check=(0, 1))
    tl.store(bart_block_ptr, bart, boundary_check=(0, 1))
    tl.store(m_block_ptr, m_acc, boundary_check=(0, 1))


@triton.autotune(
    configs=_PREPROCESS_CONFIGS,
    key=["HEAD_DIM"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["N_QUERIES"])
def parallel_parallax_attn_bwd_kernel_preprocess(
    grad_o_ptr,
    o_ptr,
    barv_ptr,
    t_ptr,
    b_ptr,
    stride_gob, stride_goq, stride_god,
    stride_ob, stride_oq, stride_od,
    stride_barv_b, stride_barv_q, stride_barv_d,
    stride_tb, stride_tq,
    stride_bb, stride_bq,
    N_QUERIES,
    HEAD_DIM: tl.constexpr,
    BD: tl.constexpr,
    ROW_TILE_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(1).to(tl.int64)
    pid_row = tl.program_id(0)
    row_offset = pid_row * ROW_TILE_SIZE

    grad_o_block_ptr = tl.make_block_ptr(
        base=grad_o_ptr + pid_batch * stride_gob,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_goq, stride_god),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, BD), order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + pid_batch * stride_ob,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_oq, stride_od),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, BD), order=(1, 0),
    )
    barv_block_ptr = tl.make_block_ptr(
        base=barv_ptr + pid_batch * stride_barv_b,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_barv_q, stride_barv_d),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, BD), order=(1, 0),
    )
    t_block_ptr = tl.make_block_ptr(
        base=t_ptr + pid_batch * stride_tb,
        shape=(N_QUERIES, 1), strides=(stride_tq, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + pid_batch * stride_bb,
        shape=(N_QUERIES, 1), strides=(stride_bq, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )

    grad_o = tl.load(grad_o_block_ptr, boundary_check=(0, 1), padding_option="zero")
    O_tile = tl.load(o_block_ptr, boundary_check=(0, 1), padding_option="zero")
    barv = tl.load(barv_block_ptr, boundary_check=(0, 1), padding_option="zero")

    grad_o_f32 = grad_o.to(tl.float32)
    t = tl.sum(grad_o_f32 * O_tile.to(tl.float32), axis=1, keep_dims=True)
    b = tl.sum(grad_o_f32 * barv.to(tl.float32), axis=1, keep_dims=True)

    tl.store(t_block_ptr, t, boundary_check=(0, 1))
    tl.store(b_block_ptr, b, boundary_check=(0, 1))


@triton.autotune(
    configs=list(_CONFIGS),
    key=["HEAD_DIM", "N_REP", "WINDOW_SIZE_LEFT"],
    prune_configs_by={"early_config_prune": _prune_bwd_configs_by_head_dim},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["N_QUERIES", "N_KEYVALS"])
def parallel_parallax_attn_bwd_kernel_dqr(
    q_ptr,
    r_ptr,
    k_ptr,
    v_ptr,
    d1_ptr,
    bart_ptr,
    m_ptr,
    t_ptr,
    b_ptr,
    grad_o_ptr,
    grad_q_ptr,
    grad_r_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_rb, stride_rq, stride_rd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_d1_b, stride_d1_q,
    stride_bart_b, stride_bart_q,
    stride_m_b, stride_m_q,
    stride_tb, stride_tq,
    stride_bb, stride_bq,
    stride_gob, stride_goq, stride_god,
    stride_gqb, stride_gqq, stride_gqd,
    stride_grb, stride_grq, stride_grd,
    qk_scale,
    N_QUERIES,
    N_KEYVALS,
    HEAD_DIM: tl.constexpr,
    BD: tl.constexpr,
    N_REP: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    ROW_TILE_SIZE: tl.constexpr,
    COL_TILE_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(1).to(tl.int64)
    kv_batch_idx = pid_batch // N_REP
    pid_row = tl.program_id(0)
    row_offset = pid_row * ROW_TILE_SIZE
    row_indices = row_offset + tl.arange(0, ROW_TILE_SIZE)
    row_mask = row_indices[:, None] < N_QUERIES
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(N_KEYVALS, row_offset + ROW_TILE_SIZE), COL_TILE_SIZE)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, N_KEYVALS) // COL_TILE_SIZE

    # SWA col-block boundaries (see fwd kernel for derivation).
    if WINDOW_SIZE_LEFT >= 0:
        leftmost_valid = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = leftmost_valid // COL_TILE_SIZE
        SAFE_LEFT_START = (leftmost_valid + COL_TILE_SIZE - 1) // COL_TILE_SIZE
    else:
        FIRST_COL_BLOCK = 0
        SAFE_LEFT_START = 0
    LEFT_BORDER_END = tl.minimum(SAFE_LEFT_START, NUM_SAFE_BLOCKS)
    SAFE_MIDDLE_START = tl.maximum(FIRST_COL_BLOCK, SAFE_LEFT_START)
    RIGHT_BORDER_START = tl.maximum(FIRST_COL_BLOCK, NUM_SAFE_BLOCKS)

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + pid_batch * stride_qb,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_qq, stride_qd),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, BD), order=(1, 0),
    )
    r_block_ptr = tl.make_block_ptr(
        base=r_ptr + pid_batch * stride_rb,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_rq, stride_rd),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, BD), order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + kv_batch_idx * stride_kb,
        shape=(N_KEYVALS, HEAD_DIM), strides=(stride_kk, stride_kd),
        offsets=(FIRST_COL_BLOCK * COL_TILE_SIZE, 0), block_shape=(COL_TILE_SIZE, BD), order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + kv_batch_idx * stride_vb,
        shape=(N_KEYVALS, HEAD_DIM), strides=(stride_vk, stride_vd),
        offsets=(FIRST_COL_BLOCK * COL_TILE_SIZE, 0), block_shape=(COL_TILE_SIZE, BD), order=(1, 0),
    )
    d1_block_ptr = tl.make_block_ptr(
        base=d1_ptr + pid_batch * stride_d1_b,
        shape=(N_QUERIES, 1), strides=(stride_d1_q, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    bart_block_ptr = tl.make_block_ptr(
        base=bart_ptr + pid_batch * stride_bart_b,
        shape=(N_QUERIES, 1), strides=(stride_bart_q, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    m_block_ptr = tl.make_block_ptr(
        base=m_ptr + pid_batch * stride_m_b,
        shape=(N_QUERIES, 1), strides=(stride_m_q, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    t_block_ptr = tl.make_block_ptr(
        base=t_ptr + pid_batch * stride_tb,
        shape=(N_QUERIES, 1), strides=(stride_tq, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + pid_batch * stride_bb,
        shape=(N_QUERIES, 1), strides=(stride_bq, 1),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    grad_o_block_ptr = tl.make_block_ptr(
        base=grad_o_ptr + pid_batch * stride_gob,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_goq, stride_god),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, BD), order=(1, 0),
    )
    grad_q_block_ptr = tl.make_block_ptr(
        base=grad_q_ptr + pid_batch * stride_gqb,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_gqq, stride_gqd),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, BD), order=(1, 0),
    )
    grad_r_block_ptr = tl.make_block_ptr(
        base=grad_r_ptr + pid_batch * stride_grb,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_grq, stride_grd),
        offsets=(row_offset, 0), block_shape=(ROW_TILE_SIZE, BD), order=(1, 0),
    )

    Q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    R = tl.load(r_block_ptr, boundary_check=(0, 1), padding_option="zero")
    d1 = tl.load(d1_block_ptr, boundary_check=(0, 1), padding_option="zero")
    bart = tl.load(bart_block_ptr, boundary_check=(0, 1), padding_option="zero")
    m = tl.load(m_block_ptr, boundary_check=(0, 1), padding_option="zero")
    t = tl.load(t_block_ptr, boundary_check=(0, 1), padding_option="zero")
    b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
    grad_o = tl.load(grad_o_block_ptr, boundary_check=(0, 1), padding_option="zero")
    grad_q_acc = tl.zeros((ROW_TILE_SIZE, BD), dtype=tl.float32)
    grad_r_acc = tl.zeros((ROW_TILE_SIZE, BD), dtype=tl.float32)
    RCP_LN2: tl.constexpr = 1.4426950216
    qk_scale_log2 = qk_scale * RCP_LN2

    inv_d1 = tl.where(row_mask, 1.0 / d1, 0.0)

    # Phase 0: left-border blocks (SWA only).
    for col_block_id in range(FIRST_COL_BLOCK, LEFT_BORDER_END):
        col_offset = col_block_id * COL_TILE_SIZE
        col_indices = col_offset + tl.arange(0, COL_TILE_SIZE)
        K = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < N_KEYVALS)
        )
        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - m)
        a = tl.dot(grad_o, tl.trans(V), out_dtype=tl.float32)
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        p = w * inv_d1
        bart_minus_rk = bart - rk
        delta = a - b
        gl = p * (a - t + bart_minus_rk * delta)
        gu = -p * delta
        grad_q_acc = tl.dot(gl.to(K.dtype), K, out_dtype=tl.float32, acc=grad_q_acc)
        grad_r_acc = tl.dot(gu.to(K.dtype), K, out_dtype=tl.float32, acc=grad_r_acc)
        k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (COL_TILE_SIZE, 0))

    # Phase A: safe blocks (no mask).
    for col_block_id in range(SAFE_MIDDLE_START, NUM_SAFE_BLOCKS):
        K = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        w = exp2(qk - m)
        a = tl.dot(grad_o, tl.trans(V), out_dtype=tl.float32)
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        p = w * inv_d1
        bart_minus_rk = bart - rk
        delta = a - b
        gl = p * (a - t + bart_minus_rk * delta)
        gu = -p * delta
        grad_q_acc = tl.dot(gl.to(K.dtype), K, out_dtype=tl.float32, acc=grad_q_acc)
        grad_r_acc = tl.dot(gu.to(K.dtype), K, out_dtype=tl.float32, acc=grad_r_acc)
        k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (COL_TILE_SIZE, 0))

    # Phase B: right-border blocks (causal + boundary + window mask).
    for col_block_id in range(RIGHT_BORDER_START, NUM_TOTAL_BLOCKS):
        col_offset = col_block_id * COL_TILE_SIZE
        col_indices = col_offset + tl.arange(0, COL_TILE_SIZE)
        K = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < N_KEYVALS)
            )
        else:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & row_mask
                & (col_indices[None, :] < N_KEYVALS)
            )
        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - m)
        a = tl.dot(grad_o, tl.trans(V), out_dtype=tl.float32)
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        p = w * inv_d1
        bart_minus_rk = bart - rk
        delta = a - b
        gl = p * (a - t + bart_minus_rk * delta)
        gu = -p * delta
        grad_q_acc = tl.dot(gl.to(K.dtype), K, out_dtype=tl.float32, acc=grad_q_acc)
        grad_r_acc = tl.dot(gu.to(K.dtype), K, out_dtype=tl.float32, acc=grad_r_acc)
        k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (COL_TILE_SIZE, 0))

    grad_q_acc = qk_scale * grad_q_acc

    tl.store(grad_q_block_ptr, grad_q_acc.to(Q.dtype), boundary_check=(0, 1))
    tl.store(grad_r_block_ptr, grad_r_acc.to(Q.dtype), boundary_check=(0, 1))


@triton.autotune(
    configs=list(_CONFIGS),
    key=["HEAD_DIM", "N_REP", "WINDOW_SIZE_LEFT"],
    prune_configs_by={"early_config_prune": _prune_bwd_configs_by_head_dim},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["N_QUERIES", "N_KEYVALS"])
def parallel_parallax_attn_bwd_kernel_dkv(
    q_ptr,
    r_ptr,
    k_ptr,
    v_ptr,
    d1_ptr,
    bart_ptr,
    m_ptr,
    t_ptr,
    b_ptr,
    grad_o_ptr,
    grad_k_ptr,
    grad_v_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_rb, stride_rq, stride_rd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_d1_b, stride_d1_q,
    stride_bart_b, stride_bart_q,
    stride_m_b, stride_m_q,
    stride_tb, stride_tq,
    stride_bb, stride_bq,
    stride_gob, stride_goq, stride_god,
    stride_gkb, stride_gkk, stride_gkd,
    stride_gvb, stride_gvk, stride_gvd,
    qk_scale,
    N_QUERIES,
    N_KEYVALS,
    HEAD_DIM: tl.constexpr,
    BD: tl.constexpr,
    N_REP: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    ROW_TILE_SIZE: tl.constexpr,
    COL_TILE_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(1).to(tl.int64)
    kv_batch_idx = pid_batch // N_REP
    pid_col = tl.program_id(0)
    col_offset = pid_col * COL_TILE_SIZE
    col_indices = col_offset + tl.arange(0, COL_TILE_SIZE)

    start_row_block = col_offset // ROW_TILE_SIZE
    start_row_offset = start_row_block * ROW_TILE_SIZE

    # SWA row-block boundaries:
    #   - num_row_blocks: cap after which rows can no longer reach this col block
    #   - WINDOW_SAFE_END: last row-block fully within W of every col in the block
    num_row_blocks_qbound = tl.cdiv(N_QUERIES, ROW_TILE_SIZE)
    if WINDOW_SIZE_LEFT >= 0:
        last_row_window = tl.cdiv(col_offset + COL_TILE_SIZE + WINDOW_SIZE_LEFT - 1, ROW_TILE_SIZE)
        num_row_blocks = tl.minimum(num_row_blocks_qbound, last_row_window)
        WINDOW_SAFE_END = (col_offset + WINDOW_SIZE_LEFT) // ROW_TILE_SIZE
    else:
        num_row_blocks = num_row_blocks_qbound
        WINDOW_SAFE_END = num_row_blocks

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + pid_batch * stride_qb,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_qq, stride_qd),
        offsets=(start_row_offset, 0), block_shape=(ROW_TILE_SIZE, BD), order=(1, 0),
    )
    r_block_ptr = tl.make_block_ptr(
        base=r_ptr + pid_batch * stride_rb,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_rq, stride_rd),
        offsets=(start_row_offset, 0), block_shape=(ROW_TILE_SIZE, BD), order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + kv_batch_idx * stride_kb,
        shape=(N_KEYVALS, HEAD_DIM), strides=(stride_kk, stride_kd),
        offsets=(col_offset, 0), block_shape=(COL_TILE_SIZE, BD), order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + kv_batch_idx * stride_vb,
        shape=(N_KEYVALS, HEAD_DIM), strides=(stride_vk, stride_vd),
        offsets=(col_offset, 0), block_shape=(COL_TILE_SIZE, BD), order=(1, 0),
    )
    d1_block_ptr = tl.make_block_ptr(
        base=d1_ptr + pid_batch * stride_d1_b,
        shape=(N_QUERIES, 1), strides=(stride_d1_q, 1),
        offsets=(start_row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    bart_block_ptr = tl.make_block_ptr(
        base=bart_ptr + pid_batch * stride_bart_b,
        shape=(N_QUERIES, 1), strides=(stride_bart_q, 1),
        offsets=(start_row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    m_block_ptr = tl.make_block_ptr(
        base=m_ptr + pid_batch * stride_m_b,
        shape=(N_QUERIES, 1), strides=(stride_m_q, 1),
        offsets=(start_row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    t_block_ptr = tl.make_block_ptr(
        base=t_ptr + pid_batch * stride_tb,
        shape=(N_QUERIES, 1), strides=(stride_tq, 1),
        offsets=(start_row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + pid_batch * stride_bb,
        shape=(N_QUERIES, 1), strides=(stride_bq, 1),
        offsets=(start_row_offset, 0), block_shape=(ROW_TILE_SIZE, 1), order=(1, 0),
    )
    grad_o_block_ptr = tl.make_block_ptr(
        base=grad_o_ptr + pid_batch * stride_gob,
        shape=(N_QUERIES, HEAD_DIM), strides=(stride_goq, stride_god),
        offsets=(start_row_offset, 0), block_shape=(ROW_TILE_SIZE, BD), order=(1, 0),
    )
    grad_k_block_ptr = tl.make_block_ptr(
        base=grad_k_ptr + pid_batch * stride_gkb,
        shape=(N_KEYVALS, HEAD_DIM), strides=(stride_gkk, stride_gkd),
        offsets=(col_offset, 0), block_shape=(COL_TILE_SIZE, BD), order=(1, 0),
    )
    grad_v_block_ptr = tl.make_block_ptr(
        base=grad_v_ptr + pid_batch * stride_gvb,
        shape=(N_KEYVALS, HEAD_DIM), strides=(stride_gvk, stride_gvd),
        offsets=(col_offset, 0), block_shape=(COL_TILE_SIZE, BD), order=(1, 0),
    )

    K = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
    V = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
    grad_k_acc = tl.zeros((COL_TILE_SIZE, BD), dtype=tl.float32)
    grad_v_acc = tl.zeros((COL_TILE_SIZE, BD), dtype=tl.float32)
    RCP_LN2: tl.constexpr = 1.4426950216
    qk_scale_log2 = qk_scale * RCP_LN2

    # Safe phase starts when min(row) > max(col) — i.e.
    # row_offset >= col_offset + COL_TILE_SIZE.
    first_safe_row_block = tl.cdiv(col_offset + COL_TILE_SIZE, ROW_TILE_SIZE)
    SAFE_MIDDLE_END = tl.minimum(WINDOW_SAFE_END, num_row_blocks)
    WINDOW_BORDER_START = tl.maximum(first_safe_row_block, WINDOW_SAFE_END)

    # Phase A: causal-border row blocks. Apply causal mask plus window mask
    # when SWA is on.
    causal_end = tl.minimum(first_safe_row_block, num_row_blocks)
    for row_block_id in range(start_row_block, causal_end):
        row_offset = row_block_id * ROW_TILE_SIZE
        row_indices = row_offset + tl.arange(0, ROW_TILE_SIZE)
        row_mask = row_indices[:, None] < N_QUERIES
        Q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        R = tl.load(r_block_ptr, boundary_check=(0, 1), padding_option="zero")
        d1 = tl.load(d1_block_ptr, boundary_check=(0, 1), padding_option="zero")
        bart = tl.load(bart_block_ptr, boundary_check=(0, 1), padding_option="zero")
        m = tl.load(m_block_ptr, boundary_check=(0, 1), padding_option="zero")
        t = tl.load(t_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_o = tl.load(grad_o_block_ptr, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / d1, 0.0)
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < N_KEYVALS)
            )
        else:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & row_mask
                & (col_indices[None, :] < N_KEYVALS)
            )
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - m)
        p = w * inv_d1
        a = tl.dot(grad_o, tl.trans(V), out_dtype=tl.float32)
        delta = a - b
        bart_minus_rk = bart - rk
        gl = p * (a - t + bart_minus_rk * delta) * qk_scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(Q.dtype), Q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(R.dtype), R, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o.dtype), grad_o, out_dtype=tl.float32, acc=grad_v_acc)

        q_block_ptr = tl.advance(q_block_ptr, (ROW_TILE_SIZE, 0))
        r_block_ptr = tl.advance(r_block_ptr, (ROW_TILE_SIZE, 0))
        d1_block_ptr = tl.advance(d1_block_ptr, (ROW_TILE_SIZE, 0))
        bart_block_ptr = tl.advance(bart_block_ptr, (ROW_TILE_SIZE, 0))
        m_block_ptr = tl.advance(m_block_ptr, (ROW_TILE_SIZE, 0))
        t_block_ptr = tl.advance(t_block_ptr, (ROW_TILE_SIZE, 0))
        b_block_ptr = tl.advance(b_block_ptr, (ROW_TILE_SIZE, 0))
        grad_o_block_ptr = tl.advance(grad_o_block_ptr, (ROW_TILE_SIZE, 0))

    # Phase B: safe row blocks (no causal/col/window mask).
    safe_b_start = tl.maximum(first_safe_row_block, start_row_block)
    for row_block_id in range(safe_b_start, SAFE_MIDDLE_END):
        row_offset = row_block_id * ROW_TILE_SIZE
        row_indices = row_offset + tl.arange(0, ROW_TILE_SIZE)
        row_mask = row_indices[:, None] < N_QUERIES
        Q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        R = tl.load(r_block_ptr, boundary_check=(0, 1), padding_option="zero")
        d1 = tl.load(d1_block_ptr, boundary_check=(0, 1), padding_option="zero")
        bart = tl.load(bart_block_ptr, boundary_check=(0, 1), padding_option="zero")
        m = tl.load(m_block_ptr, boundary_check=(0, 1), padding_option="zero")
        t = tl.load(t_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_o = tl.load(grad_o_block_ptr, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / d1, 0.0)
        w = exp2(qk - m)
        p = w * inv_d1
        a = tl.dot(grad_o, tl.trans(V), out_dtype=tl.float32)
        delta = a - b
        bart_minus_rk = bart - rk
        gl = p * (a - t + bart_minus_rk * delta) * qk_scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(Q.dtype), Q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(R.dtype), R, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o.dtype), grad_o, out_dtype=tl.float32, acc=grad_v_acc)

        q_block_ptr = tl.advance(q_block_ptr, (ROW_TILE_SIZE, 0))
        r_block_ptr = tl.advance(r_block_ptr, (ROW_TILE_SIZE, 0))
        d1_block_ptr = tl.advance(d1_block_ptr, (ROW_TILE_SIZE, 0))
        bart_block_ptr = tl.advance(bart_block_ptr, (ROW_TILE_SIZE, 0))
        m_block_ptr = tl.advance(m_block_ptr, (ROW_TILE_SIZE, 0))
        t_block_ptr = tl.advance(t_block_ptr, (ROW_TILE_SIZE, 0))
        b_block_ptr = tl.advance(b_block_ptr, (ROW_TILE_SIZE, 0))
        grad_o_block_ptr = tl.advance(grad_o_block_ptr, (ROW_TILE_SIZE, 0))

    # Phase C: window-border row blocks (SWA only). Rows past the causal
    # diagonal but straddling the right window edge.
    window_border_start = tl.maximum(WINDOW_BORDER_START, start_row_block)
    for row_block_id in range(window_border_start, num_row_blocks):
        row_offset = row_block_id * ROW_TILE_SIZE
        row_indices = row_offset + tl.arange(0, ROW_TILE_SIZE)
        row_mask = row_indices[:, None] < N_QUERIES
        Q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        R = tl.load(r_block_ptr, boundary_check=(0, 1), padding_option="zero")
        d1 = tl.load(d1_block_ptr, boundary_check=(0, 1), padding_option="zero")
        bart = tl.load(bart_block_ptr, boundary_check=(0, 1), padding_option="zero")
        m = tl.load(m_block_ptr, boundary_check=(0, 1), padding_option="zero")
        t = tl.load(t_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_o = tl.load(grad_o_block_ptr, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * qk_scale_log2
        rk = tl.dot(R, tl.trans(K), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / d1, 0.0)
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < N_KEYVALS)
        )
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - m)
        p = w * inv_d1
        a = tl.dot(grad_o, tl.trans(V), out_dtype=tl.float32)
        delta = a - b
        bart_minus_rk = bart - rk
        gl = p * (a - t + bart_minus_rk * delta) * qk_scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(Q.dtype), Q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(R.dtype), R, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o.dtype), grad_o, out_dtype=tl.float32, acc=grad_v_acc)

        q_block_ptr = tl.advance(q_block_ptr, (ROW_TILE_SIZE, 0))
        r_block_ptr = tl.advance(r_block_ptr, (ROW_TILE_SIZE, 0))
        d1_block_ptr = tl.advance(d1_block_ptr, (ROW_TILE_SIZE, 0))
        bart_block_ptr = tl.advance(bart_block_ptr, (ROW_TILE_SIZE, 0))
        m_block_ptr = tl.advance(m_block_ptr, (ROW_TILE_SIZE, 0))
        t_block_ptr = tl.advance(t_block_ptr, (ROW_TILE_SIZE, 0))
        b_block_ptr = tl.advance(b_block_ptr, (ROW_TILE_SIZE, 0))
        grad_o_block_ptr = tl.advance(grad_o_block_ptr, (ROW_TILE_SIZE, 0))

    tl.store(grad_k_block_ptr, grad_k_acc.to(K.dtype), boundary_check=(0, 1))
    tl.store(grad_v_block_ptr, grad_v_acc.to(V.dtype), boundary_check=(0, 1))


def parallax_fwd(q, r, k, v, scale, n_rep=1, window_size_left=-1):
    """Parallax forward (Triton). Folded `(B*H, T, D)` inputs; see the public
    `parallel_parallax_attn` for the `(B, T, H, D)` API.

    Returns `(o, barv, d1, bart, m)`: `o`/`barv` in the input dtype, `d1`/`bart`/`m`
    fp32 per-row scalars consumed by the backward.
    """
    batch_size, n_queries, head_dim = q.shape
    n_keyvals = k.shape[1]
    bd = triton.next_power_of_2(head_dim)
    o = torch.empty((batch_size, n_queries, head_dim), device=q.device, dtype=q.dtype)
    barv = torch.empty((batch_size, n_queries, head_dim), device=q.device, dtype=q.dtype)
    d1 = torch.empty((batch_size, n_queries, 1), device=q.device, dtype=torch.float32)
    bart = torch.empty((batch_size, n_queries, 1), device=q.device, dtype=torch.float32)
    m = torch.empty((batch_size, n_queries, 1), device=q.device, dtype=torch.float32)
    def grid(meta): return (math.ceil(n_queries / meta["ROW_TILE_SIZE"]), batch_size)
    parallel_parallax_attn_fwd_kernel[grid](
        q, r, k, v, o, barv, d1, bart, m,
        q.stride(0), q.stride(1), q.stride(2),
        r.stride(0), r.stride(1), r.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        barv.stride(0), barv.stride(1), barv.stride(2),
        d1.stride(0), d1.stride(1),
        bart.stride(0), bart.stride(1),
        m.stride(0), m.stride(1),
        scale,
        n_queries,
        n_keyvals,
        head_dim,
        bd,
        n_rep,
        window_size_left,
    )
    return o, barv, d1, bart, m


def parallax_bwd(q, r, k, v, o, barv, d1, bart, m, grad_o, scale, n_rep=1, window_size_left=-1):
    """Parallax backward (Triton). Folded `(B*H, T, D)` inputs/saved tensors.

    Returns `(grad_q, grad_r, grad_k, grad_v)` in the input dtype. Under GQA the
    per-q-head dK/dV slots are folded back to the kv-head axis with a sum reduce.
    """
    batch_size, n_queries, head_dim = q.shape
    n_keyvals = k.shape[1]
    bd = triton.next_power_of_2(head_dim)
    kv_batch_size = batch_size // n_rep

    grad_q = torch.empty_like(q)
    grad_r = torch.empty_like(r)
    grad_k_buf = torch.empty((batch_size, n_keyvals, head_dim), device=q.device, dtype=q.dtype)
    grad_v_buf = torch.empty((batch_size, n_keyvals, head_dim), device=q.device, dtype=q.dtype)

    t = torch.empty((batch_size, n_queries, 1), device=q.device, dtype=torch.float32)
    b = torch.empty((batch_size, n_queries, 1), device=q.device, dtype=torch.float32)
    def pre_grid(meta): return (math.ceil(n_queries / meta["ROW_TILE_SIZE"]), batch_size)
    parallel_parallax_attn_bwd_kernel_preprocess[pre_grid](
        grad_o, o, barv, t, b,
        grad_o.stride(0), grad_o.stride(1), grad_o.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        barv.stride(0), barv.stride(1), barv.stride(2),
        t.stride(0), t.stride(1),
        b.stride(0), b.stride(1),
        n_queries,
        head_dim,
        bd,
    )

    def rq_grid(meta): return (math.ceil(n_queries / meta["ROW_TILE_SIZE"]), batch_size)
    def kv_grid(meta): return (math.ceil(n_keyvals / meta["COL_TILE_SIZE"]), batch_size)

    parallel_parallax_attn_bwd_kernel_dqr[rq_grid](
        q, r, k, v, d1, bart, m, t, b, grad_o, grad_q, grad_r,
        q.stride(0), q.stride(1), q.stride(2),
        r.stride(0), r.stride(1), r.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        d1.stride(0), d1.stride(1),
        bart.stride(0), bart.stride(1),
        m.stride(0), m.stride(1),
        t.stride(0), t.stride(1),
        b.stride(0), b.stride(1),
        grad_o.stride(0), grad_o.stride(1), grad_o.stride(2),
        grad_q.stride(0), grad_q.stride(1), grad_q.stride(2),
        grad_r.stride(0), grad_r.stride(1), grad_r.stride(2),
        scale,
        n_queries,
        n_keyvals,
        head_dim,
        bd,
        n_rep,
        window_size_left,
    )

    parallel_parallax_attn_bwd_kernel_dkv[kv_grid](
        q, r, k, v, d1, bart, m, t, b, grad_o, grad_k_buf, grad_v_buf,
        q.stride(0), q.stride(1), q.stride(2),
        r.stride(0), r.stride(1), r.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        d1.stride(0), d1.stride(1),
        bart.stride(0), bart.stride(1),
        m.stride(0), m.stride(1),
        t.stride(0), t.stride(1),
        b.stride(0), b.stride(1),
        grad_o.stride(0), grad_o.stride(1), grad_o.stride(2),
        grad_k_buf.stride(0), grad_k_buf.stride(1), grad_k_buf.stride(2),
        grad_v_buf.stride(0), grad_v_buf.stride(1), grad_v_buf.stride(2),
        scale,
        n_queries,
        n_keyvals,
        head_dim,
        bd,
        n_rep,
        window_size_left,
    )

    if n_rep == 1:
        grad_k = grad_k_buf
        grad_v = grad_v_buf
    else:
        # Fold the n_rep per-q-head slots back to the kv-head axis (the same sum
        # autograd through repeat_kv would produce).
        grad_k = grad_k_buf.view(kv_batch_size, n_rep, n_keyvals, head_dim).sum(dim=1)
        grad_v = grad_v_buf.view(kv_batch_size, n_rep, n_keyvals, head_dim).sum(dim=1)
    return grad_q, grad_r, grad_k, grad_v


class ParallaxAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, r, k, v, scale, window_size_left):
        B, T, HQ, D = q.shape
        H = k.shape[2]
        n_rep = HQ // H
        # Fold heads into the batch axis: (B, T, H, D) -> (B*H, T, D), contiguous.
        qf = q.transpose(1, 2).reshape(B * HQ, T, D).contiguous()
        rf = r.transpose(1, 2).reshape(B * HQ, T, D).contiguous()
        kf = k.transpose(1, 2).reshape(B * H, T, D).contiguous()
        vf = v.transpose(1, 2).reshape(B * H, T, D).contiguous()
        o, barv, d1, bart, m = parallax_fwd(qf, rf, kf, vf, scale, n_rep, window_size_left)
        ctx.save_for_backward(qf, rf, kf, vf, o, barv, d1, bart, m)
        ctx.scale = scale
        ctx.n_rep = n_rep
        ctx.window_size_left = window_size_left
        ctx.shape = (B, T, HQ, H, D)
        return o.reshape(B, HQ, T, D).transpose(1, 2)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        qf, rf, kf, vf, o, barv, d1, bart, m = ctx.saved_tensors
        B, T, HQ, H, D = ctx.shape
        dof = do.transpose(1, 2).reshape(B * HQ, T, D).contiguous()
        gq, gr, gk, gv = parallax_bwd(
            qf, rf, kf, vf, o, barv, d1, bart, m, dof,
            ctx.scale, ctx.n_rep, ctx.window_size_left,
        )
        dq = gq.reshape(B, HQ, T, D).transpose(1, 2)
        dr = gr.reshape(B, HQ, T, D).transpose(1, 2)
        dk = gk.reshape(B, H, T, D).transpose(1, 2)
        dv = gv.reshape(B, H, T, D).transpose(1, 2)
        return dq, dr, dk, dv, None, None


def parallel_parallax_attn(
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
    Causal Parallax (parameterized/centered local linear attention) with autograd,
    backed by Triton kernels. See `fla.ops.parallax.naive.naive_parallax_attn` for
    the reference math.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, D]`.
        r (torch.Tensor):
            centering queries of shape `[B, T, HQ, D]` (same shape as `q`). NOTE:
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
            Variable-length packing offsets. NOT yet supported by the Parallax op
            (raises `NotImplementedError`); use the dense path. Default: `None`.

    Returns:
        o (torch.Tensor):
            output of shape `[B, T, HQ, D]`.
    """
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if cu_seqlens is not None:
        raise NotImplementedError(
            "Variable-length (cu_seqlens) Parallax is not implemented yet; "
            "use the dense path (cu_seqlens=None).",
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"parallel_parallax_attn requires bf16 or fp16 inputs, got q.dtype={q.dtype}")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    # The kernel keeps cols [i - W + 1, i] (W keys total, diagonal included),
    # matching FLA's `window_size=W` semantics exactly (no off-by-one).
    window_size_left = -1 if window_size is None else window_size
    return ParallaxAttentionFunction.apply(q, r, k, v, float(scale), window_size_left)
