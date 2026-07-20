# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Wall attention, contributed by Tilde Research (Timor Averbuch, Dhruv Pai).
# Heavily modified from fla/ops/gated_oja_rule/chunk.py.
"""Wall training/prefill kernels: forward, backward, and the autograd Function."""

import os

import torch
import triton
import triton.language as tl
from einops import reduce

from fla.ops.backends import dispatch
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.cumsum import chunk_global_cumsum
from fla.ops.utils.op import exp2, log2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, check_shared_mem, contiguous

_DEBUG_ASSERTS = os.environ.get("WALL_ATTN_DEBUG", "0") == "1"

# Set WALL_ATTN_DKV_DIAG_BF16=0 to force fp32 tensor cores in the diagonal loop
# of parallel_wall_attn_bwd_kernel_dkv (precision over speed).
_DKV_DIAG_BF16 = os.environ.get("WALL_ATTN_DKV_DIAG_BF16", "1") == "1"

WALL_FWD_AUTOTUNE_CONFIGS = [
    triton.Config({'BT': BT, 'BS': BS}, num_warps=nw, num_stages=ns)
    for BT in (64, 128)
    for BS in (32, 64)
    for nw in (2, 4, 8)
    for ns in (2, 3)
]

WALL_BWD_AUTOTUNE_CONFIGS = [
    triton.Config({'BT': BT, 'BS': BS}, num_warps=nw, num_stages=ns)
    for BT in (64, 128)
    for BS in (32, 64)
    for nw in (2, 4, 8)
    for ns in (2, 3)
]


def _prune_wall_fwd_configs(configs, nargs, **kwargs):
    # The per-program fp32 accumulator tile is BT x BV. At the Hopper BV cap of
    # 256, BT=128 configs hold a 128x256 fp32 accumulator (128 KiB), spill
    # registers massively, can never win the benchmark, and each take minutes to
    # compile under IEEE fp32 dots (TRITON_F32_DEFAULT=ieee in the test suite),
    # which reads as a multi-minute stall per autotune key. Drop configs that
    # exceed the tile budget; BV <= 128 keeps the full sweep. BV never exceeds
    # the fwd cap (256), so BT=64 always survives and the pruned set is non-empty.
    nargs = {**(nargs or {}), **kwargs}  # keyword launches leave nargs empty
    BV = nargs.get('BV') or 128
    return [c for c in configs if c.kwargs['BT'] * BV <= 16384]


@triton.jit(do_not_specialize=['N'])
def parallel_wall_attn_bwd_kernel_preprocess(
    o,
    do,
    delta,
    N,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    i_nv = tl.program_id(0).to(tl.int64)
    i_v, i_n = i_nv // N, i_nv % N
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    b_o = tl.load(o + i_n * V + o_v, mask=m_v, other=0.).to(tl.float32)
    b_do = tl.load(do + i_n * V + o_v, mask=m_v, other=0.).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_v * N + i_n, b_delta)


@triton.autotune(
    configs=WALL_FWD_AUTOTUNE_CONFIGS,
    key=['T_BUCKET', 'K', 'V', 'HQ', 'H'],
    prune_configs_by={'early_config_prune': _prune_wall_fwd_configs},
    **autotune_cache_kwargs,
)
@triton.heuristics({
    'USE_SINK_BIAS': lambda args: args['sink_bias'] is not None,
    'USE_WINDOW': lambda args: args['W'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_SCALAR_G': lambda args: args['g_scalar_cumsum'] is not None,
})
@triton.jit(do_not_specialize=['T', 'T_BUCKET'])
def parallel_wall_attn_fwd_kernel(
    q,
    k,
    v,
    o,
    g_cumsum,
    g_scalar_cumsum,
    sink_bias,
    lse,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    T_BUCKET,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_SINK_BIAS: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SCALAR_G: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    p_pq = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_R = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (1, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_pq = tl.load(p_pq, boundary_check=(0, 1)).to(tl.float32)
    b_R = tl.load(p_R, boundary_check=(0, 1)).to(tl.float32)
    # b_R is at i_t*BT; the same value is used in both off-diag and diag loops,
    # since |b_pq - b_R| within a BT-chunk is bounded by BT*|g_max|*RCP_LN2.
    # Off-diagonal: P_q - R <= 0 and R - P_k <= 0 -> exp2 <= 1 -> bf16 safe.
    b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_R)).to(b_q.dtype)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    if USE_SCALAR_G:
        p_cq = tl.make_block_ptr(g_scalar_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_cq = tl.load(p_cq, boundary_check=(0,)).to(tl.float32)

    if USE_SINK_BIAS:
        b_sink_bias = tl.load(sink_bias + i_hq).to(tl.float32)
    else:
        b_sink_bias = None

    o_q = i_t * BT + tl.arange(0, BT)
    i_start = tl.maximum((i_t * BT - W + 1) // BS * BS, 0) if USE_WINDOW else 0

    b_R_t = tl.trans(b_R)

    for i_s in range(i_start, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_pk = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (K, T), (1, HQ*K), (0, i_s), (BK, BS), (0, 1))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)
        b_k_til = (b_k.to(tl.float32) * exp2(b_R_t - b_pk)).to(b_k.dtype)
        b_s = tl.dot(b_q_til, b_k_til) * scale * RCP_LN2

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_cq[:, None] - b_ck[None, :]
        if USE_WINDOW:
            b_s = tl.where((o_q[:, None] - o_k[None, :] < W) & m_k[None, :], b_s, float('-inf'))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_mw = tl.where(b_m == float('-inf'), 0., b_m)
        b_r = exp2(b_mp - b_mw)
        b_p = exp2(b_s - b_mw[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_mp = b_m

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_pk = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (K, T), (1, HQ*K), (0, i_s), (BK, BS), (0, 1))

        # Per-sub-block local reference to prevent exp2 overflow.
        p_R_local = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (1, BK), (1, 0))
        b_R_local = tl.load(p_R_local, boundary_check=(0, 1)).to(tl.float32)
        b_R_local_bc = tl.broadcast_to(b_R_local, (BT, BK))
        b_R_local_t = tl.trans(b_R_local)

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)

        # k_til in local frame: exp2 bounded by BS positions <= 110 (fp32-safe with BK dot).
        b_exp_k = b_R_local_t - b_pk
        b_exp_k = tl.where(b_exp_k > 110.0, tl.zeros_like(b_exp_k) + 110.0, b_exp_k)
        b_k_til = b_k.to(tl.float32) * exp2(b_exp_k)

        # q_til in local frame; clamp exp for Q before sub-block (masked by causality).
        b_exp_q = tl.minimum(b_pq - b_R_local_bc, tl.zeros([BT, BK], dtype=tl.float32))
        b_q_til_local = b_q.to(tl.float32) * exp2(b_exp_q)
        b_s = tl.dot(b_q_til_local, b_k_til) * scale * RCP_LN2

        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_cq[:, None] - b_ck[None, :]

        m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        if USE_WINDOW:
            m_s = m_s & (o_q[:, None] - o_k[None, :] < W)
        b_s = tl.where(m_s, b_s, float('-inf'))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_mw = tl.where(b_m == float('-inf'), 0., b_m)
        b_r = exp2(b_mp - b_mw)
        b_p = exp2(b_s - b_mw[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_mp = b_m

    if USE_SINK_BIAS:
        b_m = tl.where(b_m == float('-inf'), 0., b_m)
        b_acc += exp2(b_sink_bias - b_m)

    b_o = b_o / b_acc[:, None]
    b_m += log2(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    if i_v == 0:
        tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=WALL_BWD_AUTOTUNE_CONFIGS,
    key=['T_BUCKET', 'K', 'V', 'HQ', 'H'],
    **autotune_cache_kwargs,
)
@triton.heuristics({
    'USE_WINDOW': lambda args: args['W'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_SCALAR_G': lambda args: args['g_scalar_cumsum'] is not None,
})
@triton.jit(do_not_specialize=['T', 'T_BUCKET'])
def parallel_wall_attn_bwd_kernel_dq(
    q,
    k,
    v,
    g_cumsum,
    g_scalar_cumsum,
    lse,
    delta,
    do,
    dq,
    dg_cumsum,
    dg_scalar_cumsum,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    T_BUCKET,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SCALAR_G: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    i_v64 = i_v.to(tl.int64)
    delta += i_v64 * B * T * HQ
    dq += i_v64 * B * T * HQ * K
    dg_cumsum += i_v64 * B * T * HQ * K
    if USE_SCALAR_G:
        dg_scalar_cumsum += i_v64 * B * T * HQ

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216
    LN2: tl.constexpr = 0.6931471805599453

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_pq = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_R = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (1, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_pq = tl.load(p_pq, boundary_check=(0, 1)).to(tl.float32)
    b_R = tl.load(p_R, boundary_check=(0, 1)).to(tl.float32)
    # Off-diagonal q_til (bf16 tensor cores, exp2 <= 1).
    b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_R)).to(b_q.dtype)

    b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
    b_lse = tl.load(p_lse, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))

    b_dq_til = tl.zeros([BT, BK], dtype=tl.float32)

    if USE_SCALAR_G:
        p_cq = tl.make_block_ptr(g_scalar_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_cq = tl.load(p_cq, boundary_check=(0,)).to(tl.float32)
        b_dc = tl.zeros([BT], dtype=tl.float32)

    o_q = i_t * BT + tl.arange(0, BT)
    i_start = tl.maximum((i_t * BT - W + 1) // BS * BS, 0) if USE_WINDOW else 0
    b_R_t = tl.trans(b_R)

    for i_s in range(i_start, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
        p_pk = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (K, T), (1, HQ*K), (0, i_s), (BK, BS), (0, 1))

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_k_til = (b_k.to(tl.float32) * exp2(b_R_t - b_pk)).to(b_k.dtype)
        b_s = tl.dot(b_q_til, b_k_til) * scale * RCP_LN2

        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_cq[:, None] - b_ck[None, :]
        if USE_WINDOW:
            b_s = tl.where((o_q[:, None] - o_k[None, :] < W) & m_k[None, :], b_s, float('-inf'))
        b_p = exp2(b_s - b_lse[:, None])
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        b_dq_til += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k_til))
        if USE_SCALAR_G:
            b_dc += tl.sum(b_ds, 1)

    # Diagonal: use per-sub-block local reference to prevent exp2 overflow.
    # Accumulate dq_diag directly in output space to avoid large frame corrections.
    b_dq_diag = tl.zeros([BT, BK], dtype=tl.float32)

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
        p_pk = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (K, T), (1, HQ*K), (0, i_s), (BK, BS), (0, 1))

        p_R_local = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (1, BK), (1, 0))
        b_R_local = tl.load(p_R_local, boundary_check=(0, 1)).to(tl.float32)
        b_R_local_bc = tl.broadcast_to(b_R_local, (BT, BK))
        b_R_local_t = tl.trans(b_R_local)

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")

        # Local-ref k_til: exp2 arg bounded by BS positions <= 110 (fp32-safe with BK dot).
        b_exp_k = b_R_local_t - b_pk
        b_exp_k = tl.where(b_exp_k > 110.0, tl.zeros_like(b_exp_k) + 110.0, b_exp_k)
        b_k_til = b_k.to(tl.float32) * exp2(b_exp_k)

        # q_til in local frame for computing scores.
        b_q_til_local = b_q.to(tl.float32) * exp2(b_pq - b_R_local_bc)
        b_s = tl.dot(b_q_til_local, b_k_til) * scale * RCP_LN2

        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_cq[:, None] - b_ck[None, :]

        if USE_WINDOW:
            b_p = tl.where(
                (o_q[:, None] >= o_k[None, :]) & (o_q[:, None] - o_k[None, :] < W) & m_k[None, :],
                exp2(b_s - b_lse[:, None]), 0
            )
        else:
            b_p = tl.where((o_q[:, None] >= o_k[None, :]) & m_k[None, :], exp2(b_s - b_lse[:, None]), 0)

        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        # dq_sub in local frame; convert to output space: dq += dq_sub * scale * exp2(Pq - R_local).
        # For Q < sub-block start: causal mask makes ds=0, but mask explicitly to kill fp noise.
        b_dq_sub = tl.dot(b_ds.to(tl.float32), tl.trans(b_k_til))
        m_q_causal = (o_q >= i_s)[:, None]
        b_dq_diag += tl.where(m_q_causal, b_dq_sub * exp2(b_pq - b_R_local_bc), 0.0)
        if USE_SCALAR_G:
            b_dc += tl.sum(b_ds, 1)

    # Off-diagonal (chunk-ref) + diagonal (direct output space).
    b_dq = (b_dq_til * scale) * exp2(b_pq - b_R) + b_dq_diag * scale

    # Structural: b_dg derived from b_dq (saves [BT,BK] fp32 accumulator).
    b_dg = LN2 * b_q.to(tl.float32) * b_dq
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))
    if USE_SCALAR_G:
        p_dc = tl.make_block_ptr(dg_scalar_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dc, b_dc.to(p_dc.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=WALL_BWD_AUTOTUNE_CONFIGS,
    key=['T_BUCKET', 'K', 'V', 'HQ', 'H'],
    **autotune_cache_kwargs,
)
@triton.heuristics({
    'USE_WINDOW': lambda args: args['W'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_SCALAR_G': lambda args: args['g_scalar_cumsum'] is not None,
})
@triton.jit(do_not_specialize=['T', 'T_BUCKET'])
def parallel_wall_attn_bwd_kernel_dkv(
    q,
    k,
    v,
    g_cumsum,
    g_scalar_cumsum,
    lse,
    delta,
    do,
    dk,
    dv,
    dg_cumsum,
    dg_scalar_cumsum,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    T_BUCKET,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SCALAR_G: tl.constexpr,
    DIAG_BF16: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    i_v64 = i_v.to(tl.int64)
    delta += i_v64 * B * T * HQ
    dk += i_v64 * B * T * HQ * K
    dg_cumsum += i_v64 * B * T * HQ * K
    if USE_SCALAR_G:
        dg_scalar_cumsum += i_v64 * B * T * HQ

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216
    LN2: tl.constexpr = 0.6931471805599453

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_pk = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dg = tl.make_block_ptr(dg_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    o_k = i_t * BT + tl.arange(0, BT)

    if USE_SCALAR_G:
        p_ck = tl.make_block_ptr(g_scalar_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_ck = tl.load(p_ck, boundary_check=(0,)).to(tl.float32)
        b_dc = tl.zeros([BT], dtype=tl.float32)

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_Rq = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (1, BK), (1, 0))
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_pq = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        b_Rq = tl.load(p_Rq, boundary_check=(0, 1)).to(tl.float32)
        b_Rq_bc_k = tl.broadcast_to(b_Rq, (BT, BK))
        b_Rq_bc_q = tl.broadcast_to(b_Rq, (BS, BK))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_pq = tl.load(p_pq, boundary_check=(0, 1)).to(tl.float32)
        b_exp_k = b_Rq_bc_k - b_pk
        # Non-causal keys (key_pos > last query in sub-block) can have
        # exp_k >> 110, overflowing the gradient headroom.  The causal mask
        # zeros their contribution in b_p, but inf*0 = NaN poisons b_dk/b_dg.
        # Clamp to 110 (fp32-safe with BK/BS dot accumulation) so products
        # in dg accumulation stay well within fp32.
        b_exp_k = tl.where(b_exp_k > 110.0, tl.zeros_like(b_exp_k) + 110.0, b_exp_k)
        b_exp_k_val = exp2(b_exp_k)  # CSE: reused for k_til and final dk scaling
        if DIAG_BF16:
            # Fast path: bf16 tensor cores, fp32 accumulation.
            b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_Rq_bc_q)).to(b_q.dtype)
            b_k_til = (b_k.to(tl.float32) * b_exp_k_val).to(b_k.dtype)
        else:
            # Precise path: fp32 tensor cores.
            b_q_til = b_q.to(tl.float32) * exp2(b_pq - b_Rq_bc_q)
            b_k_til = b_k.to(tl.float32) * b_exp_k_val
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_s = tl.dot(b_k_til, tl.trans(b_q_til)) * scale * RCP_LN2
        if USE_SCALAR_G:
            b_cq = tl.load(g_scalar_cumsum + (bos + o_q) * HQ + i_hq, mask=m_q, other=0).to(tl.float32)
            b_s += b_cq[None, :] - b_ck[:, None]
        if USE_WINDOW:
            b_p = tl.where(
                (o_k[:, None] <= o_q[None, :]) & (o_q[None, :] - o_k[:, None] < W) & m_q[None, :],
                exp2(b_s - b_lse[None, :]), 0
            )
        else:
            b_p = tl.where((o_k[:, None] <= o_q[None, :]) & m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp - b_delta[None, :])
        if DIAG_BF16:
            b_dk_til = tl.dot(b_ds.to(b_q.dtype), b_q_til)
        else:
            b_dk_til = tl.dot(b_ds.to(tl.float32), b_q_til)
        b_dk += (b_dk_til * scale) * b_exp_k_val
        if USE_SCALAR_G:
            b_dc -= tl.sum(b_ds, 1)

    i_end = min(tl.cdiv(T, BS) * BS, (i_t + 1) * BT + W - 1) if USE_WINDOW else tl.cdiv(T, BS) * BS

    for i_s in range((i_t + 1) * BT, i_end, BS):
        p_Rq = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (1, BK), (1, 0))
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_pq = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        b_Rq = tl.load(p_Rq, boundary_check=(0, 1)).to(tl.float32)
        b_Rq_bc_k = tl.broadcast_to(b_Rq, (BT, BK))
        b_Rq_bc_q = tl.broadcast_to(b_Rq, (BS, BK))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_pq = tl.load(p_pq, boundary_check=(0, 1)).to(tl.float32)
        b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_Rq_bc_q)).to(b_q.dtype)
        b_exp_k_off = b_Rq_bc_k - b_pk
        b_exp_k_off_val = exp2(b_exp_k_off)  # CSE: reused for k_til and final dk scaling
        b_k_til = (b_k.to(tl.float32) * b_exp_k_off_val).to(b_k.dtype)
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_s = tl.dot(b_k_til, tl.trans(b_q_til)) * scale * RCP_LN2
        if USE_SCALAR_G:
            b_cq = tl.load(g_scalar_cumsum + (bos + o_q) * HQ + i_hq, mask=m_q, other=0).to(tl.float32)
            b_s += b_cq[None, :] - b_ck[:, None]
        if USE_WINDOW:
            b_p = tl.where((o_q[None, :] - o_k[:, None] < W) & m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        else:
            b_p = tl.where(m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp - b_delta[None, :])
        b_dk_til = tl.dot(b_ds.to(b_q.dtype), b_q_til)
        b_dk += (b_dk_til * scale) * b_exp_k_off_val
        if USE_SCALAR_G:
            b_dc -= tl.sum(b_ds, 1)

    # Structural: b_dg derived from b_dk (saves [BT,BK] fp32 accumulator).
    b_dg = -LN2 * b_k.to(tl.float32) * b_dk
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))
    if USE_SCALAR_G:
        p_dc = tl.make_block_ptr(dg_scalar_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dc, b_dc.to(p_dc.dtype.element_ty), boundary_check=(0,))


@dispatch('attn')
def parallel_wall_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    sink_bias: torch.Tensor | None,
    scale: float,
    g_scalar_cumsum: torch.Tensor | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    if g_cumsum.dim() != 4 or g_cumsum.shape != (*q.shape[:-1], q.shape[-1]):
        raise ValueError(
            f"`g_cumsum` must be [B, T, HQ, K] matching `q`; got {g_cumsum.shape} vs q {q.shape}"
        )
    if g_cumsum.stride(-1) != 1:
        raise ValueError("`g_cumsum` must be contiguous in the last (K) dimension")
    g_cumsum = g_cumsum.contiguous()

    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    if check_shared_mem('hopper', q.device.index):
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(256, max(16, triton.next_power_of_2(V)))
    elif check_shared_mem('ampere', q.device.index):
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(128, max(16, triton.next_power_of_2(V)))
    else:
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(64, max(16, triton.next_power_of_2(V)))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    if cu_seqlens is not None:
        # Autotuning sweeps BT in {64,128}, but `chunk_indices` is precomputed for a
        # single BT. Pin BT=128 and bypass the autotuner (calling the heuristics-wrapped
        # jit directly) so the varlen chunk map stays valid.
        BT = 128
        if check_shared_mem('hopper', q.device.index):
            BS, num_warps, num_stages = min(64, max(16, triton.next_power_of_2(T))), 8, 2
        elif check_shared_mem('ampere', q.device.index):
            BS, num_warps, num_stages = min(32, max(16, triton.next_power_of_2(T))), 4, 2
        else:
            BS, num_warps, num_stages = min(32, max(16, triton.next_power_of_2(T))), 2, 2
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
        o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
        lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
        parallel_wall_attn_fwd_kernel.fn[(NV, NT, B * HQ)](
            q=q, k=k, v=v, o=o,
            g_cumsum=g_cumsum, g_scalar_cumsum=g_scalar_cumsum, sink_bias=sink_bias,
            lse=lse, scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            B=B, T=T, T_BUCKET=1, W=window_size, H=H, HQ=HQ, G=G, K=K, V=V,
            BT=BT, BS=BS, BK=BK, BV=BV, num_warps=num_warps, num_stages=num_stages,
        )
        return o, lse

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    # the bucket selects an autotune config; exact T remains the runtime loop bound
    T_BUCKET = triton.next_power_of_2(T)

    def grid(meta):
        return (NV, triton.cdiv(T, meta['BT']), B * HQ)
    parallel_wall_attn_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        g_cumsum=g_cumsum,
        g_scalar_cumsum=g_scalar_cumsum,
        sink_bias=sink_bias,
        lse=lse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        T_BUCKET=T_BUCKET,
        W=window_size,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
    )
    return o, lse


def parallel_wall_attn_bwd_preprocess(
    o: torch.Tensor,
    do: torch.Tensor,
    BV: int,
) -> torch.Tensor:
    V = o.shape[-1]
    NV = triton.cdiv(V, BV)
    N = o.numel() // V
    delta = torch.empty(NV, *o.shape[:-1], dtype=torch.float, device=o.device)
    parallel_wall_attn_bwd_kernel_preprocess[(NV * N,)](
        o=o,
        do=do,
        delta=delta,
        N=N,
        V=V,
        BV=BV,
    )
    return delta


@dispatch('attn')
def parallel_wall_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    g_cumsum: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    sink_bias: torch.Tensor | None = None,
    scale: float | None = None,
    g_scalar_cumsum: torch.Tensor | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    if g_cumsum.dim() != 4 or g_cumsum.shape != (*q.shape[:-1], q.shape[-1]):
        raise ValueError("`g_cumsum` must be [B, T, HQ, K] matching `q`")
    g_cumsum = g_cumsum.contiguous()

    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BK = max(triton.next_power_of_2(K), 16)
    if check_shared_mem('hopper', q.device.index) or check_shared_mem('ampere', q.device.index):
        BV = min(128, max(triton.next_power_of_2(V), 16))
    else:
        BV = min(64, max(triton.next_power_of_2(V), 16))

    NV = triton.cdiv(V, BV)

    # Varlen: pin BT=128 and bypass the autotuner (BT sweep would invalidate the
    # precomputed chunk map). `extra` injects the fixed launch config via `.fn`.
    is_varlen = cu_seqlens is not None
    # varlen bypasses autotuning; dense lengths share configs within power-of-two buckets
    T_BUCKET = 1 if is_varlen else triton.next_power_of_2(T)
    extra = {}
    if is_varlen:
        BT = 128
        if check_shared_mem('hopper', q.device.index):
            BS, num_warps = min(64, max(16, triton.next_power_of_2(T))), 8
        elif check_shared_mem('ampere', q.device.index):
            BS, num_warps = min(32, max(16, triton.next_power_of_2(T))), 4
        else:
            BS, num_warps = min(32, max(16, triton.next_power_of_2(T))), 2
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
        grid = (NV, NT, B * HQ)
        extra = dict(BT=BT, BS=BS, num_warps=num_warps, num_stages=2)
    else:
        def grid(meta):
            return (NV, triton.cdiv(T, meta['BT']), B * HQ)

    delta = parallel_wall_attn_bwd_preprocess(o, do, BV)

    partial_dtype = k.dtype if NV == 1 and H == HQ else torch.float
    dq = torch.empty(NV, B, T, HQ, K, dtype=partial_dtype, device=q.device)
    dq_kernel = parallel_wall_attn_bwd_kernel_dq.fn if is_varlen else parallel_wall_attn_bwd_kernel_dq
    dkv_kernel = parallel_wall_attn_bwd_kernel_dkv.fn if is_varlen else parallel_wall_attn_bwd_kernel_dkv

    dg_cumsum = torch.empty(NV, B, T, HQ, K, dtype=torch.float, device=q.device)

    if g_scalar_cumsum is not None:
        dg_scalar_cumsum = torch.empty(NV, B, T, HQ, dtype=torch.float, device=q.device)
    else:
        dg_scalar_cumsum = None

    dq_kernel[grid](
        q=q,
        k=k,
        v=v,
        g_cumsum=g_cumsum,
        g_scalar_cumsum=g_scalar_cumsum,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        dg_cumsum=dg_cumsum,
        dg_scalar_cumsum=dg_scalar_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        T_BUCKET=T_BUCKET,
        W=window_size,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        **extra,
    )
    # reduce query-side partials before allocating key-side partials to bound peak temporary memory
    if NV > 1:
        dq = dq.sum(0)
        dg_cumsum = dg_cumsum.sum(0)
        if g_scalar_cumsum is not None:
            dg_scalar_cumsum = dg_scalar_cumsum.sum(0)
    else:
        dq = dq[0]
        dg_cumsum = dg_cumsum[0]
        if g_scalar_cumsum is not None:
            dg_scalar_cumsum = dg_scalar_cumsum[0]

    dk = torch.empty(NV, B, T, HQ, K, dtype=partial_dtype, device=q.device)
    dv = torch.empty(B, T, HQ, V, dtype=v.dtype if H == HQ else torch.float, device=q.device)
    dg_cumsum_k = torch.empty(NV, B, T, HQ, K, dtype=torch.float, device=q.device)
    if g_scalar_cumsum is not None:
        dg_scalar_cumsum_k = torch.empty(NV, B, T, HQ, dtype=torch.float, device=q.device)
    else:
        dg_scalar_cumsum_k = None

    dkv_kernel[grid](
        q=q,
        k=k,
        v=v,
        g_cumsum=g_cumsum,
        g_scalar_cumsum=g_scalar_cumsum,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        dg_cumsum=dg_cumsum_k,
        dg_scalar_cumsum=dg_scalar_cumsum_k,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        T_BUCKET=T_BUCKET,
        W=window_size,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        DIAG_BF16=_DKV_DIAG_BF16,
        **extra,
    )
    if NV > 1:
        dk = dk.sum(0)
        dg_cumsum_k = dg_cumsum_k.sum(0)
        if g_scalar_cumsum is not None:
            dg_scalar_cumsum_k = dg_scalar_cumsum_k.sum(0)
    else:
        dk = dk[0]
        dg_cumsum_k = dg_cumsum_k[0]
        if g_scalar_cumsum is not None:
            dg_scalar_cumsum_k = dg_scalar_cumsum_k[0]

    dk = reduce(dk, 'b t (h g) k -> b t h k', g=G, reduction='sum')
    dv = reduce(dv, 'b t (h g) v -> b t h v', g=G, reduction='sum')
    dg_cumsum.add_(dg_cumsum_k)

    if g_scalar_cumsum is not None:
        dg_scalar_cumsum.add_(dg_scalar_cumsum_k)

    dsink_bias = None
    if sink_bias is not None:
        p_sink_bias = torch.exp2(sink_bias[None, None, :] - lse)
        delta_total = delta.sum(0) if NV > 1 else delta[0]
        dsink_bias = -(p_sink_bias * delta_total).sum((0, 1))

    return dq, dk, dv, dg_cumsum, dsink_bias, dg_scalar_cumsum


class WallParallelAttentionFunction(torch.autograd.Function):
    r"""Wall Attention: per-channel prefix :math:`P` (log_2); scores use
    :math:`\sum_n q_{in} k_{jn} \exp_2(P_{in}-P_{jn})` via per-Q-block reference
    :math:`R_n=P_{i_t\cdot BT,n}` inside :func:`parallel_wall_attn_fwd_kernel`."""

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, sink_bias, scale, window_size, cu_seqlens, g_scalar=None, chunk_indices=None):
        if g.shape[-1] != q.shape[-1] or g.shape[2] != q.shape[2]:
            raise ValueError(
                f"`g` must be [B, T, HQ, K] with same HQ and K as `q`; got {g.shape} vs q {q.shape}"
            )
        if q.shape[2] % k.shape[2] != 0:
            raise ValueError("HQ must be divisible by `k.shape[2]` for GQA")

        if _DEBUG_ASSERTS:
            assert torch.isfinite(g).all(), f"[wall-attn fwd] NaN/Inf in g: {g.abs().max()}"
            assert torch.isfinite(q).all(), f"[wall-attn fwd] NaN/Inf in q: {q.abs().max()}"
            assert torch.isfinite(k).all(), f"[wall-attn fwd] NaN/Inf in k: {k.abs().max()}"
            assert torch.isfinite(v).all(), f"[wall-attn fwd] NaN/Inf in v: {v.abs().max()}"

        P = chunk_global_cumsum(g, cu_seqlens=cu_seqlens, scale=RCP_LN2)
        if _DEBUG_ASSERTS:
            assert torch.isfinite(P).all(), (
                f"[wall-attn fwd] NaN/Inf in P=cumsum(g): P abs max={P.abs().max()}, g abs max={g.abs().max()}"
            )

        if g_scalar is not None:
            if _DEBUG_ASSERTS:
                assert torch.isfinite(g_scalar).all(), f"[wall-attn fwd] NaN/Inf in g_scalar: {g_scalar.abs().max()}"
            c = chunk_global_cumsum(g_scalar, cu_seqlens=cu_seqlens, scale=RCP_LN2)
            if _DEBUG_ASSERTS:
                assert torch.isfinite(c).all(), f"[wall-attn fwd] NaN/Inf in c=cumsum(g_scalar): c abs max={c.abs().max()}"
        else:
            c = None

        sink_bias_scaled = sink_bias * RCP_LN2 if sink_bias is not None else None
        o, lse = parallel_wall_attn_fwd(
            q=q,
            k=k,
            v=v,
            g_cumsum=P,
            sink_bias=sink_bias_scaled,
            scale=scale,
            g_scalar_cumsum=c,
            window_size=window_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        if _DEBUG_ASSERTS:
            assert torch.isfinite(o).all(), (
                f"[wall-attn fwd] NaN/Inf in output o: o abs max={o.abs().max()}, "
                f"lse finite={torch.isfinite(lse).all()}, P abs max={P.abs().max()}"
            )

        ctx.save_for_backward(q, k, v, o, P, lse, sink_bias_scaled, c if c is not None else torch.empty(0))
        ctx.scale = scale
        ctx.window_size = window_size
        ctx.cu_seqlens = cu_seqlens
        ctx.has_scalar_g = g_scalar is not None
        return o.to(q.dtype)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, o, P, lse, sink_bias_scaled, c_or_empty = ctx.saved_tensors
        c = c_or_empty if ctx.has_scalar_g else None

        if _DEBUG_ASSERTS:
            assert torch.isfinite(do).all(), f"[wall-attn bwd] NaN/Inf in do: {do.abs().max()}"

        dq, dk, dv, dP, dsink_bias, dc = parallel_wall_attn_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            g_cumsum=P,
            lse=lse,
            do=do,
            sink_bias=sink_bias_scaled,
            scale=ctx.scale,
            g_scalar_cumsum=c,
            window_size=ctx.window_size,
            cu_seqlens=ctx.cu_seqlens,
        )
        if _DEBUG_ASSERTS:
            assert torch.isfinite(dq).all(), f"[wall-attn bwd] NaN/Inf in dq: {dq.abs().max()}"
            assert torch.isfinite(dk).all(), f"[wall-attn bwd] NaN/Inf in dk: {dk.abs().max()}"
            assert torch.isfinite(dv).all(), f"[wall-attn bwd] NaN/Inf in dv: {dv.abs().max()}"
            assert torch.isfinite(dP).all(), (
                f"[wall-attn bwd] NaN/Inf in dP: dP abs max={dP.abs().max()}, "
                f"P abs max={P.abs().max()}, lse finite={torch.isfinite(lse).all()}"
            )

        # The kernel emits dP with an LN2 factor (b_dg = LN2 * q * dq); the forward set
        # P = cumsum(g) * RCP_LN2, so dL/dg = RCP_LN2 * reverse_cumsum(dP) (nets the LN2).
        dg = chunk_global_cumsum(dP, cu_seqlens=ctx.cu_seqlens, reverse=True, scale=RCP_LN2)
        if _DEBUG_ASSERTS:
            assert torch.isfinite(dg).all(), (
                f"[wall-attn bwd] NaN/Inf in dg (after rev cumsum): dg abs max={dg.abs().max()}, "
                f"dP abs max={dP.abs().max()}"
            )

        if dc is not None:
            dg_scalar = chunk_global_cumsum(dc, cu_seqlens=ctx.cu_seqlens, reverse=True)
        else:
            dg_scalar = None

        return dq.to(q), dk.to(k), dv.to(v), dg, dsink_bias, None, None, None, dg_scalar, None


def parallel_wall_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    *,
    g_scalar: torch.Tensor | None = None,
    sink_bias: torch.Tensor | None = None,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""Wall parallel attention (training / prefill forward + backward).

    Scores use a per-channel multiplicative decay: with :math:`P = \mathrm{cumsum}(g)`
    in :math:`\log_2` space, the logit for pair :math:`(i, j)` is
    :math:`\mathrm{scale} \sum_n q_{in} k_{jn} \exp_2(P_{in} - P_{jn})`.

    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, HQ, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]`` (``H`` may be < ``HQ`` for GQA).
        v (torch.Tensor):
            Values of shape ``[B, T, H, V]``.
        g (torch.Tensor):
            Per-channel log-decay of shape ``[B, T, HQ, K]`` (cumsum'd internally).
        g_scalar (torch.Tensor, Optional):
            FoX-style additive scalar gate of shape ``[B, T, HQ]``. Default: `None`.
        sink_bias (torch.Tensor, Optional):
            Attention-sink logit of shape ``[HQ]``. Default: `None`.
        scale (float, Optional):
            Softmax scale. If `None`, defaults to ``K ** -0.5``. Default: `None`.
        window_size (int, Optional):
            Sliding-window width (causal). Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative seqlens of shape ``[N + 1]`` for varlen packing
            (requires ``B == 1``). Default: `None`.

    Returns:
        Attention output of shape ``[B, T, HQ, V]``.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if g_scalar is not None and g_scalar.shape != q.shape[:-1]:
        raise ValueError(f"`g_scalar` must be [B, T, HQ] matching q.shape[:-1]; got {g_scalar.shape}")
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("`cu_seqlens` (varlen) requires batch size 1")
    if sink_bias is not None and sink_bias.shape != (q.shape[2],):
        raise ValueError(f"`sink_bias` must be [HQ]; got {sink_bias.shape}")
    return WallParallelAttentionFunction.apply(
        q, k, v, g, sink_bias, scale, window_size, cu_seqlens, g_scalar, None
    )
