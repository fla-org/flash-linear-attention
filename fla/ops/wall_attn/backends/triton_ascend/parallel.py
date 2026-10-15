# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Wall attention, contributed by Tilde Research (Timor Averbuch, Dhruv Pai).
# Heavily modified from fla/ops/gated_oja_rule/chunk.py.
"""Triton-Ascend Ascend NPU implementations of the Wall training entries.

Registered under the ``wall_attn`` operation for ``parallel_wall_attn_fwd`` and
``parallel_wall_attn_bwd``; the autograd Function and the public entry point stay
in ``fla/ops/wall_attn/parallel.py``. Kernel math is unchanged from the default
implementation; the Ascend adaptations are host-side and launch-configuration:

- Head dimensions are zero-padded to ``max(16, next_power_of_2(dim))`` on the host,
  so the in-kernel head-dim masks fold. A masked head-dim load trips a
  triton-ascend 3.2.2 ``vector.transfer_write`` permutation defect when
  ``K != BK``. Zero padding does not change the math; the backward slices the
  gradients back to the original dimensions.
- The autotune sweep covers ``BT x BS`` only: ``num_warps`` and ``num_stages``
  are unsupported on the Ascend backend
  (``.agents/skills/fla-ascend-performance``) and are omitted.
- The variable-length backward pins ``BT=64``; ``BT=128`` overflows UB in the
  ``bwd_dkv`` kernel. The forward keeps upstream's ``BT=128``, so the backward
  always rebuilds its chunk map instead of trusting a caller-supplied one.
- Every attention kernel launch passes ``ascend_compile_kwargs()`` (disables
  auto-multi-buffer) to bound UB usage.
- The backward ``dq`` kernel selects its off-diagonal term by whether the
  off-diagonal loop ran: on some Ascend targets a zero-trip loop does not honor
  the ``tl.zeros`` initializer of ``b_dq_til``, and reading the uninitialized
  accumulator corrupts ``dq``. When the loop runs the values are unchanged.
"""

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import reduce

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2, log2, unflatten_program_id
from fla.utils import ascend_compile_kwargs, autotune_cache_kwargs

# Set WALL_ATTN_DKV_DIAG_BF16=0 to force fp32 tensor cores in the diagonal loop
# of parallel_wall_attn_bwd_kernel_dkv (precision over speed).
_DKV_DIAG_BF16 = os.environ.get("WALL_ATTN_DKV_DIAG_BF16", "1") == "1"

# Ascend: num_warps/num_stages are unsupported on this backend (see
# .agents/skills/fla-ascend-performance), and the BT x BS sweep stays small so
# the autotuner does not pile up multi-minute compiles.
WALL_FWD_AUTOTUNE_CONFIGS = [
    triton.Config({'BT': BT, 'BS': BS})
    for BT in (64, 128)
    for BS in (32, 64)
]

WALL_BWD_AUTOTUNE_CONFIGS = [
    triton.Config({'BT': BT, 'BS': BS})
    for BT in (64, 128)
    for BS in (32, 64)
]


def _prune_wall_fwd_configs(configs, nargs, **kwargs):
    # The per-program fp32 accumulator tile is BT x BV. Drop configs that exceed
    # the tile budget: they spill registers massively, can never win the
    # benchmark, and each takes minutes to compile under IEEE fp32 dots, which
    # reads as a multi-minute stall per autotune key. BV never exceeds the fwd
    # cap, so BT=64 always survives and the pruned set is non-empty.
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
    i_v, i_t = unflatten_program_id(tl.cdiv(V, BV))
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    o_q = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_q = o_q < T
    m_qk = m_q[:, None] & (o_d[None, :] < K)
    p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    p_o = o + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    p_lse = lse + bos * HQ + i_hq + o_q * HQ

    p_pq = g_cumsum + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    o_r = i_t * BT + tl.arange(0, 1)
    p_R = g_cumsum + (bos * HQ + i_hq) * K + o_r[:, None] * (HQ*K) + o_d[None, :]

    b_q = tl.load(p_q, mask=m_qk, other=0.0)
    b_pq = tl.load(p_pq, mask=m_qk, other=0.0).to(tl.float32)
    b_R = tl.load(p_R, mask=(o_r[:, None] < T) & (o_d[None, :] < K), other=0.0).to(tl.float32)
    # b_R is at i_t*BT; the same value is used in both off-diag and diag loops,
    # since |b_pq - b_R| within a BT-chunk is bounded by BT*|g_max|*RCP_LN2.
    # Off-diagonal: P_q - R <= 0 and R - P_k <= 0 -> exp2 <= 1 -> bf16 safe.
    b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_R)).to(b_q.dtype)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    if USE_SCALAR_G:
        p_cq = g_scalar_cumsum + bos * HQ + i_hq + o_q * HQ
        b_cq = tl.load(p_cq, mask=m_q, other=0.0).to(tl.float32)

    if USE_SINK_BIAS:
        b_sink_bias = tl.load(sink_bias + i_hq).to(tl.float32)
    else:
        b_sink_bias = None

    i_start = tl.maximum((i_t * BT - W + 1) // BS * BS, 0) if USE_WINDOW else 0

    b_R_t = tl.trans(b_R)

    for i_s in range(i_start, i_t * BT, BS):
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]
        p_pk = g_cumsum + (bos * HQ + i_hq) * K + o_d[:, None] + o_k[None, :] * (HQ*K)

        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
        b_pk = tl.load(p_pk, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0).to(tl.float32)
        b_k_til = (b_k.to(tl.float32) * exp2(b_R_t - b_pk)).to(b_k.dtype)
        b_s = tl.dot(b_q_til, b_k_til) * scale * RCP_LN2

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
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]
        p_pk = g_cumsum + (bos * HQ + i_hq) * K + o_d[:, None] + o_k[None, :] * (HQ*K)

        # Per-sub-block local reference to prevent exp2 overflow.
        o_r = i_s + tl.arange(0, 1)
        p_R_local = g_cumsum + (bos * HQ + i_hq) * K + o_r[:, None] * (HQ*K) + o_d[None, :]
        b_R_local = tl.load(p_R_local, mask=(o_r[:, None] < T) & (o_d[None, :] < K), other=0.0).to(tl.float32)
        b_R_local_bc = tl.broadcast_to(b_R_local, (BT, BK))
        b_R_local_t = tl.trans(b_R_local)

        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
        b_pk = tl.load(p_pk, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0).to(tl.float32)

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
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_q[:, None] & (o_v[None, :] < V))
    if i_v == 0:
        tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), mask=m_q)


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
    i_v, i_t = unflatten_program_id(tl.cdiv(V, BV))
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    i_v64 = i_v.to(tl.int64)
    delta += i_v64 * B * T * HQ
    dq += i_v64 * B * T * HQ * K
    dg_cumsum += i_v64 * B * T * HQ * K
    if USE_SCALAR_G:
        dg_scalar_cumsum += i_v64 * B * T * HQ

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216
    LN2: tl.constexpr = 0.6931471805599453

    o_q = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_q = o_q < T
    m_qk = m_q[:, None] & (o_d[None, :] < K)
    m_o = m_q[:, None] & (o_v[None, :] < V)
    p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    p_pq = g_cumsum + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    o_r = i_t * BT + tl.arange(0, 1)
    p_R = g_cumsum + (bos * HQ + i_hq) * K + o_r[:, None] * (HQ*K) + o_d[None, :]
    p_dq = dq + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    p_dg = dg_cumsum + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    p_do = do + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    p_lse = lse + bos * HQ + i_hq + o_q * HQ
    p_delta = delta + bos * HQ + i_hq + o_q * HQ

    b_q = tl.load(p_q, mask=m_qk, other=0.0)
    b_pq = tl.load(p_pq, mask=m_qk, other=0.0).to(tl.float32)
    b_R = tl.load(p_R, mask=(o_r[:, None] < T) & (o_d[None, :] < K), other=0.0).to(tl.float32)
    # Off-diagonal q_til (bf16 tensor cores, exp2 <= 1).
    b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_R)).to(b_q.dtype)

    b_do = tl.load(p_do, mask=m_o, other=0.0)
    b_lse = tl.load(p_lse, mask=m_q, other=0.0)
    b_delta = tl.load(p_delta, mask=m_q, other=0.0)

    b_dq_til = tl.zeros([BT, BK], dtype=tl.float32)

    if USE_SCALAR_G:
        p_cq = g_scalar_cumsum + bos * HQ + i_hq + o_q * HQ
        b_cq = tl.load(p_cq, mask=m_q, other=0.0).to(tl.float32)
        b_dc = tl.zeros([BT], dtype=tl.float32)

    i_start = tl.maximum((i_t * BT - W + 1) // BS * BS, 0) if USE_WINDOW else 0
    b_R_t = tl.trans(b_R)

    for i_s in range(i_start, i_t * BT, BS):
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_v[:, None] + o_k[None, :] * (H*V)
        p_pk = g_cumsum + (bos * HQ + i_hq) * K + o_d[:, None] + o_k[None, :] * (HQ*K)

        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        b_pk = tl.load(p_pk, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=(o_v[:, None] < V) & m_k[None, :], other=0.0)
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
        b_dq_til = tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k_til), b_dq_til)
        if USE_SCALAR_G:
            b_dc += tl.sum(b_ds, 1)

    # Diagonal: use per-sub-block local reference to prevent exp2 overflow.
    # Accumulate dq_diag directly in output space to avoid large frame corrections.
    b_dq_diag = tl.zeros([BT, BK], dtype=tl.float32)

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_v[:, None] + o_k[None, :] * (H*V)
        p_pk = g_cumsum + (bos * HQ + i_hq) * K + o_d[:, None] + o_k[None, :] * (HQ*K)

        o_r = i_s + tl.arange(0, 1)
        p_R_local = g_cumsum + (bos * HQ + i_hq) * K + o_r[:, None] * (HQ*K) + o_d[None, :]
        b_R_local = tl.load(p_R_local, mask=(o_r[:, None] < T) & (o_d[None, :] < K), other=0.0).to(tl.float32)
        b_R_local_bc = tl.broadcast_to(b_R_local, (BT, BK))
        b_R_local_t = tl.trans(b_R_local)

        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        b_pk = tl.load(p_pk, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=(o_v[:, None] < V) & m_k[None, :], other=0.0)

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

    # Off-diagonal (chunk-ref) + diagonal (direct output space). Select the
    # off-diagonal term by whether the loop above ran: on Ascend a zero-trip loop
    # does not honor the `tl.zeros` initializer of `b_dq_til` on some targets,
    # and the uninitialized accumulator would corrupt `dq`. When the loop runs,
    # the values are unchanged.
    has_off = i_start < i_t * BT
    b_dq = tl.where(has_off, (b_dq_til * scale) * exp2(b_pq - b_R), 0.0) + b_dq_diag * scale

    # Structural: b_dg derived from b_dq (saves [BT,BK] fp32 accumulator).
    b_dg = LN2 * b_q.to(tl.float32) * b_dq
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_qk)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_qk)
    if USE_SCALAR_G:
        p_dc = dg_scalar_cumsum + bos * HQ + i_hq + o_q * HQ
        tl.store(p_dc, b_dc.to(p_dc.dtype.element_ty), mask=m_q)


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
    i_v, i_t = unflatten_program_id(tl.cdiv(V, BV))
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    i_v64 = i_v.to(tl.int64)
    delta += i_v64 * B * T * HQ
    dk += i_v64 * B * T * HQ * K
    dg_cumsum += i_v64 * B * T * HQ * K
    if USE_SCALAR_G:
        dg_scalar_cumsum += i_v64 * B * T * HQ

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216
    LN2: tl.constexpr = 0.6931471805599453

    o_k = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < T
    m_kk = m_k[:, None] & (o_d[None, :] < K)
    m_kv = m_k[:, None] & (o_v[None, :] < V)
    p_k = k + (bos * H + i_h) * K + o_k[:, None] * (H*K) + o_d[None, :]
    p_pk = g_cumsum + (bos * HQ + i_hq) * K + o_k[:, None] * (HQ*K) + o_d[None, :]
    p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]
    p_dk = dk + (bos * HQ + i_hq) * K + o_k[:, None] * (HQ*K) + o_d[None, :]
    p_dv = dv + (bos * HQ + i_hq) * V + o_k[:, None] * (HQ*V) + o_v[None, :]
    p_dg = dg_cumsum + (bos * HQ + i_hq) * K + o_k[:, None] * (HQ*K) + o_d[None, :]

    b_k = tl.load(p_k, mask=m_kk, other=0.0)
    b_pk = tl.load(p_pk, mask=m_kk, other=0.0).to(tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_v = tl.load(p_v, mask=m_kv, other=0.0)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    if USE_SCALAR_G:
        p_ck = g_scalar_cumsum + bos * HQ + i_hq + o_k * HQ
        b_ck = tl.load(p_ck, mask=m_k, other=0.0).to(tl.float32)
        b_dc = tl.zeros([BT], dtype=tl.float32)

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        o_r = i_s + tl.arange(0, 1)
        p_Rq = g_cumsum + (bos * HQ + i_hq) * K + o_r[:, None] * (HQ*K) + o_d[None, :]
        p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
        p_pq = g_cumsum + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
        p_do = do + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
        p_lse = lse + bos * HQ + i_hq + o_q * HQ
        p_delta = delta + bos * HQ + i_hq + o_q * HQ

        b_Rq = tl.load(p_Rq, mask=(o_r[:, None] < T) & (o_d[None, :] < K), other=0.0).to(tl.float32)
        b_Rq_bc_k = tl.broadcast_to(b_Rq, (BT, BK))
        b_Rq_bc_q = tl.broadcast_to(b_Rq, (BS, BK))
        b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
        b_pq = tl.load(p_pq, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0).to(tl.float32)
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
        b_do = tl.load(p_do, mask=m_q[:, None] & (o_v[None, :] < V), other=0.0)
        b_lse = tl.load(p_lse, mask=m_q, other=0.0)
        b_delta = tl.load(p_delta, mask=m_q, other=0.0)
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
        b_dv = tl.dot(b_p.to(b_do.dtype), b_do, b_dv)
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
        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        o_r = i_s + tl.arange(0, 1)
        p_Rq = g_cumsum + (bos * HQ + i_hq) * K + o_r[:, None] * (HQ*K) + o_d[None, :]
        p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
        p_pq = g_cumsum + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
        p_do = do + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
        p_lse = lse + bos * HQ + i_hq + o_q * HQ
        p_delta = delta + bos * HQ + i_hq + o_q * HQ

        b_Rq = tl.load(p_Rq, mask=(o_r[:, None] < T) & (o_d[None, :] < K), other=0.0).to(tl.float32)
        b_Rq_bc_k = tl.broadcast_to(b_Rq, (BT, BK))
        b_Rq_bc_q = tl.broadcast_to(b_Rq, (BS, BK))
        b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
        b_pq = tl.load(p_pq, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0).to(tl.float32)
        b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_Rq_bc_q)).to(b_q.dtype)
        b_exp_k_off = b_Rq_bc_k - b_pk
        b_exp_k_off_val = exp2(b_exp_k_off)  # CSE: reused for k_til and final dk scaling
        b_k_til = (b_k.to(tl.float32) * b_exp_k_off_val).to(b_k.dtype)
        b_do = tl.load(p_do, mask=m_q[:, None] & (o_v[None, :] < V), other=0.0)
        b_lse = tl.load(p_lse, mask=m_q, other=0.0)
        b_delta = tl.load(p_delta, mask=m_q, other=0.0)
        b_s = tl.dot(b_k_til, tl.trans(b_q_til)) * scale * RCP_LN2
        if USE_SCALAR_G:
            b_cq = tl.load(g_scalar_cumsum + (bos + o_q) * HQ + i_hq, mask=m_q, other=0).to(tl.float32)
            b_s += b_cq[None, :] - b_ck[:, None]
        if USE_WINDOW:
            b_p = tl.where((o_q[None, :] - o_k[:, None] < W) & m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        else:
            b_p = tl.where(m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        b_dv = tl.dot(b_p.to(b_do.dtype), b_do, b_dv)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp - b_delta[None, :])
        b_dk_til = tl.dot(b_ds.to(b_q.dtype), b_q_til)
        b_dk += (b_dk_til * scale) * b_exp_k_off_val
        if USE_SCALAR_G:
            b_dc -= tl.sum(b_ds, 1)

    # Structural: b_dg derived from b_dk (saves [BT,BK] fp32 accumulator).
    b_dg = -LN2 * b_k.to(tl.float32) * b_dk
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_kk)
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_kv)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_kk)
    if USE_SCALAR_G:
        p_dc = dg_scalar_cumsum + bos * HQ + i_hq + o_k * HQ
        tl.store(p_dc, b_dc.to(p_dc.dtype.element_ty), mask=m_k)


def parallel_wall_attn_fwd_npu(
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

    # Ascend: zero-pad the head dims so the in-kernel head-dim masks fold
    # (a masked head-dim load trips a triton-ascend 3.2.2 permutation defect
    # when K != BK). Padding is a no-op for the math.
    K_orig, V_orig = k.shape[-1], v.shape[-1]
    pad_k = max(16, triton.next_power_of_2(K_orig)) - K_orig
    pad_v = max(16, triton.next_power_of_2(V_orig)) - V_orig
    if pad_k:
        q = F.pad(q, (0, pad_k))
        k = F.pad(k, (0, pad_k))
        g_cumsum = F.pad(g_cumsum, (0, pad_k))
    if pad_v:
        v = F.pad(v, (0, pad_v))

    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
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
        BS = min(32, max(16, triton.next_power_of_2(T)))
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
        o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
        lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
        parallel_wall_attn_fwd_kernel.fn[(NV * NT, B * HQ)](
            q=q, k=k, v=v, o=o,
            g_cumsum=g_cumsum, g_scalar_cumsum=g_scalar_cumsum, sink_bias=sink_bias,
            lse=lse, scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            B=B, T=T, T_BUCKET=1, W=window_size, H=H, HQ=HQ, G=G, K=K, V=V,
            BT=BT, BS=BS, BK=BK, BV=BV,
            **ascend_compile_kwargs(),
        )
        return (o[..., :V_orig] if pad_v else o), lse

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    # the bucket selects an autotune config; exact T remains the runtime loop bound
    T_BUCKET = triton.next_power_of_2(T)

    def grid(meta):
        return (NV * triton.cdiv(T, meta['BT']), B * HQ)
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
        **ascend_compile_kwargs(),
    )
    return (o[..., :V_orig] if pad_v else o), lse


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


def parallel_wall_attn_bwd_npu(
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

    # Ascend: zero-pad the head dims so the in-kernel head-dim masks fold; the
    # gradients are sliced back to the original dimensions before returning.
    K_orig, V_orig = k.shape[-1], v.shape[-1]
    pad_k = max(16, triton.next_power_of_2(K_orig)) - K_orig
    pad_v = max(16, triton.next_power_of_2(V_orig)) - V_orig
    if pad_k:
        q = F.pad(q, (0, pad_k))
        k = F.pad(k, (0, pad_k))
        g_cumsum = F.pad(g_cumsum, (0, pad_k))
    if pad_v:
        v = F.pad(v, (0, pad_v))
        o = F.pad(o, (0, pad_v))
        do = F.pad(do, (0, pad_v))

    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BK = max(triton.next_power_of_2(K), 16)
    BV = min(64, max(triton.next_power_of_2(V), 16))

    NV = triton.cdiv(V, BV)

    # Varlen: pin BT=128 and bypass the autotuner (BT sweep would invalidate the
    # precomputed chunk map). `extra` injects the fixed launch config via `.fn`.
    is_varlen = cu_seqlens is not None
    # varlen bypasses autotuning; dense lengths share configs within power-of-two buckets
    T_BUCKET = 1 if is_varlen else triton.next_power_of_2(T)
    extra = {}
    if is_varlen:
        # Ascend: pin BT=64; BT=128 overflows UB in the bwd_dkv kernel. The
        # forward pins BT=128, so a caller-supplied `chunk_indices` map (built
        # for 128) would halve NT here and leave the tail tokens zero; always
        # rebuild the map at BT=64.
        BT = 64
        BS = min(32, max(16, triton.next_power_of_2(T)))
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
        grid = (NV * NT, B * HQ)
        extra = dict(BT=BT, BS=BS)
    else:
        def grid(meta):
            return (NV * triton.cdiv(T, meta['BT']), B * HQ)

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
        **ascend_compile_kwargs(),
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
        **ascend_compile_kwargs(),
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

    return (
        dq[..., :K_orig],
        dk[..., :K_orig],
        dv[..., :V_orig],
        dg_cumsum[..., :K_orig],
        dsink_bias,
        dg_scalar_cumsum,
    )
