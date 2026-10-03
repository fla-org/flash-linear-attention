# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


# Portions adapted from CyclicFlowAttention, Copyright (c) 2026 Yixiao Chen.
# https://github.com/Chyxx/CyclicFlowAttention

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla.ops.common.chunk_o import BKV_LIST, NUM_WARPS, chunk_bwd_dv, chunk_fwd_o
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.cumsum import chunk_global_cumsum, chunk_local_cumsum
from fla.ops.utils.op import exp2
from fla.utils import (
    IS_NVIDIA_HOPPER,
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    get_multiprocessor_count,
    input_guard,
)

from .qk_norm import qk_rmsnorm_fwd
from .utils import (
    CyFAState,
    add_segment_initial,
    build_readout_table,
    cyfa_clock_cumsum,
    cyfa_cos,
    cyfa_sin,
    modal_omega,
    modal_phase,
    prepare_cu_seqlens,
    segment_sum,
    validate_cyfa_inputs,
    validate_initial_state,
)

_CHUNK_SIZE = 64
_READOUT_BWD_AUTOTUNE_CONFIGS = [
    triton.Config({'BT': bt}, num_warps=num_warps, num_stages=num_stages)
    for bt in [16, 32, 64]
    for num_warps in NUM_WARPS
    for num_stages in [2, 3, 4]
]
_POINT_AUTOTUNE_CONFIGS = [
    triton.Config({'BT': bt}, num_warps=num_warps)
    for bt in [16, 32, 64]
    for num_warps in NUM_WARPS
]
_TABLE_GRAD_AUTOTUNE_CONFIGS = [
    triton.Config(
        {'BN': bn, 'BR': br, 'BC': bc},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for bn in [64, 128]
    for br, bc in [(32, 32), (32, 64), (64, 64)]
    for num_warps in NUM_WARPS
    for num_stages in [2, 3]
]
_FUSED_READOUT_FWD_AUTOTUNE_CONFIGS = [
    triton.Config(
        {'BK': bk, 'BR': br},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for bk in [128]
    for br in [16, 32]
    for num_warps in NUM_WARPS
    for num_stages in [2, 3, 4]
]
_RECURRENCE_BWD_AUTOTUNE_CONFIGS = [
    triton.Config(
        {'BK': bk, 'BV': bv},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for bk in BKV_LIST
    for bv in BKV_LIST
    for num_warps in NUM_WARPS
    for num_stages in [2, 3, 4]
]


@triton.jit(do_not_specialize=['N'])
def _chunk_cyfa_qk_norm_bwd_kernel(
    q,
    k,
    dq_out,
    dk_out,
    q_weight,
    k_weight,
    q_rstd,
    k_rstd,
    dq,
    dk,
    dweight,
    N,
    D: tl.constexpr,
    BS: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_s = tl.program_id(0).to(tl.int64)
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_q_weight = tl.load(q_weight + o_d, mask=m_d).to(tl.float32)
    b_k_weight = tl.load(k_weight + o_d, mask=m_d).to(tl.float32)
    b_dq_weight = tl.zeros((BT, BD), dtype=tl.float32)
    b_dk_weight = tl.zeros((BT, BD), dtype=tl.float32)

    for i_n in range(i_s * BS, i_s * BS + BS, BT):
        o_n = i_n + tl.arange(0, BT)
        m_n = o_n < N
        m_acc = o_n < min((i_s + 1) * BS, N)
        mask = m_n[:, None] & m_d[None, :]
        acc_mask = m_acc[:, None] & m_d[None, :]
        offsets = o_n[:, None] * D + o_d[None, :]

        b_q = tl.load(q + offsets, mask=mask, other=0.0).to(tl.float32)
        b_dq_out = tl.load(dq_out + offsets, mask=mask, other=0.0).to(tl.float32)
        b_q_rstd = tl.load(q_rstd + o_n, mask=m_n, other=0.0)
        b_q_hat = tl.where(m_d[None, :], b_q * b_q_rstd[:, None], 0.0)
        b_q_wdy = b_dq_out * b_q_weight[None, :]
        b_q_proj = tl.sum(b_q_hat * b_q_wdy, axis=1) / D
        b_dq = (b_q_wdy - b_q_hat * b_q_proj[:, None]) * b_q_rstd[:, None]
        tl.store(dq + offsets, b_dq.to(dq.dtype.element_ty), mask=mask)
        b_dq_weight += tl.where(acc_mask, b_dq_out * b_q_hat, 0.0)

        b_k = tl.load(k + offsets, mask=mask, other=0.0).to(tl.float32)
        b_dk_out = tl.load(dk_out + offsets, mask=mask, other=0.0).to(tl.float32)
        b_k_rstd = tl.load(k_rstd + o_n, mask=m_n, other=0.0)
        b_k_hat = tl.where(m_d[None, :], b_k * b_k_rstd[:, None], 0.0)
        b_k_wdy = b_dk_out * b_k_weight[None, :]
        b_k_proj = tl.sum(b_k_hat * b_k_wdy, axis=1) / D
        b_dk = (b_k_wdy - b_k_hat * b_k_proj[:, None]) * b_k_rstd[:, None]
        tl.store(dk + offsets, b_dk.to(dk.dtype.element_ty), mask=mask)
        b_dk_weight += tl.where(acc_mask, b_dk_out * b_k_hat, 0.0)

    partial_offset = i_s * D + o_d
    tl.store(dweight + partial_offset, tl.sum(b_dq_weight, axis=0), mask=m_d)
    tl.store(dweight + tl.num_programs(0).to(tl.int64) * D + partial_offset, tl.sum(b_dk_weight, axis=0), mask=m_d)


def _qk_rmsnorm_bwd(
    dq_out: torch.Tensor,
    dk_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.shape != k.shape or dq_out.shape != q.shape or dk_out.shape != k.shape:
        raise ValueError('Fused Q/K RMSNorm backward requires matching tensor shapes.')
    q = q.contiguous()
    k = k.contiguous()
    dq_out = dq_out.contiguous()
    dk_out = dk_out.contiguous()
    n_rows = q.numel() // q.shape[-1]
    d_head = q.shape[-1]
    n_splits = min(get_multiprocessor_count(q.device.index), n_rows)
    block_rows = triton.cdiv(n_rows, n_splits)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dweight = torch.empty((2, n_splits, d_head), device=q.device, dtype=torch.float32)

    _chunk_cyfa_qk_norm_bwd_kernel[(n_splits,)](
        q,
        k,
        dq_out,
        dk_out,
        q_weight,
        k_weight,
        q_rstd,
        k_rstd,
        dq,
        dk,
        dweight,
        N=n_rows,
        D=d_head,
        BS=block_rows,
        BT=32,
        BD=triton.next_power_of_2(d_head),
        num_warps=8,
        num_stages=3,
    )
    dq_weight, dk_weight = dweight.sum(dim=1).to(q_weight).unbind(0)
    return dq, dk, dq_weight, dk_weight


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@fla_cache_autotune(
    configs=_RECURRENCE_BWD_AUTOTUNE_CONFIGS,
    key=['H', 'K', 'V', 'BT'],
    reset_to_zero=['dg_force'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def _chunk_cyfa_bwd_dqk_kernel(
    q,
    k,
    v,
    g,
    h,
    do,
    dh,
    dq,
    dk,
    dg_force,
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
    i_k = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int64)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        nt = tl.cdiv(T, BT)
        i_tg = i_b * nt + i_t
        bos = i_b * T

    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V
    dh += (i_tg * H + i_h).to(tl.int64) * K * V
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K

    b_dq = tl.zeros((BT, BK), dtype=tl.float32)
    b_dk = tl.zeros((BT, BK), dtype=tl.float32)
    b_ds = tl.zeros((BT, BT), dtype=tl.float32)
    # g_last transports recurrent state into the next chunk; inject its boundary adjoint at the last cumsum coordinate.
    b_dg_last = tl.zeros((1,), dtype=tl.float32)
    for i_v in range(0, tl.cdiv(V, BV)):
        o_v_0 = i_t * BT + tl.arange(0, BT)
        o_v_1 = i_v * BV + tl.arange(0, BV)
        p_v = v + o_v_0[:, None] * (H * V) + o_v_1[None, :] * (1)
        m_v = (o_v_0[:, None] < T) & (o_v_1[None, :] < V)
        o_do_0 = i_t * BT + tl.arange(0, BT)
        o_do_1 = i_v * BV + tl.arange(0, BV)
        p_do = do + o_do_0[:, None] * (H * V) + o_do_1[None, :] * (1)
        m_do = (o_do_0[:, None] < T) & (o_do_1[None, :] < V)
        o_h_0 = i_v * BV + tl.arange(0, BV)
        o_h_1 = i_k * BK + tl.arange(0, BK)
        p_h = h + o_h_0[:, None] * (1) + o_h_1[None, :] * (V)
        m_h = (o_h_0[:, None] < V) & (o_h_1[None, :] < K)
        o_dh_0 = i_v * BV + tl.arange(0, BV)
        o_dh_1 = i_k * BK + tl.arange(0, BK)
        p_dh = dh + o_dh_0[:, None] * (1) + o_dh_1[None, :] * (V)
        m_dh = (o_dh_0[:, None] < V) & (o_dh_1[None, :] < K)
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_do = tl.load(p_do, mask=m_do, other=0.0)
        b_h = tl.load(p_h, mask=m_h, other=0.0)
        b_dh = tl.load(p_dh, mask=m_dh, other=0.0)
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))

    o_q_0 = i_t * BT + tl.arange(0, BT)
    o_q_1 = i_k * BK + tl.arange(0, BK)
    p_q = q + o_q_0[:, None] * (H * K) + o_q_1[None, :] * (1)
    m_q = (o_q_0[:, None] < T) & (o_q_1[None, :] < K)
    o_k_0 = i_t * BT + tl.arange(0, BT)
    o_k_1 = i_k * BK + tl.arange(0, BK)
    p_k = k + o_k_0[:, None] * (H * K) + o_k_1[None, :] * (1)
    m_k = (o_k_0[:, None] < T) & (o_k_1[None, :] < K)
    b_q = tl.load(p_q, mask=m_q, other=0.0)
    b_k = tl.load(p_k, mask=m_k, other=0.0)

    o_t = i_t * BT + tl.arange(0, BT)
    mask_t = o_t < T
    mask_a = (o_t[:, None] >= o_t[None, :]) & mask_t[:, None] & mask_t[None, :]
    o_g_0 = i_t * BT + tl.arange(0, BT)
    p_g = g + bos * H + i_h + o_g_0 * (H)
    m_g = o_g_0 < T
    b_g = tl.load(p_g, mask=m_g, other=0.0)
    b_g_last = tl.load(g + (bos + min(i_t * BT + BT, T) - 1) * H + i_h)

    b_dg_last *= exp2(b_g_last)
    b_dq *= exp2(b_g)[:, None] * scale
    b_dk *= tl.where(mask_t, exp2(-b_g + b_g_last), 0.0)[:, None]
    b_dg_last += tl.sum(b_dk * b_k)
    b_ds = tl.where(mask_a, b_ds * exp2(b_g[:, None] - b_g[None, :]), 0.0) * scale
    b_ds = b_ds.to(b_k.dtype)
    b_dq += tl.dot(b_ds, b_k)
    b_dk += tl.dot(tl.trans(b_ds), b_q)

    force = tl.sum(b_dq * b_q, axis=1) - tl.sum(b_dk * b_k, axis=1)
    last = min(i_t * BT + BT, T) - 1
    force += tl.where(o_t == last, b_dg_last, 0.0)
    tl.atomic_add(
        dg_force + (bos + o_t) * H + i_h,
        force,
        mask=mask_t,
        sem='relaxed',
    )

    o_dq_0 = i_t * BT + tl.arange(0, BT)
    o_dq_1 = i_k * BK + tl.arange(0, BK)
    p_dq = dq + o_dq_0[:, None] * (H * K) + o_dq_1[None, :] * (1)
    m_dq = (o_dq_0[:, None] < T) & (o_dq_1[None, :] < K)
    o_dk_0 = i_t * BT + tl.arange(0, BT)
    o_dk_1 = i_k * BK + tl.arange(0, BK)
    p_dk = dk + o_dk_0[:, None] * (H * K) + o_dk_1[None, :] * (1)
    m_dk = (o_dk_0[:, None] < T) & (o_dk_1[None, :] < K)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_dq)
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_dk)


def _scalar_decay_bwd_dqk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    *,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    chunk_size: int,
    chunk_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seq_len, n_heads, d_k = q.shape
    d_v = v.shape[-1]
    nt = triton.cdiv(seq_len, chunk_size) if chunk_indices is None else len(chunk_indices)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg_force = torch.zeros(bsz, seq_len, n_heads, device=q.device, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(d_k, meta['BK']), nt, bsz * n_heads)

    _chunk_cyfa_bwd_dqk_kernel[grid](
        q,
        k,
        v,
        g,
        h,
        do,
        dh,
        dq,
        dk,
        dg_force,
        cu_seqlens,
        chunk_indices,
        scale,
        T=seq_len,
        H=n_heads,
        K=d_k,
        V=d_v,
        BT=chunk_size,
    )
    return dq, dk, dg_force


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T'])
def _chunk_cyfa_bwd_dqk_dv_kernel(
    q,
    k,
    v,
    g,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    dg_force,
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
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int64)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        nt = tl.cdiv(T, BT)
        i_tg = i_b * nt + i_t
        bos = i_b * T

    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V
    dh += (i_tg * H + i_h).to(tl.int64) * K * V
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K

    o_t = i_t * BT + tl.arange(0, BT)
    mask_t = o_t < T

    o_q_0 = i_t * BT + tl.arange(0, BT)
    o_q_1 = 0 + tl.arange(0, BK)
    p_q = q + o_q_0[:, None] * (H * K) + o_q_1[None, :] * (1)
    m_q = (o_q_0[:, None] < T) & (o_q_1[None, :] < K)
    o_k_0 = i_t * BT + tl.arange(0, BT)
    o_k_1 = 0 + tl.arange(0, BK)
    p_k = k + o_k_0[:, None] * (H * K) + o_k_1[None, :] * (1)
    m_k = (o_k_0[:, None] < T) & (o_k_1[None, :] < K)
    b_q = tl.load(p_q, mask=m_q, other=0.0)
    b_k = tl.load(p_k, mask=m_k, other=0.0)

    o_g_0 = i_t * BT + tl.arange(0, BT)
    p_g = g + bos * H + i_h + o_g_0 * (H)
    m_g = o_g_0 < T
    b_g = tl.load(p_g, mask=m_g, other=0.0)
    b_g_last = tl.load(g + (bos + min(i_t * BT + BT, T) - 1) * H + i_h)

    mask_fwd = (o_t[:, None] >= o_t[None, :]) & mask_t[:, None] & mask_t[None, :]
    mask_bwd = (o_t[:, None] <= o_t[None, :]) & mask_t[:, None] & mask_t[None, :]
    b_A_dv = tl.dot(b_k, tl.trans(b_q))
    b_A_dv = tl.where(mask_bwd, b_A_dv * exp2(b_g[None, :] - b_g[:, None]) * scale, 0.0)
    b_A_dv = b_A_dv.to(do.dtype.element_ty)
    b_dv_decay = tl.where(mask_t, exp2(-b_g + b_g_last), 0.0)

    b_dq = tl.zeros((BT, BK), dtype=tl.float32)
    b_dk = tl.zeros((BT, BK), dtype=tl.float32)
    b_ds = tl.zeros((BT, BT), dtype=tl.float32)
    b_dg_last = tl.zeros((1,), dtype=tl.float32)
    for i_v in range(0, tl.cdiv(V, BV)):
        o_v_0 = i_t * BT + tl.arange(0, BT)
        o_v_1 = i_v * BV + tl.arange(0, BV)
        p_v = v + o_v_0[:, None] * (H * V) + o_v_1[None, :] * (1)
        m_v = (o_v_0[:, None] < T) & (o_v_1[None, :] < V)
        o_do_0 = i_t * BT + tl.arange(0, BT)
        o_do_1 = i_v * BV + tl.arange(0, BV)
        p_do = do + o_do_0[:, None] * (H * V) + o_do_1[None, :] * (1)
        m_do = (o_do_0[:, None] < T) & (o_do_1[None, :] < V)
        o_dv_0 = i_t * BT + tl.arange(0, BT)
        o_dv_1 = i_v * BV + tl.arange(0, BV)
        p_dv = dv + o_dv_0[:, None] * (H * V) + o_dv_1[None, :] * (1)
        m_dv = (o_dv_0[:, None] < T) & (o_dv_1[None, :] < V)
        o_h_0 = i_v * BV + tl.arange(0, BV)
        o_h_1 = 0 + tl.arange(0, BK)
        p_h = h + o_h_0[:, None] * (1) + o_h_1[None, :] * (V)
        m_h = (o_h_0[:, None] < V) & (o_h_1[None, :] < K)
        o_dh_0 = i_v * BV + tl.arange(0, BV)
        o_dh_1 = 0 + tl.arange(0, BK)
        p_dh = dh + o_dh_0[:, None] * (1) + o_dh_1[None, :] * (V)
        m_dh = (o_dh_0[:, None] < V) & (o_dh_1[None, :] < K)
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_do = tl.load(p_do, mask=m_do, other=0.0)
        b_h = tl.load(p_h, mask=m_h, other=0.0)
        b_dh = tl.load(p_dh, mask=m_dh, other=0.0)
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))

        b_dv = tl.dot(b_k, tl.trans(b_dh).to(b_k.dtype))
        b_dv *= b_dv_decay[:, None]
        b_dv += tl.dot(b_A_dv, b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_dv)

    b_dg_last *= exp2(b_g_last)
    b_dq *= exp2(b_g)[:, None] * scale
    b_dk *= b_dv_decay[:, None]
    b_dg_last += tl.sum(b_dk * b_k)
    b_ds = tl.where(mask_fwd, b_ds * exp2(b_g[:, None] - b_g[None, :]), 0.0) * scale
    b_ds = b_ds.to(b_k.dtype)
    if q.dtype.element_ty == tl.float32:
        b_dq += tl.trans(tl.dot(tl.trans(b_k), tl.trans(b_ds)))
        b_dk += tl.trans(tl.dot(tl.trans(b_q), b_ds))
    else:
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)

    force = tl.sum(b_dq * b_q, axis=1) - tl.sum(b_dk * b_k, axis=1)
    last = min(i_t * BT + BT, T) - 1
    force += tl.where(o_t == last, b_dg_last, 0.0)
    tl.store(dg_force + (bos + o_t) * H + i_h, force, mask=mask_t)

    o_dq_0 = i_t * BT + tl.arange(0, BT)
    o_dq_1 = 0 + tl.arange(0, BK)
    p_dq = dq + o_dq_0[:, None] * (H * K) + o_dq_1[None, :] * (1)
    m_dq = (o_dq_0[:, None] < T) & (o_dq_1[None, :] < K)
    o_dk_0 = i_t * BT + tl.arange(0, BT)
    o_dk_1 = 0 + tl.arange(0, BK)
    p_dk = dk + o_dk_0[:, None] * (H * K) + o_dk_1[None, :] * (1)
    m_dk = (o_dk_0[:, None] < T) & (o_dk_1[None, :] < K)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_dq)
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_dk)


def _scalar_decay_bwd_dqk_dv_single_k(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    *,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    chunk_size: int,
    chunk_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seq_len, n_heads, d_k = q.shape
    d_v = v.shape[-1]
    if d_k > 128:
        raise ValueError("single-k fused dqk/dv path requires K <= 128.")
    nt = triton.cdiv(seq_len, chunk_size) if chunk_indices is None else len(chunk_indices)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(do)
    dg_force = torch.empty(bsz, seq_len, n_heads, device=q.device, dtype=torch.float32)
    block_k = max(triton.next_power_of_2(d_k), 16)
    is_hopper_fp32 = IS_NVIDIA_HOPPER and q.dtype == torch.float32
    # Keep the fused FP32 kernel below the SM90 per-block shared-memory limit.
    block_v = min(max(triton.next_power_of_2(d_v), 16), 32 if is_hopper_fp32 else 64)

    _chunk_cyfa_bwd_dqk_dv_kernel[(nt, bsz * n_heads)](
        q,
        k,
        v,
        g,
        h,
        do,
        dh,
        dq,
        dk,
        dv,
        dg_force,
        cu_seqlens,
        chunk_indices,
        scale,
        T=seq_len,
        H=n_heads,
        K=d_k,
        V=d_v,
        BT=chunk_size,
        BK=block_k,
        BV=block_v,
        num_warps=4,
        num_stages=1 if is_hopper_fp32 else 2,
    )
    return dq, dk, dv, dg_force


def _scalar_decay_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    chunk_size: int,
    chunk_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    h, final_state = chunk_fwd_h(
        k=k,
        v=v,
        g=g_cumsum,
        h0=None if initial_state is None else initial_state.transpose(-1, -2).contiguous(),
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        states_in_fp32=False,
        state_v_first=True,
    )
    out = chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=True,
    )
    if final_state is not None:
        final_state = final_state.transpose(-1, -2).contiguous()
    return out, final_state


def _scalar_decay_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    chunk_size: int,
    chunk_indices: torch.Tensor | None,
    fuse_dv_single_k: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    h, _ = chunk_fwd_h(
        k=k,
        v=v,
        g=g_cumsum,
        h0=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        states_in_fp32=False,
    )
    dh, dh0 = chunk_bwd_dh(
        q=q,
        k=k,
        v=v,
        g=g_cumsum,
        do=do,
        h0=initial_state,
        dht=dht,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        states_in_fp32=False,
    )
    if fuse_dv_single_k and q.shape[-1] <= 128:
        dq, dk, dv, dg_force = _scalar_decay_bwd_dqk_dv_single_k(
            q,
            k,
            v,
            g_cumsum,
            h,
            do,
            dh,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
    else:
        dq, dk, dg_force = _scalar_decay_bwd_dqk(
            q,
            k,
            v,
            g_cumsum,
            h,
            do,
            dh,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        dv = chunk_bwd_dv(
            q=q,
            k=k,
            g=g_cumsum,
            g_gamma=None,
            do=do,
            dh=dh,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
    return dq, dk, dv, dg_force, dh0


@triton.jit
def _cardinal_orbit_value(lam, o_m, M: tl.constexpr):
    is_cos = (o_m % 2) == 0
    phase = modal_phase(lam, o_m, M)
    scale = tl.sqrt(tl.where((o_m // 2) == 0, 1.0, 2.0) / (M - 1))
    return scale * tl.where(is_cos, cyfa_cos(phase), -cyfa_sin(phase))


@triton.jit
def _cardinal_orbit_derivative(lam, o_m, M: tl.constexpr):
    is_cos = (o_m % 2) == 0
    omega = modal_omega(o_m, M)
    phase = modal_phase(lam, o_m, M)
    scale = tl.sqrt(tl.where((o_m // 2) == 0, 1.0, 2.0) / (M - 1))
    return -scale * omega * tl.where(is_cos, cyfa_sin(phase), cyfa_cos(phase))


@triton.autotune(
    configs=_POINT_AUTOTUNE_CONFIGS,
    key=['M'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def _chunk_cyfa_write_fwd_kernel(
    lambdas,
    beta,
    write,
    T,
    H: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    o_m = tl.arange(0, BM)
    m_t = o_t < T
    m_m = o_m < M

    b_lambda = tl.load(lambdas + (i_b * T + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
    b_beta = tl.load(beta + (i_b * T + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
    write_value = b_beta[:, None] * _cardinal_orbit_value(
        b_lambda[:, None],
        o_m[None, :],
        M,
    )
    mask = m_t[:, None] & m_m[None, :]
    p = ((i_b * T + o_t[:, None]) * H + i_h) * M + o_m[None, :]
    tl.store(write + p, write_value, mask=mask)


def _point_write(
    lambdas: torch.Tensor,
    beta: torch.Tensor,
    num_slots: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    lambdas = lambdas.contiguous()
    beta = beta.contiguous()
    bsz, seq_len, n_heads = lambdas.shape
    write = torch.empty(bsz, seq_len, n_heads, num_slots, device=lambdas.device, dtype=dtype)
    _chunk_cyfa_write_fwd_kernel[lambda meta: (triton.cdiv(seq_len, meta['BT']), bsz * n_heads)](
        lambdas,
        beta,
        write,
        T=seq_len,
        H=n_heads,
        M=num_slots,
        BM=triton.next_power_of_2(num_slots),
    )
    return write


@triton.autotune(
    configs=_POINT_AUTOTUNE_CONFIGS,
    key=['M'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def _chunk_cyfa_write_bwd_kernel(
    dwrite_a,
    dwrite_b,
    lambdas,
    beta,
    dlambda,
    dbeta,
    T,
    H: tl.constexpr,
    M: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    o_m = tl.arange(0, BM)
    m_t = o_t < T
    mask = m_t[:, None] & (o_m[None, :] < M)
    p = ((i_b * T + o_t[:, None]) * H + i_h) * M + o_m[None, :]
    dw = tl.load(dwrite_a + p, mask=mask, other=0.0).to(tl.float32)
    dw += tl.load(dwrite_b + p, mask=mask, other=0.0).to(tl.float32)
    b_lambda = tl.load(lambdas + (i_b * T + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
    b_beta = tl.load(beta + (i_b * T + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
    endpoint = _cardinal_orbit_value(b_lambda[:, None], o_m[None, :], M)
    tangent = _cardinal_orbit_derivative(b_lambda[:, None], o_m[None, :], M)
    d_phase = tl.sum(tl.where(mask, dw * b_beta[:, None] * tangent, 0.0), axis=1)
    d_write = tl.sum(tl.where(mask, dw * endpoint, 0.0), axis=1)
    offset = (i_b * T + o_t) * H + i_h
    tl.store(dlambda + offset, d_phase, mask=m_t)
    tl.store(dbeta + offset, d_write, mask=m_t)


def _point_write_backward(
    dwrite_a: torch.Tensor,
    dwrite_b: torch.Tensor,
    lambdas: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, n_heads, num_slots = dwrite_a.shape
    dlambda = torch.empty(bsz, seq_len, n_heads, device=dwrite_a.device, dtype=torch.float32)
    dbeta = torch.empty_like(dlambda)
    _chunk_cyfa_write_bwd_kernel[lambda meta: (triton.cdiv(seq_len, meta['BT']), bsz * n_heads)](
        dwrite_a.contiguous(),
        dwrite_b.contiguous(),
        lambdas.contiguous(),
        beta.contiguous(),
        dlambda,
        dbeta,
        T=seq_len,
        H=n_heads,
        M=num_slots,
        BM=triton.next_power_of_2(num_slots),
    )
    return dlambda, dbeta


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=_FUSED_READOUT_FWD_AUTOTUNE_CONFIGS,
    key=['H', 'K', 'M', 'C', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def _chunk_cyfa_fused_readout_fwd_kernel(
    q,
    q_weight,
    q_rstd,
    k,
    write,
    h,
    g,
    lambdas,
    table,
    weights,
    raw_output,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BR: tl.constexpr,
    STORE_RAW: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_r = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int64)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        n_chunks = tl.cdiv(T, BT)
        i_tg = i_b * n_chunks + i_t
        bos = i_b * T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    write += (bos * H + i_h) * M
    h += (i_tg * H + i_h).to(tl.int64) * M * K

    # the row offset is bounded by BT; global time offsets remain int64
    o_t = i_t * BT + (i_r * BR).to(tl.int32) + tl.arange(0, BR)
    o_s = i_t * BT + tl.arange(0, BT)
    o_m = tl.arange(0, M)
    m_t = o_t < T
    m_s = o_s < T
    b_x = tl.zeros((BR, M), dtype=tl.float32)

    if USE_G:
        p_g_t = g + (bos + o_t) * H + i_h
        p_g_s = g + (bos + o_s) * H + i_h
        b_g_t = tl.load(p_g_t, mask=m_t, other=0.0)
        b_g_s = tl.load(p_g_s, mask=m_s, other=0.0)
    m_A = (o_t[:, None] >= o_s[None, :]) & m_t[:, None] & m_s[None, :]
    p_write = write + o_s[:, None] * (H * M) + o_m[None, :]
    b_write = tl.load(p_write, mask=m_s[:, None], other=0.0)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
        p_k = k + o_k[:, None] + o_s[None, :] * (H * K)
        p_h = h + o_m[:, None] * K + o_k[None, :]
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_q_scale = tl.load(q_rstd + (bos + o_t) * H + i_h, mask=m_t, other=0.0)
        b_q_weight = tl.load(q_weight + o_k, mask=m_k, other=0.0).to(tl.float32)
        b_q = (b_q * b_q_scale[:, None] * b_q_weight[None, :]).to(q.dtype.element_ty)
        b_k = tl.load(p_k, mask=m_k[:, None] & m_s[None, :], other=0.0)
        b_h = tl.load(p_h, mask=m_k[None, :], other=0.0)
        b_x_partial = tl.dot(b_q, tl.trans(b_h))
        b_A_partial = tl.dot(b_q, b_k)
        if USE_G:
            b_x_partial *= exp2(b_g_t)[:, None]
            b_A_partial *= exp2(b_g_t[:, None] - b_g_s[None, :])
        b_A_partial = tl.where(m_A, b_A_partial, 0.0)
        b_x_partial = (b_x_partial + tl.dot(b_A_partial.to(b_write.dtype), b_write)) * scale
        b_x += b_x_partial.to(q.dtype.element_ty).to(tl.float32)

    p_x = ((bos + o_t[:, None]) * H + i_h) * M + o_m[None, :]
    if STORE_RAW:
        tl.store(raw_output + p_x, b_x, mask=m_t[:, None])

    o_even = 2 * tl.arange(0, M // 2)
    x_even, x_odd = tl.split(tl.reshape(b_x, (BR, M // 2, 2)))
    b_lambda = tl.load(lambdas + (bos + o_t) * H + i_h, mask=m_t, other=0.0).to(tl.float32)
    phase = modal_phase(b_lambda[:, None], o_even[None, :], M)
    c = cyfa_cos(phase)
    s = cyfa_sin(phase)
    z = tl.reshape(tl.join(c * x_even - s * x_odd, s * x_even + c * x_odd), (BR, M))
    z = tl.where(m_t[:, None], z, 0.0)

    o_r = tl.arange(0, M)
    m_r = o_r < C
    w = tl.load(
        table + i_h * C * M + o_r[:, None] * M + o_m[None, :],
        mask=m_r[:, None],
        other=0.0,
    ).to(tl.float32)
    logits = tl.dot(z.to(q.dtype.element_ty), tl.trans(w.to(q.dtype.element_ty)))
    logits = tl.where(m_r[None, :], logits, -float("inf"))
    max_logits = tl.max(logits, axis=1)
    probs = tl.where(m_r[None, :], tl.exp(logits - max_logits[:, None]), 0.0)
    denom = tl.sum(probs, axis=1)
    probs *= tl.where(denom > 0.0, 1.0 / denom, 0.0)[:, None]

    u = tl.dot(probs.to(weights.dtype.element_ty), w.to(weights.dtype.element_ty))
    u_even, u_odd = tl.split(tl.reshape(u, (BR, M // 2, 2)))
    y = tl.reshape(tl.join(c * u_even + s * u_odd, -s * u_even + c * u_odd), (BR, M))
    tl.store(weights + p_x, y, mask=m_t[:, None])


def _scalar_decay_fused_readout_fwd(
    q: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_rstd: torch.Tensor,
    k: torch.Tensor,
    write: torch.Tensor,
    g_cumsum: torch.Tensor,
    lambdas: torch.Tensor,
    readout_table: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    chunk_size: int,
    chunk_indices: torch.Tensor | None,
    *,
    save_o_prime: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    bsz, seq_len, n_heads, d_k = q.shape
    num_slots = write.shape[-1]
    readout_slots = readout_table.shape[-2]
    n_chunks = triton.cdiv(seq_len, chunk_size) if chunk_indices is None else len(chunk_indices)
    h, final_state = chunk_fwd_h(
        k=k,
        v=write,
        g=g_cumsum,
        h0=None if initial_state is None else initial_state.transpose(-1, -2).contiguous(),
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        states_in_fp32=False,
        state_v_first=True,
    )
    weights = torch.empty_like(write)
    raw_output = torch.empty_like(weights) if save_o_prime else None

    def grid(meta):
        return (triton.cdiv(chunk_size, meta['BR']), n_chunks, bsz * n_heads)

    _chunk_cyfa_fused_readout_fwd_kernel[grid](
        q=q,
        q_weight=q_norm_weight,
        q_rstd=q_rstd,
        k=k,
        write=write,
        h=h,
        g=g_cumsum,
        lambdas=lambdas.contiguous(),
        table=readout_table.contiguous(),
        weights=weights,
        raw_output=raw_output,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=seq_len,
        H=n_heads,
        K=d_k,
        M=num_slots,
        C=readout_slots,
        BT=chunk_size,
        STORE_RAW=save_o_prime,
    )
    if final_state is not None:
        final_state = final_state.transpose(-1, -2).contiguous()
    return weights, raw_output, final_state


@fla_cache_autotune(
    configs=_READOUT_BWD_AUTOTUNE_CONFIGS,
    key=['M', 'C'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['N'])
def _chunk_cyfa_fused_readout_bwd_kernel(
    lambdas,
    table,
    raw_logits,
    dweights,
    probs,
    dlogits,
    draw_logits,
    dlambda,
    N,
    H: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
):
    i_blk = tl.program_id(0)
    i_h = tl.program_id(1)
    o_n_local = tl.arange(0, BT)
    o_n = i_blk * BT + o_n_local
    o_j = tl.arange(0, BM)
    o_r = tl.arange(0, BM)
    mask_n = o_n < N
    mask_j = o_j < M
    mask_r = o_r < C
    mask_p = 2 * tl.arange(0, BM // 2) + 1 < M

    i_blk_global = i_blk.to(tl.int64)
    i_h_global = i_h.to(tl.int64)
    token_head_base = i_blk_global * BT * H + i_h_global
    p_lambda = token_head_base + o_n_local * H
    p_x = token_head_base * M + o_n_local[:, None] * (H * M) + o_j[None, :]
    b_lambda = tl.load(lambdas + p_lambda, mask=mask_n, other=0.0).to(tl.float32)
    raw = tl.load(raw_logits + p_x, mask=mask_n[:, None] & mask_j[None, :], other=0.0).to(tl.float32)
    dw = tl.load(dweights + p_x, mask=mask_n[:, None] & mask_j[None, :], other=0.0).to(tl.float32)
    raw_even, raw_odd = tl.split(tl.reshape(raw, (BT, BM // 2, 2)))
    dw_even, dw_odd = tl.split(tl.reshape(dw, (BT, BM // 2, 2)))
    o_even = 2 * tl.arange(0, BM // 2)
    omega = modal_omega(o_even, M)
    phase = modal_phase(b_lambda[:, None], o_even[None, :], M)
    c = cyfa_cos(phase)
    s = cyfa_sin(phase)
    raw_z = tl.reshape(tl.join(c * raw_even - s * raw_odd, s * raw_even + c * raw_odd), (BT, BM))
    dw_z = tl.reshape(tl.join(c * dw_even - s * dw_odd, s * dw_even + c * dw_odd), (BT, BM))
    w = tl.load(
        table + i_h_global * C * M + o_r[:, None] * M + o_j[None, :],
        mask=mask_r[:, None] & mask_j[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = tl.dot(raw_z.to(raw_logits.dtype.element_ty), tl.trans(w.to(raw_logits.dtype.element_ty)))
    scores = tl.where(mask_r[None, :], scores, -float("inf"))
    dp = tl.dot(dw_z.to(dweights.dtype.element_ty), tl.trans(w.to(dweights.dtype.element_ty)))
    max_scores = tl.max(scores, axis=1)
    p = tl.where(mask_r[None, :], tl.exp(scores - max_scores[:, None]), 0.0)
    denom = tl.sum(p, axis=1)
    p *= tl.where(denom > 0.0, 1.0 / denom, 0.0)[:, None]
    correction = tl.sum(p * dp, axis=1)
    dl = p * (dp - correction[:, None])
    p_probs = token_head_base * C + o_n_local[:, None] * (H * C) + o_r[None, :]
    p_store = p.to(probs.dtype.element_ty)
    dl_store = dl.to(dlogits.dtype.element_ty)
    tl.store(probs + p_probs, p_store, mask=mask_n[:, None] & mask_r[None, :])
    tl.store(dlogits + p_probs, dl_store, mask=mask_n[:, None] & mask_r[None, :])

    w_low = w.to(probs.dtype.element_ty)
    coeff = tl.dot(p_store, w_low)
    grad = tl.dot(dl_store, w_low)
    coeff_even, coeff_odd = tl.split(tl.reshape(coeff, (BT, BM // 2, 2)))
    grad_even, grad_odd = tl.split(tl.reshape(grad, (BT, BM // 2, 2)))
    dx_even = c * grad_even + s * grad_odd
    dx_odd = -s * grad_even + c * grad_odd
    dx = tl.reshape(tl.join(dx_even, dx_odd), (BT, BM))
    tl.store(draw_logits + p_x, dx, mask=mask_n[:, None] & mask_j[None, :])

    force_v = coeff_even * (-s * dw_even - c * dw_odd)
    force_v += coeff_odd * (c * dw_even - s * dw_odd)
    force_k = grad_even * (-s * raw_even - c * raw_odd)
    force_k += grad_odd * (c * raw_even - s * raw_odd)
    force = tl.sum(
        tl.where(mask_n[:, None] & mask_p[None, :], omega[None, :] * (force_v + force_k), 0.0),
        axis=1,
    )
    tl.store(dlambda + p_lambda, force, mask=mask_n)


def _readout_backward(
    lambdas: torch.Tensor,
    dweights: torch.Tensor,
    raw_logits: torch.Tensor,
    readout_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    draw_logits = torch.empty_like(raw_logits)
    dlambda = torch.empty_like(lambdas, dtype=torch.float32)
    n_heads = raw_logits.shape[-2]
    num_slots = raw_logits.shape[-1]
    readout_slots = readout_table.shape[-2]
    probs = torch.empty(*raw_logits.shape[:-1], readout_slots, device=raw_logits.device, dtype=raw_logits.dtype)
    dlogits = torch.empty_like(probs)
    n_tokens = raw_logits.numel() // (n_heads * num_slots)

    def grid(meta):
        return (triton.cdiv(n_tokens, meta['BT']), n_heads)

    _chunk_cyfa_fused_readout_bwd_kernel[grid](
        lambdas.contiguous(),
        readout_table.contiguous(),
        raw_logits.contiguous(),
        dweights.contiguous(),
        probs,
        dlogits,
        draw_logits,
        dlambda,
        N=n_tokens,
        H=n_heads,
        M=num_slots,
        C=readout_slots,
        BM=triton.next_power_of_2(num_slots),
    )
    return probs, dlogits, draw_logits, dlambda


def _prune_table_grad_configs(configs, nargs, **kwargs):
    args = {**(nargs or {}), **kwargs}
    right_a = args.get('right_a')
    if not IS_NVIDIA_HOPPER or right_a is None or right_a.dtype != torch.float32:
        return configs
    # Triton 3.4 rejects this four-warp FP32 dot layout on SM90 during compilation.
    return [
        config for config in configs
        if not (
            config.kwargs['BR'] == 64
            and config.kwargs['BC'] == 64
            and config.num_warps == 4
        )
    ]


@fla_cache_autotune(
    configs=_TABLE_GRAD_AUTOTUNE_CONFIGS,
    key=['M', 'C'],
    prune_configs_by={'early_config_prune': _prune_table_grad_configs},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def _chunk_cyfa_readout_table_grad_kernel(
    lambdas,
    left_a,
    right_a,
    left_b,
    right_b,
    dtable,
    T,
    H: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    i_r = tl.program_id(0)
    i_c = tl.program_id(1)
    i_hn = tl.program_id(2)
    i_h = i_hn % H
    i_n = (i_hn // H).to(tl.int64)
    o_r = i_r * BR + tl.arange(0, BR)
    o_c = i_c * BC + tl.arange(0, BC)
    m_r = o_r < C
    m_c = o_c < M
    acc = tl.zeros((BR, BC), tl.float32)
    n_splits = tl.num_programs(2) // H
    i_h_global = i_h.to(tl.int64)
    for start in range(i_n * BN, T, n_splits * BN):
        o_n = start + tl.arange(0, BN)
        m_n = o_n < T
        b_lambda = tl.load(lambdas + o_n * H + i_h_global, mask=m_n, other=0.0).to(tl.float32)
        p_l = (o_n[:, None] * H + i_h_global) * C + o_r[None, :]
        p_x = (o_n[:, None] * H + i_h_global) * M + o_c[None, :]
        l_a = tl.load(left_a + p_l, mask=m_n[:, None] & m_r[None, :], other=0.0)
        l_b = tl.load(left_b + p_l, mask=m_n[:, None] & m_r[None, :], other=0.0)
        x_a = tl.load(right_a + p_x, mask=m_n[:, None] & m_c[None, :], other=0.0).to(tl.float32)
        x_b = tl.load(right_b + p_x, mask=m_n[:, None] & m_c[None, :], other=0.0).to(tl.float32)
        a_even, a_odd = tl.split(tl.reshape(x_a, (BN, BC // 2, 2)))
        b_even, b_odd = tl.split(tl.reshape(x_b, (BN, BC // 2, 2)))
        o_even = i_c * BC + 2 * tl.arange(0, BC // 2)
        phase = modal_phase(b_lambda[:, None], o_even[None, :], M)
        co = cyfa_cos(phase)
        si = cyfa_sin(phase)
        z_a = tl.reshape(tl.join(co * a_even - si * a_odd, si * a_even + co * a_odd), (BN, BC))
        z_b = tl.reshape(tl.join(co * b_even - si * b_odd, si * b_even + co * b_odd), (BN, BC))
        acc += tl.dot(tl.trans(l_a.to(right_a.dtype.element_ty)), z_a.to(right_a.dtype.element_ty))
        acc += tl.dot(tl.trans(l_b.to(right_b.dtype.element_ty)), z_b.to(right_b.dtype.element_ty))
    tl.store(
        dtable + (i_n * H + i_h_global) * C * M + o_r[:, None] * M + o_c[None, :],
        acc,
        mask=m_r[:, None] & m_c[None, :],
    )


@triton.jit(do_not_specialize=['S'])
def _chunk_cyfa_readout_table_grad_reduce_kernel(partials, output, S, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    o_x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    o_s = tl.arange(0, 64)
    x = tl.load(partials + o_s[:, None] * SIZE + o_x[None, :],
                mask=(o_s[:, None] < S) & (o_x[None, :] < SIZE), other=0.0)
    tl.store(output + o_x, tl.sum(x, axis=0), mask=o_x < SIZE)


def _readout_table_grad_dual(
    lambdas: torch.Tensor,
    left_a: torch.Tensor,
    right_a: torch.Tensor,
    left_b: torch.Tensor,
    right_b: torch.Tensor,
    readout_table: torch.Tensor,
) -> torch.Tensor:
    n_heads = left_a.shape[-2]
    readout_slots = left_a.shape[-1]
    num_slots = right_a.shape[-1]
    n_tokens = left_a.numel() // (n_heads * readout_slots)
    n_splits = min(32, triton.cdiv(n_tokens, 128))
    partials = torch.empty((n_splits, n_heads, readout_slots, num_slots),
                           device=readout_table.device, dtype=torch.float32)
    dtable = torch.empty_like(readout_table, dtype=torch.float32)

    def grid(meta):
        return (
            triton.cdiv(readout_slots, meta['BR']),
            triton.cdiv(num_slots, meta['BC']),
            n_heads * n_splits,
        )

    _chunk_cyfa_readout_table_grad_kernel[grid](
        lambdas.contiguous(),
        left_a.contiguous(),
        right_a.contiguous(),
        left_b.contiguous(),
        right_b.contiguous(),
        partials,
        T=n_tokens,
        H=n_heads,
        M=num_slots,
        C=readout_slots,
    )
    _chunk_cyfa_readout_table_grad_reduce_kernel[(triton.cdiv(dtable.numel(), 128),)](
        partials, dtable, S=n_splits, SIZE=dtable.numel(), BLOCK=128,
    )
    return dtable


class ChunkCyFAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        delta: torch.Tensor,
        beta: torch.Tensor,
        readout_table: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        q_norm_eps: float,
        k_norm_eps: float,
        scale: float,
        initial_k: torch.Tensor | None,
        initial_v: torch.Tensor | None,
        initial_lambda: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None,
        cu_seqlens_cpu: torch.Tensor | None,
        checkpoint_level: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        chunk_size = _CHUNK_SIZE
        q_input = q
        k_input = k
        q, k, q_rstd, k_rstd = qk_rmsnorm_fwd(
            q_input,
            k_input,
            q_norm_weight,
            k_norm_weight,
            q_norm_eps,
            k_norm_eps,
            store_q=False,
            store_rstd=True,
        )

        if cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu)
        else:
            chunk_indices = None

        ctx.g_dtype = g.dtype
        g = chunk_local_cumsum(
            g.float(),
            chunk_size=chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            output_dtype=torch.float32,
        )
        lambdas = cyfa_clock_cumsum(delta, cu_seqlens=cu_seqlens)
        if initial_lambda is not None:
            lambdas = add_segment_initial(lambdas, initial_lambda, cu_seqlens=cu_seqlens)
        write = _point_write(
            lambdas,
            beta.float(),
            readout_table.shape[-1],
            dtype=q.dtype,
        )
        weights, raw_logits, hkt = _scalar_decay_fused_readout_fwd(
            q=q,
            q_norm_weight=q_norm_weight,
            q_rstd=q_rstd,
            k=k,
            write=write,
            g_cumsum=g,
            lambdas=lambdas,
            readout_table=readout_table,
            initial_state=initial_k,
            output_final_state=output_final_state,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            save_o_prime=checkpoint_level == 0,
        )
        out, hvt = _scalar_decay_fwd(
            q=weights.contiguous(),
            k=write,
            v=v,
            g_cumsum=g,
            initial_state=initial_v,
            output_final_state=output_final_state,
            scale=1.0,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        if output_final_state:
            flt = segment_sum(delta, cu_seqlens=cu_seqlens)
            if initial_lambda is not None:
                flt = flt + initial_lambda
        else:
            flt = None

        ctx.save_for_backward(
            q_input,
            k_input,
            v,
            q_norm_weight,
            k_norm_weight,
            delta,
            beta,
            g,
            lambdas,
            readout_table,
            raw_logits,
            weights,
            initial_k,
            initial_v,
            initial_lambda,
            q_rstd,
            k_rstd,
            chunk_indices,
        )
        ctx.q_norm_eps = float(q_norm_eps)
        ctx.k_norm_eps = float(k_norm_eps)
        ctx.scale = float(scale)
        ctx.cu_seqlens = cu_seqlens
        return out.to(dtype=q.dtype), hkt, hvt, flt

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dhkt=None, dhvt=None, dlt=None):
        (
            q_saved,
            k_saved,
            v,
            q_norm_weight,
            k_norm_weight,
            delta,
            beta,
            g,
            lambdas,
            readout_table,
            raw_logits,
            weights,
            initial_k,
            initial_v,
            initial_lambda,
            q_rstd,
            k_rstd,
            chunk_indices,
        ) = ctx.saved_tensors
        scale = ctx.scale
        cu_seqlens = ctx.cu_seqlens
        chunk_size = _CHUNK_SIZE
        q, k, _, _ = qk_rmsnorm_fwd(
            q_saved,
            k_saved,
            q_norm_weight,
            k_norm_weight,
            ctx.q_norm_eps,
            ctx.k_norm_eps,
            store_rstd=False,
        )

        write = _point_write(
            lambdas,
            beta.float(),
            readout_table.shape[-1],
            dtype=q.dtype,
        )
        weights_q = weights.contiguous()
        dweights, dwrite_v, dv, dg_force_v, dhv0 = _scalar_decay_bwd(
            q=weights_q,
            k=write,
            v=v,
            g_cumsum=g,
            initial_state=initial_v,
            do=do,
            dht=dhvt,
            scale=1.0,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            fuse_dv_single_k=True,
        )

        if raw_logits is None:
            _, raw_logits, _ = _scalar_decay_fused_readout_fwd(
                q=q_saved,
                q_norm_weight=q_norm_weight,
                q_rstd=q_rstd,
                k=k,
                write=write,
                g_cumsum=g,
                lambdas=lambdas,
                readout_table=readout_table,
                initial_state=initial_k,
                output_final_state=False,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_size=chunk_size,
                chunk_indices=chunk_indices,
                save_o_prime=True,
            )

        probs, dlogits, draw_logits, dlambda_readout = _readout_backward(
            lambdas,
            dweights,
            raw_logits,
            readout_table,
        )
        if ctx.needs_input_grad[6]:
            dreadout_table = _readout_table_grad_dual(
                lambdas,
                probs,
                dweights,
                dlogits,
                raw_logits,
                readout_table,
            )
        else:
            dreadout_table = None

        draw_q = draw_logits.contiguous()
        dq, dk, dwrite_k, dg_force_k, dhk0 = _scalar_decay_bwd(
            q=q,
            k=k,
            v=write,
            g_cumsum=g,
            initial_state=initial_k,
            do=draw_q,
            dht=dhkt,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            fuse_dv_single_k=False,
        )

        dlambda_write, dbeta = _point_write_backward(
            dwrite_k,
            dwrite_v,
            lambdas,
            beta.float(),
        )
        phase_force = dlambda_write.add_(dlambda_readout)
        if dlt is not None:
            dlt = dlt.float()
            if cu_seqlens is None:
                phase_force[:, -1, :].add_(dlt)
            else:
                end_indices = cu_seqlens[1:].to(device=phase_force.device, dtype=torch.long) - 1
                phase_force[0].index_add_(0, end_indices, dlt)
        ddelta = chunk_global_cumsum(
            phase_force,
            reverse=True,
            cu_seqlens=cu_seqlens,
            output_dtype=torch.float32,
        )
        dlambda0 = (
            None
            if initial_lambda is None
            else segment_sum(phase_force, cu_seqlens=cu_seqlens)
        )
        dg_force = dg_force_v.add_(dg_force_k)
        dg = chunk_local_cumsum(
            dg_force,
            chunk_size=chunk_size,
            reverse=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            output_dtype=torch.float32,
        )
        dq, dk, dq_norm_weight, dk_norm_weight = _qk_rmsnorm_bwd(
            dq,
            dk,
            q_saved,
            k_saved,
            q_norm_weight,
            k_norm_weight,
            q_rstd,
            k_rstd,
        )
        return (
            dq.to(q_saved),
            dk.to(k_saved),
            dv.to(v),
            dg.to(dtype=ctx.g_dtype),
            ddelta.to(delta),
            dbeta.to(beta),
            None if dreadout_table is None else dreadout_table.to(readout_table),
            dq_norm_weight,
            dk_norm_weight,
            None,
            None,
            None,
            dhk0,
            dhv0,
            dlambda0,
            None,
            None,
            None,
            None,
        )


@torch.compiler.disable
def chunk_cyfa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    delta: torch.Tensor,
    beta: torch.Tensor,
    readout: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    q_norm_eps: float,
    k_norm_eps: float,
    scale: float | None = None,
    initial_state: CyFAState | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    checkpoint_level: int = 0,
) -> tuple[torch.Tensor, CyFAState | None]:
    """Run the chunkwise CyclicFlowAttention recurrence with log-space decay ``g``."""
    num_slots = validate_cyfa_inputs(
        q,
        k,
        v,
        g,
        delta,
        beta,
        readout,
        require_cuda=True,
    )
    cu_seqlens = prepare_cu_seqlens(
        cu_seqlens,
        batch_size=q.shape[0],
        seq_len=q.shape[1],
        device=q.device,
    )
    _, _, n_heads, d_k = q.shape
    d_v = v.shape[-1]
    if scale is None:
        scale = d_k ** -0.5
    logical_bsz = q.shape[0] if cu_seqlens is None else int(cu_seqlens.numel() - 1)
    if initial_state is None:
        initial_k = None
        initial_v = None
        initial_lambda = None
    else:
        initial_k, initial_v, initial_lambda = validate_initial_state(
            initial_state,
            batch=logical_bsz,
            n_heads=n_heads,
            num_slots=num_slots,
            d_k=d_k,
            d_v=d_v,
            device=q.device,
        )
    readout_table = build_readout_table(readout)
    if checkpoint_level not in (0, 1):
        raise ValueError("`checkpoint_level` must be either 0 or 1.")
    effective_checkpoint_level = 0 if (
        checkpoint_level == 0
        and torch.is_grad_enabled()
        and any(tensor.requires_grad for tensor in (q, k, v, g, delta, beta, readout_table))
    ) else 1
    out, fk, fv, fl = ChunkCyFAFunction.apply(
        q,
        k,
        v,
        g,
        delta,
        beta,
        readout_table,
        q_norm_weight,
        k_norm_weight,
        float(q_norm_eps),
        float(k_norm_eps),
        float(scale),
        initial_k,
        initial_v,
        initial_lambda,
        bool(output_final_state),
        cu_seqlens,
        cu_seqlens_cpu,
        effective_checkpoint_level,
    )
    final_state = (fk, fv, fl) if output_final_state else None
    return out, final_state
