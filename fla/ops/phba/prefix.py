# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Chunkwise gated prefix states with deterministic segmented backward."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from fla.ops.phba.plans import (
    CHUNKWISE_TRAIN_MAX_DH,
    CHUNKWISE_TRAIN_MAX_M,
    CHUNKWISE_VALIDATED_C,
    BlockLayout,
    PrefixBackwardPlan,
    prefix_backward_plan,
)
from fla.ops.phba.utils import fail
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

# fixed launch geometry preserves the validated accumulation order


@dataclass(frozen=True)
class LaunchConfig:
    num_warps: int
    num_stages: int


CUMSUM_BS = 32
FWD_BK = 64
FWD_BV = 64
BWD_BLOCK_T = 32
REVERSE_BLOCK_M = 64
REVERSE_BLOCK_D = 64
SEGMENT_CHUNKS = 2

LAUNCH = {
    "chunk_local_cumsum": LaunchConfig(num_warps=4, num_stages=1),
    "chunk_local_cumsum_reverse": LaunchConfig(num_warps=4, num_stages=1),
    "chunk_fwd_h": LaunchConfig(num_warps=4, num_stages=1),
    "chunk_state_bwd": LaunchConfig(num_warps=8, num_stages=2),
    "prefix_reverse_state": LaunchConfig(num_warps=4, num_stages=1),
    "prefix_bwd_local": LaunchConfig(num_warps=8, num_stages=2),
}

STORAGE_DTYPES = (torch.float16, torch.bfloat16)


@triton.jit
def _local_cumsum_kernel(
    x_ptr,
    y_ptr,
    block_ptr,
    H: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
):
    """Compute inclusive block-local sums over tokens, or suffix sums in reverse mode, with FP32 output."""
    i_b = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    i_s = tl.program_id(2).to(tl.int64)
    bos = (tl.load(block_ptr + (i_b))).to(tl.int64)
    eos = (tl.load(block_ptr + (i_b + 1))).to(tl.int64)
    o_t = tl.arange(0, C)
    o_s = i_s * BS + tl.arange(0, BS)
    mask = (o_t[:, None] < eos - bos) & (o_s[None, :] < M)
    off = (bos + o_t)[:, None] * (H * M) + i_h * M + o_s[None, :]
    x = tl.load(x_ptr + (off), mask=mask, other=0.0).to(tl.float32)
    y = tl.cumsum(x, axis=0, reverse=REVERSE)
    tl.store(y_ptr + (off), y, mask=mask)


@triton.jit
def _prefix_fwd_kernel(
    s_ptr,
    z_ptr,
    gk_ptr,
    out_ptr,
    ht_ptr,
    seq_block_ptr,
    block_ptr,
    H: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_FINAL: tl.constexpr,
):
    """Traverse each sequence in block order with FP32 recurrent state.

    Insertion operands are rounded to storage dtype before each FP32 state update.
    Prefixes use [Nc, M, H, D] storage; optional final states use [Nseq, H, M, D] FP32.
    """
    i_nh = tl.program_id(0)
    i_k = tl.program_id(1)
    i_v = tl.program_id(2)
    i_n = i_nh // H
    i_h = i_nh % H
    b0 = tl.load(seq_block_ptr + i_n.to(tl.int64))
    b1 = tl.load(seq_block_ptr + i_n.to(tl.int64) + 1)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_t = tl.arange(0, C)
    m_k = o_k < M
    m_v = o_v < D
    m_kv = m_k[:, None] & m_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    for b in range(b0, b1):
        bos = tl.load(block_ptr + b.to(tl.int64))
        eos = tl.load(block_ptr + b.to(tl.int64) + 1)
        p_out = (
            out_ptr
            + tl.cast(b, tl.int64) * (M * H * D)
            + o_k.to(tl.int64)[:, None] * (H * D)
            + i_h.to(tl.int64) * D
            + o_v[None, :]
        )
        tl.store(p_out, b_h.to(out_ptr.dtype.element_ty), mask=m_kv)

        m_t = o_t < eos - bos
        rows = bos.to(tl.int64) + o_t
        m_kt = m_k[:, None] & m_t[None, :]
        b_s = tl.load(s_ptr + rows[None, :] * (H * M) + i_h.to(tl.int64) * M + o_k[:, None], mask=m_kt, other=0.0)
        b_g = tl.load(gk_ptr + rows[None, :] * (H * M) + i_h.to(tl.int64) * M + o_k[:, None], mask=m_kt, other=0.0)
        b_z = tl.load(
            z_ptr + rows[:, None] * (H * D) + i_h.to(tl.int64) * D + o_v[None, :],
            mask=m_t[:, None] & m_v[None, :],
            other=0.0,
        )
        last = eos.to(tl.int64) - 1
        b_glast = tl.load(gk_ptr + last * (H * M) + i_h.to(tl.int64) * M + o_k, mask=m_k, other=0.0)

        b_h *= tl.exp(b_glast)[:, None]
        b_w = (b_s * tl.exp(b_glast[:, None] - b_g)).to(b_s.dtype)
        b_h += tl.dot(b_w, b_z)

    if STORE_FINAL:
        p_ht = ht_ptr + i_nh.to(tl.int64) * (M * D) + o_k.to(tl.int64)[:, None] * D + o_v[None, :]
        tl.store(p_ht, b_h, mask=m_kv)


@triton.jit
def _prefix_bwd_kernel(
    s_ptr,
    z_ptr,
    gk_ptr,
    prefix_ptr,
    dprefix_ptr,
    ds_ptr,
    dz_ptr,
    dg_ptr,
    seq_block_ptr,
    block_ptr,
    H: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    """Traverse each sequence in fixed reverse block order.

    The value branch accumulates ds/dg after the key branch on the same stream.
    Apply the final-row gate correction before the single store of that row.
    """
    i_n = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    b0 = (tl.load(seq_block_ptr + (i_n))).to(tl.int64)
    b1 = (tl.load(seq_block_ptr + (i_n + 1))).to(tl.int64)

    o_k = (tl.arange(0, BLOCK_K)).to(tl.int64)
    o_v = (tl.arange(0, BLOCK_V)).to(tl.int64)
    m_k = o_k < M
    m_v = o_v < D
    m_kv = m_k[:, None] & m_v[None, :]

    grad_next = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
    for c in range(b1 - b0 - 1, -1, -1):
        b = b0 + c
        bos = (tl.load(block_ptr + (b))).to(tl.int64)
        eos = (tl.load(block_ptr + (b + 1))).to(tl.int64)
        last = eos - 1
        g_last = tl.load(gk_ptr + (last.to(tl.int64) * (H * M) + i_h * M + o_k), mask=m_k, other=0.0).to(tl.float32)
        exp_g_last = tl.exp(g_last)

        sum_k_gradk = tl.zeros([BLOCK_K], dtype=tl.float32)
        last_dg = tl.zeros([BLOCK_K], dtype=tl.float32)
        grad_next_st = grad_next.to(z_ptr.dtype.element_ty)
        for t0 in range(0, C, BLOCK_T):
            tok = bos + t0 + tl.arange(0, BLOCK_T)
            tok_mask = tok < eos
            rows = tok
            m_tk = tok_mask[:, None] & m_k[None, :]
            m_tv = tok_mask[:, None] & m_v[None, :]
            off_k = rows[:, None] * (H * M) + i_h * M + o_k[None, :]
            off_v = rows[:, None] * (H * D) + i_h * D + o_v[None, :]

            k_f = tl.load(s_ptr + (off_k), mask=m_tk, other=0.0).to(tl.float32)
            g_blk = tl.load(gk_ptr + (off_k), mask=m_tk, other=0.0).to(tl.float32)
            exp_diff = tl.where(m_tk, tl.exp(g_last[None, :] - g_blk), 0.0)
            v_val = tl.load(z_ptr + (off_v), mask=m_tv, other=0.0)

            dot_lk = tl.dot(v_val, tl.trans(grad_next_st))  # [BLOCK_T, K] fp32
            grad_k_blk = exp_diff * dot_lk
            k_scaled = (k_f * exp_diff).to(v_val.dtype)
            grad_v_blk = tl.dot(k_scaled, grad_next_st)  # [BLOCK_T, V] fp32
            grad_g_blk = -(k_f * grad_k_blk)
            sum_k_gradk += tl.sum(k_f * grad_k_blk, axis=0)

            is_last = tok == last
            last_dg += tl.sum(tl.where(is_last[:, None], grad_g_blk, 0.0), axis=0)
            m_tk_body = m_tk & (tok != last)[:, None]
            if ACCUMULATE:
                grad_k_blk += tl.load(ds_ptr + (off_k), mask=m_tk, other=0.0)
                grad_g_blk += tl.load(dg_ptr + (off_k), mask=m_tk_body, other=0.0)
            tl.store(ds_ptr + (off_k), grad_k_blk, mask=m_tk)
            tl.store(dz_ptr + (off_v), grad_v_blk, mask=m_tv)
            tl.store(dg_ptr + (off_k), grad_g_blk, mask=m_tk_body)

        p_pref = (b) * (M * H * D) + o_k[:, None] * (H * D) + i_h * D + o_v[None, :]
        h_val = tl.load(prefix_ptr + (p_pref), mask=m_kv, other=0.0).to(tl.float32)
        dot_state = tl.sum(h_val * grad_next, axis=1)
        dg_last = last_dg + (exp_g_last * dot_state + sum_k_gradk)
        p_dg_last = dg_ptr + (last.to(tl.int64) * (H * M) + i_h * M + o_k)
        if ACCUMULATE:
            dg_last = tl.load(p_dg_last, mask=m_k, other=0.0) + dg_last
        tl.store(p_dg_last, dg_last, mask=m_k)

        grad_h = tl.load(dprefix_ptr + (p_pref), mask=m_kv, other=0.0).to(tl.float32)
        grad_next = grad_h + exp_g_last[:, None] * grad_next


@triton.jit
def _prefix_reverse_state_kernel(
    gk_ptr,
    dprefix_ptr,
    reverse_ptr,
    seq_block_ptr,
    block_ptr,
    H: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    dm_tiles: tl.constexpr = (D + BLOCK_D - 1) // BLOCK_D
    slot_tiles: tl.constexpr = (M + BLOCK_M - 1) // BLOCK_M
    i_v = program % dm_tiles
    i_k = (program // dm_tiles) % slot_tiles
    i_h = (program // (dm_tiles * slot_tiles)) % H
    i_n = program // (dm_tiles * slot_tiles * H)
    b0 = tl.load(seq_block_ptr + (i_n)).to(tl.int64)
    b1 = tl.load(seq_block_ptr + (i_n + 1)).to(tl.int64)
    o_k = i_k * BLOCK_M + tl.arange(0, BLOCK_M)
    o_v = i_v * BLOCK_D + tl.arange(0, BLOCK_D)
    m_k = o_k < M
    m_v = o_v < D
    m_kv = m_k[:, None] & m_v[None, :]
    grad_next = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for c in range(b1 - b0 - 1, -1, -1):
        b = b0 + c
        eos = tl.load(block_ptr + (b + 1)).to(tl.int64)
        p_pref = b * (M * H * D) + o_k[:, None] * (H * D) + i_h * D + o_v[None, :]
        tl.store(reverse_ptr + (p_pref), grad_next, m_kv)
        g_last = tl.load(gk_ptr + ((eos - 1) * (H * M) + i_h * M + o_k), m_k, other=0.0).to(tl.float32)
        exp_g_last = tl.exp(g_last)
        grad_h = tl.load(dprefix_ptr + (p_pref), m_kv, other=0.0).to(tl.float32)
        grad_next = grad_h + exp_g_last[:, None] * grad_next


@triton.jit(do_not_specialize=["MAX_SEGMENTS"])
def _prefix_bwd_segment_kernel(
    s_ptr,
    z_ptr,
    gk_ptr,
    prefix_ptr,
    dprefix_ptr,
    reverse_ptr,
    ds_ptr,
    dz_ptr,
    dg_ptr,
    seq_block_ptr,
    block_ptr,
    MAX_SEGMENTS,
    H: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    SEGMENT_CHUNKS: tl.constexpr,
):
    """Keep the original dynamic reverse recurrence within independent contiguous segments."""
    program = tl.program_id(0).to(tl.int64)
    i_n = program // MAX_SEGMENTS
    segment_index = program % MAX_SEGMENTS
    i_h = tl.program_id(1).to(tl.int64)
    seq_b0 = (tl.load(seq_block_ptr + (i_n))).to(tl.int64)
    seq_b1 = (tl.load(seq_block_ptr + (i_n + 1))).to(tl.int64)
    if segment_index * SEGMENT_CHUNKS >= (seq_b1 - seq_b0):
        return
    b1 = seq_b1 - segment_index * SEGMENT_CHUNKS
    b0 = tl.maximum(seq_b0, b1 - SEGMENT_CHUNKS)

    o_k = tl.arange(0, BLOCK_K)
    o_v = tl.arange(0, BLOCK_V)
    m_k = o_k < M
    m_v = o_v < D
    m_kv = m_k[:, None] & m_v[None, :]

    carry_offset = (b1 - 1) * (M * H * D) + o_k.to(tl.int64)[:, None] * (H * D)
    carry_offset += i_h * D + o_v.to(tl.int64)[None, :]
    grad_next = tl.load(reverse_ptr + tl.cast(carry_offset, tl.int64), mask=m_kv, other=0.0).to(tl.float32)
    for c in range(b1 - b0 - 1, -1, -1):
        b = b0 + c
        bos = (tl.load(block_ptr + (b.to(tl.int64)))).to(tl.int64)
        eos = (tl.load(block_ptr + (b.to(tl.int64) + 1))).to(tl.int64)
        last = eos - 1
        g_last = tl.load(gk_ptr + (last.to(tl.int64) * (H * M) + i_h * M + o_k.to(tl.int64)), mask=m_k, other=0.0).to(
            tl.float32,
        )
        exp_g_last = tl.exp(g_last)

        sum_k_gradk = tl.zeros([BLOCK_K], dtype=tl.float32)
        last_dg = tl.zeros([BLOCK_K], dtype=tl.float32)
        grad_next_st = grad_next.to(z_ptr.dtype.element_ty)
        for t0 in range(0, C, BLOCK_T):
            tok = bos + t0 + tl.arange(0, BLOCK_T)
            tok_mask = tok < eos
            rows = (bos) + t0 + tl.arange(0, BLOCK_T)
            m_tk = tok_mask[:, None] & m_k[None, :]
            m_tv = tok_mask[:, None] & m_v[None, :]
            off_k = rows[:, None] * (H * M) + i_h * M + o_k.to(tl.int64)[None, :]
            off_v = rows[:, None] * (H * D) + i_h * D + o_v.to(tl.int64)[None, :]

            k_f = tl.load(s_ptr + (off_k), mask=m_tk, other=0.0).to(tl.float32)
            g_blk = tl.load(gk_ptr + (off_k), mask=m_tk, other=0.0).to(tl.float32)
            exp_diff = tl.where(m_tk, tl.exp(g_last[None, :] - g_blk), 0.0)
            v_val = tl.load(z_ptr + (off_v), mask=m_tv, other=0.0)

            dot_lk = tl.dot(v_val, tl.trans(grad_next_st))  # [BLOCK_T, K] fp32
            grad_k_blk = exp_diff * dot_lk
            k_scaled = (k_f * exp_diff).to(v_val.dtype)
            grad_v_blk = tl.dot(k_scaled, grad_next_st)  # [BLOCK_T, V] fp32
            grad_g_blk = -(k_f * grad_k_blk)
            sum_k_gradk += tl.sum(k_f * grad_k_blk, axis=0)

            is_last = tok == last
            last_dg += tl.sum(tl.where(is_last[:, None], grad_g_blk, 0.0), axis=0)
            m_tk_body = m_tk & (tok != last)[:, None]
            if ACCUMULATE:
                grad_k_blk += tl.load(ds_ptr + (off_k), mask=m_tk, other=0.0)
                grad_g_blk += tl.load(dg_ptr + (off_k), mask=m_tk_body, other=0.0)
            tl.store(ds_ptr + (off_k), grad_k_blk, mask=m_tk)
            tl.store(dz_ptr + (off_v), grad_v_blk, mask=m_tv)
            tl.store(dg_ptr + (off_k), grad_g_blk, mask=m_tk_body)

        p_pref = (b) * (M * H * D) + o_k.to(tl.int64)[:, None] * (H * D) + i_h * D + o_v.to(tl.int64)[None, :]
        h_val = tl.load(prefix_ptr + (p_pref), mask=m_kv, other=0.0).to(tl.float32)
        dot_state = tl.sum(h_val * grad_next, axis=1)
        dg_last = last_dg + (exp_g_last * dot_state + sum_k_gradk)
        p_dg_last = dg_ptr + (last.to(tl.int64) * (H * M) + i_h * M + o_k.to(tl.int64))
        if ACCUMULATE:
            dg_last = tl.load(p_dg_last, mask=m_k, other=0.0) + dg_last
        tl.store(p_dg_last, dg_last, mask=m_k)

        grad_h = tl.load(dprefix_ptr + (p_pref), mask=m_kv, other=0.0).to(tl.float32)
        grad_next = grad_h + exp_g_last[:, None] * grad_next


def _launch_kwargs(name: str) -> dict:
    cfg = LAUNCH[name]
    return {"num_warps": cfg.num_warps, "num_stages": cfg.num_stages}


def local_cumsum(x: torch.Tensor, layout: BlockLayout, *, reverse: bool = False) -> torch.Tensor:
    """Compute block-local inclusive token sums, or suffix sums in reverse mode, in FP32."""
    T, H, M = x.shape
    y = torch.empty((T, H, M), dtype=torch.float32, device=x.device)
    grid = (layout.Nc, H, triton.cdiv(M, CUMSUM_BS))
    name = "chunk_local_cumsum_reverse" if reverse else "chunk_local_cumsum"
    _local_cumsum_kernel[grid](
        x,
        y,
        layout.block_ptr,
        H=H,
        M=M,
        C=layout.chunk_size,
        BS=CUMSUM_BS,
        REVERSE=reverse,
        **_launch_kwargs(name),
    )
    return y


def _prefix_forward(s, z, gk, layout: BlockLayout, out, ht):
    T, H, M = s.shape
    D = z.shape[-1]
    nseq = len(layout.seq_block_ptr_host) - 1
    grid = (nseq * H, triton.cdiv(M, FWD_BK), triton.cdiv(D, FWD_BV))
    _prefix_fwd_kernel[grid](
        s,
        z,
        gk,
        out,
        ht if ht is not None else out,
        layout.seq_block_ptr,
        layout.block_ptr,
        H=H,
        M=M,
        D=D,
        C=layout.chunk_size,
        BK=FWD_BK,
        BV=FWD_BV,
        STORE_FINAL=ht is not None,
        **_launch_kwargs("chunk_fwd_h"),
    )


def _prefix_backward(s, z, gk, prefix, dprefix, ds, dz, dg, layout: BlockLayout, accumulate: bool):
    T, H, M = s.shape
    D = z.shape[-1]
    nseq = len(layout.seq_block_ptr_host) - 1
    _prefix_bwd_kernel[nseq, H](
        s,
        z,
        gk,
        prefix,
        dprefix,
        ds,
        dz,
        dg,
        layout.seq_block_ptr,
        layout.block_ptr,
        H=H,
        M=M,
        D=D,
        C=layout.chunk_size,
        BLOCK_T=BWD_BLOCK_T,
        BLOCK_K=max(16, triton.next_power_of_2(M)),
        BLOCK_V=max(16, triton.next_power_of_2(D)),
        ACCUMULATE=accumulate,
        **_launch_kwargs("chunk_state_bwd"),
    )


def _prefix_reverse_state(gk, dprefix, layout: BlockLayout, out):
    """Materialize the FP32 adjoint before each chunk."""
    _, H, M = gk.shape
    D = dprefix.shape[-1]
    nseq = len(layout.seq_block_ptr_host) - 1
    grid = (nseq * H * triton.cdiv(M, REVERSE_BLOCK_M) * triton.cdiv(D, REVERSE_BLOCK_D),)
    _prefix_reverse_state_kernel[grid](
        gk,
        dprefix,
        out,
        layout.seq_block_ptr,
        layout.block_ptr,
        H=H,
        M=M,
        D=D,
        BLOCK_M=REVERSE_BLOCK_M,
        BLOCK_D=REVERSE_BLOCK_D,
        **_launch_kwargs("prefix_reverse_state"),
    )


def _prefix_backward_segment(s, z, gk, prefix, dprefix, reverse_state, ds, dz, dg, layout: BlockLayout, accumulate: bool):
    """Launch deterministic backward over independent contiguous sequence segments."""
    _, H, M = s.shape
    D = z.shape[-1]
    boundaries = layout.seq_block_ptr_host
    nseq = len(boundaries) - 1
    max_segments = max((triton.cdiv(end - begin, SEGMENT_CHUNKS) for begin, end in zip(boundaries, boundaries[1:])), default=0)
    if max_segments == 0:
        return
    _prefix_bwd_segment_kernel[nseq * max_segments, H](
        s,
        z,
        gk,
        prefix,
        dprefix,
        reverse_state,
        ds,
        dz,
        dg,
        layout.seq_block_ptr,
        layout.block_ptr,
        max_segments,
        H=H,
        M=M,
        D=D,
        C=layout.chunk_size,
        BLOCK_T=BWD_BLOCK_T,
        BLOCK_K=max(16, triton.next_power_of_2(M)),
        BLOCK_V=max(16, triton.next_power_of_2(D)),
        ACCUMULATE=accumulate,
        SEGMENT_CHUNKS=SEGMENT_CHUNKS,
        **_launch_kwargs("prefix_bwd_local"),
    )


def _prefix_backward_pair(s, k, v, gk, pk, pv, dpk, dpv, ds, dk, dv, dg, layout, plan: PrefixBackwardPlan):
    if plan.parallel:
        nc, m, h, d = pk.shape
        if plan.workspace_bytes != 4 * nc * m * h * d:
            raise ValueError("forward-bound Prefix backward plan does not match saved tensors")
        reverse_state = torch.empty(pk.shape, dtype=torch.float32, device=pk.device)
        _prefix_reverse_state(gk, dpk, layout, reverse_state)
        _prefix_backward_segment(s, k, gk, pk, dpk, reverse_state, ds, dk, dg, layout, accumulate=False)
        _prefix_reverse_state(gk, dpv, layout, reverse_state)
        _prefix_backward_segment(s, v, gk, pv, dpv, reverse_state, ds, dv, dg, layout, accumulate=True)
    else:
        _prefix_backward(s, k, gk, pk, dpk, ds, dk, dg, layout, accumulate=False)
        _prefix_backward(s, v, gk, pv, dpv, ds, dv, dg, layout, accumulate=True)


def _aligned_contiguous(x: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    if x is None:
        return torch.zeros_like(like)
    if x.is_contiguous() and x.data_ptr() % 16 == 0 and x.dtype == like.dtype:
        return x
    out = torch.empty(like.shape, dtype=like.dtype, device=like.device)
    out.copy_(x)
    return out


class ChunkwisePrefixStateFn(torch.autograd.Function):
    """Differentiate prefix states independently of joint attention.

    Prefixes use [Nc, M, H, D] storage; optional FP32 final states are nondifferentiable.
    """

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, s_val, k_raw, v, g_log, block_ptr, seq_block_ptr, layout: BlockLayout, output_final_state: bool):
        T, H, M = s_val.shape
        D = k_raw.shape[-1]
        nc, nseq = layout.Nc, len(layout.seq_block_ptr_host) - 1
        gk = local_cumsum(g_log, layout)
        prefix_k = torch.empty((nc, M, H, D), dtype=k_raw.dtype, device=k_raw.device)
        prefix_v = torch.empty((nc, M, H, D), dtype=v.dtype, device=v.device)
        if output_final_state:
            ht_k = torch.empty((nseq, H, M, D), dtype=torch.float32, device=k_raw.device)
            ht_v = torch.empty((nseq, H, M, D), dtype=torch.float32, device=v.device)
        else:
            ht_k = ht_v = None
        _prefix_forward(s_val, k_raw, gk, layout, prefix_k, ht_k)
        _prefix_forward(s_val, v, gk, layout, prefix_v, ht_v)
        # save device tensors through save_for_backward; keep only host geometry on ctx
        ctx.layout_host = (layout.chunk_size, layout.block_ptr_host, layout.seq_block_ptr_host)
        ctx.backward_plan = prefix_backward_plan(nc, H, M, D)
        ctx.save_for_backward(s_val, k_raw, v, g_log, block_ptr, seq_block_ptr, prefix_k, prefix_v)
        if ht_k is None:
            ht_k = prefix_k.new_empty(0, dtype=torch.float32)
            ht_v = prefix_v.new_empty(0, dtype=torch.float32)
        ctx.mark_non_differentiable(ht_k, ht_v)
        return prefix_k, prefix_v, ht_k, ht_v

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, d_prefix_k, d_prefix_v, _d_ht_k, _d_ht_v):
        s_val, k_raw, v, g_log, block_ptr, seq_block_ptr, prefix_k, prefix_v = ctx.saved_tensors
        chunk_size, bp_host, sbp_host = ctx.layout_host
        layout = BlockLayout(chunk_size, bp_host, sbp_host, block_ptr=block_ptr, seq_block_ptr=seq_block_ptr)
        T, H, M = s_val.shape
        D = k_raw.shape[-1]
        if M > CHUNKWISE_TRAIN_MAX_M or D > CHUNKWISE_TRAIN_MAX_DH:
            raise fail(
                "prefix_kernel_unsupported",
                f"Prefix backward requires M<={CHUNKWISE_TRAIN_MAX_M} and Dh<={CHUNKWISE_TRAIN_MAX_DH}; got M={M}, Dh={D}",
            )
        gk = local_cumsum(g_log, layout)
        d_prefix_k = _aligned_contiguous(d_prefix_k, prefix_k)
        d_prefix_v = _aligned_contiguous(d_prefix_v, prefix_v)
        ds = torch.empty((T, H, M), dtype=torch.float32, device=s_val.device)
        dg = torch.empty((T, H, M), dtype=torch.float32, device=s_val.device)
        dk = torch.empty((T, H, D), dtype=k_raw.dtype, device=s_val.device)
        dv = torch.empty((T, H, D), dtype=v.dtype, device=s_val.device)
        _prefix_backward_pair(
            s_val,
            k_raw,
            v,
            gk,
            prefix_k,
            prefix_v,
            d_prefix_k,
            d_prefix_v,
            ds,
            dk,
            dv,
            dg,
            layout,
            ctx.backward_plan,
        )
        d_g_log = local_cumsum(dg, layout, reverse=True)
        return (ds.to(s_val.dtype), dk.to(k_raw.dtype), dv.to(v.dtype), d_g_log.to(g_log.dtype), None, None, None, None)


def check_inputs(
    s_val: torch.Tensor,
    k_raw: torch.Tensor,
    v: torch.Tensor,
    g_log: torch.Tensor,
    layout: BlockLayout,
    *,
    needs_backward: bool,
) -> None:
    """Validate prefix geometry, storage types and device layout before launching kernels."""
    c = layout.chunk_size
    if c not in CHUNKWISE_VALIDATED_C:
        raise fail("prefix_kernel_unsupported", f"Chunk size C={c} is outside the validated set {CHUNKWISE_VALIDATED_C}")
    for name, x in (("s_val", s_val), ("k_raw", k_raw), ("v", v), ("g_log", g_log)):
        if x.device.type != "cuda":
            raise fail("prefix_kernel_unsupported", f"{name} is on {x.device}; prefix kernels require CUDA")
        if x.dim() != 3:
            raise fail("prefix_kernel_unsupported", f"{name} must have shape [T, H, *], got {tuple(x.shape)}")
        if not x.is_contiguous():
            raise fail("non_contiguous_input", f"{name} must be contiguous for row-major [T, H, *] addressing")
    if s_val.dtype not in STORAGE_DTYPES or k_raw.dtype != s_val.dtype or v.dtype != s_val.dtype:
        raise fail(
            "prefix_kernel_unsupported",
            f"s_val, k_raw and v must share FP16/BF16 dtype; got {s_val.dtype}, {k_raw.dtype}, {v.dtype}",
        )
    if g_log.dtype != torch.float32:
        raise fail("prefix_kernel_unsupported", f"g_log must be FP32, got {g_log.dtype}")
    T, H, M = s_val.shape
    if g_log.shape != s_val.shape or k_raw.shape[:2] != (T, H) or v.shape != k_raw.shape:
        raise fail(
            "prefix_kernel_unsupported",
            f"Shape mismatch: s {tuple(s_val.shape)}, g {tuple(g_log.shape)}, k {tuple(k_raw.shape)} v {tuple(v.shape)}",
        )
    if layout.block_ptr_host[-1] != T:
        raise fail("invalid_cu_seqlens", f"Block layout length {layout.block_ptr_host[-1]} does not match input length {T}")
    D = k_raw.shape[-1]
    if needs_backward and (M > CHUNKWISE_TRAIN_MAX_M or D > CHUNKWISE_TRAIN_MAX_DH):
        raise fail(
            "prefix_kernel_unsupported",
            f"Prefix training requires M<={CHUNKWISE_TRAIN_MAX_M} and Dh<={CHUNKWISE_TRAIN_MAX_DH}; got M={M}, Dh={D}",
        )
    if layout.block_ptr.device != s_val.device:
        raise fail(
            "device_mismatch",
            f"Block layout device {layout.block_ptr.device} does not match input device {s_val.device}",
        )


def chunkwise_prefix_states(
    s_val: torch.Tensor,
    k_raw: torch.Tensor,
    v: torch.Tensor,
    g_log: torch.Tensor,
    layout: BlockLayout,
    *,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Return key/value prefixes [Nc, M, H, D] and optional FP32 final states [Nseq, H, M, D]."""
    needs_backward = torch.is_grad_enabled() and any(x.requires_grad for x in (s_val, k_raw, v, g_log))
    check_inputs(s_val, k_raw, v, g_log, layout, needs_backward=needs_backward)
    pk, pv, hk, hv = ChunkwisePrefixStateFn.apply(
        s_val,
        k_raw,
        v,
        g_log,
        layout.block_ptr,
        layout.seq_block_ptr,
        layout,
        output_final_state,
    )
    if not output_final_state:
        hk = hv = None
    return pk, pv, hk, hv


__all__ = [
    "BWD_BLOCK_T",
    "CUMSUM_BS",
    "FWD_BK",
    "FWD_BV",
    "LAUNCH",
    "ChunkwisePrefixStateFn",
    "LaunchConfig",
    "check_inputs",
    "chunkwise_prefix_states",
    "local_cumsum",
]


@triton.jit(do_not_specialize=["N"])
def _gate_neg_store_kernel(u, out, N, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    value = tl.load(u + (i), i < N, other=0)
    negated = (value.to(tl.int32, bitcast=True) ^ -2147483648).to(tl.float32, bitcast=True)
    tl.store(out + (i), negated, i < N)


@triton.jit(do_not_specialize=["N"])
def _gate_algebra_backward_kernel(u, ds, dg, out, N, HAS_S: tl.constexpr, HAS_G: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    if HAS_S:
        storage_grad = tl.load(ds + (i), i < N, other=0).to(tl.float32)
        saved = tl.load(u + (i), i < N, other=0)
        factor = saved + 1.0
        negated = (storage_grad.to(tl.int32, bitcast=True) ^ -2147483648).to(tl.float32, bitcast=True)
        value = negated * factor
    if HAS_G:
        direct = tl.load(dg + (i), i < N, other=0)
        if HAS_S:
            value = direct + value
        else:
            value = direct
    value = value * 0.125
    tl.store(out + (i), value, i < N)


class GateAlgebraFn(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, log_sigmoid, storage_dtype):
        g = log_sigmoid / 8
        u = torch.expm1(g)
        s = torch.empty_like(u, dtype=storage_dtype)
        _gate_neg_store_kernel[triton.cdiv(u.numel(), 256),](
            u,
            s,
            u.numel(),
            B=256,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(g, u)
        ctx.set_materialize_grads(False)
        return g, s

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dg, ds):
        g, u = ctx.saved_tensors
        if ds is None and dg is None:
            return None, None
        if torch.is_grad_enabled():
            value = (-ds.float()) * (torch.expm1(g) + 1) if ds is not None else None
            if dg is not None:
                value = dg + value if value is not None else dg
            return value / 8, None
        ds = ds.contiguous() if ds is not None else None
        dg = dg.contiguous() if dg is not None else None
        out = torch.empty_like(u)
        _gate_algebra_backward_kernel[triton.cdiv(u.numel(), 256),](
            u,
            ds,
            dg,
            out,
            u.numel(),
            HAS_S=ds is not None,
            HAS_G=dg is not None,
            B=256,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        return out, None


def gate_algebra(log_sigmoid, storage_dtype, tau):
    if (
        tau == 8
        and log_sigmoid.is_cuda
        and log_sigmoid.dtype == torch.float32
        and log_sigmoid.is_contiguous()
        and storage_dtype in (torch.bfloat16, torch.float16)
    ):
        return GateAlgebraFn.apply(log_sigmoid, storage_dtype)
    g = log_sigmoid / tau
    return g, (-torch.expm1(g)).to(storage_dtype)
