# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from . import invcum

BLOCK_SIZE_C = 512


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    u: torch.Tensor,
    qk_scale: float,
    beta: torch.Tensor,
    row_start: int,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    B, C, D = q.size()
    _B, T, _D = k.size()
    __B, __C = beta.size()
    assert B == _B and D == _D and B == __B and __C == C
    w = torch.empty(B, C, C, device=q.device, dtype=q.dtype)
    lse = torch.empty(B, C, device=q.device, dtype=torch.float)
    delta_flash_attn_compileable(q, k, v, u, w, lse, qk_scale, beta, row_start, cu_seqlens)
    return w, lse


def backward_u_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    grad_v: torch.Tensor,
    fa_scale: float,
    beta: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    row_start: int = 0,
):
    B, C, D = q.size()
    _B, T, _D = k.size()
    grad_u = torch.empty_like(q)

    def grid(META):
        return (triton.cdiv(C, META['BLOCK_C']), B)

    backward_u_chunk_kernel[grid](
        grad_u,
        grad_u.stride(0), grad_u.stride(1), grad_u.stride(2),
        q,
        q.stride(0), q.stride(1), q.stride(2),
        k,
        k.stride(0), k.stride(1), k.stride(2),
        grad_v,
        grad_v.stride(0), grad_v.stride(1), grad_v.stride(2),
        lse,
        lse.stride(0), lse.stride(1),
        beta,
        beta.stride(0), beta.stride(1),
        B, T, C, D, fa_scale,
        row_start=row_start,
        cu_seqlens=cu_seqlens
    )
    return grad_u


def backward_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    lse: torch.Tensor,
    grad_v: torch.Tensor,
    qk_scale: float,
    fa_scale: float,
    beta: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    row_start: int = 0,
):
    B, T, D = k.size()
    row_dot_sum = torch.empty_like(lse)

    def grid_bp(META):
        return (triton.cdiv(T, META['BLOCK_C']), B)

    backward_p_row_sum_kernel[grid_bp](
        row_dot_sum,
        row_dot_sum.stride(0), row_dot_sum.stride(1),
        q,
        q.stride(0), q.stride(1), q.stride(2),
        k,
        k.stride(0), k.stride(1), k.stride(2),
        grad_v,
        grad_v.stride(0), grad_v.stride(1), grad_v.stride(2),
        u,
        u.stride(0), u.stride(1), u.stride(2),
        lse,
        lse.stride(0), lse.stride(1),
        B, T, D,
        fa_scale,
        row_start=row_start,
        cu_seqlens=cu_seqlens
    )
    grad_k = torch.empty_like(k)
    grad_q = torch.empty_like(q)

    backward_qk_kernel[grid_bp](
        grad_q,
        grad_q.stride(0), grad_q.stride(1), grad_q.stride(2),
        grad_k,
        grad_k.stride(0), grad_k.stride(1), grad_k.stride(2),
        q,
        q.stride(0), q.stride(1), q.stride(2),
        k,
        k.stride(0), k.stride(1), k.stride(2),
        grad_v,
        grad_v.stride(0), grad_v.stride(1), grad_v.stride(2),
        u,
        u.stride(0), u.stride(1), u.stride(2),
        lse,
        lse.stride(0), lse.stride(1),
        beta,
        beta.stride(0), beta.stride(1),
        row_dot_sum,
        row_dot_sum.stride(0), row_dot_sum.stride(1),
        B, T, D,
        fa_scale, qk_scale,
        row_start=row_start,
        cu_seqlens=cu_seqlens
    )
    return grad_q, grad_k, row_dot_sum  # row_dot_sum is the gradient w.r.t. the per-row beta scaling


def delta_flash_attn_compileable(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    lse: torch.Tensor,
    qk_scale: float,
    beta: torch.Tensor,
    row_start: int,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> None:
    B, C, D = q.size()
    _B, T, _D = k.size()

    def grid(META):
        return (triton.cdiv(C, META['BLOCK_C']), B)

    flash_attn_kernel[grid](
        q,
        q.stride(0), q.stride(1), q.stride(2),
        k,
        k.stride(0), k.stride(1), k.stride(2),
        v,
        v.stride(0), v.stride(1), v.stride(2),
        u,
        u.stride(0), u.stride(1), u.stride(2),
        w,
        w.stride(0), w.stride(1),
        lse,
        lse.stride(0),
        beta,
        beta.stride(0),
        B, T, C, D, qk_scale,
        row_start=row_start,
        cu_seqlens=cu_seqlens
    )


def _config_delta_flash_attn():
    return [
        triton.Config({'BLOCK_C': BC, 'BLOCK_T': BT}, num_stages=ns, num_warps=nw)
        for BC in [128, 64]
        for BT in [64, 32]
        for ns in [3, 2]
        for nw in [8, 4]
    ]


@triton.autotune(configs=_config_delta_flash_attn(), key=['C', 'D'])
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def flash_attn_kernel(
    q_ptr,
    stride_qh,
    stride_qc,
    stride_qd,
    k_ptr,
    stride_kh,
    stride_kt,
    stride_kd,
    v_ptr,
    stride_vh,
    stride_vc,
    stride_vd,
    u_ptr,
    stride_uh,
    stride_ut,
    stride_ud,
    w_ptr,
    stride_wh,
    stride_wc,
    lse_ptr,
    stride_lse_r,
    beta_ptr,
    beta_stride_r,
    B,
    T,
    C,
    D: tl.constexpr,
    qk_scale: float,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
    row_start: tl.constexpr,
    cu_seqlens,
    IS_VARLEN: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    rowid_block = tl.arange(0, BLOCK_C) + pid_c * BLOCK_C
    colid_block = tl.arange(0, BLOCK_T)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + pid_b).to(tl.int32)
        eos = tl.load(cu_seqlens + pid_b + 1).to(tl.int32)
        T_local = eos - bos
        local_row_start = row_start - bos
    else:
        bos = pid_b * T
        eos = bos + T
        T_local = T
        local_row_start = row_start

    cb_local = tl.maximum(0, tl.minimum(C, T_local - local_row_start))
    valid_row = rowid_block < cb_local

    rowmax = tl.full([BLOCK_C], -float('inf'), dtype=tl.float32)
    rowsum = tl.full([BLOCK_C], 1, dtype=tl.float32)
    acc = tl.zeros([BLOCK_C, D], dtype=tl.float32)

    q_blk_ptr = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qh,
        shape=(C, D),
        strides=(stride_qc, stride_qd),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    q = tl.load(q_blk_ptr, boundary_check=(0,))

    k_base = k_ptr + pid_b * stride_kh + (bos if IS_VARLEN else 0) * stride_kt
    u_base = u_ptr + pid_b * stride_uh + (bos if IS_VARLEN else 0) * stride_ut

    for kv_i in range(0, T_local, BLOCK_T):
        k_blk_ptr = tl.make_block_ptr(
            base=k_base,
            shape=(D, T_local),
            strides=(stride_kd, stride_kt),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        k = tl.load(k_blk_ptr, boundary_check=(1,))
        qk = tl.dot(q, k) * qk_scale

        if kv_i >= local_row_start:
            mask = (local_row_start - kv_i + rowid_block[:, None] - colid_block[None, :] < 1)
            qk = tl.where(mask, -1e6, qk)

        rowmax_i = tl.maximum(rowmax, tl.max(qk, axis=1))
        qk -= rowmax_i[:, None]
        p = tl.math.exp2(qk)

        rowsum_i = tl.sum(p, axis=1)
        alpha = tl.math.exp2(rowmax - rowmax_i)
        rowsum = rowsum * alpha + rowsum_i
        acc = acc * alpha[:, None]
        rowmax = rowmax_i

        if kv_i < local_row_start:
            u_blk_ptr = tl.make_block_ptr(
                base=u_base,
                shape=(T_local, D),
                strides=(stride_ut, stride_ud),
                offsets=(kv_i, 0),
                block_shape=(BLOCK_T, D),
                order=(1, 0),
            )
            u = tl.load(u_blk_ptr, boundary_check=(0,))
            acc = tl.dot(p.to(u_ptr.dtype.element_ty), u, acc)

    lse = rowmax + tl.math.log2(rowsum)
    lse_block_ptr = lse_ptr + stride_lse_r * pid_b + rowid_block
    lse_mask = valid_row
    tl.store(lse_block_ptr, lse, mask=lse_mask)

    v_ptr = tl.make_block_ptr(
        base=v_ptr + pid_b * stride_vh,
        shape=(C, D),
        strides=(stride_vc, stride_vd),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    acc = acc / rowsum[:, None]

    beta_blk_ptr = tl.make_block_ptr(
        base=beta_ptr + pid_b * beta_stride_r,
        shape=(C,),
        strides=(1,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,)
    )
    beta = tl.load(beta_blk_ptr, boundary_check=(0,))
    acc = acc * beta[:, None]

    v = tl.load(v_ptr, boundary_check=(0,))
    u = v - acc.to(v_ptr.dtype.element_ty)
    u_block_ptr = tl.make_block_ptr(
        base=u_base,
        shape=(T_local, D),
        strides=(stride_ut, stride_ud),
        offsets=(local_row_start + pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    tl.store(u_block_ptr, u, boundary_check=(0, 1))

    for kv_i in range(local_row_start, T_local, BLOCK_T):
        k_blk_ptr = tl.make_block_ptr(
            base=k_base,
            shape=(D, T_local),
            strides=(stride_kd, stride_kt),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        k = tl.load(k_blk_ptr, boundary_check=(1,))
        qk = tl.dot(q, k) * qk_scale

        mask = (local_row_start - kv_i + rowid_block[:, None] - colid_block[None, :] < 1)
        qk -= rowmax[:, None]
        p = tl.math.exp2(qk) / rowsum[:, None]
        p = tl.where(mask, 0, p)
        w_blk_ptr = tl.make_block_ptr(
            base=w_ptr + pid_b * stride_wh,
            shape=(C, C),
            strides=(stride_wc, 1),
            offsets=(pid_c * BLOCK_C, kv_i - local_row_start),
            block_shape=(BLOCK_C, BLOCK_T),
            order=(1, 0)
        )
        tl.store(w_blk_ptr, p.to(w_ptr.dtype.element_ty), boundary_check=(0, 1))


def _config_backward_u_chunk():
    return [
        triton.Config({'BLOCK_C': BC, 'BLOCK_T': BT}, num_stages=ns, num_warps=nw)
        for BC in [128, 64]
        for BT in [64, 32]
        for ns in [3, 2]
        for nw in [8, 4]
    ]


@triton.autotune(configs=_config_backward_u_chunk(), key=['C', 'D'])
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def backward_u_chunk_kernel(
    o_ptr,
    stride_oh,
    stride_oc,
    stride_od,
    q_ptr,
    stride_qh,
    stride_qc,
    stride_qd,
    k_ptr,
    stride_kh,
    stride_kt,
    stride_kd,
    v_ptr,
    stride_vh,
    stride_vt,
    stride_vd,
    lse_ptr,
    stride_lse_h,
    stride_lse_t,
    beta_ptr,
    stride_beta_h,
    stride_beta_t,
    B,
    T,
    C,
    D: tl.constexpr,
    fa_scale,
    row_start: tl.constexpr,
    cu_seqlens,
    IS_VARLEN: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + pid_b).to(tl.int32)
        eos = tl.load(cu_seqlens + pid_b + 1).to(tl.int32)
        local_row_start = row_start - bos
        tail_start = local_row_start + C
        T_local = tl.maximum(0, eos - bos - tail_start)
    else:
        T_local = T

    local_row_start = row_start

    acc = tl.zeros([BLOCK_C, D], dtype=tl.float32)

    q_blk_ptr = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qh,
        shape=(C, D),
        strides=(stride_qc, stride_qd),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    q = tl.load(q_blk_ptr)

    for kv_i in range(0, T_local, BLOCK_T):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_b * stride_kh,
            shape=(D, T_local),
            strides=(stride_kd, stride_kt),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        k = tl.load(k_blk_ptr)
        qk = tl.dot(q, k) * fa_scale

        lse_blk_ptr = tl.make_block_ptr(
            base=lse_ptr + pid_b * stride_lse_h,
            shape=(T_local,),
            strides=(stride_lse_t,),
            offsets=(kv_i,),
            block_shape=(BLOCK_T,),
            order=(0,),
        )
        lse = tl.load(lse_blk_ptr)
        beta_blk_ptr = tl.make_block_ptr(
            base=beta_ptr + pid_b * stride_beta_h,
            shape=(T_local,),
            strides=(stride_beta_t,),
            offsets=(kv_i,),
            block_shape=(BLOCK_T,),
            order=(0,),
        )
        beta = tl.load(beta_blk_ptr)

        p = tl.math.exp2(qk - lse[None, :]) * beta[None, :]

        v_blk_ptr = tl.make_block_ptr(
            base=v_ptr + pid_b * stride_vh,
            shape=(T_local, D),
            strides=(stride_vt, stride_vd),
            offsets=(kv_i, 0),
            block_shape=(BLOCK_T, D),
            order=(1, 0),
        )
        v = tl.load(v_blk_ptr)
        acc = tl.dot(p.to(v_ptr.dtype.element_ty), v, acc)

    o_blk_ptr = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_oh,
        shape=(C, D),
        strides=(stride_oc, stride_od),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    tl.store(o_blk_ptr, acc.to(o_ptr.dtype.element_ty))


def _config_backward_p_row_sum():
    return [
        triton.Config({'BLOCK_C': BC, 'BLOCK_T': BT}, num_stages=ns, num_warps=nw)
        for BC in [128, 64]
        for BT in [64, 32]
        for ns in [4, 3, 2]
        for nw in [8, 4]
    ]


@triton.autotune(configs=_config_backward_p_row_sum(), key=['T', 'D'])
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def backward_p_row_sum_kernel(
    row_dot_ptr,
    stride_row_dot_h,
    stride_row_dot_t,
    q_ptr,
    stride_qh,
    stride_qt,
    stride_qd,
    k_ptr,
    stride_kh,
    stride_kt,
    stride_kd,
    grad_v_ptr,
    stride_grad_vh,
    stride_grad_vt,
    stride_grad_vd,
    u_ptr,
    stride_uh,
    stride_ut,
    stride_ud,
    lse_ptr,
    stride_lse_h,
    stride_lse_t,
    B,
    T,
    D: tl.constexpr,
    fa_scale,
    row_start: tl.constexpr,
    cu_seqlens,
    IS_VARLEN: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    rowid_block = tl.arange(0, BLOCK_C) + pid_c * BLOCK_C
    colid_block = tl.arange(0, BLOCK_T)

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + pid_b).to(tl.int32)
        eos = tl.load(cu_seqlens + pid_b + 1).to(tl.int32)
        T_local = eos - bos
        local_row_start = row_start - bos
    else:
        bos = pid_b * T
        eos = bos + T
        T_local = T
        local_row_start = row_start

    k_row_blk_ptr = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qh + (bos if IS_VARLEN else 0) * stride_qt,
        shape=(T_local, D),
        strides=(stride_qt, stride_qd),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    k_row = tl.load(k_row_blk_ptr)
    lse_blk_ptr = tl.make_block_ptr(
        base=lse_ptr + pid_b * stride_lse_h,
        shape=(T_local,),
        strides=(stride_lse_t,),
        offsets=(local_row_start + pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    lse = tl.load(lse_blk_ptr)
    grad_v_blk_ptr = tl.make_block_ptr(
        base=grad_v_ptr + pid_b * stride_grad_vh + (bos if IS_VARLEN else 0) * stride_grad_vt,
        shape=(T_local, D),
        strides=(stride_grad_vt, stride_grad_vd),
        offsets=(local_row_start + pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    grad_v_row = -tl.load(grad_v_blk_ptr)

    for kv_i in range(0, (pid_c + 1) * BLOCK_C, BLOCK_T):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_b * stride_kh + (bos if IS_VARLEN else 0) * stride_kt,
            shape=(D, T_local),
            strides=(stride_kd, stride_kt),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        k = tl.load(k_blk_ptr)
        qk = tl.dot(k_row, k) * fa_scale
        p = tl.math.exp2(qk - lse[:, None])

        u_blk_ptr = tl.make_block_ptr(
            base=u_ptr + pid_b * stride_uh + (bos if IS_VARLEN else 0) * stride_ut,
            shape=(D, T_local),
            strides=(stride_ud, stride_ut),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        ut = tl.load(u_blk_ptr)
        dp = tl.dot(grad_v_row, ut)
        if kv_i + BLOCK_T >= pid_c * BLOCK_C:
            mask = (rowid_block[:, None] <= colid_block[None, :] + kv_i)
            p = tl.where(mask, 0., p)
            dp = tl.where(mask, 0., dp)
        acc += tl.sum(p * dp, axis=1)
    row_dot_block_ptr = tl.make_block_ptr(
        base=row_dot_ptr + pid_b * stride_row_dot_h,
        shape=(T_local,),
        strides=(stride_row_dot_t,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    tl.store(row_dot_block_ptr, acc)


def _config_backward_k():
    return [
        triton.Config({'BLOCK_C': BC}, num_stages=ns, num_warps=nw)
        for BC in [64, 32]
        for ns in [4, 3]
        for nw in [4]
    ]


@triton.autotune(configs=_config_backward_k(), key=['T', 'D'])
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def backward_qk_kernel(
    grad_q_ptr,
    stride_grad_qh,
    stride_grad_qt,
    stride_grad_qd,
    grad_k_ptr,
    stride_grad_kh,
    stride_grad_kt,
    stride_grad_kd,
    q_ptr,
    stride_qh,
    stride_qt,
    stride_qd,
    k_ptr,
    stride_kh,
    stride_kt,
    stride_kd,
    grad_v_ptr,
    stride_grad_vh,
    stride_grad_vt,
    stride_grad_vd,
    u_ptr,
    stride_uh,
    stride_ut,
    stride_ud,
    lse_ptr,
    stride_lse_h,
    stride_lse_t,
    beta_ptr,
    stride_beta_h,
    stride_beta_t,
    row_dot_ptr,
    stride_row_dot_h,
    stride_row_dot_t,
    B,
    T,
    D: tl.constexpr,
    fa_scale: tl.constexpr,
    qk_scale: tl.constexpr,
    BLOCK_C: tl.constexpr,
    row_start: tl.constexpr,
    cu_seqlens,
    IS_VARLEN: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    block_i = tl.arange(0, BLOCK_C)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + pid_b).to(tl.int32)
        eos = tl.load(cu_seqlens + pid_b + 1).to(tl.int32)
        T_local = eos - bos
        local_row_start = row_start - bos
    else:
        bos = pid_b * T
        eos = bos + T
        T_local = T
        local_row_start = row_start

    acc = tl.zeros([BLOCK_C, D], dtype=tl.float32)

    k_row_blk_ptr = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qh + (bos if IS_VARLEN else 0) * stride_qt,
        shape=(T_local, D),
        strides=(stride_qt, stride_qd),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    k_row = tl.load(k_row_blk_ptr)
    lse_blk_ptr = tl.make_block_ptr(
        base=lse_ptr + pid_b * stride_lse_h,
        shape=(T_local,),
        strides=(stride_lse_t,),
        offsets=(local_row_start + pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    lse = tl.load(lse_blk_ptr)
    beta_blk_ptr = tl.make_block_ptr(
        base=beta_ptr + pid_b * stride_beta_h,
        shape=(T_local,),
        strides=(stride_beta_t,),
        offsets=(local_row_start + pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    beta = tl.load(beta_blk_ptr)
    grad_v_blk_ptr = tl.make_block_ptr(
        base=grad_v_ptr + pid_b * stride_grad_vh + (bos if IS_VARLEN else 0) * stride_grad_vt,
        shape=(T_local, D),
        strides=(stride_grad_vt, stride_grad_vd),
        offsets=(local_row_start + pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    grad_v_row = -tl.load(grad_v_blk_ptr)
    row_dot_blk_ptr = tl.make_block_ptr(
        base=row_dot_ptr + pid_b * stride_row_dot_h,
        shape=(T_local,),
        strides=(stride_row_dot_t,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    row_dot_row = tl.load(row_dot_blk_ptr).to(k_ptr.dtype.element_ty)

    for kv_i in range(0, pid_c * BLOCK_C, BLOCK_C):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_b * stride_kh + (bos if IS_VARLEN else 0) * stride_kt,
            shape=(D, T_local),
            strides=(stride_kd, stride_kt),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_C),
            order=(0, 1),
        )
        kt = tl.load(k_blk_ptr)
        qk = tl.dot(k_row, kt) * fa_scale
        p = tl.math.exp2(qk - lse[:, None]) * beta[:, None]

        u_blk_ptr = tl.make_block_ptr(
            base=u_ptr + pid_b * stride_uh + (bos if IS_VARLEN else 0) * stride_ut,
            shape=(D, T_local),
            strides=(stride_ud, stride_ut),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_C),
            order=(0, 1),
        )
        ut = tl.load(u_blk_ptr)
        dp = tl.dot(grad_v_row, ut)
        da = p * (dp - row_dot_row[:, None])
        k = tl.trans(kt, 1, 0)
        acc = tl.dot(da.to(k.dtype), k, acc)

    k_row_blk_ptr = tl.make_block_ptr(
        base=k_ptr + pid_b * stride_kh + (bos if IS_VARLEN else 0) * stride_kt,
        shape=(T_local, D),
        strides=(stride_kt, stride_kd),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    k_row_true = tl.load(k_row_blk_ptr)
    qk = tl.dot(k_row, tl.trans(k_row_true, 1, 0)) * fa_scale
    p = tl.math.exp2(qk - lse[:, None]) * beta[:, None]
    u_blk_ptr = tl.make_block_ptr(
        base=u_ptr + pid_b * stride_uh + (bos if IS_VARLEN else 0) * stride_ut,
        shape=(D, T_local),
        strides=(stride_ud, stride_ut),
        offsets=(0, pid_c * BLOCK_C),
        block_shape=(D, BLOCK_C),
        order=(0, 1),
    )
    ut = tl.load(u_blk_ptr)
    dp = tl.dot(grad_v_row, ut)
    dpm = dp - row_dot_row[:, None]
    mask = block_i[None, :] < block_i[:, None]
    p = tl.where(mask, p, 0.)
    dpm = tl.where(mask, dpm, 0.)
    da = p * dpm
    daat = da
    acc = tl.dot(daat.to(k_row.dtype), k_row_true, acc)

    grad_q_blk_ptr = tl.make_block_ptr(
        base=grad_q_ptr + pid_b * stride_grad_qh,
        shape=(T, D),
        strides=(stride_grad_qt, stride_grad_qd),
        offsets=(BLOCK_C * pid_c, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    acc = acc * qk_scale
    tl.store(grad_q_blk_ptr, acc.to(grad_q_ptr.dtype.element_ty))

    daat = tl.trans(da, 1, 0)
    acc = tl.dot(daat.to(k_row.dtype), k_row)
    k_row = k_row_true
    nu = -tl.trans(ut, 1, 0)
    for kv_i in range((pid_c + 1) * BLOCK_C, T_local, BLOCK_C):
        k_blk_ptr = tl.make_block_ptr(
            base=q_ptr + pid_b * stride_qh + (bos if IS_VARLEN else 0) * stride_qt,
            shape=(D, T_local),
            strides=(stride_qd, stride_qt),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_C),
            order=(0, 1),
        )
        kt = tl.load(k_blk_ptr)
        lse_blk_ptr = tl.make_block_ptr(
            base=lse_ptr + pid_b * stride_lse_h,
            shape=(T_local,),
            strides=(stride_lse_t,),
            offsets=(kv_i,),
            block_shape=(BLOCK_C,),
            order=(0,),
        )
        lse = tl.load(lse_blk_ptr)
        beta_blk_ptr = tl.make_block_ptr(
            base=beta_ptr + pid_b * stride_beta_h,
            shape=(T_local,),
            strides=(stride_beta_t,),
            offsets=(kv_i,),
            block_shape=(BLOCK_C,),
            order=(0,),
        )
        beta = tl.load(beta_blk_ptr)
        qk = tl.dot(k_row, kt) * fa_scale
        p = tl.math.exp2(qk - lse[None, :]) * beta[None, :]

        grad_vt_blk_ptr = tl.make_block_ptr(
            base=grad_v_ptr + pid_b * stride_grad_vh + (bos if IS_VARLEN else 0) * stride_grad_vt,
            shape=(D, T_local),
            strides=(stride_grad_vd, stride_grad_vt),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_C),
            order=(0, 1),
        )
        grad_vt = tl.load(grad_vt_blk_ptr)
        row_dot_blk_ptr = tl.make_block_ptr(
            base=row_dot_ptr + pid_b * stride_row_dot_h,
            shape=(T_local,),
            strides=(stride_row_dot_t,),
            offsets=(kv_i,),
            block_shape=(BLOCK_C,),
            order=(0,),
        )
        row_dot = tl.load(row_dot_blk_ptr).to(k_ptr.dtype.element_ty)
        dp = tl.dot(nu, grad_vt)
        da = p * (dp - row_dot[None, :])
        k = tl.trans(kt, 1, 0)
        acc = tl.dot(da.to(k.dtype), k, acc)

    grad_k_blk_ptr = tl.make_block_ptr(
        base=grad_k_ptr + pid_b * stride_grad_kh,
        shape=(T, D),
        strides=(stride_grad_kt, stride_grad_kd),
        offsets=(BLOCK_C * pid_c, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    acc = acc * qk_scale
    tl.store(grad_k_blk_ptr, acc.to(grad_k_ptr.dtype.element_ty))


class _DeltaPreAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qo: torch.Tensor,
        ko: torch.Tensor,
        vo: torch.Tensor,
        betao: Optional[torch.Tensor] = None,
        C: int = BLOCK_SIZE_C,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        C = min(C, ko.size(2))
        u, ws, lses = _DeltaPreAttnFunction._forward_impl(qo, ko, vo, betao, C, need_aux=True)
        BS, NH, T, _ = ko.size()
        saved_beta = betao if betao is not None else torch.ones(BS, NH, T, device=ko.device, dtype=ko.dtype)
        ctx.save_for_backward(qo, ko, vo, u, ws, lses, saved_beta)
        ctx.C = C
        ctx.beta_is_none = betao is None
        ctx.cu_seqlens = cu_seqlens
        return u

    @staticmethod
    def backward(
        ctx,
        grad_u: torch.Tensor
    ):
        qo, ko, vo, u, ws, lses, betao = ctx.saved_tensors
        C = ctx.C
        BS, NH, T_max, D = ko.size()
        cu_seqlens = ctx.cu_seqlens

        qk_scale = 1.0 / math.sqrt(D)
        fa_scale = qk_scale / math.log(2)

        grad_qo = torch.zeros_like(qo)
        grad_ko = torch.zeros_like(ko)
        grad_vo = torch.zeros_like(vo)
        grad_beta_full = torch.zeros_like(betao) if not ctx.beta_is_none else None

        q = qo.flatten(0, 1)
        k = ko.flatten(0, 1)
        v = vo.flatten(0, 1)
        beta = betao.flatten(0, 1)
        grad_o = grad_u.flatten(0, 1)

        cu_flat = None
        if cu_seqlens is not None:
            lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
            lens = lens.unsqueeze(1).repeat(1, NH).reshape(-1)
            cu_flat = torch.empty(lens.numel() + 1, dtype=torch.int32, device=cu_seqlens.device)
            cu_flat[0] = 0
            torch.cumsum(lens, dim=0, out=cu_flat[1:])

        grad_v = torch.empty_like(v)
        num_chunks = (T_max + C - 1) // C
        for chunk_idx in range(num_chunks - 1, -1, -1):
            i = chunk_idx * C
            cb = min(C, T_max - i)
            do = grad_o[:, i:i + cb, :]
            if i + cb < T_max:
                qi = k[:, i:i + cb, :]
                ki = q[:, i + cb:, :]
                lse_slice = lses[:, i + cb:]
                beta_single = beta[:, i + cb:]
                du = backward_u_chunk(
                    qi,
                    ki,
                    lse_slice,
                    grad_v[:, i + cb:, :],
                    fa_scale,
                    beta_single,
                    cu_seqlens=cu_flat,
                    row_start=i,
                )
                do = do - du
            w_block = ws[chunk_idx][:, :cb, :cb]
            du = invcum.solve_unit_upper_triangular_system(do, w_block)
            grad_v[:, i:i + cb, :].copy_(du)

        gq, gk, gbeta = backward_qk(
            q,
            k,
            u.flatten(0, 1),
            lses,
            grad_v,
            qk_scale,
            fa_scale,
            beta,
            cu_seqlens=cu_flat,
            row_start=0,
        )
        grad_qo.copy_(gq.view_as(qo))
        grad_ko.copy_(gk.view_as(ko))
        grad_vo.copy_(grad_v.view_as(vo))
        if not ctx.beta_is_none:
            grad_beta_full.copy_(gbeta.view_as(betao))

        grad_beta_out = None if ctx.beta_is_none else grad_beta_full
        return grad_qo, grad_ko, grad_vo, grad_beta_out, None, None

    @staticmethod
    def _forward_impl(
        qo: torch.Tensor,
        ko: torch.Tensor,
        vo: torch.Tensor,
        betao: Optional[torch.Tensor],
        C: int,
        need_aux: bool,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        BS, NH, T, D = ko.size()
        C = min(C, T)
        q = qo.flatten(0, 1)
        k = ko.flatten(0, 1)
        v = vo.flatten(0, 1)
        if betao is None:
            beta = torch.ones(BS, NH, T_max, device=k.device, dtype=k.dtype)
        else:
            beta = betao
        beta = beta.flatten(0, 1)

        u = torch.empty_like(v)
        qk_scale = 1.0 / math.sqrt(D)
        fa_scale = qk_scale / math.log(2)
        # ws = torch.empty(T // C, BS * NH, C, C, device=k.device, dtype=k.dtype) if need_aux else None
        ws = torch.empty((T + C - 1) // C, BS * NH, C, C, device=k.device, dtype=k.dtype) if need_aux else None
        lses = torch.empty(BS * NH, T, device=k.device, dtype=torch.float) if need_aux else None

        for chunk_idx in range(num_chunks):
            i = chunk_idx * C
            cb = min(C, T_max - i)
            qi = q[:, i:i + cb, :]
            vi = v[:, i:i + cb, :]
            betai = beta[:, i:i + cb]
            w, lse_chunk = forward(qi, k, vi, u_flat, fa_scale, betai, row_start=i, cu_seqlens=cu_flat)
            w = w * betai.unsqueeze(-1)
            if need_aux:
                ws[chunk_idx, :, :cb, :cb].copy_(w)
                lses[:, i:i + cb].copy_(lse_chunk)
            invcum.solve_unit_lower_triangular_system_inplace(u_flat[:, i:i + cb, :], w[:, :cb, :cb])

        u = u_flat.view(BS, NH, T_max, D)
        return u, ws, lses


def delta_pre_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    C: int = BLOCK_SIZE_C,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fixed-length DeltaFormer pre-attention. Computes u given q, k, v, beta.
    q,k,v: [B, H, T, D]
    beta: [B, H, T]
    Returns u with shape [B, H, T, D]
    """
    C = min(C, k.size(2))
    if k.requires_grad or q.requires_grad or v.requires_grad or (beta is not None and beta.requires_grad):
        return _DeltaPreAttnFunction.apply(q, k, v, beta, C, cu_seqlens)
    u, _, _ = _DeltaPreAttnFunction._forward_impl(q, k, v, beta, C, need_aux=False, cu_seqlens=cu_seqlens)
    return u


__all__ = [
    'delta_pre_attn',
]
