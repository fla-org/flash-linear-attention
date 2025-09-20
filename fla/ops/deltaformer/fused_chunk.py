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
    beta: torch.Tensor
):
    B, C, D = q.size()
    _B, T, _D = k.size()
    __B, __C = beta.size()
    assert B == _B and D == _D and B == __B and __C == C
    w = torch.empty(B, C, C, device=q.device, dtype=q.dtype)
    lse = torch.empty(B, C, device=q.device, dtype=torch.float)
    delta_flash_attn_compileable(q, k, v, u, w, lse, qk_scale, beta)
    return w, lse


def backward_u_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    grad_v: torch.Tensor,
    fa_scale: float,
    beta: torch.Tensor
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
        B, T, C, D, fa_scale
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
    beta: torch.Tensor
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
        fa_scale
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
        fa_scale, qk_scale
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
    beta: torch.Tensor
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
        B, T, C, D, qk_scale
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
):
    pid_c = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    rowid_block = tl.arange(0, BLOCK_C) + pid_c * BLOCK_C
    colid_block = tl.arange(0, BLOCK_T)

    rowmax = tl.zeros([BLOCK_C], dtype=tl.float32) - float('inf')
    rowsum = tl.zeros([BLOCK_C], dtype=tl.float32) + 1
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

    for kv_i in range(0, T, BLOCK_T):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_b * stride_kh,
            shape=(D, T),
            strides=(stride_kd, stride_kt),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        k = tl.load(k_blk_ptr, boundary_check=(1,))
        qk = tl.dot(q, k) * qk_scale

        if kv_i >= T - C:
            mask = (T - C - kv_i + rowid_block[:, None] - colid_block[None, :] < 1)
            qk = tl.where(mask, -1e6, qk)

        rowmax_i = tl.maximum(rowmax, tl.max(qk, axis=1))
        qk -= rowmax_i[:, None]
        p = tl.math.exp2(qk)

        rowsum_i = tl.sum(p, axis=1)
        alpha = tl.math.exp2(rowmax - rowmax_i)
        rowsum = rowsum * alpha + rowsum_i
        acc = acc * alpha[:, None]
        rowmax = rowmax_i

        if kv_i < T - C:
            u_blk_ptr = tl.make_block_ptr(
                base=u_ptr + pid_b * stride_uh,
                shape=(T, D),
                strides=(stride_ut, stride_ud),
                offsets=(kv_i, 0),
                block_shape=(BLOCK_T, D),
                order=(1, 0),
            )
            u = tl.load(u_blk_ptr, boundary_check=(0,))
            acc = tl.dot(p.to(u_ptr.dtype.element_ty), u, acc)

    lse = rowmax + tl.math.log2(rowsum)
    lse_block_ptr = lse_ptr + stride_lse_r * pid_b + rowid_block
    lse_mask = rowid_block < C
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

    beta_ptr = tl.make_block_ptr(
        base=beta_ptr + pid_b * beta_stride_r,
        shape=(C,),
        strides=(1,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,)
    )
    beta = tl.load(beta_ptr, boundary_check=(0,))
    acc = acc * beta[:, None]

    v = tl.load(v_ptr, boundary_check=(0,))
    u = v - acc.to(v_ptr.dtype.element_ty)
    u_block_ptr = tl.make_block_ptr(
        base=u_ptr + pid_b * stride_uh,
        shape=(T, D),
        strides=(stride_ut, stride_ud),
        offsets=(T - C + pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    tl.store(u_block_ptr, u, boundary_check=(0, 1))

    for kv_i in range(T - C, T, BLOCK_T):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_b * stride_kh,
            shape=(D, T),
            strides=(stride_kd, stride_kt),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        k = tl.load(k_blk_ptr, boundary_check=(1,))
        qk = tl.dot(q, k) * qk_scale

        mask = (T - C - kv_i + rowid_block[:, None] - colid_block[None, :] < 1)
        qk -= rowmax[:, None]
        p = tl.math.exp2(qk) / rowsum[:, None]
        p = tl.where(mask, 0, p)
        w_blk_ptr = tl.make_block_ptr(
            base=w_ptr + pid_b * stride_wh,
            shape=(C, C),
            strides=(stride_wc, 1),
            offsets=(pid_c * BLOCK_C, kv_i - (T - C)),
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
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

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

    for kv_i in range(0, T, BLOCK_T):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_b * stride_kh,
            shape=(D, T),
            strides=(stride_kd, stride_kt),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        k = tl.load(k_blk_ptr)
        qk = tl.dot(q, k) * fa_scale

        lse_blk_ptr = tl.make_block_ptr(
            base=lse_ptr + pid_b * stride_lse_h,
            shape=(T,),
            strides=(stride_lse_t,),
            offsets=(kv_i,),
            block_shape=(BLOCK_T,),
            order=(0,),
        )
        lse = tl.load(lse_blk_ptr)
        beta_blk_ptr = tl.make_block_ptr(
            base=beta_ptr + pid_b * stride_beta_h,
            shape=(T,),
            strides=(stride_beta_t,),
            offsets=(kv_i,),
            block_shape=(BLOCK_T,),
            order=(0,),
        )
        beta = tl.load(beta_blk_ptr)

        p = tl.math.exp2(qk - lse[None, :]) * beta[None, :]

        v_blk_ptr = tl.make_block_ptr(
            base=v_ptr + pid_b * stride_vh,
            shape=(T, D),
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
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    rowid_block = tl.arange(0, BLOCK_C) + pid_c * BLOCK_C
    colid_block = tl.arange(0, BLOCK_T)

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    k_row_blk_ptr = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qh,
        shape=(T, D),
        strides=(stride_qt, stride_qd),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    k_row = tl.load(k_row_blk_ptr)
    lse_blk_ptr = tl.make_block_ptr(
        base=lse_ptr + pid_b * stride_lse_h,
        shape=(T,),
        strides=(stride_lse_t,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    lse = tl.load(lse_blk_ptr)
    grad_v_blk_ptr = tl.make_block_ptr(
        base=grad_v_ptr + pid_b * stride_grad_vh,
        shape=(T, D),
        strides=(stride_grad_vt, stride_grad_vd),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    grad_v_row = -tl.load(grad_v_blk_ptr)

    for kv_i in range(0, (pid_c + 1) * BLOCK_C, BLOCK_T):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_b * stride_kh,
            shape=(D, T),
            strides=(stride_kd, stride_kt),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        k = tl.load(k_blk_ptr)
        qk = tl.dot(k_row, k) * fa_scale
        p = tl.math.exp2(qk - lse[:, None])

        u_blk_ptr = tl.make_block_ptr(
            base=u_ptr + pid_b * stride_uh,
            shape=(D, T),
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
        shape=(T,),
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
):
    pid_c = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    block_i = tl.arange(0, BLOCK_C)

    acc = tl.zeros([BLOCK_C, D], dtype=tl.float32)

    k_row_blk_ptr = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qh,
        shape=(T, D),
        strides=(stride_qt, stride_qd),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    k_row = tl.load(k_row_blk_ptr)
    lse_blk_ptr = tl.make_block_ptr(
        base=lse_ptr + pid_b * stride_lse_h,
        shape=(T,),
        strides=(stride_lse_t,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    lse = tl.load(lse_blk_ptr)
    beta_blk_ptr = tl.make_block_ptr(
        base=beta_ptr + pid_b * stride_beta_h,
        shape=(T,),
        strides=(stride_beta_t,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    beta = tl.load(beta_blk_ptr)
    grad_v_blk_ptr = tl.make_block_ptr(
        base=grad_v_ptr + pid_b * stride_grad_vh,
        shape=(T, D),
        strides=(stride_grad_vt, stride_grad_vd),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    grad_v_row = -tl.load(grad_v_blk_ptr)
    row_dot_blk_ptr = tl.make_block_ptr(
        base=row_dot_ptr + pid_b * stride_row_dot_h,
        shape=(T,),
        strides=(stride_row_dot_t,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    row_dot_row = tl.load(row_dot_blk_ptr).to(k_ptr.dtype.element_ty)

    for kv_i in range(0, pid_c * BLOCK_C, BLOCK_C):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_b * stride_kh,
            shape=(D, T),
            strides=(stride_kd, stride_kt),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_C),
            order=(0, 1),
        )
        kt = tl.load(k_blk_ptr)
        qk = tl.dot(k_row, kt) * fa_scale
        p = tl.math.exp2(qk - lse[:, None]) * beta[:, None]

        u_blk_ptr = tl.make_block_ptr(
            base=u_ptr + pid_b * stride_uh,
            shape=(D, T),
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
        base=k_ptr + pid_b * stride_kh,
        shape=(T, D),
        strides=(stride_kt, stride_kd),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    k_row_true = tl.load(k_row_blk_ptr)
    qk = tl.dot(k_row, tl.trans(k_row_true, 1, 0)) * fa_scale
    p = tl.math.exp2(qk - lse[:, None]) * beta[:, None]
    u_blk_ptr = tl.make_block_ptr(
        base=u_ptr + pid_b * stride_uh,
        shape=(D, T),
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
    for kv_i in range((pid_c + 1) * BLOCK_C, T, BLOCK_C):
        k_blk_ptr = tl.make_block_ptr(
            base=q_ptr + pid_b * stride_qh,
            shape=(D, T),
            strides=(stride_qd, stride_qt),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_C),
            order=(0, 1),
        )
        kt = tl.load(k_blk_ptr)
        lse_blk_ptr = tl.make_block_ptr(
            base=lse_ptr + pid_b * stride_lse_h,
            shape=(T,),
            strides=(stride_lse_t,),
            offsets=(kv_i,),
            block_shape=(BLOCK_C,),
            order=(0,),
        )
        lse = tl.load(lse_blk_ptr)
        beta_blk_ptr = tl.make_block_ptr(
            base=beta_ptr + pid_b * stride_beta_h,
            shape=(T,),
            strides=(stride_beta_t,),
            offsets=(kv_i,),
            block_shape=(BLOCK_C,),
            order=(0,),
        )
        beta = tl.load(beta_blk_ptr)
        qk = tl.dot(k_row, kt) * fa_scale
        p = tl.math.exp2(qk - lse[None, :]) * beta[None, :]

        grad_vt_blk_ptr = tl.make_block_ptr(
            base=grad_v_ptr + pid_b * stride_grad_vh,
            shape=(D, T),
            strides=(stride_grad_vd, stride_grad_vt),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_C),
            order=(0, 1),
        )
        grad_vt = tl.load(grad_vt_blk_ptr)
        row_dot_blk_ptr = tl.make_block_ptr(
            base=row_dot_ptr + pid_b * stride_row_dot_h,
            shape=(T,),
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
        cu_seqlens: Optional[torch.LongTensor] = None
    ):
        C = min(C, ko.size(2))
        ctx.C = C
        ctx.cu_seqlens = cu_seqlens

        if cu_seqlens is not None:
            need_aux = qo.requires_grad or ko.requires_grad or vo.requires_grad or (betao is not None and betao.requires_grad)
            u, ws, lses = _DeltaPreAttnFunction._forward_impl(qo, ko, vo, betao, C, need_aux=need_aux, cu_seqlens=cu_seqlens)
            BS, NH, T_max, _ = ko.size()
            saved_beta = betao if betao is not None else torch.ones(BS, NH, T_max, device=ko.device, dtype=ko.dtype)
            ctx.beta_is_none = betao is None
            if need_aux:
                ctx.save_for_backward(qo, ko, vo, u, ws, lses, saved_beta)
            else:
                ctx.save_for_backward()
            return u

        u, ws, lses = _DeltaPreAttnFunction._forward_impl(qo, ko, vo, betao, C, need_aux=True)
        BS, NH, T, _ = ko.size()
        saved_beta = betao if betao is not None else torch.ones(BS, NH, T, device=ko.device, dtype=ko.dtype)
        ctx.save_for_backward(qo, ko, vo, u, ws, lses, saved_beta)
        ctx.beta_is_none = betao is None
        return u

    @staticmethod
    def backward(
        ctx,
        grad_u: torch.Tensor
    ):
        if getattr(ctx, 'cu_seqlens', None) is not None:
            cu = ctx.cu_seqlens
            qo, ko, vo, u_full, ws, lses, betao = ctx.saved_tensors
            BS, NH, T_max, D = ko.size()
            qk_scale = 1.0 / math.sqrt(D)
            fa_scale = qk_scale / math.log(2)

            dq = torch.zeros_like(qo)
            dk = torch.zeros_like(ko)
            dv = torch.zeros_like(vo)
            dbeta = None if ctx.beta_is_none else torch.zeros_like(betao)

            C = ctx.C
            N = len(cu) - 1
            chunk_bases = []
            total = 0
            lengths = []
            for b in range(N):
                L = int(cu[b + 1].item() - cu[b].item())
                lengths.append(L)
                chunk_bases.append(total)
                if L > 0:
                    total += (L + C - 1) // C

            for b in range(N):
                L = lengths[b]
                if L == 0:
                    continue
                base = chunk_bases[b]
                seq_start = int(cu[b].item())

                seq_end = seq_start + L
                q_seq = qo[0, :, seq_start:seq_end, :]
                k_seq = ko[0, :, seq_start:seq_end, :]
                u_seq = u_full[0, :, seq_start:seq_end, :]
                beta_seq = betao[0, :, seq_start:seq_end]
                lse_seq = lses[0, :, seq_start:seq_end]
                go_seq = grad_u[0, :, seq_start:seq_end, :]

                gv_seq = torch.zeros_like(u_seq)
                start = ((L - 1) // C) * C
                for i_local in range(start, -1, -C):
                    Ci = min(C, L - i_local)
                    i0 = i_local
                    i1 = i_local + Ci
                    do = go_seq[:, i0:i1, :]
                    if i_local < L - C:
                        qi = k_seq[:, i0:i1, :]
                        ki = q_seq[:, i1:L, :]
                        lse_tail = lse_seq[:, i1:L]
                        beta_tail = beta_seq[:, i1:L]
                        du_tail = backward_u_chunk(qi, ki, lse_tail, gv_seq[:, i1:L, :], fa_scale, beta_tail)
                        do = do - du_tail
                    Wpad = ws[base + (i_local // C)]
                    W = Wpad[:, :Ci, :Ci]
                    du_chunk = invcum.backward_x(do, W)
                    gv_seq[:, i0:i1, :].copy_(du_chunk)

                gq, gk, gbeta = backward_qk(q_seq, k_seq, u_seq, lse_seq, gv_seq, qk_scale, fa_scale, beta_seq)
                dq[0, :, seq_start:seq_end, :].copy_(gq)
                dk[0, :, seq_start:seq_end, :].copy_(gk)
                dv[0, :, seq_start:seq_end, :].copy_(gv_seq)
                if dbeta is not None:
                    dbeta[0, :, seq_start:seq_end].copy_(gbeta)

            return dq, dk, dv, dbeta, None, None
        qo, ko, vo, u, ws, lses, betao = ctx.saved_tensors
        C = ctx.C
        BS, NH, T, D = ko.size()

        grad_q = torch.zeros_like(qo)
        grad_k = torch.zeros_like(ko)
        grad_v = torch.zeros_like(vo)
        grad_beta_out = None if ctx.beta_is_none else torch.zeros_like(betao)

        qk_scale = 1.0 / math.sqrt(D)
        fa_scale = qk_scale / math.log(2)

        chunk_base = 0
        for b in range(BS):
            grad_v_seq = torch.empty(NH, T, D, device=ko.device, dtype=ko.dtype)
            for i in range(T - C, -1, -C):
                Ci = min(C, T - i)
                do = grad_u[b, :, i:i + Ci, :]

                if i < T - C:
                    qi = ko[b, :, i:i + Ci, :]
                    ki = qo[b, :, i + Ci:, :]
                    lse = lses[b, :, i + Ci:]
                    if not ctx.beta_is_none:
                        beta_single = betao[b, :, i + Ci:]
                    else:
                        beta_single = torch.ones(NH, T - i - Ci, device=ko.device, dtype=ko.dtype)
                    du = backward_u_chunk(qi, ki, lse, grad_v_seq[:, i + Ci:, :], fa_scale, beta_single)
                    do = grad_u[b, :, i:i + Ci, :] - du

                du = invcum.backward_x(do, ws[chunk_base + (i // C)])
                grad_v_seq[:, i:i + Ci, :].copy_(du)

            q_seq = qo[b]
            k_seq = ko[b]
            u_seq = u[b]
            lse_seq = lses[b]
            beta_seq = betao[b] if not ctx.beta_is_none else torch.ones(NH, T, device=ko.device, dtype=ko.dtype)

            gq, gk, gbeta = backward_qk(q_seq, k_seq, u_seq, lse_seq, grad_v_seq, qk_scale, fa_scale, beta_seq)

            grad_q[b].copy_(gq)
            grad_k[b].copy_(gk)
            grad_v[b].copy_(grad_v_seq)
            if not ctx.beta_is_none:
                grad_beta_out[b].copy_(gbeta)

            chunk_base += (T + C - 1) // C

        return grad_q, grad_k, grad_v, grad_beta_out, None, None

    @staticmethod
    def _forward_impl(
        qo: torch.Tensor,
        ko: torch.Tensor,
        vo: torch.Tensor,
        betao: Optional[torch.Tensor],
        C: int,
        need_aux: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        BS, NH, T_max, D = ko.size()
        C = min(C, T_max)
        qk_scale = 1.0 / math.sqrt(D)
        fa_scale = qk_scale / math.log(2)

        if cu_seqlens is None:
            if betao is None:
                beta_full = torch.ones(BS, NH, T_max, device=ko.device, dtype=ko.dtype)
            else:
                beta_full = betao

            u_full = torch.empty_like(vo)
            if need_aux:
                total_chunks = BS * ((T_max + C - 1) // C)
                ws = torch.empty(total_chunks, NH, C, C, device=ko.device, dtype=ko.dtype)
                lses = torch.empty(BS, NH, T_max, device=ko.device, dtype=torch.float)
                chunk_base = 0
            else:
                ws = None
                lses = None
                chunk_base = 0

            for b in range(BS):
                for i in range(0, T_max, C):
                    Ci = min(C, T_max - i)

                    qi = qo[b, :, i:i + Ci, :]
                    ki = ko[b, :, :i + Ci, :]
                    vi = vo[b, :, i:i + Ci, :]
                    ui_prev = u_full[b, :, :i + Ci, :]
                    betai = beta_full[b, :, i:i + Ci]

                    w, lse_chunk = forward(qi, ki, vi, ui_prev, fa_scale, betai)
                    w = w * betai.unsqueeze(-1)
                    if need_aux:
                        wpad = torch.zeros(NH, C, C, device=ko.device, dtype=ko.dtype)
                        wpad[:, :Ci, :Ci].copy_(w)
                        ws[chunk_base + (i // C)].copy_(wpad)
                        lses[b, :, i:i + Ci].copy_(lse_chunk)

                    u_chunk_view = u_full[b, :, i:i + Ci, :]
                    invcum.forward_inplace(u_chunk_view, w)

                chunk_base += (T_max + C - 1) // C

            return u_full, ws, lses

        # Varlen path
        N = len(cu_seqlens) - 1
        assert cu_seqlens.dim() == 1 and cu_seqlens.size(0) == N + 1, "cu_seqlens must be [N+1]"
        device = ko.device
        dtype_k = ko.dtype
        if betao is None:
            beta_full = torch.ones(BS, NH, T_max, device=device, dtype=dtype_k)
        else:
            beta_full = betao

        u_full = torch.empty_like(vo)
        if need_aux:
            total_chunks = sum((max(0, int(cu_seqlens[b + 1].item() - cu_seqlens[b].item())) + C - 1) // C
                               for b in range(N))
            ws = torch.empty(total_chunks, NH, C, C, device=device, dtype=dtype_k)
            lses = torch.empty(BS, NH, T_max, device=device, dtype=torch.float)
            chunk_base = 0
        else:
            ws = None
            lses = None
            chunk_base = 0

        for b in range(N):
            seq_start = int(cu_seqlens[b].item())
            seq_end = int(cu_seqlens[b + 1].item())
            L = max(0, seq_end - seq_start)
            if L == 0:
                continue

            for i_local in range(0, L, C):
                Ci = min(C, L - i_local)
                li0 = i_local
                li1 = i_local + Ci

                abs_start = seq_start + li0
                abs_end = seq_start + li1
                abs_context_end = seq_start + li1

                qi = qo[0, :, abs_start:abs_end, :]
                ki = ko[0, :, seq_start:abs_context_end, :]
                vi = vo[0, :, abs_start:abs_end, :]
                ui_prev = u_full[0, :, seq_start:abs_context_end, :]
                betai = beta_full[0, :, abs_start:abs_end]

                w, lse_chunk = forward(qi, ki, vi, ui_prev, fa_scale, betai)
                w = w * betai.unsqueeze(-1)
                if need_aux:
                    wpad = torch.zeros(NH, C, C, device=device, dtype=dtype_k)
                    wpad[:, :Ci, :Ci].copy_(w)
                    ws[chunk_base + (i_local // C)].copy_(wpad)
                    lses[0, :, abs_start:abs_end].copy_(lse_chunk)

                u_chunk_view = u_full[0, :, abs_start:abs_end, :]
                invcum.forward_inplace(u_chunk_view, w)

            chunk_base += (L + C - 1) // C

        return u_full, ws, lses


def delta_pre_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    C: int = BLOCK_SIZE_C,
    cu_seqlens: Optional[torch.LongTensor] = None
) -> torch.Tensor:
    """
    Fixed-length and varlen DeltaFormer pre-attention. Computes u given q, k, v, beta.

    Fixed-length mode (cu_seqlens=None):
        - q,k,v: [B, H, T, D]
        - beta: [B, H, T]
        - Returns: u with shape [B, H, T, D]

    Varlen mode (cu_seqlens provided):
        - q,k,v: [B, H, T_max, D] with padding on the right
        - beta: [B, H, T_max] (or None, treated as ones)
        - cu_seqlens: [B+1] cumulative true sequence lengths
        - Returns: u with shape [B, H, T_max, D]; padded positions remain unused
    """
    C = min(C, k.size(2))
    if k.requires_grad or q.requires_grad or v.requires_grad or (beta is not None and beta.requires_grad):
        return _DeltaPreAttnFunction.apply(q, k, v, beta, C, cu_seqlens)
    u, _, _ = _DeltaPreAttnFunction._forward_impl(q, k, v, beta, C, need_aux=False, cu_seqlens=cu_seqlens)
    return u


__all__ = [
    'delta_pre_attn',
]
