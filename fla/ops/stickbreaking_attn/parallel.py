# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import math
from typing import Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices

ALLOW_TF32 = True
inv_log2 = 1.0 / math.log(2.0)


def _get_configs():
    return [triton.Config({}, num_stages=s, num_warps=w) for s in [4] for w in [4]]


@triton.autotune(configs=_get_configs(), key=["token_size", "head_size"])
@triton.jit
def stickbreaking_attn_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    R_ptr,
    A_ptr,
    CU_ptr,
    CI_ptr,
    logit_scale: tl.constexpr,
    attend_current: tl.constexpr,
    batch_size,
    token_size,
    head_size: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    inv_log2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    no_grad: tl.constexpr = False,
    acc_dtype: tl.constexpr = tl.float32,
    is_compiling: tl.constexpr = False,
    IS_VARLEN: tl.constexpr = False,
):
    tl.static_assert(BLOCK_M % BLOCK_N == 0)
    batch_id = 0 if IS_VARLEN else tl.program_id(0)
    head_pid = tl.program_id(1)
    prog_id = tl.program_id(2)
    tl.num_programs(2)
    if IS_VARLEN:
        i_n = tl.load(CI_ptr + prog_id * 2).to(tl.int32)
        seq_block_id = tl.load(CI_ptr + prog_id * 2 + 1).to(tl.int32)
        bos = tl.load(CU_ptr + i_n).to(tl.int32)
        eos = tl.load(CU_ptr + i_n + 1).to(tl.int32)
        seq_length = eos - bos
    else:
        bos = 0
        seq_block_id = prog_id
        seq_length = token_size
    qk_scale = inv_log2 * logit_scale
    M_range = tl.arange(0, BLOCK_M)
    N_range = tl.arange(0, BLOCK_N)
    D_range = tl.arange(0, BLOCK_D)
    D_mask = D_range < head_size
    cm = tl.where(N_range[:, None] >= N_range[None, :], 1.0, 0.0).to(Q_ptr.type.element_ty)

    head_id = head_pid
    seq_prog_id = seq_block_id
    batch_offset = batch_id * token_size
    Q_head_seq_ptr = Q_ptr + ((batch_offset + bos) * num_heads + head_id) * head_size
    K_head_seq_ptr = K_ptr + ((batch_offset + bos) * num_heads + head_id) * head_size
    V_head_seq_ptr = V_ptr + ((batch_offset + bos) * num_heads + head_id) * head_size
    O_head_seq_ptr = O_ptr + ((batch_offset + bos) * num_heads + head_id) * head_size
    R_head_seq_ptr = R_ptr + ((batch_offset + bos) * num_heads + head_id)
    A_head_seq_ptr = A_ptr + ((batch_offset + bos) * num_heads + head_id)

    stickbreaking_attn_fwd_one_row_kernel(
        seq_prog_id,
        seq_length,
        qk_scale,
        M_range,
        N_range,
        D_range,
        D_mask,
        cm,
        Q_head_seq_ptr,
        K_head_seq_ptr,
        V_head_seq_ptr,
        O_head_seq_ptr,
        R_head_seq_ptr,
        A_head_seq_ptr,
        head_size,
        num_heads,
        BLOCK_D,
        NO_D_MASK,
        NO_M_MASK,
        NO_N_MASK,
        ALLOW_TF32,
        BLOCK_M,
        BLOCK_N,
        no_grad,
        acc_dtype,
        False,
        attend_current=attend_current,
        is_compiling=is_compiling,
    )


@triton.jit
def stickbreaking_attn_fwd_one_row_kernel(
    seq_block_id,
    seq_length,
    qk_scale,
    M_range,
    N_range,
    D_range,
    D_mask,
    cm,
    Q_head_seq_ptr,
    K_head_seq_ptr,
    V_head_seq_ptr,
    O_head_seq_ptr,
    R_head_seq_ptr,
    A_head_seq_ptr,
    head_size: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    no_grad: tl.constexpr = False,
    acc_dtype: tl.constexpr = tl.float32,
    return_attention: tl.constexpr = False,
    attend_current: tl.constexpr = False,
    is_compiling: tl.constexpr = False,
):
    block_start_offset = BLOCK_M * seq_block_id
    M_blk_idxs = block_start_offset + M_range
    M_mask = M_blk_idxs < seq_length
    N_blk_idxs_start = block_start_offset + BLOCK_M
    N_blk_idxs = N_blk_idxs_start + N_range

    Q_blk_ptrs = Q_head_seq_ptr + (
        (num_heads * head_size) * M_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    K_blk_ptrs = K_head_seq_ptr + (
        (num_heads * head_size) * N_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    V_blk_ptrs = V_head_seq_ptr + (
        (num_heads * head_size) * N_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    O_blk_ptrs = O_head_seq_ptr + (
        (num_heads * head_size) * M_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    R_blk_ptrs = R_head_seq_ptr + num_heads * M_blk_idxs
    A_blk_ptrs = A_head_seq_ptr + num_heads * M_blk_idxs

    if NO_D_MASK:
        if NO_M_MASK:
            q = tl.load(Q_blk_ptrs)
        else:
            q = tl.load(Q_blk_ptrs, mask=M_mask[:, None], other=0.0)
    else:
        q = tl.load(Q_blk_ptrs, mask=M_mask[:, None] & D_mask[None, :], other=0.0)

    iters = N_blk_idxs_start // BLOCK_N
    neg_log_acc = tl.zeros([BLOCK_M], dtype=acc_dtype)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=acc_dtype)

    for i in range(iters):
        N_blk_idxs -= BLOCK_N
        N_blk_idxs_start -= BLOCK_N
        K_blk_ptrs -= BLOCK_N * (num_heads * head_size)
        V_blk_ptrs -= BLOCK_N * (num_heads * head_size)

        N_mask = N_blk_idxs < seq_length
        k, v = load_kv(
            K_blk_ptrs,
            V_blk_ptrs,
            N_mask=N_mask,
            NO_N_MASK=N_blk_idxs_start + BLOCK_N - 1 < seq_length,
            D_mask=D_mask,
            NO_D_MASK=NO_D_MASK,
        )
        on_band = i < BLOCK_M // BLOCK_N
        p, _log_om_beta, neg_log_acc = compute_block(
            q,
            k,
            qk_scale,
            neg_log_acc,
            M_blk_idxs,
            N_blk_idxs,
            cm,
            on_band,
            ALLOW_TF32,
            backward=False,
            attend_current=attend_current,
            is_compiling=is_compiling,
            use_cumsum=False,
        )
        acc = tl.dot(p.to(v.dtype), v, acc, allow_tf32=ALLOW_TF32)

    if NO_M_MASK:
        tl.store(R_blk_ptrs, tl.math.exp2(neg_log_acc))
        tl.store(A_blk_ptrs, neg_log_acc.to(A_head_seq_ptr.type.element_ty))
    else:
        tl.store(R_blk_ptrs, tl.math.exp2(neg_log_acc), mask=M_mask)
        tl.store(A_blk_ptrs, neg_log_acc.to(A_head_seq_ptr.type.element_ty), mask=M_mask)
    if NO_D_MASK:
        tl.store(O_blk_ptrs, acc.to(O_head_seq_ptr.type.element_ty), mask=M_mask[:, None])
    else:
        tl.store(O_blk_ptrs, acc.to(O_head_seq_ptr.type.element_ty), mask=M_mask[:, None] & D_mask[None, :])


def _get_bwd_configs():
    return [triton.Config({}, num_stages=s, num_warps=w) for s in [8] for w in [4]]


@triton.autotune(configs=_get_bwd_configs(), key=["token_size", "head_size"])
@triton.jit()
def stickbreaking_attn_bwd_kernel(
    DO_ptr,
    DR_ptr,
    A_ptr,
    Q_ptr,
    K_ptr,
    V_ptr,
    DQ_ptr,
    DK_ptr,
    DV_ptr,
    CU_ptr,
    CI_ptr,
    logit_scale,
    batch_size,
    token_size,
    head_size: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    inv_log2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    acc_dtype: tl.constexpr = tl.float32,
    is_compiling: tl.constexpr = False,
    attend_current: tl.constexpr = False,
    IS_VARLEN: tl.constexpr = False,
):
    tl.static_assert(BLOCK_M % BLOCK_N == 0)
    batch_id = 0 if IS_VARLEN else tl.program_id(0)
    head_pid = tl.program_id(1)
    prog_id = tl.program_id(2)
    qk_scale = inv_log2 * logit_scale
    M_range = tl.arange(0, BLOCK_M)
    N_range = tl.arange(0, BLOCK_N)
    D_range = tl.arange(0, BLOCK_D)
    D_mask = D_range < head_size
    cm = tl.where(N_range[:, None] >= N_range[None, :], 1.0, 0.0).to(Q_ptr.type.element_ty)

    if IS_VARLEN:
        i_n = tl.load(CI_ptr + prog_id * 2).to(tl.int32)
        seq_block_id = tl.load(CI_ptr + prog_id * 2 + 1).to(tl.int32)
        bos = tl.load(CU_ptr + i_n).to(tl.int32)
        eos = tl.load(CU_ptr + i_n + 1).to(tl.int32)
        seq_length = eos - bos
    else:
        bos = 0
        seq_block_id = prog_id
        seq_length = token_size

    head_id = head_pid
    seq_prog_id = seq_block_id

    batch_offset = batch_id * token_size
    DO_head_seq_ptr = DO_ptr + ((batch_offset + bos) * num_heads + head_id) * head_size
    DR_head_seq_ptr = DR_ptr + ((batch_offset + bos) * num_heads + head_id)
    A_head_seq_ptr = A_ptr + ((batch_offset + bos) * num_heads + head_id)
    Q_head_seq_ptr = Q_ptr + ((batch_offset + bos) * num_heads + head_id) * head_size
    K_head_seq_ptr = K_ptr + ((batch_offset + bos) * num_heads + head_id) * head_size
    V_head_seq_ptr = V_ptr + ((batch_offset + bos) * num_heads + head_id) * head_size
    DQ_head_seq_ptr = DQ_ptr + ((batch_offset + bos) * num_heads + head_id) * head_size
    DK_head_seq_ptr = DK_ptr + (
        seq_prog_id * batch_size * token_size * num_heads + (batch_offset + bos) * num_heads + head_id
    ) * head_size
    DV_head_seq_ptr = DV_ptr + (
        seq_prog_id * batch_size * token_size * num_heads + (batch_offset + bos) * num_heads + head_id
    ) * head_size

    stickbreaking_attn_bwd_one_row_kernel(
        seq_prog_id,
        seq_length,
        qk_scale,
        M_range,
        N_range,
        D_range,
        D_mask,
        cm,
        DO_head_seq_ptr,
        DR_head_seq_ptr,
        A_head_seq_ptr,
        Q_head_seq_ptr,
        K_head_seq_ptr,
        V_head_seq_ptr,
        DQ_head_seq_ptr,
        DK_head_seq_ptr,
        DV_head_seq_ptr,
        logit_scale,
        head_size,
        num_heads,
        BLOCK_D,
        NO_D_MASK,
        NO_M_MASK,
        NO_N_MASK,
        ALLOW_TF32,
        BLOCK_M,
        BLOCK_N,
        acc_dtype,
        is_compiling=is_compiling,
        attend_current=attend_current,
    )


@triton.jit
def load_kv(K_blk_ptrs, V_blk_ptrs, N_mask, NO_N_MASK, D_mask, NO_D_MASK: tl.constexpr):
    if NO_D_MASK:
        if NO_N_MASK:
            k = tl.load(K_blk_ptrs)
            v = tl.load(V_blk_ptrs)
        else:
            k = tl.load(K_blk_ptrs, mask=N_mask[:, None])
            v = tl.load(V_blk_ptrs, mask=N_mask[:, None])
    else:
        mask = N_mask[:, None] & D_mask[None, :]
        k = tl.load(K_blk_ptrs, mask=mask)
        v = tl.load(V_blk_ptrs, mask=mask)
    return k, v


@triton.jit
def compute_block(
    q,
    k,
    qk_scale,
    neg_log_acc,
    M_blk_idxs,
    N_blk_idxs,
    cm,
    on_band: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    backward: tl.constexpr,
    attend_current: tl.constexpr = False,
    use_cumsum: tl.constexpr = False,
    is_compiling: tl.constexpr = False,
):
    qk = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * qk_scale
    log_om_beta = -softplus(qk, is_compiling=is_compiling)

    if on_band:
        if attend_current:
            block_mask = M_blk_idxs[:, None] >= N_blk_idxs[None, :]
        else:
            block_mask = M_blk_idxs[:, None] > N_blk_idxs[None, :]
        log_om_beta = tl.where(block_mask, log_om_beta, 0.0)
        if backward:
            neg_log_acc -= tl.sum(log_om_beta, axis=1)
        log_p = qk + neg_log_acc[:, None]
        if use_cumsum:
            log_p += tl.cumsum(log_om_beta.to(q.dtype), axis=1, reverse=True)
        else:
            log_p = tl.dot(log_om_beta.to(q.dtype), cm, acc=log_p, allow_tf32=ALLOW_TF32)
        p = tl.math.exp2(log_p)
        p = tl.where(block_mask, p, 0.0)
    else:
        if backward:
            neg_log_acc -= tl.sum(log_om_beta, axis=1)
        log_p = qk + neg_log_acc[:, None]
        if use_cumsum:
            log_p += tl.cumsum(log_om_beta.to(q.dtype), axis=1, reverse=True)
        else:
            log_p = tl.dot(log_om_beta.to(q.dtype), cm, acc=log_p, allow_tf32=ALLOW_TF32)
        p = tl.math.exp2(log_p)
    if not backward:
        neg_log_acc += tl.sum(log_om_beta, axis=1)
    return p, log_om_beta, neg_log_acc


@triton.jit
def softplus(x, is_compiling: tl.constexpr = False):
    return tl.where(x < 15.0, tl.math.log2(1 + tl.math.exp2(x)), x)


@triton.jit
def stickbreaking_attn_bwd_one_row_kernel(
    seq_prog_id,
    seq_length,
    qk_scale,
    M_range,
    N_range,
    D_range,
    D_mask,
    cm,
    DO_head_seq_ptr,
    DR_head_seq_ptr,
    A_head_seq_ptr,
    Q_head_seq_ptr,
    K_head_seq_ptr,
    V_head_seq_ptr,
    DQ_head_seq_ptr,
    DK_head_seq_ptr,
    DV_head_seq_ptr,
    logit_scale,
    head_size: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    acc_dtype: tl.constexpr = tl.float32,
    is_compiling: tl.constexpr = False,
    attend_current: tl.constexpr = False,
):
    block_start_offset = BLOCK_M * seq_prog_id
    M_blk_idxs = block_start_offset + M_range
    M_mask = M_blk_idxs < seq_length

    N_blk_idxs_start = 0
    N_blk_idxs = N_blk_idxs_start + N_range

    DO_blk_ptrs = DO_head_seq_ptr + (
        (num_heads * head_size) * M_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    K_blk_ptrs = K_head_seq_ptr + (
        (num_heads * head_size) * N_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    Q_blk_ptrs = Q_head_seq_ptr + (
        (num_heads * head_size) * M_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    V_blk_ptrs = V_head_seq_ptr + (
        (num_heads * head_size) * N_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    A_blk_ptrs = A_head_seq_ptr + num_heads * M_blk_idxs
    DQ_blk_ptrs = DQ_head_seq_ptr + (
        (num_heads * head_size) * M_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    DK_blk_ptrs = DK_head_seq_ptr + (
        (num_heads * head_size) * N_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    DV_blk_ptrs = DV_head_seq_ptr + (
        (num_heads * head_size) * N_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    DR_blk_ptrs = DR_head_seq_ptr + num_heads * M_blk_idxs

    if NO_D_MASK:
        if NO_N_MASK:
            q = tl.load(Q_blk_ptrs)
            do = tl.load(DO_blk_ptrs)
            dr = tl.load(DR_blk_ptrs)
            neg_log_acc = tl.load(A_blk_ptrs, mask=M_mask)
        else:
            q = tl.load(Q_blk_ptrs, mask=M_mask[:, None])
            do = tl.load(DO_blk_ptrs, mask=M_mask[:, None])
            dr = tl.load(DR_blk_ptrs, mask=M_mask)
            neg_log_acc = tl.load(A_blk_ptrs, mask=M_mask)
    else:
        MD_mask = M_mask[:, None] & D_mask[None, :]
        q = tl.load(Q_blk_ptrs, mask=MD_mask)
        do = tl.load(DO_blk_ptrs, mask=MD_mask)
        dr = tl.load(DR_blk_ptrs, mask=M_mask)
        neg_log_acc = tl.load(A_blk_ptrs, mask=M_mask)

    neg_log_acc = neg_log_acc.to(dtype=acc_dtype)
    grad_prev_acc = tl.zeros((BLOCK_M,), dtype=acc_dtype)
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=acc_dtype)

    fwd_cm = tl.trans(cm)
    iters = (block_start_offset + BLOCK_M) // BLOCK_N
    for i in range(iters):
        on_band = (iters - i - 1) < BLOCK_M // BLOCK_N
        N_mask = N_blk_idxs < seq_length
        local_no_n_mask = (N_blk_idxs_start + BLOCK_N - 1) < seq_length
        k, v = load_kv(
            K_blk_ptrs,
            V_blk_ptrs,
            N_mask=N_mask,
            NO_N_MASK=local_no_n_mask,
            D_mask=D_mask,
            NO_D_MASK=NO_D_MASK,
        )
        p, log_om_beta, neg_log_acc = compute_block(
            q,
            k,
            qk_scale,
            neg_log_acc,
            M_blk_idxs,
            N_blk_idxs,
            cm,
            on_band,
            ALLOW_TF32,
            attend_current=attend_current,
            backward=True,
            is_compiling=is_compiling,
        )

        if not NO_M_MASK:
            neg_log_acc = tl.where(M_mask, neg_log_acc, 0.0)

        att_dA = p * (tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32) - dr[:, None])
        cumul_att_dA = tl.dot(att_dA.to(cm.dtype), fwd_cm, allow_tf32=ALLOW_TF32) + grad_prev_acc[:, None]
        grad_prev_acc += tl.sum(att_dA, axis=1)
        beta = 1 - tl.exp2(log_om_beta)
        dqk = att_dA - beta * cumul_att_dA

        dq = tl.dot(dqk.to(k.dtype), k, acc=dq, allow_tf32=ALLOW_TF32)
        block_dk = tl.dot(tl.trans(dqk).to(q.dtype), q, allow_tf32=ALLOW_TF32) * logit_scale
        block_dv = tl.dot(tl.trans(p), do.to(p.dtype), allow_tf32=ALLOW_TF32)

        if NO_D_MASK:
            tl.store(DK_blk_ptrs, block_dk, mask=N_mask[:, None])
            tl.store(DV_blk_ptrs, block_dv, mask=N_mask[:, None])
        else:
            mask = N_mask[:, None] & D_mask[None, :]
            tl.store(DK_blk_ptrs, block_dk, mask=mask)
            tl.store(DV_blk_ptrs, block_dv, mask=mask)

        N_blk_idxs += BLOCK_N
        N_blk_idxs_start += BLOCK_N
        K_blk_ptrs += BLOCK_N * (num_heads * head_size)
        V_blk_ptrs += BLOCK_N * (num_heads * head_size)
        DK_blk_ptrs += BLOCK_N * (num_heads * head_size)
        DV_blk_ptrs += BLOCK_N * (num_heads * head_size)

    dq = (logit_scale * dq).to(DQ_head_seq_ptr.type.element_ty)

    if NO_D_MASK:
        tl.store(DQ_blk_ptrs, dq, mask=M_mask[:, None])
    else:
        tl.store(DQ_blk_ptrs, dq, mask=M_mask[:, None] & D_mask[None, :])


def stickbreaking_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    inv_temp: float,
    attend_current: bool,
    cu_seqlens: torch.LongTensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run forward Triton kernel and return (o, rem, neg_log_acc).

    q, k, v: [B, T, H, D]
    Returns: o [B, T, H, D], rem [B, T, H], neg_log_acc [B, T, H]
    """
    batch_size, token_size, num_heads, dim_size = q.size()
    o = torch.empty_like(q)
    rem = torch.zeros_like(q[:, :, :, 0], device=q.device)
    neg_log_acc = torch.zeros_like(rem, device=q.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    if cu_seqlens is None:
        num_seq_blocks = triton.cdiv(token_size, BLOCK_M)
        grid = (batch_size, num_heads, num_seq_blocks)
        CI = None
    else:
        CI = prepare_chunk_indices(cu_seqlens, BLOCK_M)
        num_seq_blocks = int(CI.shape[0])
        grid = (1, num_heads, num_seq_blocks)
    BLOCK_D = triton.next_power_of_2(dim_size)

    stickbreaking_attn_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        rem,
        neg_log_acc,
        CU_ptr=cu_seqlens if cu_seqlens is not None else q,
        CI_ptr=CI if CI is not None else q,
        logit_scale=inv_temp,
        attend_current=attend_current,
        batch_size=batch_size,
        token_size=token_size,
        head_size=dim_size,
        num_heads=num_heads,
        BLOCK_D=BLOCK_D,
        NO_D_MASK=BLOCK_D == dim_size,
        NO_M_MASK=(token_size % BLOCK_M) == 0,
        NO_N_MASK=(token_size % BLOCK_N) == 0,
        ALLOW_TF32=ALLOW_TF32,
        inv_log2=inv_log2,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        no_grad=False,
        is_compiling=False,
        IS_VARLEN=cu_seqlens is not None,
    )

    return o, rem, neg_log_acc


def stickbreaking_attn_bwd(
    do: torch.Tensor,
    dr: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    neg_log_acc: torch.Tensor,
    inv_temp: float,
    attend_current: bool,
    cu_seqlens: torch.LongTensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, token_size, num_heads, dim_size = q.size()
    BLOCK_M = 64
    BLOCK_N = 64
    if cu_seqlens is None:
        M_count = triton.cdiv(token_size, BLOCK_M)
        grid = (batch_size, num_heads, M_count)
        CI = None
    else:
        CI = prepare_chunk_indices(cu_seqlens, BLOCK_M)
        M_count = int(CI.shape[0])
        grid = (1, num_heads, M_count)
    dq = torch.zeros_like(q)
    dk = torch.zeros((M_count, batch_size, token_size, num_heads, dim_size), dtype=k.dtype, device=k.device)
    dv = torch.zeros((M_count, batch_size, token_size, num_heads, dim_size), dtype=v.dtype, device=v.device)

    BLOCK_D = triton.next_power_of_2(dim_size)
    stickbreaking_attn_bwd_kernel[grid](
        do,
        dr,
        neg_log_acc,
        q,
        k,
        v,
        dq,
        dk,
        dv,
        CU_ptr=cu_seqlens if cu_seqlens is not None else q,
        CI_ptr=CI if CI is not None else q,
        logit_scale=inv_temp,
        attend_current=attend_current,
        batch_size=batch_size,
        token_size=token_size,
        head_size=dim_size,
        num_heads=num_heads,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        NO_D_MASK=BLOCK_D == dim_size,
        NO_M_MASK=(token_size % BLOCK_M) == 0,
        NO_N_MASK=(token_size % BLOCK_N) == 0,
        ALLOW_TF32=ALLOW_TF32,
        inv_log2=inv_log2,
        acc_dtype=tl.float32,
        is_compiling=False,
        IS_VARLEN=cu_seqlens is not None,
    )

    dk_final = dk.sum(0)
    dv_final = dv.sum(0)

    return dq.to(q.dtype), dk_final, dv_final


class StickBreakingAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        inv_temp: float,
        attend_current: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        o, rem, neg_log_acc = stickbreaking_attn_fwd(q, k, v, inv_temp, attend_current, cu_seqlens)
        ctx.save_for_backward(q, k, v, neg_log_acc)
        ctx.inv_temp = inv_temp
        ctx.attend_current = attend_current
        ctx.cu_seqlens = cu_seqlens
        return o, rem

    @staticmethod
    def backward(ctx, do: torch.Tensor, drem: torch.Tensor):
        q, k, v, neg_log_acc = ctx.saved_tensors
        dq, dk, dv = stickbreaking_attn_bwd(do, drem, q, k, v, neg_log_acc, ctx.inv_temp, ctx.attend_current, ctx.cu_seqlens)
        return dq, dk, dv, None, None, None


def sb_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    inv_temp: float,
    attend_current: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return StickBreakingAttentionFunction.apply(q, k, v, inv_temp, attend_current, cu_seqlens)


__all__ = [
    'sb_attn',
]
