# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from fla.ops.stickbreaking_attn.softplus import softplus
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous

ALLOW_TF32 = True


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
    use_cumsum: tl.constexpr = False,
):
    qk = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * qk_scale
    log_om_beta = -softplus(qk)

    if on_band:
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
    H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    no_grad: tl.constexpr = False,
    acc_dtype: tl.constexpr = tl.float32,
    return_attention: tl.constexpr = False,
):
    block_start_offset = BT * seq_block_id
    M_blk_idxs = block_start_offset + M_range
    M_mask = M_blk_idxs < seq_length
    N_blk_idxs_start = block_start_offset + BT
    N_blk_idxs = N_blk_idxs_start + N_range

    Q_blk_ptrs = Q_head_seq_ptr + (
        (H * head_size) * M_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    K_blk_ptrs = K_head_seq_ptr + (
        (H * head_size) * N_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    V_blk_ptrs = V_head_seq_ptr + (
        (H * head_size) * N_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    O_blk_ptrs = O_head_seq_ptr + (
        (H * head_size) * M_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    R_blk_ptrs = R_head_seq_ptr + H * M_blk_idxs
    A_blk_ptrs = A_head_seq_ptr + H * M_blk_idxs

    if NO_D_MASK:
        if NO_M_MASK:
            q = tl.load(Q_blk_ptrs)
        else:
            q = tl.load(Q_blk_ptrs, mask=M_mask[:, None], other=0.0)
    else:
        q = tl.load(Q_blk_ptrs, mask=M_mask[:, None] & D_mask[None, :], other=0.0)

    iters = N_blk_idxs_start // BS
    neg_log_acc = tl.zeros([BT], dtype=acc_dtype)
    acc = tl.zeros([BT, BLOCK_D], dtype=acc_dtype)

    for i in range(iters):
        N_blk_idxs -= BS
        N_blk_idxs_start -= BS
        K_blk_ptrs -= BS * (H * head_size)
        V_blk_ptrs -= BS * (H * head_size)

        N_mask = N_blk_idxs < seq_length
        k, v = load_kv(
            K_blk_ptrs,
            V_blk_ptrs,
            N_mask=N_mask,
            NO_N_MASK=N_blk_idxs_start + BS - 1 < seq_length,
            D_mask=D_mask,
            NO_D_MASK=NO_D_MASK,
        )
        on_band = i < BT // BS
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
    scale,
    head_size: tl.constexpr,
    H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    acc_dtype: tl.constexpr = tl.float32,
):
    block_start_offset = BT * seq_prog_id
    M_blk_idxs = block_start_offset + M_range
    M_mask = M_blk_idxs < seq_length

    N_blk_idxs_start = 0
    N_blk_idxs = N_blk_idxs_start + N_range

    DO_blk_ptrs = DO_head_seq_ptr + (
        (H * head_size) * M_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    K_blk_ptrs = K_head_seq_ptr + (
        (H * head_size) * N_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    Q_blk_ptrs = Q_head_seq_ptr + (
        (H * head_size) * M_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    V_blk_ptrs = V_head_seq_ptr + (
        (H * head_size) * N_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    A_blk_ptrs = A_head_seq_ptr + H * M_blk_idxs
    DQ_blk_ptrs = DQ_head_seq_ptr + (
        (H * head_size) * M_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    DK_blk_ptrs = DK_head_seq_ptr + (
        (H * head_size) * N_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    DV_blk_ptrs = DV_head_seq_ptr + (
        (H * head_size) * N_blk_idxs[:, None] + 1 * D_range[None, :]
    )
    DR_blk_ptrs = DR_head_seq_ptr + H * M_blk_idxs

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
    grad_prev_acc = tl.zeros((BT,), dtype=acc_dtype)
    dq = tl.zeros((BT, BLOCK_D), dtype=acc_dtype)

    fwd_cm = tl.trans(cm)
    iters = (block_start_offset + BT) // BS
    for i in range(iters):
        on_band = (iters - i - 1) < BT // BS
        N_mask = N_blk_idxs < seq_length
        local_no_n_mask = (N_blk_idxs_start + BS - 1) < seq_length
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
            backward=True,
        )

        if not NO_M_MASK:
            neg_log_acc = tl.where(M_mask, neg_log_acc, 0.0)

        att_dA = p * (tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32) - dr[:, None])
        cumul_att_dA = tl.dot(att_dA.to(cm.dtype), fwd_cm, allow_tf32=ALLOW_TF32) + grad_prev_acc[:, None]
        grad_prev_acc += tl.sum(att_dA, axis=1)
        beta = 1 - tl.exp2(log_om_beta)
        dqk = att_dA - beta * cumul_att_dA

        dq = tl.dot(dqk.to(k.dtype), k, acc=dq, allow_tf32=ALLOW_TF32)
        block_dk = tl.dot(tl.trans(dqk).to(q.dtype), q, allow_tf32=ALLOW_TF32) * scale
        block_dv = tl.dot(tl.trans(p), do.to(p.dtype), allow_tf32=ALLOW_TF32)

        if NO_D_MASK:
            tl.store(DK_blk_ptrs, block_dk, mask=N_mask[:, None])
            tl.store(DV_blk_ptrs, block_dv, mask=N_mask[:, None])
        else:
            mask = N_mask[:, None] & D_mask[None, :]
            tl.store(DK_blk_ptrs, block_dk, mask=mask)
            tl.store(DV_blk_ptrs, block_dv, mask=mask)

        N_blk_idxs += BS
        N_blk_idxs_start += BS
        K_blk_ptrs += BS * (H * head_size)
        V_blk_ptrs += BS * (H * head_size)
        DK_blk_ptrs += BS * (H * head_size)
        DV_blk_ptrs += BS * (H * head_size)

    dq = (scale * dq).to(DQ_head_seq_ptr.type.element_ty)

    if NO_D_MASK:
        tl.store(DQ_blk_ptrs, dq, mask=M_mask[:, None])
    else:
        tl.store(DQ_blk_ptrs, dq, mask=M_mask[:, None] & D_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [4]
        for w in [4]
    ],
    key=["T", "head_size"]
)
@triton.jit
def parallel_stickbreaking_attn_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    R_ptr,
    A_ptr,
    CU_ptr,
    CI_ptr,
    scale: tl.constexpr,
    B,
    T,
    head_size: tl.constexpr,
    H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    no_grad: tl.constexpr = False,
    acc_dtype: tl.constexpr = tl.float32,
    IS_VARLEN: tl.constexpr = False,
):
    tl.static_assert(BT % BS == 0)
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
        bos = tl.full([], 0, dtype=tl.int32)
        seq_block_id = prog_id
        seq_length = T
    RCP_LN2: tl.constexpr = 1.4426950216

    qk_scale = RCP_LN2 * scale
    M_range = tl.arange(0, BT)
    N_range = tl.arange(0, BS)
    D_range = tl.arange(0, BLOCK_D)
    D_mask = D_range < head_size
    cm = tl.where(N_range[:, None] >= N_range[None, :], 1.0, 0.0).to(Q_ptr.type.element_ty)

    head_id = head_pid
    seq_prog_id = seq_block_id
    batch_offset = batch_id * T
    Q_head_seq_ptr = Q_ptr + ((batch_offset + bos) * H + head_id) * head_size
    K_head_seq_ptr = K_ptr + ((batch_offset + bos) * H + head_id) * head_size
    V_head_seq_ptr = V_ptr + ((batch_offset + bos) * H + head_id) * head_size
    O_head_seq_ptr = O_ptr + ((batch_offset + bos) * H + head_id) * head_size
    R_head_seq_ptr = R_ptr + ((batch_offset + bos) * H + head_id)
    A_head_seq_ptr = A_ptr + ((batch_offset + bos) * H + head_id)

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
        H,
        BLOCK_D,
        NO_D_MASK,
        NO_M_MASK,
        NO_N_MASK,
        ALLOW_TF32,
        BT,
        BS,
        no_grad,
        acc_dtype,
        False,
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [8]
        for w in [4]
    ],
    key=["T", "head_size"]
)
@triton.jit()
def parallel_stickbreaking_attn_bwd_kernel(
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
    scale,
    B,
    T,
    head_size: tl.constexpr,
    H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr,
    NO_M_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    acc_dtype: tl.constexpr = tl.float32,
    IS_VARLEN: tl.constexpr = False,
):
    tl.static_assert(BT % BS == 0)
    batch_id = 0 if IS_VARLEN else tl.program_id(0)
    head_pid = tl.program_id(1)
    prog_id = tl.program_id(2)
    RCP_LN2: tl.constexpr = 1.4426950216

    qk_scale = RCP_LN2 * scale
    M_range = tl.arange(0, BT)
    N_range = tl.arange(0, BS)
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
        seq_length = T

    head_id = head_pid
    seq_prog_id = seq_block_id

    batch_id_i64 = batch_id.to(tl.int64)
    head_id_i64 = head_id.to(tl.int64)
    seq_prog_id_i64 = seq_prog_id.to(tl.int64)
    bos_i64 = bos.to(tl.int64)

    batch_offset = batch_id_i64 * T
    head_offset = (batch_offset + bos_i64) * H + head_id_i64
    block_offset = seq_prog_id_i64 * B * T * H

    DO_head_seq_ptr = DO_ptr + head_offset * head_size
    DR_head_seq_ptr = DR_ptr + head_offset
    A_head_seq_ptr = A_ptr + head_offset
    Q_head_seq_ptr = Q_ptr + head_offset * head_size
    K_head_seq_ptr = K_ptr + head_offset * head_size
    V_head_seq_ptr = V_ptr + head_offset * head_size
    DQ_head_seq_ptr = DQ_ptr + head_offset * head_size
    DK_head_seq_ptr = DK_ptr + (block_offset + head_offset) * head_size
    DV_head_seq_ptr = DV_ptr + (block_offset + head_offset) * head_size

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
        scale,
        head_size,
        H,
        BLOCK_D,
        NO_D_MASK,
        NO_M_MASK,
        NO_N_MASK,
        ALLOW_TF32,
        BT,
        BS,
        acc_dtype,
    )


def parallel_stickbreaking_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run forward Triton kernel and return (o, rem, neg_log_acc).

    q, k, v: [B, T, H, D]
    Returns: o [B, T, H, D], rem [B, T, H], neg_log_acc [B, T, H]
    """
    B, T, H, D = q.size()
    o = torch.empty_like(q)
    rem = torch.zeros_like(q[:, :, :, 0], device=q.device)
    neg_log_acc = torch.zeros_like(rem, device=q.device, dtype=torch.float32)

    BT = 64
    BS = 64
    if cu_seqlens is None:
        NT = triton.cdiv(T, BT)
        grid = (B, H, NT)
        CI = None
    else:
        CI = prepare_chunk_indices(cu_seqlens, BT)
        NT = int(CI.shape[0])
        grid = (1, H, NT)
    BLOCK_D = triton.next_power_of_2(D)

    NO_M_MASK = (T % BT) == 0
    NO_N_MASK = (T % BS) == 0
    if cu_seqlens is not None:
        NO_M_MASK = False
        NO_N_MASK = False

    parallel_stickbreaking_attn_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        rem,
        neg_log_acc,
        CU_ptr=cu_seqlens if cu_seqlens is not None else q,
        CI_ptr=CI if CI is not None else q,
        scale=scale,
        B=B,
        T=T,
        head_size=D,
        H=H,
        BLOCK_D=BLOCK_D,
        NO_D_MASK=D == BLOCK_D,
        NO_M_MASK=NO_M_MASK,
        NO_N_MASK=NO_N_MASK,
        ALLOW_TF32=ALLOW_TF32,
        BT=BT,
        BS=BS,
        no_grad=False,
        IS_VARLEN=cu_seqlens is not None,
    )

    return o, rem, neg_log_acc


def parallel_stickbreaking_attn_bwd(
    do: torch.Tensor,
    dr: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    neg_log_acc: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, D = q.size()
    BT = 64
    BS = 64
    if cu_seqlens is None:
        M_count = triton.cdiv(T, BT)
        grid = (B, H, M_count)
        CI = None
    else:
        CI = prepare_chunk_indices(cu_seqlens, BT)
        M_count = int(CI.shape[0])
        grid = (1, H, M_count)
    dq = torch.zeros_like(q)
    dk = torch.zeros((M_count, B, T, H, D), dtype=k.dtype, device=k.device)
    dv = torch.zeros((M_count, B, T, H, D), dtype=v.dtype, device=v.device)

    BLOCK_D = triton.next_power_of_2(D)

    NO_M_MASK = (T % BT) == 0
    NO_N_MASK = (T % BS) == 0
    if cu_seqlens is not None:
        NO_M_MASK = False
        NO_N_MASK = False

    parallel_stickbreaking_attn_bwd_kernel[grid](
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
        scale=scale,
        B=B,
        T=T,
        head_size=D,
        H=H,
        BT=BT,
        BS=BS,
        BLOCK_D=BLOCK_D,
        NO_D_MASK=D == BLOCK_D,
        NO_M_MASK=NO_M_MASK,
        NO_N_MASK=NO_N_MASK,
        ALLOW_TF32=ALLOW_TF32,
        acc_dtype=tl.float32,
        IS_VARLEN=cu_seqlens is not None,
    )

    dk_final = dk.sum(0)
    dv_final = dv.sum(0)

    return dq.to(q.dtype), dk_final, dv_final


class StickBreakingAttentionFunction(torch.autograd.Function):

    @staticmethod
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        o, rem, neg_log_acc = parallel_stickbreaking_attn_fwd(q, k, v, scale, cu_seqlens)
        ctx.save_for_backward(q, k, v, neg_log_acc)
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        return o, rem

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, drem: torch.Tensor):
        q, k, v, neg_log_acc = ctx.saved_tensors
        dq, dk, dv = parallel_stickbreaking_attn_bwd(do, drem, q, k, v, neg_log_acc, ctx.scale, ctx.cu_seqlens)
        return dq, dk, dv, None, None


def parallel_stickbreaking_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
    return StickBreakingAttentionFunction.apply(q, k, v, scale, cu_seqlens)


__all__ = [
    'parallel_stickbreaking_attn',
]
