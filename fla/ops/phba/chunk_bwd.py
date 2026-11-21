# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"Deterministic PHBA attention backward with tape and bounded-memory query owners."

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from fla.ops.phba.plans import (
    OwnerBranchTapePlan,
    OwnerTapePlan,
    owner_tape_index32_safe,
    owner_tape_producer_stages,
    owner_tape_uses_tile_major,
)
from fla.ops.phba.utils import fail

OWNER_NUM_WARPS = 4


OWNER_NUM_STAGES = 2


@triton.jit(do_not_specialize=["T", "K", "RH", "RT"])
def query_owner_dq_kernel(
    q,
    kmem,
    vmem,
    dout,
    lse,
    delta,
    route,
    block_ptr,
    prefix_valid,
    dq,
    T,
    K,
    RH,
    RT,
    SQT,
    SQH,
    SKT,
    SKH,
    SVT,
    SVH,
    SDT,
    SDH,
    SPB,
    SPM,
    SPH,
    SCALE,
    H: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    BQ: tl.constexpr,
    BC: tl.constexpr,
    BD: tl.constexpr,
    KP: tl.constexpr,
    BRANCH: tl.constexpr,
):
    start = tl.program_id(0).to(tl.int64) * BQ
    head = tl.program_id(1).to(tl.int64)
    lane = tl.arange(0, BQ)
    rows = start + lane
    qvalid = rows < T
    d = tl.arange(0, BD)
    qm = qvalid[:, None] & (d[None, :] < D)
    qt = tl.load(q + (rows[:, None] * SQT + head * SQH + d[None, :]), qm, other=0)
    dot = tl.load(dout + (rows[:, None] * SDT + head * SDH + d[None, :]), qm, other=0)
    joint_l = tl.load(lse + (head * T + rows), qvalid, other=0)
    de = tl.load(delta + (head * T + rows), qvalid, other=0)
    acc = tl.zeros((BQ, BD), tl.float32)

    flat = tl.arange(0, BQ * KP)
    flat_row = start + flat // KP
    flat_slot = flat % KP
    ids = tl.load(route + (head * RH + flat_row * RT + flat_slot), (flat_row < T) & (flat_slot < K), other=-1)
    remaining = ids
    position = tl.min(tl.where(remaining >= 0, flat, BQ * KP), 0)
    memory_lane = tl.arange(0, BC)
    while position < BQ * KP:
        candidate = tl.load(route + (head * RH + (start + position // KP) * RT + position % KP))
        leader = candidate >= 0
        safe = tl.where(leader, candidate, 0)
        if BC <= 32 or D == BD:
            member = tl.sum((tl.reshape(ids, (BQ, KP)) == safe).to(tl.int32), axis=1) > 0
        else:
            member = tl.full((BQ,), False, tl.int1)
            for slot in tl.static_range(KP):
                row_id = tl.load(route + (head * RH + rows * RT + slot), qvalid & (slot < K), other=-1)
                member = member | (row_id == safe)
        row_valid = qvalid & member & leader
        permitted = leader
        if BRANCH == 1:
            permitted = permitted & (tl.load(prefix_valid + tl.cast(safe, tl.int64)) != 0)
        if permitted:
            if BRANCH == 0:
                begin = tl.load(block_ptr + tl.cast(safe, tl.int64))
                end = tl.load(block_ptr + tl.cast(safe + 1, tl.int64))
                extent = C
            else:
                begin = 0
                end = M
                extent = M
            for m0 in range(0, extent, BC):
                memory = m0 + memory_lane
                ix = tl.cast(begin, tl.int64) + memory
                valid_memory = ix < end
                mask = valid_memory[:, None] & (d[None, :] < D)
                if BRANCH == 0:
                    kaddr = ix[:, None] * SKT + head * SKH + d[None, :]
                    vaddr = ix[:, None] * SVT + head * SVH + d[None, :]
                else:
                    kaddr = safe.to(tl.int64) * SPB + memory.to(tl.int64)[:, None] * SPM + head * SPH + d[None, :]
                    vaddr = kaddr
                kt = tl.load(kmem + (kaddr), mask, other=0)
                vt = tl.load(vmem + (vaddr), mask, other=0)
                score = tl.dot(qt, tl.trans(kt), out_dtype=tl.float32) * SCALE
                score = tl.where(row_valid[:, None] & valid_memory[None, :], score, float("-inf"))
                probability = tl.exp(score - joint_l[:, None])
                dp = tl.dot(dot, tl.trans(vt), out_dtype=tl.float32)
                ds = probability * (dp - de[:, None])
                acc += tl.dot(ds.to(qt.dtype), kt, out_dtype=tl.float32) * SCALE
        remaining = tl.where(remaining == candidate, -1, remaining)
        position = tl.min(tl.where(remaining >= 0, flat, BQ * KP), 0)
    tl.store(dq + ((rows[:, None] * H + head) * D + d[None, :]), acc, qm)


def query_owner_dq(
    q, k, v, dout, lse, delta, route, block_ptr, prefix_valid, *, chunk_size, num_slots, branch, bc=32, out=None
):
    t, h, d = q.shape
    if out is None:
        out = torch.empty((t, h, d), device=q.device, dtype=torch.float32)
    width = route.shape[-1]
    if not width:
        return out.zero_()
    prefix_strides = k.stride()[:3] if branch else (0, 0, 0)
    query_owner_dq_kernel[triton.cdiv(t, 16), h](
        q,
        k,
        v,
        dout,
        lse,
        delta,
        route,
        block_ptr,
        prefix_valid.view(torch.uint8),
        out,
        t,
        width,
        route.stride(0),
        route.stride(1),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        dout.stride(0),
        dout.stride(1),
        *prefix_strides,
        d ** (-0.5),
        H=h,
        D=d,
        M=num_slots,
        C=chunk_size,
        BQ=16,
        BC=bc,
        BD=max(16, triton.next_power_of_2(d)),
        KP=triton.next_power_of_2(width),
        BRANCH=branch,
        num_warps=OWNER_NUM_WARPS,
        num_stages=OWNER_NUM_STAGES,
    )
    return out


def deterministic_sparse_dq(
    q_rope,
    q_raw,
    k_rope,
    value,
    pk,
    pv,
    dout,
    lse,
    delta,
    rtok,
    rstate,
    block_ptr,
    prefix_valid,
    *,
    chunk_size,
    bc=32,
    outputs=None,
):
    results = []
    for branch, q, k, v, route in ((0, q_rope, k_rope, value, rtok), (1, q_raw, pk, pv, rstate)):
        results.append(
            query_owner_dq(
                q,
                k,
                v,
                dout,
                lse,
                delta,
                route,
                block_ptr,
                prefix_valid,
                chunk_size=chunk_size,
                num_slots=pk.shape[1],
                branch=branch,
                bc=bc,
                out=outputs[branch] if outputs is not None else None,
            ),
        )
    return tuple(results)


NUM_WARPS = 4


NUM_STAGES = 1


@triton.jit(do_not_specialize=["T"])
def _delta_kernel(do, out, delta, T, sot, soh, sdt, sdh, H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    h = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, BD)
    mask = (rows[:, None] < T) & (d[None, :] < D)
    r = rows
    a = tl.load(do + (r[:, None] * sdt + h * sdh + d[None, :]), mask, other=0).to(tl.float32)
    b = tl.load(out + (r[:, None] * sot + h * soh + d[None, :]), mask, other=0).to(tl.float32)
    tl.store(delta + (h * T + r), tl.sum(a * b, 1), rows < T)


@triton.jit(do_not_specialize=["T"])
def _sparse_bwd_kernel(
    q,
    kmem,
    vmem,
    do,
    lse,
    delta,
    count,
    offset,
    rho,
    block_ptr,
    prefix_valid,
    dk,
    dv,
    T,
    scale,
    sqt,
    sqh,
    skt,
    skh,
    svt,
    svh,
    sdt,
    sdh,
    spb,
    spm,
    sph,
    sgb,
    sgm,
    sgh,
    H: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    BC: tl.constexpr,
    BQ: tl.constexpr,
    BD: tl.constexpr,
    BRANCH: tl.constexpr,
):
    e = tl.program_id(0).to(tl.int64)
    block = e // H
    h = e % H
    tile = tl.program_id(1).to(tl.int64)
    n = tl.load(count + (e))
    if n == 0:
        return
    mem = tile * BC + tl.arange(0, BC)
    d = tl.arange(0, BD)
    if BRANCH == 0:
        start = tl.load(block_ptr + (block))
        end = tl.load(block_ptr + (block + 1))
        if start + tile * BC >= end:
            return
        ix = start + mem
        valid_mem = ix < end
        kaddr = ix[:, None] * skt + h * skh + d[None, :]
        vaddr = ix[:, None] * svt + h * svh + d[None, :]
        grad_addr = (ix[:, None] * H + h) * D + d[None, :]
    else:
        if tl.load(prefix_valid + (block)) == 0:
            return
        valid_mem = mem < M
        kaddr = (block) * spb + (mem)[:, None] * spm + h * sph + d[None, :]
        vaddr = kaddr
        grad_addr = (block) * sgb + (mem)[:, None] * sgm + h * sgh + d[None, :]
    mm = valid_mem[:, None] & (d[None, :] < D)
    k = tl.load(kmem + (kaddr), mm, other=0)
    v = tl.load(vmem + (vaddr), mm, other=0)
    dk_acc = tl.zeros((BC, BD), tl.float32)
    dv_acc = tl.zeros((BC, BD), tl.float32)
    begin = (tl.load(offset + (e))).to(tl.int64)
    finish = (tl.load(offset + (e + 1))).to(tl.int64)
    lane = tl.arange(0, BQ)
    for p in range(begin, finish, BQ):
        pos = p + lane
        valid_q = pos < finish
        r = tl.load(rho + (pos.to(tl.int64)), valid_q, other=0)
        valid_q = valid_q & (r % H == h)
        t = (r // H).to(tl.int64)
        qm = valid_q[:, None] & (d[None, :] < D)
        qt = tl.load(q + (t[:, None] * sqt + h * sqh + d[None, :]), qm, other=0)
        dot = tl.load(do + (t[:, None] * sdt + h * sdh + d[None, :]), qm, other=0)
        l = tl.load(lse + (h * T + t), valid_q, other=0)
        de = tl.load(delta + (h * T + t), valid_q, other=0)
        score = tl.dot(qt, tl.trans(k), out_dtype=tl.float32) * scale
        score = tl.where(valid_q[:, None] & valid_mem[None, :], score, float("-inf"))
        prob = tl.exp(score - l[:, None])
        dp = tl.dot(dot, tl.trans(v), out_dtype=tl.float32)
        ds = prob * (dp - de[:, None])
        ds_st = ds.to(qt.dtype)
        dk_acc += tl.dot(tl.trans(ds_st), qt, out_dtype=tl.float32) * scale
        dv_acc += tl.dot(tl.trans(prob.to(dot.dtype)), dot, out_dtype=tl.float32)
    tl.store(dk + (grad_addr), dk_acc, mm)
    tl.store(dv + (grad_addr), dv_acc, mm)


@triton.jit(do_not_specialize=["ROWS"])
def _merge_kernel(local, sparse, out, ROWS, D: tl.constexpr, BLOCK: tl.constexpr):
    N = tl.cast(ROWS, tl.int64) * D
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(local + (i), i < N, other=0).to(tl.float32)
    b = tl.load(sparse + (i), i < N, other=0).to(tl.float32)
    tl.store(out + (i), a + b, i < N)


def compute_delta(dout, out):
    T, H, D = out.shape
    delta = torch.empty((H, T), device=out.device, dtype=torch.float32)
    _delta_kernel[triton.cdiv(T, 32), H](
        dout,
        out,
        delta,
        T,
        out.stride(0),
        out.stride(1),
        dout.stride(0),
        dout.stride(1),
        H=H,
        D=D,
        BD=max(16, triton.next_power_of_2(D)),
        BT=32,
        num_warps=4,
        num_stages=1,
    )
    return delta


def merge_gradient(local, sparse, *, output_dtype=torch.float32):
    if not local.is_contiguous() or not sparse.is_contiguous():
        raise fail("non_contiguous_input", "gradient merge requires contiguous buffers")
    out = torch.empty(local.shape, device=local.device, dtype=output_dtype)
    n, d = local.numel(), local.shape[-1]
    _merge_kernel[(triton.cdiv(n, 256),)](local, sparse, out, n // d, D=d, BLOCK=256, num_warps=4, num_stages=1)
    return out


@dataclass
class SparseGradients:
    dq_rope: torch.Tensor
    dk_rope: torch.Tensor
    dv: torch.Tensor
    dq_raw: torch.Tensor
    dpk: torch.Tensor
    dpv: torch.Tensor


def launch_branch(
    q, k, v, dout, lse, delta, meta, block_ptr, prefix_valid, dk, dv, *, branch, chunk_size, num_slots, bc=32, bq=16
):
    T, H, D = q.shape
    extent = chunk_size if branch == 0 else num_slots
    ps = k.stride()[:3] if branch else (0, 0, 0)
    gs = dk.stride()[:3] if branch else (0, 0, 0)
    _sparse_bwd_kernel[meta.count.numel(), triton.cdiv(extent, bc), 1](
        q,
        k,
        v,
        dout,
        lse,
        delta,
        meta.count,
        meta.offset,
        meta.rho,
        block_ptr,
        prefix_valid.view(torch.uint8),
        dk,
        dv,
        T,
        D ** (-0.5),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        dout.stride(0),
        dout.stride(1),
        *ps,
        *gs,
        H=H,
        D=D,
        M=num_slots,
        BC=bc,
        BQ=bq,
        BD=max(16, triton.next_power_of_2(D)),
        BRANCH=branch,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )


def direct_sparse_backward(
    q_rope,
    q_raw,
    k_rope,
    v,
    pk,
    pv,
    dout,
    lse,
    delta,
    metadata,
    block_ptr,
    prefix_valid,
    *,
    chunk_size,
    bc=32,
    bq=16,
    rtok,
    rstate,
    tape_plan,
):
    if bq not in (16, 32) or bc not in (16, 32, 64):
        raise fail("invalid_config", "unsupported PHBA backward tiles")
    if not lse.is_contiguous() or not delta.is_contiguous():
        raise fail("non_contiguous_input", "L/Delta must be contiguous [H,T]")
    if rtok is None or rstate is None:
        raise fail("invalid_config", "query-owner backward requires its saved forward routes")
    T, H, D = q_rope.shape

    def z():
        return torch.zeros((T, H, D), device=q_rope.device, dtype=torch.float32)

    def q_buffer():
        return torch.empty((T, H, D), device=q_rope.device, dtype=torch.float32)

    grads = SparseGradients(
        q_buffer(),
        z(),
        z(),
        q_buffer(),
        torch.zeros_like(pk, dtype=torch.float32),
        torch.zeros_like(pv, dtype=torch.float32),
    )
    if tape_plan.enabled:
        return backward_with_tape(
            q_rope,
            q_raw,
            k_rope,
            v,
            pk,
            pv,
            dout,
            lse,
            delta,
            metadata,
            block_ptr,
            prefix_valid,
            rtok,
            rstate,
            grads,
            tape_plan,
            chunk_size=chunk_size,
            bc=bc,
            bq=bq,
        )
    args = (dout, lse, delta)
    if metadata.tok is not None:
        launch_branch(
            q_rope,
            k_rope,
            v,
            *args,
            metadata.tok,
            block_ptr,
            prefix_valid,
            grads.dk_rope,
            grads.dv,
            branch=0,
            chunk_size=chunk_size,
            num_slots=pk.shape[1],
            bc=bc,
            bq=bq,
        )
    launch_branch(
        q_raw,
        pk,
        pv,
        *args,
        metadata.state,
        block_ptr,
        prefix_valid,
        grads.dpk,
        grads.dpv,
        branch=1,
        chunk_size=chunk_size,
        num_slots=pk.shape[1],
        bc=bc,
        bq=bq,
    )
    deterministic_sparse_dq(
        q_rope,
        q_raw,
        k_rope,
        v,
        pk,
        pv,
        dout,
        lse,
        delta,
        rtok,
        rstate,
        block_ptr,
        prefix_valid,
        chunk_size=chunk_size,
        bc=bc,
        outputs=(grads.dq_rope, grads.dq_raw),
    )
    return grads


def attention_backward(ctx, dout, _dlse):
    from fla.ops.phba.flash_attention import local_backward
    from fla.ops.phba.metadata import BranchMetadata, DirectMetadata
    from fla.ops.phba.utils import prepare_grad_output

    qr, qw, kr, v, pk, pv, out, lse, bp, valid, sc, so, sr, *token = ctx.saved_tensors
    c, bc, bq = ctx.settings
    rtok, rstate = token[-2:]
    token = token[:-2]
    state = BranchMetadata(sc, so, sr, sr.numel())
    tok = BranchMetadata(*token, token[-1].numel()) if ctx.has_token else None
    metadata = DirectMetadata(tok, state)
    dout = prepare_grad_output(dout, out)
    dout = dout.contiguous()
    delta = compute_delta(dout, out)
    local = local_backward(dout, qr, kr, v, out, lse, bp, ctx.batches, deterministic=True)
    sparse = direct_sparse_backward(
        qr,
        qw,
        kr,
        v,
        pk,
        pv,
        dout,
        lse,
        delta,
        metadata,
        bp,
        valid,
        chunk_size=c,
        bc=bc,
        bq=bq,
        rtok=rtok,
        rstate=rstate,
        tape_plan=ctx.owner_tape_plan,
    )
    dq = merge_gradient(local[0], sparse.dq_rope, output_dtype=qr.dtype)
    dk = merge_gradient(local[1], sparse.dk_rope, output_dtype=kr.dtype)
    dv = merge_gradient(local[2], sparse.dv, output_dtype=v.dtype)
    return (
        dq.to(qr.dtype),
        sparse.dq_raw.to(qw.dtype),
        dk.to(kr.dtype),
        dv.to(v.dtype),
        sparse.dpk.to(pk.dtype),
        sparse.dpv.to(pv.dtype),
        None,
        None,
        None,
        None,
        None,
    )


__all__ = [
    "SparseGradients",
    "attention_backward",
    "compute_delta",
    "deterministic_sparse_dq",
    "direct_sparse_backward",
    "launch_branch",
    "merge_gradient",
]


@triton.jit(do_not_specialize=["NC", "HEAD_START"])
def _group_count_kernel(
    count, group_count, group_offset, NC, HEAD_START, H: tl.constexpr, GROUP_SIZE: tl.constexpr, BT: tl.constexpr
):
    index = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT).to(tl.int64)
    block = index // GROUP_SIZE
    head = tl.cast(HEAD_START, tl.int64) + index % GROUP_SIZE
    valid = (block < NC) & (head < H)
    value = tl.load(count + (block * H + head), valid, other=0)
    tl.store(group_count + (index), value, index < tl.cast(NC, tl.int64) * GROUP_SIZE)
    if tl.program_id(0).to(tl.int64) == 0:
        tl.store(group_offset, 0)


@triton.jit(do_not_specialize=["N"])
def _group_offset_kernel(count, offset, N, BT: tl.constexpr):
    start = tl.program_id(0).to(tl.int64) * BT
    lane = tl.arange(0, BT).to(tl.int64)
    carry = tl.full((), 0, tl.int32)
    previous = tl.full((), 0, tl.int64)
    while previous < start:
        value = tl.load(count + (previous + lane), previous + lane < N, other=0)
        carry += tl.sum(value, axis=0)
        previous += BT
    current = tl.load(count + (start + lane), start + lane < N, other=0)
    inclusive = tl.cumsum(current, axis=0) + carry
    tl.store(offset + (start + lane + 1), inclusive, start + lane < N)


@triton.jit(do_not_specialize=["T", "K", "RH", "RT", "HEAD_START", "SIZE"])
def _route_edge_map_kernel(
    route,
    offset,
    rho,
    group_offset,
    edge_map,
    T,
    K,
    RH,
    RT,
    HEAD_START,
    SIZE,
    H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    MAP_WIDTH: tl.constexpr,
    BT: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT).to(tl.int64)
    slot = index % MAP_WIDTH
    row = (index // MAP_WIDTH) % T
    local_head = index // (tl.cast(T, tl.int64) * MAP_WIDTH)
    head = tl.cast(HEAD_START, tl.int64) + local_head
    valid = (index < SIZE) & (head < H) & (slot < K)
    block = tl.load(route + (head * RH + row * RT + slot), valid, other=-1).to(tl.int64)
    valid = valid & (block >= 0)
    safe_block = tl.maximum(block, 0)
    expert = safe_block * H + head
    begin = tl.load(offset + (expert), valid, other=0).to(tl.int64)
    finish = tl.load(offset + (expert + 1), valid, other=0).to(tl.int64)
    lower = begin
    upper = finish
    target = row * H + head
    while tl.sum((lower < upper).to(tl.int32), axis=0) > 0:
        middle = (lower + upper) // 2
        searching = lower < upper
        value = tl.load(rho + tl.cast(middle, tl.int64), searching, other=2147483647).to(tl.int64)
        right = searching & (value < target)
        lower = tl.where(right, middle + 1, lower)
        upper = tl.where(searching & ~right, middle, upper)
    found_value = tl.load(rho + tl.cast(lower, tl.int64), valid & (lower < finish), other=-1).to(tl.int64)
    found = valid & (lower < finish) & (found_value == target)
    group_base = tl.load(group_offset + (safe_block * GROUP_SIZE + local_head), found, other=0).to(tl.int64)
    edge = tl.where(found, group_base + lower - begin, -1)
    tl.store(edge_map + (index), edge.to(tl.int32), index < SIZE)


@triton.jit(do_not_specialize=["T", "HEAD_START", "TAPE_EDGES"])
def _sparse_bwd_tape_kernel(
    q,
    kmem,
    vmem,
    do,
    lse,
    delta,
    count,
    offset,
    rho,
    block_ptr,
    prefix_valid,
    dq,
    dk,
    dv,
    tape,
    group_offset,
    HEAD_START,
    T,
    scale,
    sqt,
    sqh,
    skt,
    skh,
    svt,
    svh,
    sdt,
    sdh,
    spb,
    spm,
    sph,
    sgb,
    sgm,
    sgh,
    TAPE_EDGES,
    H: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    BC: tl.constexpr,
    BQ: tl.constexpr,
    BD: tl.constexpr,
    BRANCH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TAPE_STRIDE: tl.constexpr,
    INDEX32: tl.constexpr,
    TILE_MAJOR: tl.constexpr = True,
    TASK_ORDER=None,
    ORDERED: tl.constexpr = False,
):
    if ORDERED:
        group_expert = tl.load(TASK_ORDER + (tl.program_id(1).to(tl.int64))).to(tl.int64)
    else:
        group_expert = tl.program_id(0).to(tl.int64)
    block = group_expert // GROUP_SIZE
    h = tl.cast(HEAD_START, tl.int64) + group_expert % GROUP_SIZE
    if h >= H:
        return
    e = block * H + h
    tile_id = tl.program_id(0) if ORDERED else tl.program_id(1)
    if INDEX32:
        tile = tile_id.to(tl.int32)
    else:
        tile = tile_id.to(tl.int64)
    n = tl.load(count + (e))
    if n == 0:
        return
    mem = tile * BC + tl.arange(0, BC)
    d = tl.arange(0, BD)
    if BRANCH == 0:
        start = tl.load(block_ptr + (block))
        end = tl.load(block_ptr + (block + 1))
        if start.to(tl.int64) + tile.to(tl.int64) * BC >= end:
            return
        ix = start.to(tl.int64) + mem.to(tl.int64)
        valid_mem = ix < end
        kaddr = ix[:, None] * skt + h * skh + d[None, :]
        vaddr = ix[:, None] * svt + h * svh + d[None, :]
        grad_addr = (ix[:, None] * H + h) * D + d[None, :]
    else:
        if tl.load(prefix_valid + (block)) == 0:
            return
        valid_mem = mem < M
        kaddr = (block) * spb + mem.to(tl.int64)[:, None] * spm + h * sph + d[None, :]
        vaddr = kaddr
        grad_addr = (block) * sgb + mem.to(tl.int64)[:, None] * sgm + h * sgh + d[None, :]
    mm = valid_mem[:, None] & (d[None, :] < D)
    k = tl.load(kmem + (kaddr), mm, other=0)
    v = tl.load(vmem + (vaddr), mm, other=0)
    dk_acc = tl.zeros((BC, BD), tl.float32)
    dv_acc = tl.zeros((BC, BD), tl.float32)
    group_begin = tl.load(group_offset + (group_expert)).to(tl.int64)
    begin = tl.load(offset + (e))
    finish = tl.load(offset + (e + 1))
    if not INDEX32:
        begin = begin.to(tl.int64)
        finish = finish.to(tl.int64)
    lane = tl.arange(0, BQ)
    for p in range(begin, finish, BQ):
        pos = p + lane
        valid_q = pos < finish
        r = tl.load(rho + (pos.to(tl.int64)), valid_q, other=0)
        valid_q = valid_q & (r % H == h)
        t = (r // H).to(tl.int64)
        qm = valid_q[:, None] & (d[None, :] < D)
        qt = tl.load(q + (t[:, None] * sqt + h * sqh + d[None, :]), qm, other=0)
        dot = tl.load(do + (t[:, None] * sdt + h * sdh + d[None, :]), qm, other=0)
        l = tl.load(lse + (h * T + t), valid_q, other=0)
        de = tl.load(delta + (h * T + t), valid_q, other=0)
        score = tl.dot(qt, tl.trans(k), out_dtype=tl.float32) * scale
        score = tl.where(valid_q[:, None] & valid_mem[None, :], score, float("-inf"))
        prob = tl.exp(score - l[:, None])
        dp = tl.dot(dot, tl.trans(v), out_dtype=tl.float32)
        ds = prob * (dp - de[:, None])
        ds_st = ds.to(qt.dtype)
        edge = group_begin + pos.to(tl.int64) - tl.cast(begin, tl.int64)
        if TILE_MAJOR:
            tape_address = (tile.to(tl.int64) * tl.cast(TAPE_EDGES, tl.int64) + edge[:, None]) * BC
            tape_address += tl.arange(0, BC).to(tl.int64)[None, :]
        else:
            tape_address = edge[:, None] * TAPE_STRIDE + mem[None, :]
        tl.store(tape + tl.cast(tape_address, tl.int64), ds_st, valid_q[:, None] & valid_mem[None, :])
        dk_acc += tl.dot(tl.trans(ds_st), qt, out_dtype=tl.float32) * scale
        dv_acc += tl.dot(tl.trans(prob.to(dot.dtype)), dot, out_dtype=tl.float32)
    tl.store(dk + (grad_addr), dk_acc, mm)
    tl.store(dv + (grad_addr), dv_acc, mm)


@triton.jit(do_not_specialize=["T", "K", "RH", "RT", "HEAD_START", "TAPE_EDGES"])
def _owner_from_tape_kernel(
    q,
    kmem,
    route,
    edge_map,
    tape,
    block_ptr,
    prefix_valid,
    dq,
    T,
    K,
    RH,
    RT,
    HEAD_START,
    SKT,
    SKH,
    SPB,
    SPM,
    SPH,
    SCALE,
    TAPE_EDGES,
    H: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    BQ: tl.constexpr,
    BC: tl.constexpr,
    BD: tl.constexpr,
    KP: tl.constexpr,
    BRANCH: tl.constexpr,
    MAP_WIDTH: tl.constexpr,
    TAPE_STRIDE: tl.constexpr,
    TILE_MAJOR: tl.constexpr = True,
):
    start = tl.program_id(0).to(tl.int64) * BQ
    local_head = tl.program_id(1).to(tl.int64)
    head = tl.cast(HEAD_START, tl.int64) + local_head
    if head >= H:
        return
    lane = tl.arange(0, BQ)
    rows = start + lane
    qvalid = rows < T
    d = tl.arange(0, BD)
    qm = qvalid[:, None] & (d[None, :] < D)
    acc = tl.zeros((BQ, BD), tl.float32)
    flat = tl.arange(0, BQ * KP)
    flat_row = start + flat // KP
    flat_slot = flat % KP
    ids = tl.load(route + (head * RH + flat_row * RT + flat_slot), (flat_row < T) & (flat_slot < K), other=-1)
    remaining = ids
    position = tl.min(tl.where(remaining >= 0, flat, BQ * KP), 0)
    memory_lane = tl.arange(0, BC)
    slots = tl.arange(0, KP)
    while position < BQ * KP:
        selected = position.to(tl.int64)
        candidate = tl.load(route + (head * RH + (start + selected // KP) * RT + selected % KP))
        leader = candidate >= 0
        safe = tl.where(leader, candidate, 0)
        matches = tl.reshape(ids, (BQ, KP)) == safe
        first_slot = tl.min(tl.where(matches, slots[None, :], KP), axis=1)
        member = first_slot < K
        edge_addr = (local_head * T + rows) * MAP_WIDTH + first_slot.to(tl.int64)
        edge = tl.load(edge_map + (edge_addr), qvalid & member & leader, other=-1).to(tl.int64)
        row_valid = qvalid & member & leader & (edge >= 0)
        permitted = leader
        if BRANCH == 1:
            permitted = permitted & (tl.load(prefix_valid + (safe.to(tl.int64))) != 0)
        if permitted:
            if BRANCH == 0:
                begin = tl.load(block_ptr + (safe.to(tl.int64)))
                end = tl.load(block_ptr + (safe.to(tl.int64) + 1))
                extent = C
            else:
                begin = 0
                end = M
                extent = M
            for m0 in range(0, extent, BC):
                memory = m0 + memory_lane
                ix = tl.cast(begin, tl.int64) + memory
                valid_memory = ix < end
                mask = valid_memory[:, None] & (d[None, :] < D)
                if BRANCH == 0:
                    kaddr = ix[:, None] * SKT + head * SKH + d[None, :]
                else:
                    kaddr = safe.to(tl.int64) * SPB + memory.to(tl.int64)[:, None] * SPM + head * SPH + d[None, :]
                kt = tl.load(kmem + (kaddr), mask, other=0)
                if TILE_MAJOR:
                    tape_address = (
                        tl.cast(m0, tl.int64) // BC * tl.cast(TAPE_EDGES, tl.int64) + tl.maximum(edge, 0)[:, None]
                    ) * BC + memory_lane.to(tl.int64)[None, :]
                else:
                    tape_address = tl.maximum(edge, 0)[:, None] * TAPE_STRIDE + memory[None, :]
                ds_st = tl.load(tape + (tape_address), row_valid[:, None] & valid_memory[None, :], other=0)
                acc += tl.dot(ds_st, kt, out_dtype=tl.float32) * SCALE
        remaining = tl.where(remaining == candidate, -1, remaining)
        position = tl.min(tl.where(remaining >= 0, flat, BQ * KP), 0)
    tl.store(dq + ((rows[:, None] * H + head) * D + d[None, :]), acc, qm)


@dataclass(frozen=True)
class TapeWorkspace:
    tape: torch.Tensor
    edge_map: torch.Tensor
    count: torch.Tensor
    offset: torch.Tensor

    @property
    def nbytes(self):
        return sum(value.numel() * value.element_size() for value in (self.tape, self.edge_map, self.count, self.offset))


def allocate_tape_workspace(q, plan: OwnerBranchTapePlan):
    if not plan.enabled:
        raise ValueError("disabled owner tape plan must use the original algorithm")
    workspace = TapeWorkspace(
        torch.empty(plan.tape_elements, dtype=q.dtype, device=q.device),
        torch.empty(plan.map_elements, dtype=torch.int32, device=q.device),
        torch.empty(plan.count_elements, dtype=torch.int32, device=q.device),
        torch.empty(plan.offset_elements, dtype=torch.int32, device=q.device),
    )
    if workspace.nbytes != plan.workspace_bytes:
        raise ValueError("owner tape dtype/allocation differs from the forward-bound byte plan")
    return workspace


def build_group_metadata(route, meta, workspace, plan: OwnerTapePlan, *, head_start):
    h, t, width = route.shape
    if width == 0:
        raise ValueError("zero-width token branch must not build tape metadata")
    nc = meta.count.numel() // h
    _group_count_kernel[triton.cdiv(plan.count_elements, 128),](
        meta.count,
        workspace.count,
        workspace.offset,
        nc,
        head_start,
        H=h,
        GROUP_SIZE=plan.group_size,
        BT=128,
        num_warps=4,
        num_stages=1,
    )
    _group_offset_kernel[triton.cdiv(plan.count_elements, 128),](
        workspace.count,
        workspace.offset,
        plan.count_elements,
        BT=128,
        num_warps=4,
        num_stages=1,
    )
    _route_edge_map_kernel[triton.cdiv(plan.map_elements, 256),](
        route,
        meta.offset,
        meta.rho,
        workspace.offset,
        workspace.edge_map,
        t,
        width,
        route.stride(0),
        route.stride(1),
        head_start,
        plan.map_elements,
        H=h,
        GROUP_SIZE=plan.group_size,
        MAP_WIDTH=plan.map_width,
        BT=256,
        num_warps=4,
        num_stages=1,
    )


def run_tape_group(
    q,
    k,
    v,
    dout,
    lse,
    delta,
    route,
    meta,
    block_ptr,
    prefix_valid,
    dq,
    dk,
    dv,
    workspace,
    plan: OwnerTapePlan,
    *,
    head_start,
    branch,
    chunk_size,
    num_slots,
    bc,
    bq,
):
    """Compute one deterministic producer/consumer head group using its saved forward plan."""
    producer_stages = owner_tape_producer_stages(plan, bc)
    tile_major = owner_tape_uses_tile_major(plan, branch)
    t, h, d = q.shape
    extent = chunk_size if branch == 0 else num_slots
    tape_stride = plan.token_extent if branch == 0 else plan.state_extent
    if meta.capacity != t * h * route.shape[-1]:
        raise ValueError("canonical capacity differs from forward-bound route geometry")
    index32 = owner_tape_index32_safe(meta.capacity, tape_stride, bq)
    build_group_metadata(route, meta, workspace, plan, head_start=head_start)
    prefix_strides = k.stride()[:3] if branch else (0, 0, 0)
    gradient_strides = dk.stride()[:3] if branch else (0, 0, 0)
    nc = meta.count.numel() // h
    # reorder only independent CTAs for the validated 8K geometry
    # retain empty CTAs and each CTA's canonical row traversal
    ordered = (
        t == 8192
        and h == 16
        and d == 128
        and num_slots == 128
        and bq == 16
        and bc == 32
        and chunk_size in (64, 128, 256, 512)
        and nc * plan.group_size <= 2048
        and workspace.nbytes <= 255 * 1024**2
    )
    task_order = torch.argsort(workspace.count[: nc * plan.group_size], descending=True, stable=True) if ordered else None
    producer_grid = (
        (triton.cdiv(extent, bc), nc * plan.group_size, 1) if ordered else (nc * plan.group_size, triton.cdiv(extent, bc), 1)
    )
    _sparse_bwd_tape_kernel[producer_grid](
        q,
        k,
        v,
        dout,
        lse,
        delta,
        meta.count,
        meta.offset,
        meta.rho,
        block_ptr,
        prefix_valid.view(torch.uint8),
        dq,
        dk,
        dv,
        workspace.tape,
        workspace.offset,
        head_start,
        t,
        d ** (-0.5),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        dout.stride(0),
        dout.stride(1),
        *prefix_strides,
        *gradient_strides,
        H=h,
        D=d,
        M=num_slots,
        BC=bc,
        BQ=bq,
        BD=max(16, triton.next_power_of_2(d)),
        BRANCH=branch,
        GROUP_SIZE=plan.group_size,
        TAPE_STRIDE=tape_stride,
        INDEX32=index32,
        TAPE_EDGES=t * plan.group_size * route.shape[-1],
        TILE_MAJOR=tile_major,
        TASK_ORDER=task_order,
        ORDERED=ordered,
        num_warps=4,
        num_stages=producer_stages,
    )
    _owner_from_tape_kernel[triton.cdiv(t, 16), plan.group_size](
        q,
        k,
        route,
        workspace.edge_map,
        workspace.tape,
        block_ptr,
        prefix_valid.view(torch.uint8),
        dq,
        t,
        route.shape[-1],
        route.stride(0),
        route.stride(1),
        head_start,
        k.stride(0),
        k.stride(1),
        *prefix_strides,
        d ** (-0.5),
        H=h,
        D=d,
        M=num_slots,
        C=chunk_size,
        BQ=16,
        BC=bc,
        BD=max(16, triton.next_power_of_2(d)),
        KP=triton.next_power_of_2(route.shape[-1]),
        BRANCH=branch,
        MAP_WIDTH=plan.map_width,
        TAPE_STRIDE=tape_stride,
        TAPE_EDGES=t * plan.group_size * route.shape[-1],
        TILE_MAJOR=tile_major,
        num_warps=OWNER_NUM_WARPS,
        num_stages=OWNER_NUM_STAGES,
    )


def backward_with_tape(
    qr,
    qw,
    kr,
    value,
    pk,
    pv,
    dout,
    lse,
    delta,
    metadata,
    block_ptr,
    prefix_valid,
    rtok,
    rstate,
    grads,
    plan: OwnerBranchTapePlan,
    *,
    chunk_size,
    bc,
    bq,
):
    t, h, _ = qr.shape
    geometry = (t, h, pk.shape[0], chunk_size, pk.shape[1], rtok.shape[-1], rstate.shape[-1], bc, qr.element_size())
    if plan.geometry != geometry:
        raise ValueError("owner tape invocation differs from forward-bound geometry")
    workspace = allocate_tape_workspace(qr, plan)
    branches = (
        (0, qr, kr, value, rtok, metadata.tok, grads.dq_rope, grads.dk_rope, grads.dv),
        (1, qw, pk, pv, rstate, metadata.state, grads.dq_raw, grads.dpk, grads.dpv),
    )
    for branch, q, k, v, route, meta, dq, dk, dv in branches:
        if route.shape[-1] == 0:
            dq.zero_()
            continue
        branch_plan = plan.token if branch == 0 else plan.state
        if not branch_plan.enabled:
            raise ValueError("enabled owner plan lacks a required branch")
        for head_start in range(0, h, branch_plan.group_size):
            run_tape_group(
                q,
                k,
                v,
                dout,
                lse,
                delta,
                route,
                meta,
                block_ptr,
                prefix_valid,
                dq,
                dk,
                dv,
                workspace,
                branch_plan,
                head_start=head_start,
                branch=branch,
                chunk_size=chunk_size,
                num_slots=pk.shape[1],
                bc=bc,
                bq=bq,
            )
    return grads
