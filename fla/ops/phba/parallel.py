# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Deterministic block-major PHBA attention and bounded forward workspace."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from fla.ops.phba.flash_attention import local_attention
from fla.ops.phba.plans import CHUNKWISE_VALIDATED_C, INT32_MAX, _local_fa_batches, kpad_tok
from fla.ops.phba.utils import fail
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@dataclass(frozen=True)
class BlockMajorConfig:
    window_tokens: int = 2048
    bq: int = 64
    bc: int = 32
    owner_br: int = 16
    workspace_budget_bytes: int = 256 * 1024 * 1024
    bq_backward: int = 16
    fa_segment_batch_max: int = 65535
    fa_rows_per_call_max: int = 1 << 20

    def __post_init__(self):
        for name in ("window_tokens", "workspace_budget_bytes", "fa_segment_batch_max", "fa_rows_per_call_max"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if any(type(getattr(self, name)) is not int for name in ("bq", "bc", "owner_br", "bq_backward")):
            raise ValueError("kernel tile sizes must be integers")
        if self.bq not in (16, 32, 64) or self.bc not in (16, 32, 64) or self.owner_br not in (16, 32):
            raise ValueError("unsupported PHBA forward tile")
        if self.bq_backward not in (16, 32):
            raise ValueError("unsupported backward query tile")


def workspace_plan(t, h, d, kh, nc, config):
    if any(type(x) is not int or x < 0 for x in (t, h, d, kh, nc)) or min(t, h, d, nc) < 1:
        raise ValueError("invalid operator geometry")
    w = min(t, config.window_tokens)
    edges, experts = w * h * kh, nc * h
    if t * h > INT32_MAX or edges > INT32_MAX or experts + 1 > INT32_MAX:
        raise fail("int32_overflow", "PHBA row/edge/expert indices exceed int32")
    task_capacity = triton.cdiv(edges, config.bq) + experts if kh else 0
    if task_capacity > INT32_MAX:
        raise fail("int32_overflow", "PHBA task count exceeds int32")
    owner_capacity = triton.cdiv(w, config.owner_br) + min(w, nc) + 1
    metadata_bytes = 4 * (5 * experts + 2 + 2 * edges + 2 * task_capacity) if kh else 0
    partial_bytes = 4 * edges * (d + 2)
    owner_bytes = 12 * owner_capacity
    total = metadata_bytes + partial_bytes + owner_bytes
    if total > config.workspace_budget_bytes:
        raise fail("invalid_config", f"PHBA workspace {total} exceeds budget {config.workspace_budget_bytes}")
    return dict(
        window_tokens=w,
        edge_capacity=edges,
        expert_count=experts,
        task_capacity=task_capacity,
        owner_capacity=owner_capacity,
        metadata_bytes=metadata_bytes,
        partial_bytes=partial_bytes,
        workspace_bytes=total,
    )


@triton.jit(do_not_specialize=["T", "Q0", "W", "K", "RH", "RT"])
def _edge_count(route, count, T, Q0, W, K, RH, RT, H: tl.constexpr, BT: tl.constexpr, KP: tl.constexpr):
    h = tl.program_id(1).to(tl.int64)
    row = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    slot = tl.arange(0, KP)
    b = tl.load(
        route + (h.to(tl.int64) * RH + (Q0 + row).to(tl.int64)[:, None] * RT + slot[None, :]),
        (row < W)[:, None] & (slot < K)[None, :],
        other=-1,
    ).to(
        tl.int64,
    )
    tl.atomic_add(count + (tl.maximum(b, 0) * H + h), 1, b >= 0)


@triton.jit(do_not_specialize=["T", "Q0", "W", "K", "RH", "RT"])
def _edge_scatter(
    route, offset, cursor, rho, edge_pos, T, Q0, W, K, RH, RT, H: tl.constexpr, BT: tl.constexpr, KP: tl.constexpr
):
    h = tl.program_id(1).to(tl.int64)
    row = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    slot = tl.arange(0, KP)
    valid = (row < W)[:, None] & (slot < K)[None, :]
    b = tl.load(route + (h.to(tl.int64) * RH + (Q0 + row).to(tl.int64)[:, None] * RT + slot[None, :]), valid, other=-1).to(
        tl.int64,
    )
    valid = valid & (b >= 0)
    expert = tl.maximum(b, 0) * H + h
    position = (tl.atomic_add(cursor + (expert), 1, valid)).to(tl.int64)
    base = (tl.load(offset + (expert), valid, other=0)).to(tl.int64)
    edge = base + position
    r = (Q0 + row) * H + h
    tl.store(rho + (edge.to(tl.int64)), r[:, None] + tl.zeros((BT, KP), tl.int32), valid)
    address = ((row) * H + h)[:, None] * K + slot[None, :]
    tl.store(edge_pos + (address), edge, valid)


@triton.jit(do_not_specialize=["EXPERTS"])
def _task_counts(count, tasks, EXPERTS, BQ: tl.constexpr, BT: tl.constexpr):
    e = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    n = tl.load(count + (e), e < EXPERTS, other=0)
    tl.store(tasks + (e), tl.cdiv(n, BQ), e < EXPERTS)


@triton.jit
def _task_scatter(offset, task_count, task_offset, work_expert, work_edge, BQ: tl.constexpr, BT: tl.constexpr):
    expert = tl.program_id(0).to(tl.int64)
    count = tl.load(task_count + (expert))
    start = (tl.load(task_offset + (expert))).to(tl.int64)
    edge = (tl.load(offset + (expert))).to(tl.int64)
    lane = (tl.arange(0, BT)).to(tl.int64)
    for i in range(0, count, BT):
        tile = i + lane
        tl.store(work_expert + (start + tile), expert, tile < count)
        tl.store(work_edge + (start + tile), edge + tile * BQ, tile < count)


@triton.jit
def _update(m, z, u, scores, values):
    maximum = tl.maximum(m, tl.max(scores, 1, keep_dims=True))
    maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    alpha = tl.where(z > 0, tl.exp(m - maximum), 0.0)
    p = tl.exp(scores - maximum)
    update = tl.dot(p.to(values.dtype), values, out_dtype=tl.float32)
    return maximum, alpha * z + tl.sum(p, 1, keep_dims=True), alpha * u + update


@triton.jit(do_not_specialize=["T", "EXPERTS"])
def _history_partial(
    qr,
    qw,
    kr,
    v,
    pk,
    pv,
    rho,
    offset,
    task_offset,
    work_expert,
    work_edge,
    bp,
    prefix_valid,
    pm,
    pz,
    pu,
    T,
    EXPERTS,
    SCALE,
    SQT,
    SQH,
    SWT,
    SWH,
    SKT,
    SKH,
    SVT,
    SVH,
    SPB,
    SPM,
    SPH,
    H: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    BQ: tl.constexpr,
    BC: tl.constexpr,
    BD: tl.constexpr,
    INDEX32: tl.constexpr,
):
    task = tl.program_id(0)
    if task >= tl.load(task_offset + tl.cast(EXPERTS, tl.int64)):
        return
    expert = tl.load(work_expert + task.to(tl.int64))
    block, head = expert // H, (expert % H).to(tl.int64)
    begin = tl.load(work_edge + task.to(tl.int64))
    finish = tl.load(offset + expert.to(tl.int64) + 1)
    row_lane = tl.arange(0, BQ)
    if INDEX32:
        pos = begin + row_lane
    else:
        pos = begin.to(tl.int64) + row_lane
    valid = pos < finish
    r = tl.load(rho + pos.to(tl.int64), valid, other=0)
    t = (r // H).to(tl.int64)
    d = tl.arange(0, BD)
    qm = valid[:, None] & (d < D)[None, :]
    qrt = tl.load(qr + t[:, None] * SQT + head * SQH + d[None, :], qm, other=0)
    qwt = tl.load(qw + t[:, None] * SWT + head * SWH + d[None, :], qm, other=0)
    m = tl.full((BQ, 1), float("-inf"), tl.float32)
    z = tl.zeros((BQ, 1), tl.float32)
    u = tl.zeros((BQ, BD), tl.float32)
    lane = tl.arange(0, BC)
    start, end = tl.load(bp + block.to(tl.int64)), tl.load(bp + block.to(tl.int64) + 1)
    for k0 in range(0, C, BC):
        memory = k0 + lane < end - start
        ix = start.to(tl.int64) + k0 + lane
        mask = memory[:, None] & (d < D)[None, :]
        kt = tl.load(kr + ix.to(tl.int64)[:, None] * SKT + head * SKH + d[None, :], mask, other=0)
        vt = tl.load(v + ix.to(tl.int64)[:, None] * SVT + head * SVH + d[None, :], mask, other=0)
        score = tl.dot(qrt, tl.trans(kt), out_dtype=tl.float32) * SCALE
        score = tl.where(valid[:, None] & memory[None, :], score, float("-inf"))
        m, z, u = _update(m, z, u, score, vt)
    if tl.load(prefix_valid + block.to(tl.int64)) != 0:
        for k0 in range(0, M, BC):
            slot = k0 + lane
            mask = (slot < M)[:, None] & (d < D)[None, :]
            address = block.to(tl.int64) * SPB + head * SPH + slot.to(tl.int64)[:, None] * SPM + d[None, :]
            kt = tl.load(pk + address, mask, other=0)
            vt = tl.load(pv + address, mask, other=0)
            score = tl.dot(qwt, tl.trans(kt), out_dtype=tl.float32) * SCALE
            score = tl.where(valid[:, None] & (slot < M)[None, :], score, float("-inf"))
            m, z, u = _update(m, z, u, score, vt)
    tl.store(pm + pos.to(tl.int64)[:, None], m, valid[:, None])
    tl.store(pz + pos.to(tl.int64)[:, None], z, valid[:, None])
    tl.store(pu + pos.to(tl.int64)[:, None] * D + d[None, :], u, qm)


@triton.jit(do_not_specialize=["T", "Q0", "K"])
def _owner_merge(
    qw,
    pk,
    pv,
    oloc,
    lloc,
    edge_pos,
    pm,
    pz,
    pu,
    owner_block,
    owner_start,
    owner_end,
    prefix_valid,
    out,
    lse,
    T,
    Q0,
    K,
    SCALE,
    SWT,
    SWH,
    SPB,
    SPM,
    SPH,
    H: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    BD: tl.constexpr,
    HISTORY: tl.constexpr,
):
    work, head = tl.program_id(0), tl.program_id(1).to(tl.int64)
    block = tl.load(owner_block + work.to(tl.int64))
    start, end = tl.load(owner_start + work.to(tl.int64)), tl.load(owner_end + work.to(tl.int64))
    row_lane = tl.arange(0, BR)
    valid = row_lane < end - start
    row = start.to(tl.int64) + row_lane
    d = tl.arange(0, BD)
    mask = valid[:, None] & (d < D)[None, :]
    t = row.to(tl.int64)
    q = tl.load(qw + t[:, None] * SWT + head * SWH + d[None, :], mask, other=0)
    m = tl.load(lloc + head * T + t[:, None], valid[:, None], other=0)
    z = tl.full((BR, 1), 1.0, tl.float32)
    u = tl.load(oloc + (t[:, None] * H + head) * D + d[None, :], mask, other=0).to(tl.float32)
    if tl.load(prefix_valid + block.to(tl.int64)) != 0:
        lane = tl.arange(0, BC)
        for k0 in range(0, M, BC):
            slot = k0 + lane
            mm = (slot < M)[:, None] & (d < D)[None, :]
            address = block.to(tl.int64) * SPB + head * SPH + slot.to(tl.int64)[:, None] * SPM + d[None, :]
            kt = tl.load(pk + address, mm, other=0)
            vt = tl.load(pv + address, mm, other=0)
            score = tl.dot(q, tl.trans(kt), out_dtype=tl.float32) * SCALE
            score = tl.where(valid[:, None] & (slot < M)[None, :], score, float("-inf"))
            m, z, u = _update(m, z, u, score, vt)
    if HISTORY:
        for slot in range(K):
            row_edge = ((t - Q0) * H + head) * K + slot
            edge = tl.load(edge_pos + row_edge, valid, other=-1)
            live = valid & (edge >= 0)
            safe = tl.maximum(edge, 0).to(tl.int64)
            hm = tl.load(pm + safe[:, None], live[:, None], other=float("-inf"))
            hz = tl.load(pz + safe[:, None], live[:, None], other=0)
            hu = tl.load(pu + safe[:, None] * D + d[None, :], mask & live[:, None], other=0)
            maximum = tl.maximum(m, hm)
            a = tl.exp(m - maximum)
            b = tl.where(hz > 0, tl.exp(hm - maximum), 0.0)
            z, u = a * z + b * hz, a * u + b * hu
            m = maximum
    value = u / z
    tl.store(out + (t[:, None] * H + head) * D + d[None, :], value, mask)
    tl.store(lse + head * T + t[:, None], m + tl.log(z), valid[:, None])


class WindowWorkspace:
    def __init__(self, plan, d, device):
        self.plan = plan
        e, n, tasks = plan["edge_capacity"], plan["expert_count"], plan["task_capacity"]
        self.count = torch.zeros(n if e else 0, dtype=torch.int32, device=device)
        self.offset = torch.zeros(n + 1 if e else 0, dtype=torch.int32, device=device)
        self.cursor = torch.zeros_like(self.count)
        self.rho = torch.empty(e, dtype=torch.int32, device=device)
        self.edge_pos = torch.full((e,), -1, dtype=torch.int32, device=device)
        self.tasks = torch.empty(n if e else 0, dtype=torch.int32, device=device)
        self.task_offset = torch.zeros(n + 1 if e else 0, dtype=torch.int32, device=device)
        self.work_expert = torch.empty(tasks, dtype=torch.int32, device=device)
        self.work_edge = torch.empty_like(self.work_expert)
        self.pm = torch.empty(e, dtype=torch.float32, device=device)
        self.pz = torch.empty_like(self.pm)
        self.pu = torch.empty((e, d), dtype=torch.float32, device=device)
        self.owner_block = torch.empty(plan["owner_capacity"], dtype=torch.int32, device=device)
        self.owner_start = torch.empty_like(self.owner_block)
        self.owner_end = torch.empty_like(self.owner_block)

    def bytes(self):
        return sum(value.numel() * value.element_size() for value in vars(self).values() if isinstance(value, torch.Tensor))


def _owners(layout, q0, q1, br):
    block, start, end = [], [], []
    b0 = bisect_right(layout.block_ptr_host, q0) - 1
    for b in range(b0, layout.Nc):
        begin = max(q0, layout.block_ptr_host[b])
        finish = min(q1, layout.block_ptr_host[b + 1])
        if begin >= q1:
            break
        for row in range(begin, finish, br):
            block.append(b)
            start.append(row)
            end.append(min(row + br, finish))
    return block, start, end


@triton.jit(do_not_specialize=["Q0", "Q1", "EXPERTS"])
def _canonical_window_bounds(
    global_offset, global_rho, begin, count, tasks, Q0, Q1, EXPERTS, H: tl.constexpr, BQ: tl.constexpr, BT: tl.constexpr
):
    expert = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT).to(tl.int64)
    valid = expert < EXPERTS
    left = tl.load(global_offset + (expert), valid, other=0).to(tl.int64)
    right = tl.load(global_offset + (expert + 1), valid, other=0).to(tl.int64)
    lo, hi = left, right
    low_target = tl.cast(Q0, tl.int64) * H
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        active = lo < hi
        value = tl.load(global_rho + tl.cast(mid, tl.int64), active, other=2147483647).to(tl.int64)
        take = active & (value < low_target)
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(active & ~take, mid, hi)
    first = lo
    hi = right
    high_target = tl.cast(Q1, tl.int64) * H
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        active = lo < hi
        value = tl.load(global_rho + tl.cast(mid, tl.int64), active, other=2147483647).to(tl.int64)
        take = active & (value < high_target)
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(active & ~take, mid, hi)
    size = (lo - first).to(tl.int32)
    tl.store(begin + (expert), first.to(tl.int32), valid)
    tl.store(count + (expert), size, valid)
    tl.store(tasks + (expert), tl.cdiv(size, BQ), valid)


@triton.jit(do_not_specialize=["Q0", "K", "EXPERTS"])
def _canonical_window_materialize(
    global_rho,
    source_slots,
    begin,
    offset,
    task_offset,
    work_expert,
    work_edge,
    rho,
    edge_pos,
    Q0,
    K,
    EXPERTS,
    H: tl.constexpr,
    BQ: tl.constexpr,
):
    task = tl.program_id(0).to(tl.int64)
    if task >= tl.load(task_offset + tl.cast(EXPERTS, tl.int64)):
        return
    expert = tl.load(work_expert + (task)).to(tl.int64)
    local_begin = tl.load(offset + (expert)).to(tl.int64)
    local_end = tl.load(offset + (expert + 1)).to(tl.int64)
    pos = tl.load(work_edge + (task)).to(tl.int64) + tl.arange(0, BQ).to(tl.int64)
    valid = pos < local_end
    original = tl.load(begin + (expert)).to(tl.int64) + pos - local_begin
    row = tl.load(global_rho + (original), valid, other=0)
    slot = tl.load(source_slots + (original), valid, other=0).to(tl.int64)
    token = (row // H).to(tl.int64)
    head = expert % H
    address = ((token - tl.cast(Q0, tl.int64)) * H + head) * tl.cast(K, tl.int64) + slot
    tl.store(rho + (pos), row, valid)
    tl.store(edge_pos + (address), pos.to(tl.int32), valid)


def build_window_metadata(rtok, layout, q0, q1, ws, config, *, canonical_token=None):
    w, h, kh = q1 - q0, rtok.shape[0], rtok.shape[-1]
    if kh and canonical_token is not None:
        n = ws.plan["expert_count"]
        ws.edge_pos.fill_(-1)
        _canonical_window_bounds[triton.cdiv(n, 128),](
            canonical_token.offset,
            canonical_token.rho,
            ws.cursor,
            ws.count,
            ws.tasks,
            q0,
            q1,
            n,
            H=h,
            BQ=config.bq,
            BT=128,
            num_warps=4,
            num_stages=1,
        )
        torch.cumsum(ws.count, 0, dtype=torch.int32, out=ws.offset[1:])
        torch.cumsum(ws.tasks, 0, dtype=torch.int32, out=ws.task_offset[1:])
        _task_scatter[n,](ws.offset, ws.tasks, ws.task_offset, ws.work_expert, ws.work_edge, BQ=config.bq, BT=32, num_warps=4)
        _canonical_window_materialize[ws.plan["task_capacity"],](
            canonical_token.rho,
            canonical_token.source_slots,
            ws.cursor,
            ws.offset,
            ws.task_offset,
            ws.work_expert,
            ws.work_edge,
            ws.rho,
            ws.edge_pos,
            q0,
            kh,
            n,
            H=h,
            BQ=config.bq,
            num_warps=4,
            num_stages=1,
        )
    elif kh:
        ws.count.zero_()
        ws.cursor.zero_()
        ws.edge_pos.fill_(-1)
        grid = (triton.cdiv(w, 32), h)
        options = dict(H=h, BT=32, KP=kpad_tok(kh), num_warps=4)
        scalars = (layout.T, q0, w, kh, rtok.stride(0), rtok.stride(1))
        _edge_count[grid](rtok, ws.count, *scalars, **options)
        torch.cumsum(ws.count, 0, dtype=torch.int32, out=ws.offset[1:])
        _edge_scatter[grid](rtok, ws.offset, ws.cursor, ws.rho, ws.edge_pos, *scalars, **options)
        n = ws.plan["expert_count"]
        _task_counts[(triton.cdiv(n, 128),)](ws.count, ws.tasks, n, BQ=config.bq, BT=128, num_warps=4)
        torch.cumsum(ws.tasks, 0, dtype=torch.int32, out=ws.task_offset[1:])
        _task_scatter[n,](ws.offset, ws.tasks, ws.task_offset, ws.work_expert, ws.work_edge, BQ=config.bq, BT=32, num_warps=4)
    owners = _owners(layout, q0, q1, config.owner_br)
    if len(owners[0]) > ws.plan["owner_capacity"]:
        raise RuntimeError("owner worklist exceeds planned capacity")
    for name, values in zip(("owner_block", "owner_start", "owner_end"), owners):
        host = torch.tensor(values, dtype=torch.int32, device="cpu", pin_memory=rtok.device.type == "cuda")
        getattr(ws, name)[: len(values)].copy_(host, non_blocking=rtok.device.type == "cuda")
    return len(owners[0])


def launch_history(qr, qw, kr, v, pk, pv, layout, ws, config):
    t, h, d = qr.shape
    _history_partial[ws.plan["task_capacity"],](
        qr,
        qw,
        kr,
        v,
        pk,
        pv,
        ws.rho,
        ws.offset,
        ws.task_offset,
        ws.work_expert,
        ws.work_edge,
        layout.block_ptr,
        layout.prefix_valid.view(torch.uint8),
        ws.pm,
        ws.pz,
        ws.pu,
        t,
        ws.plan["expert_count"],
        d ** (-0.5),
        qr.stride(0),
        qr.stride(1),
        qw.stride(0),
        qw.stride(1),
        kr.stride(0),
        kr.stride(1),
        v.stride(0),
        v.stride(1),
        pk.stride(0),
        pk.stride(1),
        pk.stride(2),
        H=h,
        D=d,
        M=pk.shape[1],
        C=layout.chunk_size,
        BQ=config.bq,
        BC=config.bc,
        BD=triton.next_power_of_2(d),
        INDEX32=ws.plan["edge_capacity"] <= INT32_MAX - config.bq,
        num_warps=4,
        num_stages=1,
    )


def launch_owner(qw, pk, pv, oloc, lloc, layout, ws, config, q0, kh, owners, out, lse):
    t, h, d = qw.shape
    _owner_merge[owners, h](
        qw,
        pk,
        pv,
        oloc,
        lloc,
        ws.edge_pos,
        ws.pm,
        ws.pz,
        ws.pu,
        ws.owner_block,
        ws.owner_start,
        ws.owner_end,
        layout.prefix_valid.view(torch.uint8),
        out,
        lse,
        t,
        q0,
        kh,
        d ** (-0.5),
        qw.stride(0),
        qw.stride(1),
        pk.stride(0),
        pk.stride(1),
        pk.stride(2),
        H=h,
        D=d,
        M=pk.shape[1],
        BR=config.owner_br,
        BC=config.bc,
        BD=triton.next_power_of_2(d),
        HISTORY=kh > 0,
        num_warps=4,
        num_stages=1,
    )


def validate_inputs(qr, qw, kr, v, pk, pv, rtok, layout):
    if qr.ndim != 3 or qr.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("PHBA inputs require storage fp16/bf16 [T,H,D]")
    t, h, d = qr.shape
    supported_chunk = layout.chunk_size in CHUNKWISE_VALIDATED_C
    if t != layout.T or d < 16 or d > 128 or d % 8 or not supported_chunk:
        raise ValueError("unsupported PHBA geometry")
    if any(
        (
            x.shape != qr.shape or x.dtype != qr.dtype or x.device != qr.device or (not x.is_contiguous())
            for x in (qr, qw, kr, v)
        ),
    ):
        raise ValueError("matching contiguous Q/K/V required")
    if pk.ndim != 4 or pk.shape[0] != layout.Nc or pk.shape[2:] != (h, d) or not 1 <= pk.shape[1] <= 128:
        raise ValueError("invalid prefix shape")
    if pv.shape != pk.shape or pv.stride() != pk.stride() or pk.stride(-1) != 1:
        raise ValueError("prefix K/V layout must match")
    if any(x.device != qr.device or x.dtype != qr.dtype for x in (pk, pv)):
        raise ValueError("prefix dtype/device mismatch")
    if (
        rtok.ndim != 3
        or rtok.shape[:2] != (h, t)
        or rtok.dtype != torch.int32
        or rtok.device != qr.device
        or rtok.shape[-1] > 15
    ):
        raise ValueError("invalid frozen route contract")
    if rtok.shape[-1] and rtok.stride(-1) != 1:
        raise ValueError("route slots must be contiguous")
    if layout.device != qr.device:
        raise ValueError("layout device mismatch")
    if qr.device.type != "cuda":
        raise RuntimeError("PHBA kernels require CUDA")


def block_major_forward(qr, qw, kr, v, pk, pv, rtok, layout, *, config=BlockMajorConfig(), canonical_token=None):
    validate_inputs(qr, qw, kr, v, pk, pv, rtok, layout)
    t, h, d = qr.shape
    kh = rtok.shape[-1]
    plan = workspace_plan(t, h, d, kh, layout.Nc, config)
    if canonical_token is not None:
        slots = canonical_token.source_slots
        if (
            slots is None
            or slots.dtype != torch.uint8
            or not slots.is_contiguous()
            or slots.numel() != t * h * kh
            or slots.device != qr.device
        ):
            raise ValueError("invalid canonical route-slot metadata")
        if plan["workspace_bytes"] + slots.numel() * slots.element_size() > config.workspace_budget_bytes:
            raise ValueError("canonical reuse exceeds forward workspace budget")
    ws = WindowWorkspace(plan, d, qr.device)
    if ws.bytes() != plan["workspace_bytes"]:
        raise RuntimeError("workspace allocation does not match budget plan")
    out, lse = torch.empty_like(qr), torch.empty((h, t), dtype=torch.float32, device=qr.device)
    oloc, lloc = local_attention(
        qr,
        kr,
        v,
        layout,
        scale=d ** (-0.5),
        fa_segment_batch_max=config.fa_segment_batch_max,
        fa_rows_per_call_max=config.fa_rows_per_call_max,
    )
    for q0 in range(0, t, plan["window_tokens"]):
        q1 = min(q0 + plan["window_tokens"], t)
        owners = build_window_metadata(rtok, layout, q0, q1, ws, config, canonical_token=canonical_token)
        if kh:
            launch_history(qr, qw, kr, v, pk, pv, layout, ws, config)
        launch_owner(qw, pk, pv, oloc, lloc, layout, ws, config, q0, kh, owners, out, lse)
    return out, lse


class BlockMajorAttentionFn(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, qr, qw, kr, v, pk, pv, rtok, rstate, layout, config, needs_backward):
        from fla.ops.phba.metadata import build_direct_metadata

        retain_slots = False
        if needs_backward and rtok.shape[-1] > 0:
            t, h, d = qr.shape
            estimate = workspace_plan(t, h, d, rtok.shape[-1], layout.Nc, config)
            retain_slots = estimate["workspace_bytes"] + rtok.numel() <= config.workspace_budget_bytes
        meta = (
            build_direct_metadata(rtok, rstate, layout.Nc, retain_token_slots=retain_slots, derived_state_layout=layout)
            if needs_backward
            else None
        )
        out, lse = block_major_forward(
            qr,
            qw,
            kr,
            v,
            pk,
            pv,
            rtok,
            layout,
            config=config,
            canonical_token=meta.tok if retain_slots else None,
        )
        ctx.mark_non_differentiable(lse)
        if needs_backward:
            ctx.settings = (layout.chunk_size, config.bc, config.bq_backward)
            from fla.ops.phba.plans import owner_branch_tape_plan

            ctx.owner_tape_plan = owner_branch_tape_plan(
                qr.shape[0],
                qr.shape[1],
                layout.Nc,
                layout.chunk_size,
                pk.shape[1],
                rtok.shape[-1],
                rstate.shape[-1],
                config.bc,
                qr.element_size(),
            )
            ctx.batches = _local_fa_batches(
                layout.block_ptr_host,
                layout.chunk_size,
                config.fa_segment_batch_max,
                config.fa_rows_per_call_max,
            )
            ctx.has_token = meta.tok is not None
            saved = [
                qr,
                qw,
                kr,
                v,
                pk,
                pv,
                out,
                lse,
                layout.block_ptr,
                layout.prefix_valid,
                meta.state.count,
                meta.state.offset,
                meta.state.rho,
            ]
            if ctx.has_token:
                saved.extend((meta.tok.count, meta.tok.offset, meta.tok.rho))
            saved.extend((rtok, rstate))
            ctx.save_for_backward(*saved)
        return out, lse

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout, dlse):
        from fla.ops.phba.chunk_bwd import attention_backward

        return attention_backward(ctx, dout, dlse)


def block_major_attention(qr, qw, kr, v, pk, pv, rtok, rstate, layout, *, config=BlockMajorConfig()):
    if rstate.shape != (*rtok.shape[:2], rtok.shape[-1] + 1):
        raise ValueError("Rstate width differs from Kh+1")
    needs_backward = torch.is_grad_enabled() and any(x.requires_grad for x in (qr, qw, kr, v, pk, pv))
    return BlockMajorAttentionFn.apply(qr, qw, kr, v, pk, pv, rtok, rstate, layout, config, needs_backward)
