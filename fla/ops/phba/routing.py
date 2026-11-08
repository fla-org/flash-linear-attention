# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"Block geometry, key centroids and strict-history PHBA routing."

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import triton
import triton.language as tl

from fla.ops.phba.plans import MAX_KH, MIN_DOT, SequenceLengthPlan, _block_layout, _require_int, kpad_tok
from fla.ops.phba.utils import fail


@dataclass(frozen=True)
class RoutingLayout:
    chunk_size: int
    br: int
    Nseq: int
    Nc: int
    Nwork: int
    T: int
    block_ptr_host: tuple[int, ...] = field(repr=False)
    seq_block_ptr_host: tuple[int, ...] = field(repr=False)
    block_ptr: torch.Tensor = field(repr=False, compare=False)
    seq_block_ptr: torch.Tensor = field(repr=False, compare=False)
    block_seq: torch.Tensor = field(repr=False, compare=False)
    current_block: torch.Tensor = field(repr=False, compare=False)
    prefix_valid: torch.Tensor = field(repr=False, compare=False)
    work_block: torch.Tensor = field(repr=False, compare=False)
    work_start: torch.Tensor = field(repr=False, compare=False)
    work_end: torch.Tensor = field(repr=False, compare=False)
    work_seq: torch.Tensor = field(repr=False, compare=False)

    @property
    def device(self) -> torch.device:
        return self.block_ptr.device


def _i32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.int32).contiguous()


@torch.inference_mode(False)
def build_block_layout(length_plan: SequenceLengthPlan, chunk_size: int, br: int) -> RoutingLayout:
    """Build or reuse the routing layout cached by a sequence plan."""
    c = _require_int("chunk_size", chunk_size)
    br = _require_int("Br", br)
    cache = length_plan._routing_layouts  # noqa: SLF001
    key = (c, br)
    layout = cache.get(key)
    if layout is None:
        layout = _construct(length_plan, c, br)
        cache[key] = layout
    return layout


def _construct(lp: SequenceLengthPlan, c: int, br: int) -> RoutingLayout:
    block_ptr_host, seq_block_ptr_host, nwork = _block_layout(lp.lengths, c, br)
    nc, nseq, T = len(block_ptr_host) - 1, lp.Nseq, lp.T
    dev = lp.device
    # reuse the sequence plan's device block pointers for routing and prefix states
    blayout = lp.block_layout(c)
    block_ptr = blayout.block_ptr.to(torch.int64)
    seq_block_ptr = blayout.seq_block_ptr.to(torch.int64)
    nblk = seq_block_ptr[1:] - seq_block_ptr[:-1]
    block_seq = torch.repeat_interleave(torch.arange(nseq, device=dev), nblk, output_size=nc)  # [Nc]
    block_len = block_ptr[1:] - block_ptr[:-1]  # [Nc], including partial tails
    current_block = torch.repeat_interleave(torch.arange(nc, device=dev), block_len, output_size=T)  # [T]
    prefix_valid = torch.arange(nc, device=dev) > seq_block_ptr[block_seq]  # [Nc] bool
    # query tiles per block
    nq = torch.div(block_len + (br - 1), br, rounding_mode="floor")
    work_block = torch.repeat_interleave(torch.arange(nc, device=dev), nq, output_size=nwork)  # [Nwork]
    work_ptr = torch.zeros(nc + 1, dtype=torch.int64, device=dev)
    work_ptr[1:] = torch.cumsum(nq, 0)
    tile = torch.arange(nwork, device=dev) - work_ptr[work_block]
    work_start = block_ptr[work_block] + tile * br
    work_end = torch.minimum(work_start + br, block_ptr[work_block + 1])
    work_seq = block_seq[work_block]
    return RoutingLayout(
        chunk_size=c,
        br=br,
        Nseq=nseq,
        Nc=nc,
        Nwork=nwork,
        T=T,
        block_ptr_host=block_ptr_host,
        seq_block_ptr_host=seq_block_ptr_host,
        block_ptr=_i32(block_ptr),
        seq_block_ptr=_i32(seq_block_ptr),
        block_seq=_i32(block_seq),
        current_block=_i32(current_block),
        prefix_valid=prefix_valid.contiguous(),
        work_block=_i32(work_block),
        work_start=_i32(work_start),
        work_end=_i32(work_end),
        work_seq=_i32(work_seq),
    )


CENTROID_NUM_WARPS = 4


CENTROID_NUM_STAGES = 2


@triton.jit
def _centroid_kernel(
    k_ptr,
    block_ptr,
    out_ptr,
    stride_kt,
    stride_kh,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    i_b = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    start = (tl.load(block_ptr + (i_b))).to(tl.int64)
    end = (tl.load(block_ptr + (i_b + 1))).to(tl.int64)
    n = end - start
    inv = 1.0 / tl.maximum(n, 1).to(tl.float32)
    o_t = tl.arange(0, BLOCK_TOK)
    o_d = tl.arange(0, BLOCK_D)
    for d0 in range(0, D, BLOCK_D):
        m_d = (d0 + o_d) < D
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for t0 in range(0, n, BLOCK_TOK):
            m_t = (t0 + o_t) < n
            rows = start + t0 + o_t
            x = tl.load(
                k_ptr + (rows[:, None] * stride_kt + i_h * stride_kh + (d0 + o_d)[None, :]),
                mask=m_t[:, None] & m_d[None, :],
                other=0.0,
            )
            acc += tl.sum(x.to(tl.float32), axis=0)
        dst = out_ptr + ((tl.cast(i_b, tl.int64) * H + i_h) * D + d0 + o_d)
        tl.store(dst, acc * inv, mask=m_d)


def compute_block_centroids(
    k_rope: torch.Tensor, layout: RoutingLayout, *, block_tok: int = 64, block_d: int = 64
) -> torch.Tensor:
    """Average detached keys [T, H, Dh] into FP32 block centroids [Nc, H, Dh]."""
    if k_rope.dim() != 3 or k_rope.shape[0] != layout.T:
        raise fail("invalid_shape", f"Krope must be [T, H, Dh] with T={layout.T}, got {tuple(k_rope.shape)}")
    if k_rope.stride(-1) != 1:
        raise fail("non_contiguous_input", "Krope must have a contiguous final dimension")
    T, H, D = k_rope.shape
    out = torch.empty((layout.Nc, H, D), dtype=torch.float32, device=k_rope.device)
    if layout.Nc == 0:
        return out
    _centroid_kernel[layout.Nc, H](
        k_rope.detach(),
        layout.block_ptr,
        out,
        k_rope.stride(0),
        k_rope.stride(1),
        H=H,
        D=D,
        BLOCK_TOK=block_tok,
        BLOCK_D=block_d,
        num_warps=CENTROID_NUM_WARPS,
        num_stages=CENTROID_NUM_STAGES,
    )
    return out


TOPK_NUM_WARPS = 4


TOPK_NUM_STAGES = 2


@triton.jit(do_not_specialize=["Kh", "stride_rt_h", "stride_rt_t", "stride_rs_h", "stride_rs_t"])
def _topk_kernel(
    q_ptr,
    cen_ptr,
    work_block_ptr,
    work_start_ptr,
    work_end_ptr,
    work_seq_ptr,
    seq_block_ptr,
    prefix_valid_ptr,
    rtok_ptr,
    rstate_ptr,
    stride_qt,
    stride_qh,
    Kh,
    stride_rt_h,
    stride_rt_t,
    stride_rs_h,
    stride_rs_t,
    H: tl.constexpr,
    D: tl.constexpr,
    BR: tl.constexpr,
    BR_DOT: tl.constexpr,
    BC: tl.constexpr,
    BLOCK_D: tl.constexpr,
    KPAD: tl.constexpr,
):
    w = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    cb = tl.load(work_block_ptr + (w))
    start = (tl.load(work_start_ptr + (w))).to(tl.int64)
    end = (tl.load(work_end_ptr + (w))).to(tl.int64)
    seq = tl.load(work_seq_ptr + (w))
    b0 = tl.load(seq_block_ptr + tl.cast(seq, tl.int64))

    lane = tl.arange(0, BR_DOT)
    rows = start + lane
    qvalid = (lane < BR) & (rows < end)
    rows64 = rows
    o_d = tl.arange(0, BLOCK_D)
    kcol = tl.arange(0, KPAD)
    bcol = tl.arange(0, BC)

    best_s = tl.full([BR_DOT, KPAD], float("-inf"), tl.float32)
    best_i = tl.full([BR_DOT, KPAD], -1, tl.int32)
    for blk0 in range(b0, cb, BC):
        bids = blk0 + bcol
        bvalid = bids < cb
        score = tl.zeros([BR_DOT, BC], dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            m_d = (d0 + o_d) < D
            q = tl.load(
                q_ptr + (rows64[:, None] * stride_qt + h * stride_qh + (d0 + o_d)[None, :]),
                mask=qvalid[:, None] & m_d[None, :],
                other=0.0,
            )
            c = tl.load(
                cen_ptr + ((bids.to(tl.int64)[:, None] * H + h) * D + (d0 + o_d)[None, :]),
                mask=bvalid[:, None] & m_d[None, :],
                other=0.0,
            )
            score += tl.dot(q, tl.trans(c.to(q.dtype)), out_dtype=tl.float32)
        score = tl.where(qvalid[:, None] & bvalid[None, :], score, float("-inf"))

        # merge ranked entries with the ascending block-ID tile in a fixed number of selection rounds
        old_s = best_s
        old_i = best_i
        new_s = tl.full([BR_DOT, KPAD], float("-inf"), tl.float32)
        new_i = tl.full([BR_DOT, KPAD], -1, tl.int32)
        for r in range(KPAD):
            m_old = tl.max(old_s, axis=1)
            a_old = tl.argmax(old_s, axis=1, tie_break_left=True)
            m_new = tl.max(score, axis=1)
            a_new = tl.argmax(score, axis=1, tie_break_left=True)
            # ties retain the earlier candidate with the smaller block id
            take_new = m_new > m_old
            sel_s = tl.where(take_new, m_new, m_old)
            id_old = tl.sum(tl.where(kcol[None, :] == a_old[:, None], old_i, 0), axis=1)
            id_new = blk0 + a_new
            sel_i = tl.where(take_new, id_new, id_old)
            # invalid scores retain the route sentinel
            sel_i = tl.where(sel_s > float("-inf"), sel_i, -1)
            new_s = tl.where(kcol[None, :] == r, sel_s[:, None], new_s)
            new_i = tl.where(kcol[None, :] == r, sel_i[:, None], new_i)
            old_s = tl.where((m_new <= m_old)[:, None] & (kcol[None, :] == a_old[:, None]), float("-inf"), old_s)
            score = tl.where(take_new[:, None] & (bcol[None, :] == a_new[:, None]), float("-inf"), score)
        best_s = new_s
        best_i = new_i

    slot_ok = kcol < Kh
    m_store = qvalid[:, None] & slot_ok[None, :]
    tl.store(rtok_ptr + (h * stride_rt_h + rows64[:, None] * stride_rt_t + kcol[None, :]), best_i, mask=m_store)

    pv_cur = tl.load(prefix_valid_ptr + tl.cast(cb, tl.int64)).to(tl.int32)
    cur = tl.where(pv_cur != 0, cb, -1)
    tl.store(rstate_ptr + (h * stride_rs_h + rows64 * stride_rs_t), tl.zeros([BR_DOT], dtype=tl.int32) + cur, mask=qvalid)
    has = best_i >= 0
    pv = tl.load(prefix_valid_ptr + tl.cast(tl.where(has, best_i, 0), tl.int64), mask=has, other=0).to(tl.int32)
    hist = tl.where(has & (pv != 0), best_i, -1)
    tl.store(rstate_ptr + (h * stride_rs_h + rows64[:, None] * stride_rs_t + 1 + kcol[None, :]), hist, mask=m_store)


def route_topk(
    q_rope: torch.Tensor, centroids: torch.Tensor, layout: RoutingLayout, kh: int, *, bc: int = 32, block_d: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nondifferentiable int32 token/state routes, with -1 marking invalid slots."""
    if not 0 <= kh <= MAX_KH:
        raise fail("routing_unsupported", f"Kh={kh} must be in [0, {MAX_KH}] with KPAD_tok<=16")
    T, H, D = q_rope.shape
    if T != layout.T or centroids.shape != (layout.Nc, H, D):
        raise fail("invalid_shape", f"Qrope {tuple(q_rope.shape)} / Centroid {tuple(centroids.shape)} do not match the layout")
    if q_rope.stride(-1) != 1 or not centroids.is_contiguous():
        raise fail("non_contiguous_input", "Qrope must have a contiguous final dimension and Centroid must be contiguous")
    if centroids.dtype != torch.float32:
        raise fail("dtype_mismatch", f"Centroid must use FP32, got {centroids.dtype}")
    if bc < MIN_DOT or block_d < MIN_DOT:
        raise fail("invalid_config", f"Bc and BLOCK_D must be >= {MIN_DOT}")
    dev = q_rope.device
    rtok = torch.empty((H, T, kh), dtype=torch.int32, device=dev)
    if kh == 0:
        cur = torch.where(
            layout.prefix_valid[layout.current_block.long()],
            layout.current_block,
            torch.full_like(layout.current_block, -1),
        )
        rstate = cur.view(1, T, 1).expand(H, T, 1).contiguous()
        return rtok, rstate
    rstate = torch.empty((H, T, kh + 1), dtype=torch.int32, device=dev)
    br = layout.br
    _topk_kernel[layout.Nwork, H](
        q_rope.detach(),
        centroids,
        layout.work_block,
        layout.work_start,
        layout.work_end,
        layout.work_seq,
        layout.seq_block_ptr,
        layout.prefix_valid.view(torch.uint8),
        rtok,
        rstate,
        q_rope.stride(0),
        q_rope.stride(1),
        kh,
        rtok.stride(0),
        rtok.stride(1),
        rstate.stride(0),
        rstate.stride(1),
        H=H,
        D=D,
        BR=br,
        BR_DOT=max(br, MIN_DOT),
        BC=bc,
        BLOCK_D=block_d,
        KPAD=kpad_tok(kh),
        num_warps=TOPK_NUM_WARPS,
        num_stages=TOPK_NUM_STAGES,
    )
    return rtok, rstate


def build_routes(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    length_plan,
    *,
    chunk_size: int,
    topk: int,
    br: int = 8,
    bc: int = 32,
    block_d: int = 64,
    block_tok: int = 64,
):
    """Return the layout, token routes [H, T, topk - 1] and state routes [H, T, topk]."""
    layout: RoutingLayout = build_block_layout(length_plan, chunk_size, br)
    with torch.no_grad():
        cen = compute_block_centroids(k_rope, layout, block_tok=block_tok, block_d=block_d)
        rtok, rstate = route_topk(q_rope, cen, layout, int(topk) - 1, bc=bc, block_d=block_d)
    return layout, rtok, rstate


__all__ = [
    "RoutingLayout",
    "build_block_layout",
    "build_routes",
    "compute_block_centroids",
    "route_topk",
]
