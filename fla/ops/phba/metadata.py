# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"Canonical token and derived prefix-state metadata for deterministic PHBA."

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from fla.ops.phba.plans import INT32_MAX, kpad_state, kpad_tok
from fla.ops.phba.utils import fail


@triton.jit(do_not_specialize=["T", "K", "RH", "RT", "CAPACITY"])
def _route_keys_kernel(route, keys, T, K, RH, RT, CAPACITY, H: tl.constexpr, BT: tl.constexpr):
    SENTINEL: tl.constexpr = 2147483647 if keys.dtype.element_ty == tl.int32 else 9223372036854775807
    i = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    slot = i % K
    token = (i // K) % T
    head = i // (tl.cast(T, tl.int64) * K)
    block = tl.load(route + (head * RH + token * RT + slot), i < CAPACITY, other=-1).to(tl.int64)
    row_stride = tl.cast(T, tl.int64) * H
    expert = tl.maximum(block, 0) * H + head
    key = expert * row_stride + token * H + head
    sentinel = tl.full((BT,), SENTINEL, tl.int64)
    tl.store(keys + (i), tl.where(block >= 0, key, sentinel), i < CAPACITY)


@triton.jit(do_not_specialize=["CAPACITY", "ROW_STRIDE"])
def _unpack_rho_kernel(
    keys,
    rho,
    CAPACITY,
    ROW_STRIDE,
    BT: tl.constexpr,
    permutation=None,
    slots=None,
    K: tl.constexpr = 1,
    STORE_SLOTS: tl.constexpr = False,
):
    SENTINEL: tl.constexpr = 2147483647 if keys.dtype.element_ty == tl.int32 else 9223372036854775807
    i = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    key = tl.load(keys + (i), i < CAPACITY, other=SENTINEL)
    row = tl.where(key != SENTINEL, key % tl.cast(ROW_STRIDE, tl.int64), -1)
    tl.store(rho + (i), row.to(tl.int32), i < CAPACITY)
    if STORE_SLOTS:
        source = tl.load(permutation + (i), i < CAPACITY, other=0)
        tl.store(slots + (i), (source % K).to(tl.uint8), i < CAPACITY)


@triton.jit
def _sorted_bounds_kernel(
    keys, count, offset, CAP: tl.constexpr, ROW: tl.constexpr, NE: tl.constexpr, B: tl.constexpr, STEPS: tl.constexpr
):
    SENTINEL: tl.constexpr = 2147483647 if keys.dtype.element_ty == tl.int32 else 9223372036854775807
    e = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    lo = tl.full((B,), 0, tl.int64)
    hi = tl.full((B,), CAP, tl.int64)
    lo2 = tl.full((B,), 0, tl.int64)
    hi2 = tl.full((B,), CAP, tl.int64)
    target = e * ROW
    target2 = (e + 1) * ROW
    for _ in range(STEPS):
        mid = lo + (hi - lo) // 2
        mid2 = lo2 + (hi2 - lo2) // 2
        v = tl.load(keys + tl.cast(mid, tl.int64), (e <= NE) & (mid < CAP), other=SENTINEL)
        v2 = tl.load(keys + tl.cast(mid2, tl.int64), (e < NE) & (mid2 < CAP), other=SENTINEL)
        left = v < target
        left2 = v2 < target2
        next_lo = tl.where((lo < hi) & left, mid + 1, lo)
        hi = tl.where((lo < hi) & ~left, mid, hi)
        next_lo2 = tl.where((lo2 < hi2) & left2, mid2 + 1, lo2)
        hi2 = tl.where((lo2 < hi2) & ~left2, mid2, hi2)
        lo = next_lo
        lo2 = next_lo2
    tl.store(offset + (e), lo.to(tl.int32), e <= NE)
    tl.store(count + (e), (lo2 - lo).to(tl.int32), e < NE)


def canonical_rho(route: torch.Tensor, nc: int, *, return_slots: bool = False, return_counts: bool = False):
    """Sort full capacity, including invalid sentinels; never read a device scalar on the host."""
    if return_slots and not return_counts:
        raise fail("invalid_config", "Source slots require counts and offsets")
    h, t, k = route.shape
    if k < 1 or t < 1 or h < 1:
        raise fail("invalid_shape", "canonical metadata requires a nonempty branch")
    if (nc * h) * (t * h) >= 2**63 - 1:
        raise fail("int32_overflow", "canonical expert/query sort key exceeds int64")
    capacity = t * h * k
    key_dtype = torch.int32 if (nc * h) * (t * h) < 2**31 - 1 else torch.int64
    keys = torch.empty(capacity, dtype=key_dtype, device=route.device)
    _route_keys_kernel[triton.cdiv(capacity, 256),](
        route,
        keys,
        t,
        k,
        route.stride(0),
        route.stride(1),
        capacity,
        H=h,
        BT=256,
        num_warps=4,
        num_stages=1,
    )
    sorted_keys, permutation = torch.sort(keys)
    rho = torch.empty(capacity, dtype=torch.int32, device=route.device)
    slots = torch.empty(capacity, dtype=torch.uint8, device=route.device) if return_slots else None
    _unpack_rho_kernel[triton.cdiv(capacity, 256),](
        sorted_keys,
        rho,
        capacity,
        t * h,
        BT=256,
        permutation=permutation if return_slots else None,
        slots=slots,
        K=k,
        STORE_SLOTS=return_slots,
        num_warps=4,
        num_stages=1,
    )
    if return_counts:
        count = torch.empty(nc * h, dtype=torch.int32, device=route.device)
        offset = torch.empty(nc * h + 1, dtype=torch.int32, device=route.device)
        _sorted_bounds_kernel[triton.cdiv(nc * h + 1, 128),](
            sorted_keys,
            count,
            offset,
            CAP=capacity,
            ROW=t * h,
            NE=nc * h,
            B=128,
            STEPS=(capacity + 1).bit_length(),
            num_warps=4,
            num_stages=1,
        )
        return rho, slots, count, offset
    return rho


META_BT = 128


META_NUM_WARPS = 4


@triton.jit(do_not_specialize=["T", "K", "stride_rh", "stride_rt"])
def _count_kernel(route_ptr, count_ptr, T, K, stride_rh, stride_rt, H: tl.constexpr, BT: tl.constexpr, KPAD: tl.constexpr):
    pid_t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    t = pid_t * BT + tl.arange(0, BT)
    slot = tl.arange(0, KPAD)
    m = (t < T)[:, None] & (slot < K)[None, :]
    b = tl.load(route_ptr + (h * stride_rh + t.to(tl.int64)[:, None] * stride_rt + slot[None, :]), mask=m, other=-1).to(
        tl.int64,
    )
    valid = b >= 0
    e = tl.where(valid, b, 0) * H + h
    tl.atomic_add(count_ptr + (e), 1, mask=valid)


@dataclass(frozen=True)
class BranchMetadata:
    count: torch.Tensor  # [Nc * H] int32
    offset: torch.Tensor  # [Nc * H + 1] int32; offset[-1] = Nedge(device)
    rho: torch.Tensor  # [capacity] int32; valid prefix of length Nedge
    capacity: int
    # retain original route slots for forward metadata reuse
    source_slots: torch.Tensor | None = None


@dataclass(frozen=True)
class DirectMetadata:
    tok: BranchMetadata | None  # None when Kh is zero
    state: BranchMetadata


def _branch(
    route: torch.Tensor,
    nc: int,
    kpad: int,
    *,
    retain_slots: bool = False,
    sorted_counts: bool = False,
) -> BranchMetadata:
    H, T, K = route.shape
    if route.dtype != torch.int32 or route.stride(-1) != 1:
        raise fail(
            "invalid_shape",
            f"Routes must be int32 with a contiguous final dimension, got {route.dtype} and strides {route.stride()}",
        )
    cap = T * H * K
    if cap > INT32_MAX or nc * H + 1 > INT32_MAX:
        raise fail("int32_overflow", f"Rho capacity T*H*K={cap} or expert count Nc*H exceeds int32")
    if sorted_counts:
        rho, slots, count, offset = canonical_rho(route, nc, return_slots=retain_slots, return_counts=True)
        return BranchMetadata(count=count, offset=offset, rho=rho, capacity=cap, source_slots=slots)
    dev = route.device
    n_e = nc * H
    count = torch.zeros(n_e, dtype=torch.int32, device=dev)
    # initialize offset[0] without a synchronizing scalar write
    offset = torch.zeros(n_e + 1, dtype=torch.int32, device=dev)
    grid = (triton.cdiv(T, META_BT), H)
    args = (T, K, route.stride(0), route.stride(1))
    _count_kernel[grid](route, count, *args, H=H, BT=META_BT, KPAD=kpad, num_warps=META_NUM_WARPS)
    torch.cumsum(count, dim=0, dtype=torch.int32, out=offset[1:])

    rho = canonical_rho(route, nc)
    return BranchMetadata(count=count, offset=offset, rho=rho, capacity=cap)


def build_direct_metadata(
    rtok: torch.Tensor, rstate: torch.Tensor, nc: int, *, retain_token_slots: bool = False, derived_state_layout
) -> DirectMetadata:
    """Build canonical metadata for token routes [H, T, Kh] and state routes [H, T, Kh + 1]."""
    H, T, kh = rtok.shape
    if rstate.shape != (H, T, kh + 1):
        raise fail("invalid_shape", f"Rstate must have shape [H, T, Kh + 1]={(H, T, kh + 1)}, got {tuple(rstate.shape)}")
    tok = _branch(rtok, nc, kpad_tok(kh), retain_slots=retain_token_slots, sorted_counts=True) if kh > 0 else None
    if tok is not None:
        state = derive_state(tok, derived_state_layout, H, T * H * (kh + 1))
    else:
        state = _branch(rstate, nc, kpad_state(kh + 1))
    return DirectMetadata(tok=tok, state=state)


@triton.jit
def state_counts(tc, bp, pv, sc, N: tl.constexpr, H: tl.constexpr, B: tl.constexpr):
    e = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    block = e // H
    valid = e < N
    n = tl.load(tc + (e), valid, other=0)
    length = tl.load(bp + (block + 1), valid, other=0) - tl.load(bp + (block), valid, other=0)
    active = tl.load(pv + (block), valid, other=0)
    tl.store(sc + (e), tl.where(active, n + length, 0), valid)


@triton.jit
def state_rows(to, tr, bp, pv, so, sr, H: tl.constexpr, B: tl.constexpr):
    e = tl.program_id(0).to(tl.int64)
    block = e // H
    head = e % H
    if tl.load(pv + (block)) == 0:
        return
    start = tl.load(bp + (block)).to(tl.int64)
    end = tl.load(bp + (block + 1)).to(tl.int64)
    first = tl.load(to + (e)).to(tl.int64)
    last = tl.load(to + (e + 1)).to(tl.int64)
    dst = tl.load(so + (e)).to(tl.int64)
    cur = end - start
    total = cur + last - first
    lane = tl.arange(0, B).to(tl.int64)
    for base in range(0, total, B):
        ix = base + lane
        hist = tl.load(tr + (first + ix - cur), (ix >= cur) & (ix < total), other=0)
        row = tl.where(ix < cur, (start + ix) * H + head, hist)
        tl.store(sr + (dst + ix), row.to(tl.int32), ix < total)


def derive_state(tok, layout, h, capacity):
    if capacity > 2147483647 or tok.count.numel() + 1 > 2147483647:
        raise ValueError("state metadata exceeds int32 capacity")
    count = torch.empty_like(tok.count)
    offset = torch.zeros_like(tok.offset)
    state_counts[triton.cdiv(count.numel(), 128),](
        tok.count,
        layout.block_ptr,
        layout.prefix_valid,
        count,
        N=count.numel(),
        H=h,
        B=128,
    )
    torch.cumsum(count, 0, dtype=torch.int32, out=offset[1:])
    rho = torch.full((capacity,), -1, device=count.device, dtype=torch.int32)
    state_rows[(count.numel(),)](tok.offset, tok.rho, layout.block_ptr, layout.prefix_valid, offset, rho, H=h, B=128)
    return BranchMetadata(count, offset, rho, capacity)


__all__ = [
    "BranchMetadata",
    "DirectMetadata",
    "build_direct_metadata",
]
