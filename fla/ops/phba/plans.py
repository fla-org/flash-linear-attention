# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"Immutable sequence layouts and bounded deterministic backward workspaces."

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any
from weakref import ref

import torch

from fla.ops.phba.utils import fail

MIN_DOT = 16
INT32_MAX = 2**31 - 1
# FA varlen uses the grid y dimension for the batch
FA_SEGMENT_BATCH_LIMIT = 65535
# KPAD_tok <= 16
MAX_KH = 15
CHUNKWISE_VALIDATED_C = (32, 64, 128, 256, 512)
# prefix backward supports state and value widths up to 128
CHUNKWISE_TRAIN_MAX_M = 128
CHUNKWISE_TRAIN_MAX_DH = 128
PREFIX_REVERSE_WORKSPACE_CAP = 256 * 1024**2
OWNER_TAPE_CAP_BYTES = 256 * 1024**2
OWNER_TAPE_PRODUCER_STAGES = ((16, 2), (32, 2), (64, 1))


_CONSTRUCT_TOKEN = object()


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def kpad_tok(kh: int) -> int:
    """Round the token-route width to a power of two, with a minimum tile width of one."""
    return 1 if kh <= 1 else next_pow2(kh)


def kpad_state(ks: int) -> int:
    """Round state-route width independently; token-route padding cannot determine it."""
    return next_pow2(ks)


@dataclass(frozen=True)
class PrefixBackwardPlan:
    parallel: bool
    workspace_bytes: int


def prefix_backward_plan(nc: int, h: int, m: int, d: int) -> PrefixBackwardPlan:
    """Select a fixed backward algorithm using only host geometry, including all packed sequences."""
    required = 4 * nc * h * m * d
    parallel = 0 < required <= PREFIX_REVERSE_WORKSPACE_CAP
    return PrefixBackwardPlan(parallel, required if parallel else 0)


def owner_tape_producer_stages(plan: OwnerTapePlan, bc: int) -> int:
    """Read the single-stage wide-tile compatibility choice from the saved forward policy."""
    if bc != plan.geometry[7] or bc not in (16, 32, 64):
        raise fail("invalid_config", "owner producer tile differs from forward plan", bc=bc)
    try:
        entries = dict(plan.policy)["producer_stages_by_bc"]
        stages = dict(entries)
    except (KeyError, TypeError, ValueError) as error:
        raise fail("invalid_config", "owner producer stage policy is missing or malformed") from error
    if (
        len(entries) != 3
        or set(stages) != {16, 32, 64}
        or any(type(value) is not int or value not in (1, 2) for value in stages.values())
    ):
        raise fail("invalid_config", "unsupported frozen owner producer stage mapping")
    return stages[bc]


def owner_tape_uses_tile_major(plan: OwnerTapePlan, branch: int) -> bool:
    """Keep the high-fanout state consumer on edge-major storage using its saved policy."""
    policy = dict(plan.policy)
    if branch not in (0, 1) or policy.get("tape_layout") != "token_tile_state_route_bound_v2":
        raise fail("invalid_config", "unsupported frozen owner tape layout")
    bound = policy.get("state_tile_major_max_routes")
    if type(bound) is not int or not 1 <= bound <= MAX_KH + 1:
        raise fail("invalid_config", "invalid frozen state tape layout bound")
    return branch == 0 or plan.geometry[6] <= bound


def owner_branch_tape_policy_signature() -> tuple[tuple[str, Any], ...]:
    return (
        ("owner_algorithm", "branch_group_ds_tape_fused_dkv_v3"),
        ("tape_cap_bytes", OWNER_TAPE_CAP_BYTES),
        ("tape_layout", "token_tile_state_route_bound_v2"),
        ("state_tile_major_max_routes", 8),
        ("tape_edges", "dynamic_int64_address_stride"),
        ("map", "canonical_lower_bound_v1"),
        ("offset_scan", "tile_cumsum_int32_v1"),
        ("owner_bq", 16),
        ("count_bt", 128),
        ("offset_bt", 128),
        ("map_bt", 256),
        ("metadata_warps", 4),
        ("metadata_stages", 1),
        ("producer_warps", 4),
        ("producer_stages_by_bc", OWNER_TAPE_PRODUCER_STAGES),
        ("producer_writes_dk", True),
        ("dk_order", "canonical_bq_per_dot_scale_v1"),
        ("producer_indices", "capacity_guarded_int32_v1"),
        ("producer_padding_margin", "BQ"),
        ("owner_warps", 4),
        ("owner_stages", 2),
        ("branch_group_revision", 1),
        ("token_group_limit", 8),
        ("state_group_limit", 16),
        ("group_candidates", "powers_of_two_three_full_heads"),
        ("group_selection", "minimum_group_count_then_shared_bytes"),
        ("workspace_reuse", "per_buffer_maximum_across_branches"),
    )


def owner_tape_index32_safe(capacity: int, padded_extent: int, bq: int) -> bool:
    """Include the masked query lanes and final loop increment in the signed-index bound."""
    if capacity < 0 or padded_extent < 1 or bq < 1:
        raise fail("invalid_config", "invalid producer index bounds")
    return capacity <= INT32_MAX - bq and padded_extent <= INT32_MAX


@dataclass(frozen=True)
class OwnerTapePlan:
    geometry: tuple[int, ...]
    group_size: int
    tape_elements: int
    map_elements: int
    count_elements: int
    offset_elements: int
    map_width: int
    token_extent: int
    state_extent: int
    workspace_bytes: int
    policy: tuple[tuple[str, Any], ...]

    @property
    def enabled(self) -> bool:
        return self.group_size > 0


@dataclass(frozen=True)
class OwnerBranchTapePlan:
    geometry: tuple[int, ...]
    token: OwnerTapePlan
    state: OwnerTapePlan
    tape_elements: int
    map_elements: int
    count_elements: int
    offset_elements: int
    workspace_bytes: int
    policy: tuple[tuple[str, Any], ...]

    @property
    def enabled(self):
        kh, ks = self.geometry[5:7]
        return bool((kh or ks) and (not kh or self.token.enabled) and (not ks or self.state.enabled))


def _owner_group_candidates(h: int, limit: int) -> tuple[int, ...]:
    groups = {g for g in (1, 2, 3, 4, 8, 16) if g <= min(h, limit)}
    if h <= limit:
        groups.add(h)
    return tuple(sorted(groups, reverse=True))


def _owner_branch_subplan(geometry, group, branch, policy):
    t, h, nc, c, m, kh, ks, bc, storage_bytes = geometry
    token_extent, state_extent = cdiv(c, bc) * bc, cdiv(m, bc) * bc
    width = max(kh, ks)
    extent = kh * token_extent if branch == 0 else ks * state_extent
    tape = t * group * extent
    mapping, count = t * group * width, nc * group
    offsets = count + 1 if group else 0
    size = tape * storage_bytes + (mapping + count + offsets) * 4
    return OwnerTapePlan(geometry, group, tape, mapping, count, offsets, width, token_extent, state_extent, size, policy)


def owner_branch_tape_plan(
    t: int,
    h: int,
    nc: int,
    c: int,
    m: int,
    kh: int,
    ks: int,
    bc: int,
    storage_bytes: int,
    *,
    token_group_limit: int = 8,
    state_group_limit: int = 16,
) -> OwnerBranchTapePlan:
    """Freeze independent head groups under the actual shared allocation bound."""
    geometry = (t, h, nc, c, m, kh, ks, bc, storage_bytes)
    if min(h, c, m, bc, storage_bytes) <= 0 or min(t, nc, kh, ks) < 0 or ks > MAX_KH + 1:
        raise fail("invalid_config", "invalid branch owner tape geometry", geometry=geometry)
    if not 1 <= token_group_limit <= 8 or not 1 <= state_group_limit <= 16:
        raise fail("invalid_config", "owner branch limits exceed the frozen policy")
    policy = owner_branch_tape_policy_signature()
    empty = tuple(_owner_branch_subplan(geometry, 0, branch, policy) for branch in (0, 1))
    fallback = OwnerBranchTapePlan(geometry, *empty, 0, 0, 0, 0, 0, policy)
    if not t or not nc or not (kh or ks):
        return fallback
    token_groups = _owner_group_candidates(h, token_group_limit) if kh else (0,)
    state_groups = _owner_group_candidates(h, state_group_limit) if ks else (0,)
    choices = []
    for gt in token_groups:
        for gs in state_groups:
            token = _owner_branch_subplan(geometry, gt, 0, policy)
            state = _owner_branch_subplan(geometry, gs, 1, policy)
            sizes = tuple(
                (
                    max(getattr(token, key), getattr(state, key))
                    for key in ("tape_elements", "map_elements", "count_elements", "offset_elements")
                ),
            )
            tape, mapping, count, offsets = sizes
            size = tape * storage_bytes + (mapping + count + offsets) * 4
            if size > OWNER_TAPE_CAP_BYTES or mapping > INT32_MAX or count > INT32_MAX - 1:
                continue
            groups = (cdiv(h, gt) if gt else 0) + (cdiv(h, gs) if gs else 0)
            selected = OwnerBranchTapePlan(geometry, token, state, *sizes, size, policy)
            choices.append(((groups, size, -gt, -gs), selected))
    return min(choices, key=lambda row: row[0])[1] if choices else fallback


def _require_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise fail("invalid_config", f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise fail("invalid_config", f"{name} must be >= {minimum}, got {value}")
    return value


def _check_int32(reason_detail: str, value: int) -> None:
    if value > INT32_MAX:
        raise fail("int32_overflow", f"{reason_detail} = {value} exceeds int32", quantity=reason_detail, value=value)


def _validate_cu_seqlens_host(cu: Sequence[int]) -> tuple[int, ...]:
    values = tuple(cu)
    if len(values) < 2:
        raise fail("invalid_cu_seqlens", "cu_seqlens must contain at least two elements; empty batches are unsupported")
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int):
            raise fail("invalid_cu_seqlens", f"cu_seqlens_host must contain ints, got {type(v).__name__}")
    if values[0] != 0:
        raise fail("invalid_cu_seqlens", f"cu_seqlens_host[0] must be zero, got {values[0]}")
    for i in range(len(values) - 1):
        if values[i + 1] <= values[i]:
            raise fail(
                "invalid_cu_seqlens",
                f"cu_seqlens_host must be strictly increasing; empty sequence at index {i}: {values[i]} -> {values[i + 1]}",
                index=i,
            )
    _check_int32("T (cu_seqlens[-1])", values[-1])
    return values


class SequenceLengthPlan:
    """Immutable sequence geometry built with from_host or from_device.

    The plan owns a contiguous int32 cumulative-length tensor; callers must not modify it in place.
    """

    __slots__ = (
        "_cu_host",
        "_lengths",
        "_cu_device",
        "_block_layouts",
        "_routing_layouts",
        "_source_ref",
        "_source_version",
        "__weakref__",
    )

    def __init__(self, *_args: Any, _token: object = None, **_kwargs: Any) -> None:
        if _token is not _CONSTRUCT_TOKEN:
            raise TypeError("Use SequenceLengthPlan.from_host / from_device")

    @classmethod
    def _build(
        cls,
        cu_host: tuple[int, ...],
        cu_device: torch.Tensor,
        source: torch.Tensor | None = None,
    ) -> SequenceLengthPlan:
        obj = cls(_token=_CONSTRUCT_TOKEN)
        object.__setattr__(obj, "_cu_host", cu_host)
        object.__setattr__(obj, "_lengths", tuple(cu_host[i + 1] - cu_host[i] for i in range(len(cu_host) - 1)))
        object.__setattr__(obj, "_cu_device", cu_device)
        object.__setattr__(obj, "_block_layouts", {})
        object.__setattr__(obj, "_routing_layouts", {})
        object.__setattr__(obj, "_source_ref", ref(source) if source is not None else None)
        try:
            version = source._version if source is not None else None
        except RuntimeError:
            version = None
        object.__setattr__(obj, "_source_version", version)
        return obj

    def _matches_source(self, source: torch.Tensor) -> bool:
        # identity-only rotary caches are safe for fresh source tensors without in-place writes
        if self._source_version != 0 or self._source_ref is None or self._source_ref() is not source:
            return False
        return source._version == self._source_version

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("SequenceLengthPlan is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("SequenceLengthPlan is immutable")

    @classmethod
    @torch.inference_mode(False)
    def from_host(cls, lengths: Iterable[int], device: Any) -> SequenceLengthPlan:
        """Construct a plan from host lengths using one asynchronous pinned-memory copy on CUDA."""
        lens = tuple(lengths)
        for v in lens:
            if isinstance(v, bool) or not isinstance(v, int):
                raise fail("invalid_cu_seqlens", f"lengths must contain ints, got {type(v).__name__}")
            if v <= 0:
                raise fail("invalid_cu_seqlens", f"Sequence lengths must be positive, got {v}")
        cu = [0]
        for v in lens:
            cu.append(cu[-1] + v)
        cu_host = _validate_cu_seqlens_host(cu)
        cu_device = _to_device_async(torch.tensor(cu_host, dtype=torch.int32), device)
        return cls._build(cu_host, cu_device)

    @classmethod
    @torch.inference_mode(False)
    def from_device(cls, cu_seqlens: torch.Tensor) -> SequenceLengthPlan:
        """Validate device lengths with one host read, then retain owned host/device copies."""
        _check_cu_seqlens_tensor(cu_seqlens)
        cu_host = _validate_cu_seqlens_host([int(x) for x in cu_seqlens.tolist()])
        return cls._build(cu_host, cu_seqlens.detach().clone(memory_format=torch.contiguous_format), source=cu_seqlens)

    @property
    def cu_seqlens_host(self) -> tuple[int, ...]:
        return self._cu_host

    @property
    def cu_seqlens(self) -> torch.Tensor:
        return self._cu_device

    @property
    def lengths(self) -> tuple[int, ...]:
        return self._lengths

    @property
    def Nseq(self) -> int:  # noqa: N802
        return len(self._lengths)

    @property
    def T(self) -> int:  # noqa: N802
        return self._cu_host[-1]

    @property
    def max_seqlen(self) -> int:
        return max(self._lengths)

    @property
    def device(self) -> torch.device:
        return self._cu_device.device

    def __repr__(self) -> str:
        return f"SequenceLengthPlan(Nseq={self.Nseq}, T={self.T}, max_seqlen={self.max_seqlen}, device={self.device})"

    def block_layout(self, chunk_size: int) -> BlockLayout:
        """Cache host block boundaries and device int32 pointers without copying device values to the host."""
        c = _require_int("chunk_size", chunk_size)
        layout = self._block_layouts.get(c)
        if layout is None:
            block_ptr, seq_block_ptr, _ = _block_layout(self._lengths, c, 1)
            layout = BlockLayout(
                chunk_size=c,
                block_ptr_host=block_ptr,
                seq_block_ptr_host=seq_block_ptr,
                **dict(
                    zip(
                        ("block_ptr", "seq_block_ptr"),
                        device_block_ptrs(self._cu_device, c, len(block_ptr) - 1, self._cu_host[-1]),
                    ),
                ),
            )
            self._block_layouts[c] = layout
        return layout


def _resolve_length_plan(
    batch: int,
    tokens: int,
    device: torch.device,
    cu_seqlens: torch.Tensor | None = None,
    length_plan: SequenceLengthPlan | None = None,
) -> SequenceLengthPlan:
    """Resolve sequence boundaries without crossing dense batch elements.

    Packed inputs require batch size one. If both representations are supplied,
    their boundaries must agree; passing the plan's owned tensor avoids a device comparison.
    """
    supplied_plan = length_plan is not None
    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError("Packed PHBA inputs require batch dimension1")
        _check_cu_seqlens_tensor(cu_seqlens)
        if cu_seqlens.device != device:
            raise ValueError("PHBA cumulative lengths must be on the input device")
    if length_plan is None:
        length_plan = (
            SequenceLengthPlan.from_device(cu_seqlens)
            if cu_seqlens is not None
            else SequenceLengthPlan.from_host([tokens] * batch, device)
        )
    elif not isinstance(length_plan, SequenceLengthPlan):
        raise TypeError("PHBA length_plan must be a SequenceLengthPlan")
    if batch * tokens != length_plan.T or length_plan.device != device:
        raise ValueError("PHBA length plan does not match the input geometry or device")
    if batch > 1 and length_plan.lengths != (tokens,) * batch:
        raise ValueError("PHBA length plan must preserve each dense batch boundary")
    if (
        supplied_plan
        and cu_seqlens is not None
        and cu_seqlens is not length_plan.cu_seqlens
        and not length_plan._matches_source(cu_seqlens)
    ):
        if not torch.equal(cu_seqlens, length_plan.cu_seqlens):
            raise ValueError("PHBA length_plan and cu_seqlens have different boundaries")
    return length_plan


def _to_device_async(x: torch.Tensor, device: Any) -> torch.Tensor:
    """Use pinned host memory for an asynchronous CUDA copy; its lifetime is managed by the host allocator."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return x.to(dev)
    return x.pin_memory().to(dev, non_blocking=True)


def device_block_ptrs(cu: torch.Tensor, c: int, nc: int, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct int32 block and sequence pointers on device using host-known output sizes.

    Explicit repeat_interleave sizes avoid reading device values to determine allocation sizes.
    """
    cu = cu.to(torch.int64)
    nseq = cu.numel() - 1
    nblk = torch.div(cu[1:] - cu[:-1] + (c - 1), c, rounding_mode="floor")
    seq_block_ptr = torch.cat([nblk.new_zeros(1), torch.cumsum(nblk, 0)])
    block_seq = torch.repeat_interleave(torch.arange(nseq, device=cu.device), nblk, output_size=nc)
    local = torch.arange(nc, device=cu.device) - seq_block_ptr[block_seq]
    block_ptr = torch.cat([cu[block_seq] + local * c, cu[-1:]])
    return block_ptr.to(torch.int32).contiguous(), seq_block_ptr.to(torch.int32).contiguous()


@dataclass(frozen=True)
class BlockLayout:
    """Block boundaries for one chunk size, with host tuples and device tensors owned by the sequence plan."""

    chunk_size: int
    block_ptr_host: tuple[int, ...]
    seq_block_ptr_host: tuple[int, ...]
    block_ptr: torch.Tensor = field(repr=False, compare=False)
    seq_block_ptr: torch.Tensor = field(repr=False, compare=False)

    @property
    def Nc(self) -> int:  # noqa: N802
        return len(self.block_ptr_host) - 1


def _check_cu_seqlens_tensor(cu_seqlens: torch.Tensor) -> None:
    if not isinstance(cu_seqlens, torch.Tensor):
        raise fail("invalid_cu_seqlens", f"cu_seqlens must be a torch.Tensor, got {type(cu_seqlens).__name__}")
    if cu_seqlens.dim() != 1:
        raise fail("invalid_cu_seqlens", f"cu_seqlens must be one-dimensional, got shape {tuple(cu_seqlens.shape)}")
    if cu_seqlens.dtype != torch.int32:
        raise fail("invalid_cu_seqlens", f"cu_seqlens must use int32, got {cu_seqlens.dtype}")


@dataclass(frozen=True)
class LocalFABatch:
    """A local-attention batch covering blocks [b0, b1) and tokens [t0, t1)."""

    b0: int
    b1: int
    t0: int
    t1: int
    max_seqlen: int


def _block_layout(lengths: Sequence[int], c: int, br: int) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Compute host block boundaries, sequence block boundaries and query-tile count."""
    block_ptr = [0]
    seq_block_ptr = [0]
    nwork = 0
    q_tiles_full = cdiv(c, br)
    t = 0
    for length in lengths:
        nblk = cdiv(length, c)
        for j in range(nblk):
            t += min(c, length - j * c)
            block_ptr.append(t)
        seq_block_ptr.append(seq_block_ptr[-1] + nblk)
        nwork += (nblk - 1) * q_tiles_full + cdiv(length - (nblk - 1) * c, br)
    return tuple(block_ptr), tuple(seq_block_ptr), nwork


def _local_fa_batches(block_ptr: Sequence[int], c: int, seg_max: int, rows_per_call_max: int) -> tuple[LocalFABatch, ...]:
    """Partition consecutive blocks within the segment and row limits, keeping each block intact."""
    nc = len(block_ptr) - 1
    rows_cap = max(rows_per_call_max, c)
    batches: list[LocalFABatch] = []
    b0 = 0
    while b0 < nc:
        b1 = b0
        max_len = 0
        while b1 < nc and (b1 - b0) < seg_max:
            blk = block_ptr[b1 + 1] - block_ptr[b1]
            if b1 > b0 and (block_ptr[b1 + 1] - block_ptr[b0]) > rows_cap:
                break
            max_len = max(max_len, blk)
            b1 += 1
        batches.append(LocalFABatch(b0=b0, b1=b1, t0=block_ptr[b0], t1=block_ptr[b1], max_seqlen=max_len))
        b0 = b1
    return tuple(batches)
