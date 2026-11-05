# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Gated delta rule state updates for Ascend."""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl

from fla.backends import TritonAscendBackend, register
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp2
from fla.ops.utils.triton_ascend.addressing import (
    add_token_rows,
    chunk_token_rows,
    clamp_token_rows,
    mload,
    mstore,
)
from fla.utils import ascend_compile_kwargs, input_guard
from fla.utils.ascend_ub_manager import (
    compute_row_tile_block_size,
    get_npu_properties,
    get_ub_manager,
)


@lru_cache(maxsize=4)
def _cached_cu_seqlens_cpu(cu_seqlens, version):
    return cu_seqlens.cpu()


def _recurrence_cu_seqlens_cpu(cu_seqlens, cu_seqlens_cpu):
    if cu_seqlens_cpu is not None or cu_seqlens is None:
        return cu_seqlens_cpu
    if cu_seqlens.device.type == 'cpu':
        return cu_seqlens
    if torch.npu.is_current_stream_capturing():
        return None
    # include the version so eager in-place length updates cannot reuse stale bounds.
    if cu_seqlens.is_inference():
        return cu_seqlens.cpu()
    return _cached_cu_seqlens_cpu(cu_seqlens, cu_seqlens._version)


@lru_cache(maxsize=4)
def _padded_token_indices(offsets: tuple[int, ...], chunk_size: int, device: torch.device):
    gather, valid, restore, padded_offsets = [], [], [], [0]
    for bos, eos in zip(offsets[:-1], offsets[1:]):
        length = eos - bos
        padded_length = triton.cdiv(length, chunk_size) * chunk_size
        gather.extend(range(bos, eos))
        gather.extend([eos - 1] * (padded_length - length))
        valid.extend([True] * length + [False] * (padded_length - length))
        restore.extend(range(padded_offsets[-1], padded_offsets[-1] + length))
        padded_offsets.append(padded_offsets[-1] + padded_length)
    cu_cpu = torch.tensor(padded_offsets, dtype=torch.int32)
    return (
        torch.tensor(gather, device=device),
        torch.tensor(valid, device=device),
        torch.tensor(restore, device=device),
        cu_cpu.to(device),
        cu_cpu,
    )


def _varlen_padding(cu_seqlens, cu_seqlens_cpu, chunk_size, device):
    # capture keeps device-driven lengths; eager can use the caller's CPU metadata.
    if (cu_seqlens is None or cu_seqlens_cpu is None or cu_seqlens_cpu.device.type != "cpu"
            or torch.npu.is_current_stream_capturing()):
        return None
    offsets = tuple(cu_seqlens_cpu.tolist())
    lengths = [end - start for start, end in zip(offsets[:-1], offsets[1:])]
    if not lengths or min(lengths) <= 0 or all(length % chunk_size == 0 for length in lengths):
        return None
    return _padded_token_indices(offsets, chunk_size, device)


def _recurrence_padding(k, v, g, gk, state_v_first, cu_seqlens, cu_seqlens_cpu, chunk_size):
    # token padding also enables the aligned feature-padding path for partial two-slab heads.
    if not (64 <= k.shape[-1] <= 256 and 64 <= v.shape[-1] <= 256):
        return None
    if state_v_first and any(d & (d - 1) for d in (k.shape[-1], v.shape[-1])):
        return None
    if k.shape[-1] == 64 and v.shape[-1] == 64 and cu_seqlens_cpu is not None:
        offsets = cu_seqlens_cpu.tolist()
        if all(end - start <= chunk_size for start, end in zip(offsets[:-1], offsets[1:])):
            return None
    return _varlen_padding(cu_seqlens, cu_seqlens_cpu, chunk_size, k.device)


def _pad_recurrence_tokens(x, indices, valid=None):
    if x is None:
        return None
    return _pad_token_rows(x, indices, valid)


@triton.jit(do_not_specialize=["rows"])
def _pad_token_rows_kernel(
    x, y, indices, valid, rows,
    WIDTH: tl.constexpr, BLOCK: tl.constexpr, ZERO_PAD: tl.constexpr,
):
    blocks: tl.constexpr = tl.cdiv(WIDTH, BLOCK)
    lanes = tl.arange(0, BLOCK)
    for task in tl.range(tl.program_id(0), rows * blocks, tl.num_programs(0)):
        row = task // blocks
        col = (task % blocks).to(tl.int64) * BLOCK
        source = tl.load(indices + row).to(tl.int64)
        values = tl.load((x + source * WIDTH + col) + lanes, col + lanes < WIDTH, other=0)
        if ZERO_PAD:
            values = tl.where(tl.load(valid + row).to(tl.int1), values, 0)
        tl.store((y + row.to(tl.int64) * WIDTH + col) + lanes, values, col + lanes < WIDTH)


def _pad_token_rows(x, indices, valid=None):
    if x is None:
        return None
    if x.shape[0] != 1 or not x.is_contiguous():
        if x.ndim not in (2, 3, 4):
            raise ValueError(f"Triton tail gather expects rank 2-4 tensors, got rank {x.ndim}")
        B, T = x.shape[:2]
        width = x.numel() // (B * T)
        feature = x.shape[-1] if x.ndim == 4 else 1
        # Rank-2 inputs have no feature axis; keep its synthetic stride nonzero
        # so the shared kernel's flattened-column address remains well-formed.
        strides = (*x.stride(), 1, 1)[:4]
        padded = x.new_empty(B, len(indices), *x.shape[2:])
        _pad_token_rows_strided_kernel[(get_npu_properties()["num_aicore"],)](
            x, padded, indices, valid, len(indices),
            B=B,
            WIDTH=width,
            FEATURE=feature,
            S0=strides[0],
            S1=strides[1],
            S2=strides[2],
            S3=strides[3],
            RANK=x.ndim,
            BLOCK=min(4096, triton.next_power_of_2(width)),
            ZERO_PAD=valid is not None,
        )
        return padded
    width = x.numel() // x.shape[1]
    padded = x.new_empty(1, len(indices), *x.shape[2:])
    _pad_token_rows_kernel[(get_npu_properties()["num_aicore"],)](
        x, padded, indices, valid, len(indices),
        WIDTH=width, BLOCK=min(4096, triton.next_power_of_2(width)), ZERO_PAD=valid is not None,
    )
    return padded


@triton.jit(do_not_specialize=["rows"])
def _pad_token_rows_strided_kernel(
    x, y, indices, valid, rows,
    B: tl.constexpr, WIDTH: tl.constexpr, FEATURE: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
    RANK: tl.constexpr, BLOCK: tl.constexpr, ZERO_PAD: tl.constexpr,
):
    blocks: tl.constexpr = tl.cdiv(WIDTH, BLOCK)
    lanes = tl.arange(0, BLOCK)
    for task in tl.range(tl.program_id(0), B * rows * blocks, tl.num_programs(0)):
        batch = (task // (rows * blocks)).to(tl.int64)
        row = (task // blocks % rows).to(tl.int64)
        col = (task % blocks).to(tl.int64) * BLOCK + lanes
        source = tl.load(indices + row).to(tl.int64)
        if RANK == 4:
            pointer = x + batch * S0 + source * S1 + (col // FEATURE) * S2 + (col % FEATURE) * S3
        else:
            pointer = x + batch * S0 + source * S1 + col * S2
        values = tl.load(pointer, col < WIDTH, other=0)
        if ZERO_PAD:
            values = tl.where(tl.load(valid + row).to(tl.int1), values, 0)
        tl.store(y + (batch * rows + row) * WIDTH + col, values, col < WIDTH)


def _pad_dense_tail_tokens(x, padded_T, *, repeat_last=False):
    """Pad dense token rows; repeated gates keep the final chunk anchor unchanged."""
    if x is None or padded_T == x.shape[1]:
        return x
    pad = padded_T - x.shape[1]
    if repeat_last:
        tail = x[:, -1:].expand(-1, pad, *x.shape[2:])
        return torch.cat((x, tail), dim=1)
    return torch.nn.functional.pad(x, (0, 0) * (x.ndim - 2) + (0, pad))


@triton.jit
def _gather_recurrence_operand(x, y, source, row, col, lanes, valid, WIDTH: tl.constexpr, ZERO_PAD: tl.constexpr):
    values = tl.load((x + source * WIDTH + col) + lanes, col + lanes < WIDTH, other=0)
    if ZERO_PAD:
        values = tl.where(valid, values, 0)
    tl.store((y + row * WIDTH + col) + lanes, values, col + lanes < WIDTH)


@triton.jit(do_not_specialize=['rows'])
def _pad_recurrence_inputs_kernel(
    k, w, v, gk, q, do, pk, pw, pv, pgk, pq, pdo, indices, valid, rows,
    KW: tl.constexpr, WW: tl.constexpr, VW: tl.constexpr, BLOCK: tl.constexpr, BACKWARD: tl.constexpr,
    HAS_GK: tl.constexpr,
):
    blocks: tl.constexpr = tl.cdiv(max(KW, max(WW, VW)), BLOCK)
    operands: tl.constexpr = 6 if BACKWARD else 4
    lanes = tl.arange(0, BLOCK)
    for task in tl.range(tl.program_id(0), rows * blocks * operands, tl.num_programs(0)):
        operand = task // (rows * blocks)
        row = (task // blocks % rows).to(tl.int64)
        col = (task % blocks).to(tl.int64) * BLOCK
        source = tl.load(indices + row).to(tl.int64)
        is_valid = tl.load(valid + row).to(tl.int1)
        if operand == 0:
            _gather_recurrence_operand(k, pk, source, row, col, lanes, is_valid, KW, True)
        elif operand == 1:
            _gather_recurrence_operand(w, pw, source, row, col, lanes, is_valid, WW, True)
        elif operand == 2:
            _gather_recurrence_operand(v, pv, source, row, col, lanes, is_valid, VW, True)
        elif operand == 3:
            if HAS_GK:
                _gather_recurrence_operand(gk, pgk, source, row, col, lanes, is_valid, WW, False)
        else:
            if BACKWARD:
                if operand == 4:
                    _gather_recurrence_operand(q, pq, source, row, col, lanes, is_valid, KW, True)
                else:
                    _gather_recurrence_operand(do, pdo, source, row, col, lanes, is_valid, VW, True)


def _pad_recurrence_inputs(k, w, v, gk, indices, valid, q=None, do=None):
    tensors = (k, w, v, gk, q, do)
    widths = tuple(x.numel() // x.shape[1] for x in (k, w, v))
    # launch savings dominate for small copies; larger copies retain independent DMA streams.
    if len(indices) * max(widths) > 512 * 1024 or not all(
        x is None or (x.shape[0] == 1 and x.is_contiguous()) for x in tensors
    ):
        return tuple(_pad_recurrence_tokens(x, indices, None if i == 3 else valid) for i, x in enumerate(tensors))
    outputs = tuple(x.new_empty(1, len(indices), *x.shape[2:]) if x is not None else None for x in tensors)
    kw, ww, vw = widths
    _pad_recurrence_inputs_kernel[(get_npu_properties()['num_aicore'],)](
        *tensors, *outputs, indices, valid, len(indices),
        KW=kw, WW=ww, VW=vw, BLOCK=min(4096, triton.next_power_of_2(max(kw, ww, vw))), BACKWARD=q is not None,
        HAS_GK=gk is not None,
    )
    return outputs


@lru_cache(maxsize=4)
def _padded_gate_tail_indices(offsets: tuple[int, ...], chunk_size: int, device: torch.device):
    sources, destinations, padded_base = [], [], 0
    for bos, eos in zip(offsets[:-1], offsets[1:]):
        length = eos - bos
        padded_length = triton.cdiv(length, chunk_size) * chunk_size
        sources.extend(range(padded_base + length, padded_base + padded_length))
        destinations.extend([eos - 1] * (padded_length - length))
        padded_base += padded_length
    return (torch.tensor(sources, dtype=torch.int64, device=device),
            torch.tensor(destinations, dtype=torch.int64, device=device))


def _restore_gate_gradient(x, restore, cu_seqlens_cpu, chunk_size):
    result = _pad_recurrence_tokens(x, restore)
    sources, destinations = _padded_gate_tail_indices(tuple(cu_seqlens_cpu.tolist()), chunk_size, x.device)
    if len(sources):
        result.index_add_(1, destinations, _pad_recurrence_tokens(x, sources))
    return result


def _zero_padded_view(x: torch.Tensor, width: int) -> torch.Tensor:
    """Retain zero-padded WY storage for subsequent Ascend kernels."""
    view = x[..., :width]
    if width != x.shape[-1] and not torch.is_inference_mode_enabled():
        view._fla_ascend_zero_pad = (x, view._version, x._version)
    return view


def _pad_head_dim(x: torch.Tensor, width: int) -> torch.Tensor:
    metadata = getattr(x, "_fla_ascend_zero_pad", None)
    if metadata is not None and not torch.is_inference_mode_enabled():
        padded, view_version, padded_version = metadata
        if (x._version == view_version and padded._version == padded_version
                and padded.shape == (*x.shape[:-1], width) and padded.is_contiguous()
                and x.dtype == padded.dtype and x.device == padded.device
                and x.data_ptr() == padded.data_ptr() and x.stride() == padded.stride()):
            return padded
    return torch.nn.functional.pad(x, (0, width - x.shape[-1]))


def _varlen_has_tail_chunk(
    cu_seqlens: torch.LongTensor | None,
    BT: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> bool:
    if cu_seqlens is None or len(cu_seqlens) < 2:
        return False
    cpu = _recurrence_cu_seqlens_cpu(cu_seqlens, cu_seqlens_cpu)
    src = cpu if cpu is not None else cu_seqlens
    if src.device.type != "cpu":
        return True
    offsets = src.tolist()
    return any((end - start) % BT != 0 for start, end in zip(offsets[:-1], offsets[1:]))


def time_axis_needs_mask(
    T: int,
    BT: int,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> bool:
    """True when any launched chunk can contain invalid token rows."""
    if T % BT != 0:
        return True
    return _varlen_has_tail_chunk(cu_seqlens, BT, cu_seqlens_cpu)


def need_kv_dma_mask(K: int, BK: int, V: int, BV: int) -> bool:
    return K % BK != 0 or V % BV != 0


def need_dma_mask(
    T: int,
    BT: int,
    K: int,
    BK: int,
    V: int,
    BV: int,
    varlen: bool,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    *,
    time_dma: bool | None = None,
) -> bool:
    if need_kv_dma_mask(K, BK, V, BV):
        return True
    if time_dma is False:
        return False
    if time_dma is True:
        return True
    if varlen:
        return time_axis_needs_mask(T, BT, cu_seqlens, cu_seqlens_cpu)
    return T % BT != 0


def mask_dma_heuristic(args) -> bool:
    """Constexpr gate so aligned launches DCE masked DMA. ``T`` is runtime, so this stays on the host."""
    cu = args.get("cu_seqlens")
    cu_cpu = args.get("cu_seqlens_cpu")
    return need_dma_mask(
        args["T"],
        args["BT"],
        args["K"],
        args["BK"],
        args["V"],
        args["BV"],
        cu is not None,
        cu,
        cu_cpu,
    )


# b_h[BK,BV] fp32 × n_slabs stay live; w/k are sequential one-slab tiles.
# ``enable_ubuf_saving`` on the fwd launch packs that live set under 192KiB
# (without it, BK=256/BV=128 reports 256KiB and fails to compile).
# ``unit_flag`` overlaps Cube/Fixpipe waits (``--enable-hivm-unit-flag-sync``).
_FWD_H_MEM_MULT = 8.0
_SAFETY_MARGIN = 0.80
_FALLBACK_BV = 16
_MAX_BV = 64
_FWD_H_BT = 64
# live-range packing fits BK=256/BV=128 despite a peak estimate of 1.18 times physical UB capacity.
_FWD_UB_SOFT = 1.20
# Masked blockdim64 (unaligned T / varlen) is fatter than oneslab. D128 BK=128/BV=128
# analytical peak is 132352 B, but the compiler asks for 196864 B (256 B over 192 KiB).
_FWD_UB_MASKED = 0.25
_DHU_BT = 64
# ubuf_saving packs the D256 / BV=128 live set (about 1.17× 192 KiB) the way fwd_h does.
_DHU_UB_SOFT = 1.20
# Masked DMA (varlen or T not divisible by BT) keeps pad buffers the packer does not
# reclaim. D256 / 4 slabs / BV=128 asks for 263168 B against 192 KiB, so stay inside raw UB.
_DHU_UB_MASKED = 1.0
# Tighter fraction when in-kernel exp2(g) inflates live UB (~1.7× analytical).
_DHU_UB_GATE_INLINE = 0.60


def _get_bv(K: int, V: int) -> int:
    return compute_row_tile_block_size(
        row_dim=min(K, 64),
        fixed_dim=V,
        memory_multiplier=_FWD_H_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BV,
        min_block=16,
        max_block=min(_MAX_BV, triton.next_power_of_2(V)),
    )


def _fwd_h_peak_bytes(BK: int, BV: int, n_slabs: int, BT: int = _FWD_H_BT) -> int:
    """Estimate forward UB usage with all state slabs live and one w/k slab at a time."""
    return n_slabs * BK * BV * 4 + BK * BT * 4 + BT * BV * 4 + BK * 4 + 12 * BT


def _fwd_h_tile_cost(K: int, V: int, BK: int, BV: int) -> int:
    """Scalar proxy: nv × (base ptrs + ~4 per K-slab), same family as bwd dhu."""
    return triton.cdiv(V, BV) * (3 + 4 * triton.cdiv(K, BK))


def _select_fwd_h_tiles(
    K: int,
    V: int,
    state_v_first: bool,
    *,
    masked: bool = False,
) -> tuple[int, int]:
    """Pick (BK, BV) minimizing ptr/scalar cost under a UB cap.

    ``STATE_V_FIRST`` keeps BK=64 because ``tl.trans(b_h)`` copies inflate live UB.
    At most four K-slabs (kernel unroll); BK starts at 64.
    ``masked`` is the blockdim64 path (unaligned T or varlen), whose compiler UB
    exceeds the analytical peak; oneslab keeps the soft over-admit.
    """
    if state_v_first:
        return 64, _get_bv(K=K, V=V)

    soft_cap = int(get_ub_manager().ub_capacity_bytes * (_FWD_UB_MASKED if masked else _FWD_UB_SOFT))
    if masked and K <= 64 and V <= 64:
        # The 64x64 state tile fits the UB budget for small heads and halves V slabs.
        soft_cap = max(soft_cap, int(get_ub_manager().ub_capacity_bytes * 0.28))
    max_bk = min(256, triton.next_power_of_2(max(K, 64)))
    desired_v = triton.next_power_of_2(V)
    if masked and (K % 64 or V % 64):
        desired_v = min(desired_v, 64)
    best: tuple[int, int, int] | None = None  # cost, BK, BV

    bk = 64
    while bk <= max_bk:
        n_slabs = triton.cdiv(K, bk)
        if n_slabs <= 4:
            bv = 16
            while bv <= min(desired_v, 256):
                if _fwd_h_peak_bytes(BK=bk, BV=bv, n_slabs=n_slabs) <= soft_cap:
                    cost = _fwd_h_tile_cost(K=K, V=V, BK=bk, BV=bv)
                    if (
                        best is None
                        or cost < best[0]
                        or (cost == best[0] and (bk > best[1] or (bk == best[1] and bv > best[2])))
                    ):
                        best = (cost, bk, bv)
                bv *= 2
        bk *= 2

    if best is None:
        return 64, _get_bv(K=K, V=V)
    return best[1], best[2]


def _dhu_peak_bytes(
    BK: int,
    BV: int,
    n_slabs: int,
    BT: int = _DHU_BT,
    partial_kv: bool = False,
    row_select: bool = False,
    use_gate: bool = False,
) -> int:
    # dh slabs stay live. Non-v-first keeps dv, do, and the dot acc as [BT, BV] fp32.
    # The extra do/dv copies are v-first only (lhs clobber), so they are not charged here.
    # +512 B is the address tile. D256 / BV=128 / 4 slabs is ~230 KiB and relies on ubuf_saving.
    # Ragged K/V keeps one more [BK, BV] fp32 tile. D100 / 2 slabs / BV=128 asks for
    # 197120 B, 512 B over 192 KiB, and ubuf_saving does not reclaim it.
    # Masked launches select invalid token rows of b_dv to 0. That select does not alias
    # b_dv: D128 / 2 slabs / BV=128 asks for 213504 B against 192 KiB.
    # USE_G on a ragged head (two K slabs) keeps the gate scales and the where()
    # result. D96 / BV=64 asks for 221696 B (1773568 bits) against 192 KiB.
    peak = n_slabs * BK * BV * 4 + 3 * BT * BV * 4 + 512
    if partial_kv:
        peak += BK * BV * 4
    if row_select:
        peak += BT * BV * 4
    if use_gate and partial_kv and n_slabs >= 2:
        peak += 7 * BT * BV * 4
    return peak


def _dhu_tile_cost(K: int, V: int, BK: int, BV: int) -> int:
    """Reverse-loop scalar proxy: nv × (base ptrs + ~4 per K-slab)."""
    return triton.cdiv(V, BV) * (3 + 4 * triton.cdiv(K, BK))


def _select_bwd_dhu_tiles(
    K: int,
    V: int,
    state_v_first: bool,
    *,
    min_nv: int = 1,
    gate_inline: bool = False,
    masked: bool = False,
    use_gate: bool = False,
) -> tuple[int, int]:
    """Pick (BK, BV) minimizing launch×ptr cost under a soft UB cap.

    ``gate_inline`` covers USE_G without host-precomputed exp2 (unaligned T / varlen).
    ``use_gate`` is any USE_G path. Ragged heads with two K slabs need a smaller BV.
    ``masked`` is blockdim64 DMA (varlen or unaligned T); those pad buffers are not packed.
    Ragged K/V tiles are scored against raw UB: the extra tile is not packed.
    """
    if state_v_first:
        return 64, _get_bv(K=K, V=V)

    ub = get_ub_manager().ub_capacity_bytes
    inline_cap = int(ub * _DHU_UB_GATE_INLINE)
    soft_cap = int(ub * _DHU_UB_SOFT)
    masked_cap = int(ub * _DHU_UB_MASKED)
    # bwd kernel is blockdim64: K is covered by up to four BK=64 slabs only.
    max_bk = 64
    desired_v = min(
        triton.next_power_of_2(V),
        max(16, triton.next_power_of_2(triton.cdiv(V, min_nv))),
    )
    # partial K/V slabs plus tail-token masking can exceed UB with BV > 64 (e.g. D=100).
    if K % 64 or V % 64:
        desired_v = min(desired_v, 64)
    best: tuple[int, int, int] | None = None  # cost, BK, BV

    bk = 64
    while bk <= max_bk:
        n_slabs = triton.cdiv(K, bk)
        bv = 16
        while bv <= min(desired_v, 256):
            partial = (K % bk != 0) or (V % bv != 0)
            if gate_inline:
                cap = inline_cap
            elif partial:
                cap = ub
            elif masked:
                cap = masked_cap
            else:
                cap = soft_cap
            # The 2D row select is compiled whenever DMA is masked: varlen, a ragged
            # chunk, or a partial K/V tile.
            if _dhu_peak_bytes(
                bk, bv, n_slabs, partial_kv=partial, row_select=masked or partial,
                use_gate=use_gate,
            ) <= cap:
                cost = _dhu_tile_cost(K, V, bk, bv)
                if best is None or cost < best[0] or (cost == best[0] and bk > best[1]):
                    best = (cost, bk, bv)
            bv *= 2
        bk *= 2

    if best is None:
        return 64, _get_bv(K=K, V=V)
    return best[1], best[2]


def _launch_core_grid(kernel, *, task_num: int, kernel_kwargs: dict, **compile_opts) -> None:
    num_core = min(get_npu_properties()["num_aicore"], max(1, task_num))
    kernel[(num_core,)](task_num=task_num, num_core=num_core, **compile_opts, **kernel_kwargs)


# enable_ubuf_saving packs BK=256/BV=128 into 192 KiB; unit_flag overlaps Cube/Fixpipe waits.
_FWD_H_COMPILE = dict(enable_ubuf_saving=True, unit_flag=True)
_DHU_COMPILE = dict(enable_ubuf_saving=True, unit_flag=True)

_FWD_H_HEURISTICS = {
    "USE_G": lambda args: args["g"] is not None or args["g_ratio"] is not None,
    "USE_G_PRECOMP": lambda args: args["g_ratio"] is not None,
    "USE_GK": lambda args: args["gk"] is not None or args["gk_last_exp"] is not None,
    "USE_GK_PRECOMP": lambda args: args["gk_last_exp"] is not None,
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
}


@triton.heuristics(_FWD_H_HEURISTICS)
@triton.jit(do_not_specialize=["T", "task_num", "num_core"])
def chunk_gated_delta_rule_h_fwd_kernel(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    g_ratio,
    g_last_exp,
    gk_last_exp,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_PRECOMP: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GK_PRECOMP: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MASK_DMA: tl.constexpr,
):
    core_id = tl.program_id(0)
    MASK_KV: tl.constexpr = K % BK != 0 or V % BV != 0
    NV: tl.constexpr = tl.cdiv(V, BV)
    DH_CS: tl.constexpr = HV * K * V
    stride_v: tl.constexpr = HV * V
    stride_k: tl.constexpr = H * K
    stride_w: tl.constexpr = HV * K
    T_max = T
    for task_id in tl.range(core_id, task_num, num_core):
        i_v, i_nh = task_id % NV, (task_id // NV).to(tl.int64)
        i_n, i_h = i_nh // HV, i_nh % HV
        if IS_VARLEN:
            bos, eos = (
                tl.load(cu_seqlens + i_n).to(tl.int64),
                tl.load(cu_seqlens + i_n + 1).to(tl.int64),
            )
            T = (eos - bos).to(tl.int32)
            NT = tl.cdiv(T, BT)
            boh = tl.load(chunk_offsets + i_n).to(tl.int64)
        else:
            bos = tl.cast(i_n, tl.int64) * T
            NT = tl.cdiv(T, BT)
            boh = tl.cast(i_n, tl.int64) * NT

        v_start = i_v * BV
        # rebind GM bases per task to avoid Ascend MTE faults from accumulated pointer offsets.
        w_base = w + (bos * HV + i_h) * K
        k_base = k + (bos * H + i_h // (HV // H)) * K
        v_base = v + (bos * HV + i_h) * V
        if SAVE_NEW_VALUE:
            v_new_base = v_new + (bos * HV + i_h) * V
        h_nh = h + (boh * HV + i_h) * K * V

        if USE_G:
            if USE_G_PRECOMP:
                g_ratio_ptr = g_ratio + (tl.cast(i_n, tl.int64) * HV + i_h) * T_max
                g_last_exp_ptr = g_last_exp + (tl.cast(i_n, tl.int64) * HV + i_h) * NT
        if USE_GK:
            if USE_GK_PRECOMP:
                gk_last_exp_nh = gk_last_exp + (tl.cast(i_n, tl.int64) * HV + i_h) * NT * K

        # state tiles are [BK, BV], or [BV, BK] with STATE_V_FIRST.
        if STATE_V_FIRST:
            b_h1 = tl.zeros([BV, BK], dtype=tl.float32)
            if K > BK:
                b_h2 = tl.zeros([BV, BK], dtype=tl.float32)
            if K > BK * 2:
                b_h3 = tl.zeros([BV, BK], dtype=tl.float32)
            if K > BK * 3:
                b_h4 = tl.zeros([BV, BK], dtype=tl.float32)
        else:
            b_h1 = tl.zeros([BK, BV], dtype=tl.float32)
            if K > BK:
                b_h2 = tl.zeros([BK, BV], dtype=tl.float32)
            if K > BK * 2:
                b_h3 = tl.zeros([BK, BV], dtype=tl.float32)
            if K > BK * 3:
                b_h4 = tl.zeros([BK, BV], dtype=tl.float32)

        if USE_INITIAL_STATE:
            h0_ptr = h0 + i_nh * K * V
            if STATE_V_FIRST:
                # h0 layout [V, K]: block (V_seg=BV, K_seg=BK)
                o_v = v_start + tl.arange(0, BV)
                m_v = (o_v >= 0) & (o_v < V)
                o_i = tl.arange(0, BK)
                m_i = (o_i >= 0) & (o_i < K)
                m_p_h0_1 = m_v[:, None] & m_i[None, :]
                p_h0_1 = h0_ptr + o_v[:, None] * K + o_i[None, :]
                b_h1 += mload(p_h0_1, m_p_h0_1, MASK_KV).to(tl.float32)
                if K > BK:
                    o_k = BK + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h0_2 = m_v[:, None] & m_k[None, :]
                    p_h0_2 = h0_ptr + o_v[:, None] * K + o_k[None, :]
                    b_h2 += mload(p_h0_2, m_p_h0_2, MASK_KV).to(tl.float32)
                if K > BK * 2:
                    o_k = BK * 2 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h0_3 = m_v[:, None] & m_k[None, :]
                    p_h0_3 = h0_ptr + o_v[:, None] * K + o_k[None, :]
                    b_h3 += mload(p_h0_3, m_p_h0_3, MASK_KV).to(tl.float32)
                if K > BK * 3:
                    o_k = BK * 3 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h0_4 = m_v[:, None] & m_k[None, :]
                    p_h0_4 = h0_ptr + o_v[:, None] * K + o_k[None, :]
                    b_h4 += mload(p_h0_4, m_p_h0_4, MASK_KV).to(tl.float32)
            else:
                # h0 layout [K, V]: block (K_seg=BK, V_seg=BV)
                o_i = tl.arange(0, BK)
                m_i = (o_i >= 0) & (o_i < K)
                o_v = v_start + tl.arange(0, BV)
                m_v = (o_v >= 0) & (o_v < V)
                m_p_h0_1 = m_i[:, None] & m_v[None, :]
                p_h0_1 = h0_ptr + o_i[:, None] * V + o_v[None, :]
                b_h1 += mload(p_h0_1, m_p_h0_1, MASK_KV).to(tl.float32)
                if K > BK:
                    o_k = BK + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h0_2 = m_k[:, None] & m_v[None, :]
                    p_h0_2 = h0_ptr + o_k[:, None] * V + o_v[None, :]
                    b_h2 += mload(p_h0_2, m_p_h0_2, MASK_KV).to(tl.float32)
                if K > BK * 2:
                    o_k = BK * 2 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h0_3 = m_k[:, None] & m_v[None, :]
                    p_h0_3 = h0_ptr + o_k[:, None] * V + o_v[None, :]
                    b_h3 += mload(p_h0_3, m_p_h0_3, MASK_KV).to(tl.float32)
                if K > BK * 3:
                    o_k = BK * 3 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h0_4 = m_k[:, None] & m_v[None, :]
                    p_h0_4 = h0_ptr + o_k[:, None] * V + o_v[None, :]
                    b_h4 += mload(p_h0_4, m_p_h0_4, MASK_KV).to(tl.float32)

        # main recurrence
        for phase in tl.static_range(2):
            if phase == 0:
                chunk_start, chunk_end = 0, T // BT
            else:
                chunk_start, chunk_end = T // BT, NT
            for i_t in range(chunk_start, chunk_end):
                h_base = h_nh + tl.cast(i_t, tl.int64) * DH_CS

                if STATE_V_FIRST:
                    o_v = v_start + tl.arange(0, BV)
                    m_v = (o_v >= 0) & (o_v < V)
                    o_i = tl.arange(0, BK)
                    m_i = (o_i >= 0) & (o_i < K)
                    m_p_h1 = m_v[:, None] & m_i[None, :]
                    p_h1 = h_base + o_v[:, None] * K + o_i[None, :]
                    mstore(p_h1, b_h1.to(p_h1.dtype.element_ty), m_p_h1, MASK_KV)
                    if K > BK:
                        o_k = BK + tl.arange(0, BK)
                        m_k = (o_k >= 0) & (o_k < K)
                        m_p_h2 = m_v[:, None] & m_k[None, :]
                        p_h2 = h_base + o_v[:, None] * K + o_k[None, :]
                        mstore(p_h2, b_h2.to(p_h2.dtype.element_ty), m_p_h2, MASK_KV)
                    if K > BK * 2:
                        o_k = BK * 2 + tl.arange(0, BK)
                        m_k = (o_k >= 0) & (o_k < K)
                        m_p_h3 = m_v[:, None] & m_k[None, :]
                        p_h3 = h_base + o_v[:, None] * K + o_k[None, :]
                        mstore(p_h3, b_h3.to(p_h3.dtype.element_ty), m_p_h3, MASK_KV)
                    if K > BK * 3:
                        o_k = BK * 3 + tl.arange(0, BK)
                        m_k = (o_k >= 0) & (o_k < K)
                        m_p_h4 = m_v[:, None] & m_k[None, :]
                        p_h4 = h_base + o_v[:, None] * K + o_k[None, :]
                        mstore(p_h4, b_h4.to(p_h4.dtype.element_ty), m_p_h4, MASK_KV)
                else:
                    o_i = tl.arange(0, BK)
                    m_i = (o_i >= 0) & (o_i < K)
                    o_v = v_start + tl.arange(0, BV)
                    m_v = (o_v >= 0) & (o_v < V)
                    m_p_h1 = m_i[:, None] & m_v[None, :]
                    p_h1 = h_base + o_i[:, None] * V + o_v[None, :]
                    mstore(p_h1, b_h1.to(p_h1.dtype.element_ty), m_p_h1, MASK_KV)
                    if K > BK:
                        o_k = BK + tl.arange(0, BK)
                        m_k = (o_k >= 0) & (o_k < K)
                        m_p_h2 = m_k[:, None] & m_v[None, :]
                        p_h2 = h_base + o_k[:, None] * V + o_v[None, :]
                        mstore(p_h2, b_h2.to(p_h2.dtype.element_ty), m_p_h2, MASK_KV)
                    if K > BK * 2:
                        o_k = BK * 2 + tl.arange(0, BK)
                        m_k = (o_k >= 0) & (o_k < K)
                        m_p_h3 = m_k[:, None] & m_v[None, :]
                        p_h3 = h_base + o_k[:, None] * V + o_v[None, :]
                        mstore(p_h3, b_h3.to(p_h3.dtype.element_ty), m_p_h3, MASK_KV)
                    if K > BK * 3:
                        o_k = BK * 3 + tl.arange(0, BK)
                        m_k = (o_k >= 0) & (o_k < K)
                        m_p_h4 = m_k[:, None] & m_v[None, :]
                        p_h4 = h_base + o_k[:, None] * V + o_v[None, :]
                        mstore(p_h4, b_h4.to(p_h4.dtype.element_ty), m_p_h4, MASK_KV)

                if not (MASK_DMA and phase == 1):
                    row_base = tl.cast(i_t, tl.int64) * BT
                    rows = tl.arange(0, BT)
                    m_rows = tl.full((BT,), True, tl.int1)
                else:
                    row_base, rows_local, m_rows = chunk_token_rows(i_t, T, BT)
                    rows = clamp_token_rows(row_base + rows_local, T)
                    row_base = tl.cast(0, tl.int64)
                o_i = tl.arange(0, BK)
                m_i = (o_i >= 0) & (o_i < K)
                m_p_w1 = m_rows[:, None] & m_i[None, :]
                p_w1 = add_token_rows(w_base, row_base, rows[:, None], stride_w, True) + o_i[None, :]
                b_w = mload(p_w1, m_p_w1, (MASK_DMA and phase == 1) or K % BK != 0 or V % BV != 0).to(tl.float32)
                if STATE_V_FIRST:
                    b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
                else:
                    b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
                if K > BK:
                    o_k = BK + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_w2 = m_rows[:, None] & m_k[None, :]
                    p_w2 = add_token_rows(w_base, row_base, rows[:, None], stride_w, True) + o_k[None, :]
                    b_w = mload(p_w2, m_p_w2, (MASK_DMA and phase == 1) or K % BK != 0 or V % BV != 0).to(tl.float32)
                    if STATE_V_FIRST:
                        b_v = tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype), b_v)
                    else:
                        b_v = tl.dot(b_w, b_h2.to(b_w.dtype), b_v)
                if K > BK * 2:
                    o_k = BK * 2 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_w3 = m_rows[:, None] & m_k[None, :]
                    p_w3 = add_token_rows(w_base, row_base, rows[:, None], stride_w, True) + o_k[None, :]
                    b_w = mload(p_w3, m_p_w3, (MASK_DMA and phase == 1) or K % BK != 0 or V % BV != 0).to(tl.float32)
                    if STATE_V_FIRST:
                        b_v = tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype), b_v)
                    else:
                        b_v = tl.dot(b_w, b_h3.to(b_w.dtype), b_v)
                if K > BK * 3:
                    o_k = BK * 3 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_w4 = m_rows[:, None] & m_k[None, :]
                    p_w4 = add_token_rows(w_base, row_base, rows[:, None], stride_w, True) + o_k[None, :]
                    b_w = mload(p_w4, m_p_w4, (MASK_DMA and phase == 1) or K % BK != 0 or V % BV != 0).to(tl.float32)
                    if STATE_V_FIRST:
                        b_v = tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype), b_v)
                    else:
                        b_v = tl.dot(b_w, b_h4.to(b_w.dtype), b_v)

                if USE_G:
                    if USE_G_PRECOMP:
                        p_g_ratio = add_token_rows(g_ratio_ptr, row_base, rows, 1, True)
                        b_g_ratio = mload(p_g_ratio, m_rows, MASK_DMA and phase == 1).to(tl.float32)
                        b_g_last = tl.load(g_last_exp_ptr + i_t).to(tl.float32)
                    else:
                        last_idx = min((i_t + 1) * BT, T) - 1
                        # g is transposed to [B, HV, T] in wrapper for contiguous T-load.
                        if IS_VARLEN:
                            g_ptr = g + bos + i_h * T_max
                        else:
                            g_ptr = g + tl.cast(i_n, tl.int64) * HV * T_max + i_h * T_max
                        b_g_last = tl.load(g_ptr + last_idx).to(tl.float32)
                        p_g = add_token_rows(g_ptr, row_base, rows, 1, True)
                        b_g = mload(p_g, m_rows, MASK_DMA and phase == 1).to(tl.float32)

                # load v and compute v_new = v - b_v
                o_v = v_start + tl.arange(0, BV)
                m_v = (o_v >= 0) & (o_v < V)
                m_p_v = m_rows[:, None] & m_v[None, :]
                p_v = add_token_rows(v_base, row_base, rows[:, None], stride_v, True) + o_v[None, :]
                b_v = mload(p_v, m_p_v, (MASK_DMA and phase == 1) or K % BK != 0 or V % BV != 0).to(tl.float32) - b_v

                if SAVE_NEW_VALUE:
                    p_v_new = add_token_rows(v_new_base, row_base, rows[:, None], stride_v, True) + o_v[None, :]
                    mstore(p_v_new, b_v.to(p_v_new.dtype.element_ty), m_p_v,
                           (MASK_DMA and phase == 1) or K % BK != 0 or V % BV != 0)

                if USE_G:
                    if USE_G_PRECOMP:
                        b_v = b_v * b_g_ratio[:, None]
                    else:
                        if MASK_DMA and phase == 1:
                            b_v = b_v * tl.where(m_rows, exp2(b_g_last - b_g), 0)[:, None]
                        else:
                            b_v = b_v * exp2(b_g_last - b_g)[:, None]
                        b_g_last = exp2(b_g_last)
                if MASK_DMA and phase == 1:
                    b_v = tl.where(m_rows[:, None], b_v, 0)
                if USE_GK:
                    o_gk = tl.arange(0, BK)
                    if USE_GK_PRECOMP:
                        # gk_last_exp layout [B, HV, NT, K]
                        gk_base = gk_last_exp_nh + tl.cast(i_t, tl.int64) * K
                    else:
                        last_idx = min((i_t + 1) * BT, T) - 1
                        gk_base = gk + (bos + last_idx) * HV * K + i_h * K

                # Scale one slab, then add k @ v into it, then the next slab.
                # hivm vmul writes a new fp32[BK, BV]. Doing every slab first keeps
                # two of those products live across the first vadd (64 KiB). On the
                # masked BK=64/BV=128 path that is what asks for 205952 B. The
                # product dies at that slab's vadd, so the next slab reuses it.
                if USE_G:
                    b_h1 *= b_g_last
                if USE_GK:
                    if USE_GK_PRECOMP:
                        b_gk_last1 = mload(gk_base + o_gk, (o_gk < K), K % BK != 0).to(tl.float32)
                    else:
                        b_gk_last1 = exp2(mload(gk_base + o_gk, (o_gk < K), K % BK != 0).to(tl.float32))
                    if STATE_V_FIRST:
                        b_h1 *= b_gk_last1[None, :]
                    else:
                        b_h1 *= b_gk_last1[:, None]
                m_p_k1 = m_i[:, None] & m_rows[None, :]
                p_k1 = add_token_rows(k_base, row_base, rows[None, :], stride_k, True) + o_i[:, None]
                b_k = mload(p_k1, m_p_k1, (MASK_DMA and phase == 1) or K % BK != 0 or V % BV != 0).to(tl.float32)
                if STATE_V_FIRST:
                    b_h1 += tl.trans(tl.dot(b_k, b_v))
                else:
                    b_h1 = tl.dot(b_k, b_v, b_h1)
                if K > BK:
                    o_k = BK + tl.arange(0, BK)
                    if USE_G:
                        b_h2 *= b_g_last
                    if USE_GK:
                        if USE_GK_PRECOMP:
                            b_gk_last2 = mload(gk_base + o_k, (o_k < K), K % BK != 0).to(tl.float32)
                        else:
                            b_gk_last2 = exp2(mload(gk_base + o_k, (o_k < K), K % BK != 0).to(tl.float32))
                        if STATE_V_FIRST:
                            b_h2 *= b_gk_last2[None, :]
                        else:
                            b_h2 *= b_gk_last2[:, None]
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_k2 = m_k[:, None] & m_rows[None, :]
                    p_k2 = add_token_rows(k_base, row_base, rows[None, :], stride_k, True) + o_k[:, None]
                    b_k = mload(p_k2, m_p_k2, (MASK_DMA and phase == 1) or K % BK != 0 or V % BV != 0).to(tl.float32)
                    if STATE_V_FIRST:
                        b_h2 += tl.trans(tl.dot(b_k, b_v))
                    else:
                        b_h2 = tl.dot(b_k, b_v, b_h2)
                if K > BK * 2:
                    o_k = BK * 2 + tl.arange(0, BK)
                    if USE_G:
                        b_h3 *= b_g_last
                    if USE_GK:
                        if USE_GK_PRECOMP:
                            b_gk_last3 = mload(gk_base + o_k, (o_k < K), K % BK != 0).to(tl.float32)
                        else:
                            b_gk_last3 = exp2(mload(gk_base + o_k, (o_k < K), K % BK != 0).to(tl.float32))
                        if STATE_V_FIRST:
                            b_h3 *= b_gk_last3[None, :]
                        else:
                            b_h3 *= b_gk_last3[:, None]
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_k3 = m_k[:, None] & m_rows[None, :]
                    p_k3 = add_token_rows(k_base, row_base, rows[None, :], stride_k, True) + o_k[:, None]
                    b_k = mload(p_k3, m_p_k3, (MASK_DMA and phase == 1) or K % BK != 0 or V % BV != 0).to(tl.float32)
                    if STATE_V_FIRST:
                        b_h3 += tl.trans(tl.dot(b_k, b_v))
                    else:
                        b_h3 = tl.dot(b_k, b_v, b_h3)
                if K > BK * 3:
                    o_k = BK * 3 + tl.arange(0, BK)
                    if USE_G:
                        b_h4 *= b_g_last
                    if USE_GK:
                        if USE_GK_PRECOMP:
                            b_gk_last4 = mload(gk_base + o_k, (o_k < K), K % BK != 0).to(tl.float32)
                        else:
                            b_gk_last4 = exp2(mload(gk_base + o_k, (o_k < K), K % BK != 0).to(tl.float32))
                        if STATE_V_FIRST:
                            b_h4 *= b_gk_last4[None, :]
                        else:
                            b_h4 *= b_gk_last4[:, None]
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_k4 = m_k[:, None] & m_rows[None, :]
                    p_k4 = add_token_rows(k_base, row_base, rows[None, :], stride_k, True) + o_k[:, None]
                    b_k = mload(p_k4, m_p_k4, (MASK_DMA and phase == 1) or K % BK != 0 or V % BV != 0).to(tl.float32)
                    if STATE_V_FIRST:
                        b_h4 += tl.trans(tl.dot(b_k, b_v))
                    else:
                        b_h4 = tl.dot(b_k, b_v, b_h4)

        if STORE_FINAL_STATE:
            ht_ptr = ht + i_nh * K * V

            if STATE_V_FIRST:
                o_v = v_start + tl.arange(0, BV)
                m_v = (o_v >= 0) & (o_v < V)
                o_i = tl.arange(0, BK)
                m_i = (o_i >= 0) & (o_i < K)
                m_p_ht1 = m_v[:, None] & m_i[None, :]
                p_ht1 = ht_ptr + o_v[:, None] * K + o_i[None, :]
                mstore(p_ht1, b_h1.to(p_ht1.dtype.element_ty), m_p_ht1, MASK_KV)
                if K > BK:
                    o_k = BK + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_ht2 = m_v[:, None] & m_k[None, :]
                    p_ht2 = ht_ptr + o_v[:, None] * K + o_k[None, :]
                    mstore(p_ht2, b_h2.to(p_ht2.dtype.element_ty), m_p_ht2, MASK_KV)
                if K > BK * 2:
                    o_k = BK * 2 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_ht3 = m_v[:, None] & m_k[None, :]
                    p_ht3 = ht_ptr + o_v[:, None] * K + o_k[None, :]
                    mstore(p_ht3, b_h3.to(p_ht3.dtype.element_ty), m_p_ht3, MASK_KV)
                if K > BK * 3:
                    o_k = BK * 3 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_ht4 = m_v[:, None] & m_k[None, :]
                    p_ht4 = ht_ptr + o_v[:, None] * K + o_k[None, :]
                    mstore(p_ht4, b_h4.to(p_ht4.dtype.element_ty), m_p_ht4, MASK_KV)
            else:
                o_i = tl.arange(0, BK)
                m_i = (o_i >= 0) & (o_i < K)
                o_v = v_start + tl.arange(0, BV)
                m_v = (o_v >= 0) & (o_v < V)
                m_p_ht1 = m_i[:, None] & m_v[None, :]
                p_ht1 = ht_ptr + o_i[:, None] * V + o_v[None, :]
                mstore(p_ht1, b_h1.to(p_ht1.dtype.element_ty), m_p_ht1, MASK_KV)
                if K > BK:
                    o_k = BK + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_ht2 = m_k[:, None] & m_v[None, :]
                    p_ht2 = ht_ptr + o_k[:, None] * V + o_v[None, :]
                    mstore(p_ht2, b_h2.to(p_ht2.dtype.element_ty), m_p_ht2, MASK_KV)
                if K > BK * 2:
                    o_k = BK * 2 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_ht3 = m_k[:, None] & m_v[None, :]
                    p_ht3 = ht_ptr + o_k[:, None] * V + o_v[None, :]
                    mstore(p_ht3, b_h3.to(p_ht3.dtype.element_ty), m_p_ht3, MASK_KV)
                if K > BK * 3:
                    o_k = BK * 3 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_ht4 = m_k[:, None] & m_v[None, :]
                    p_ht4 = ht_ptr + o_k[:, None] * V + o_v[None, :]
                    mstore(p_ht4, b_h4.to(p_ht4.dtype.element_ty), m_p_ht4, MASK_KV)


@triton.jit
def _dense_chunk_load(base, row_base, lanes, cols, T, STRIDE: tl.constexpr, BT: tl.constexpr, RAGGED: tl.constexpr):
    if RAGGED:
        if row_base + BT <= T:
            values = tl.load((base + row_base * STRIDE) + lanes * STRIDE + cols)
        else:
            rows = clamp_token_rows(row_base + lanes, T)
            values = tl.load(base + rows * STRIDE + cols, row_base + lanes < T, other=0)
    else:
        values = tl.load((base + row_base * STRIDE) + lanes * STRIDE + cols)
    return values


@triton.jit
def _dense_chunk_store(base, row_base, lanes, cols, values, T, STRIDE: tl.constexpr, BT: tl.constexpr,
                       RAGGED: tl.constexpr):
    if RAGGED:
        if row_base + BT <= T:
            tl.store((base + row_base * STRIDE) + lanes * STRIDE + cols, values)
        else:
            rows = clamp_token_rows(row_base + lanes, T)
            tl.store(base + rows * STRIDE + cols, values, row_base + lanes < T)
    else:
        tl.store((base + row_base * STRIDE) + lanes * STRIDE + cols, values)


def _dense_single_slab_tail(K, V, state_v_first, g, cu_seqlens):
    if cu_seqlens is not None or state_v_first or g is not None:
        return False
    BK, BV = _select_fwd_h_tiles(K, V, False)
    return K == BK and V % BV == 0


@triton.heuristics(
    {
        "USE_G": lambda args: args["g_ratio"] is not None,
        "USE_GK": lambda args: args["gk_last_exp"] is not None or args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "task_num", "num_core"])
def chunk_gated_delta_rule_h_fwd_kernel_single_slab(
    k,
    v,
    w,
    v_new,
    g_ratio,
    g_last_exp,
    gk_last_exp,
    gk,
    h,
    h0,
    ht,
    T,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    RAGGED: tl.constexpr,
):
    """Update K-first states for aligned dense sequences with one K slab and host-precomputed gates."""
    core_id = tl.program_id(0)
    DH_CS: tl.constexpr = HV * K * V
    stride_v: tl.constexpr = HV * V
    stride_k: tl.constexpr = H * K
    stride_w: tl.constexpr = HV * K
    NV: tl.constexpr = tl.cdiv(V, BV)
    NT = tl.cdiv(T, BT)
    if USE_GK:
        o_k = tl.arange(0, BK)

    for task_id in tl.range(core_id, task_num, num_core):
        i_v, i_nh = task_id % NV, (task_id // NV).to(tl.int64)
        i_n, i_h = i_nh // HV, i_nh % HV
        bos = tl.cast(i_n, tl.int64) * T
        v_start = i_v * BV

        w_base = w + (bos * HV + i_h) * K
        k_base = k + (bos * H + i_h // (HV // H)) * K
        v_base = v + (bos * HV + i_h) * V
        if SAVE_NEW_VALUE:
            v_new_base = v_new + (bos * HV + i_h) * V
        h_nh = h + (tl.cast(i_n, tl.int64) * NT * HV + i_h) * K * V

        if USE_G:
            g_ratio_ptr = g_ratio + (tl.cast(i_n, tl.int64) * HV + i_h) * T
            g_last_exp_ptr = g_last_exp + (tl.cast(i_n, tl.int64) * HV + i_h) * NT
        if USE_GK:
            if RAGGED:
                gk_base = gk + (bos * HV + i_h) * K
            else:
                gk_last_exp_nh = gk_last_exp + (tl.cast(i_n, tl.int64) * HV + i_h) * NT * K

        b_h1 = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_INITIAL_STATE:
            h0_ptr = h0 + i_nh * K * V
            o_i = tl.arange(0, BK)
            o_v = v_start + tl.arange(0, BV)
            p_h0_1 = h0_ptr + o_i[:, None] * V + o_v[None, :]
            b_h1 += tl.load(p_h0_1).to(tl.float32)

        for i_t in range(NT):
            h_base = h_nh + tl.cast(i_t, tl.int64) * DH_CS
            o_i = tl.arange(0, BK)
            o_v = v_start + tl.arange(0, BV)
            p_h1 = h_base + o_i[:, None] * V + o_v[None, :]
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty))

            row_base = tl.cast(i_t, tl.int64) * BT
            rows = tl.arange(0, BT)
            b_w = _dense_chunk_load(w_base, row_base, rows[:, None], o_i[None, :], T,
                                    stride_w, BT, RAGGED).to(tl.float32)
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))

            if USE_G:
                p_g_ratio = g_ratio_ptr + row_base + rows
                b_g_ratio = tl.load(p_g_ratio).to(tl.float32)
                b_g_last = tl.load(g_last_exp_ptr + i_t).to(tl.float32)

            b_v = _dense_chunk_load(v_base, row_base, rows[:, None], o_v[None, :], T,
                                    stride_v, BT, RAGGED).to(tl.float32) - b_v
            if RAGGED:
                b_v = tl.where((row_base + rows < T)[:, None], b_v, 0)
            if SAVE_NEW_VALUE:
                _dense_chunk_store(v_new_base, row_base, rows[:, None], o_v[None, :],
                                   b_v.to(v_new.dtype.element_ty), T, stride_v, BT, RAGGED)

            if USE_G:
                b_v = b_v * b_g_ratio[:, None]
                b_h1 *= b_g_last

            if USE_GK:
                if RAGGED:
                    last_idx = min((i_t + 1) * BT, T) - 1
                    b_gk_last1 = exp2(tl.load(gk_base + last_idx.to(tl.int64) * stride_w + o_k).to(tl.float32))
                else:
                    gk_chunk = gk_last_exp_nh + tl.cast(i_t, tl.int64) * K
                    b_gk_last1 = tl.load(gk_chunk + o_k).to(tl.float32)
                b_h1 *= b_gk_last1[:, None]

            b_k = _dense_chunk_load(k_base, row_base, rows[None, :], o_i[:, None], T,
                                    stride_k, BT, RAGGED).to(tl.float32)
            b_h1 = tl.dot(b_k, b_v, b_h1)

        if STORE_FINAL_STATE:
            ht_ptr = ht + i_nh * K * V
            o_i = tl.arange(0, BK)
            o_v = v_start + tl.arange(0, BV)
            p_ht1 = ht_ptr + o_i[:, None] * V + o_v[None, :]
            tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty))


def _prepare_fwd_g_gates(
    g: torch.Tensor | None,
    *,
    B: int,
    T: int,
    HV: int,
    BT: int,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Prepare contiguous scalar gates, precomputing fp32 scales for aligned dense chunks."""
    if g is None:
        return None, None, None
    g_log = g.transpose(1, 2).contiguous()
    if cu_seqlens is None and (T % BT == 0):
        NT = T // BT
        g_f = g_log.float()
        g_view = g_f.view(B, HV, NT, BT)
        g_last = g_view[:, :, :, -1:]
        # fp32 required: bf16 precomp drifts through the recurrence at large NT.
        g_ratio = torch.exp2(g_last - g_view).reshape(B, HV, T)
        g_last_exp = torch.exp2(g_last).reshape(B, HV, NT)
        return None, g_ratio, g_last_exp
    return g_log, None, None


def _prepare_fwd_gk_last_exp(
    gk: torch.Tensor | None,
    *,
    B: int,
    T: int,
    HV: int,
    K: int,
    BT: int,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Precompute fp32 key-gate scales at chunk ends for aligned dense sequences."""
    if gk is None:
        return None, None
    if cu_seqlens is None and (T % BT == 0):
        NT = T // BT
        gk_last = gk.view(B, NT, BT, HV, K)[:, :, -1].permute(0, 2, 1, 3).contiguous()
        return None, torch.exp2(gk_last.float())
    return gk, None


@register(chunk_gated_delta_rule_fwd_h, backend=TritonAscendBackend)
@input_guard(no_guard_contiguous=["k", "w", "u"])
def chunk_gated_delta_rule_fwd_h_npu(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
    use_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size
    if use_graph and (cu_seqlens is None or chunk_indices is None or chunk_offsets is None):
        raise ValueError("GDN graph recurrence requires fixed chunk indices, offsets, and cu_seqlens.")
    cu_seqlens_cpu = None if use_graph else _recurrence_cu_seqlens_cpu(cu_seqlens, cu_seqlens_cpu)
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens=cu_seqlens, chunk_size=BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    # blockdim64 keeps T/K/V masks (and inline exp2 when T % BT != 0).
    masked = use_graph or time_axis_needs_mask(T, BT, cu_seqlens, cu_seqlens_cpu)
    dense_tail = T % BT != 0 and _dense_single_slab_tail(K, V, state_v_first, g, cu_seqlens)
    BK, BV = _select_fwd_h_tiles(K, V, state_v_first, masked=masked and not dense_tail)
    # Partial K/V tiles take the masked DMA path. That path keeps a second image
    # of the state plus the w/v operands: BK=128/BV=128 asks for 230528 B against
    # 192 KiB. Zero-pad onto the tile already chosen so the unmasked kernel runs
    # at the same block size.
    K_orig, V_orig = K, V
    k_pad = triton.cdiv(K, BK) * BK
    v_pad = triton.cdiv(V, BV) * BV
    if (
        not state_v_first
        and not (use_graph or time_axis_needs_mask(T, BT, cu_seqlens, cu_seqlens_cpu))
        and k_pad <= 256
        and (k_pad != K or v_pad != V)
    ):
        if k_pad != K:
            k = _pad_head_dim(k, k_pad)
            w = _pad_head_dim(w, k_pad)
            if gk is not None:
                gk = torch.nn.functional.pad(gk, (0, k_pad - K))
        if v_pad != V:
            u = _pad_head_dim(u, v_pad)
        if initial_state is not None:
            initial_state = torch.nn.functional.pad(initial_state, (0, v_pad - V, 0, k_pad - K))
        K, V = k_pad, v_pad

    k, w, u = k.contiguous(), w.contiguous(), u.contiguous()
    if state_v_first:
        h = (k.new_zeros if use_graph else k.new_empty)(B, NT, HV, V, K)
        final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = (k.new_zeros if use_graph else k.new_empty)(B, NT, HV, K, V)
        final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None

    v_new = (torch.zeros_like if use_graph else torch.empty_like)(u) if save_new_value else None
    g, g_ratio, g_last_exp = _prepare_fwd_g_gates(g=g, B=B, T=T, HV=HV, BT=BT, cu_seqlens=cu_seqlens)
    gk, gk_last_exp = _prepare_fwd_gk_last_exp(gk=gk, B=B, T=T, HV=HV, K=K, BT=BT, cu_seqlens=cu_seqlens)
    oneslab = (
        cu_seqlens is None
        and not state_v_first
        and (T % BT == 0 or dense_tail)
        and K == BK
        and V % BV == 0
    )
    kwargs = dict(
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g_ratio=g_ratio,
        g_last_exp=g_last_exp,
        gk_last_exp=gk_last_exp,
        h=h,
        h0=initial_state,
        ht=final_state,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    if oneslab:
        _launch_core_grid(
            kernel=chunk_gated_delta_rule_h_fwd_kernel_single_slab,
            task_num=N * HV * triton.cdiv(V, BV),
            kernel_kwargs={**kwargs, "gk": gk, "RAGGED": dense_tail},
            **_FWD_H_COMPILE,
        )
    else:
        _launch_core_grid(
            kernel=chunk_gated_delta_rule_h_fwd_kernel,
            task_num=N * HV * triton.cdiv(V, BV),
            kernel_kwargs={
                **kwargs,
                "g": g,
                "gk": gk,
                "cu_seqlens": cu_seqlens,
                "chunk_offsets": chunk_offsets,
                "STATE_V_FIRST": state_v_first,
                "MASK_DMA": use_graph or need_dma_mask(
                    T, BT, K, BK, V, BV, cu_seqlens is not None, cu_seqlens, cu_seqlens_cpu,
                ),
            },
            **_FWD_H_COMPILE,
            **ascend_compile_kwargs(),
        )
    if K_orig != K or V_orig != V:
        h = h[..., :K_orig, :V_orig].contiguous()
        if v_new is not None:
            v_new = v_new[..., :V_orig].contiguous()
        if final_state is not None:
            final_state = final_state[..., :K_orig, :V_orig].contiguous()
    return h, v_new, final_state


# Keep scale a runtime fp32. A Python int (tests pass scale=1) is specialized
# and folded into `b_do * scale`; that compile hangs the masked D=100 kernel.
@triton.jit(do_not_specialize=["T", "task_num", "num_core", "scale"])
def chunk_gated_delta_rule_h_bwd_kernel(
    q,
    k,
    w,
    g,
    g_exp,
    g_ratio,
    g_last_exp,
    gk,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_PRECOMP: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MASK_DMA: tl.constexpr,
):
    core_id = tl.program_id(0)
    MASK_KV: tl.constexpr = K % BK != 0 or V % BV != 0
    T_max = T
    NV: tl.constexpr = tl.cdiv(V, BV)
    for task_id in tl.range(core_id, task_num, num_core):
        i_nh = (task_id // NV).to(tl.int64)
        i_v_task = task_id % NV
        i_n, i_h = i_nh // HV, i_nh % HV
        if IS_VARLEN:
            bos, eos = (
                tl.load(cu_seqlens + i_n).to(tl.int64),
                tl.load(cu_seqlens + i_n + 1).to(tl.int64),
            )
            T = (eos - bos).to(tl.int32)
            NT = tl.cdiv(T, BT)
            boh = tl.load(chunk_offsets + i_n).to(tl.int64)
        else:
            bos = tl.cast(i_n, tl.int64) * T_max
            eos = bos + T_max
            NT = tl.cdiv(T, BT)
            boh = tl.cast(i_n, tl.int64) * NT

        stride_v = HV * V
        stride_k = H * K
        stride_w = HV * K

        q_base = q + (bos * H + i_h // (HV // H)) * K
        k_base = k + (bos * H + i_h // (HV // H)) * K
        w_base = w + (bos * HV + i_h) * K
        do_base = do + (bos * HV + i_h) * V
        dv_base = dv + (bos * HV + i_h) * V
        dv2_base = dv2 + (bos * HV + i_h) * V
        dh_base = dh + (boh * HV + i_h) * K * V
        if USE_GK:
            gk_base = gk + (bos * HV + i_h) * K
        if USE_G:
            if USE_G_PRECOMP:
                g_exp_base = g_exp + (tl.cast(i_n, tl.int64) * HV + i_h) * T_max
                g_ratio_base = g_ratio + (tl.cast(i_n, tl.int64) * HV + i_h) * T_max
                g_last_exp_base = g_last_exp + (tl.cast(i_n, tl.int64) * HV + i_h) * NT
                g_base = g
            elif IS_VARLEN:
                g_base = g + (bos + tl.cast(i_h, tl.int64) * T_max)
                g_exp_base = g_exp
                g_ratio_base = g_ratio
                g_last_exp_base = g_last_exp
            else:
                g_base = g + (tl.cast(i_n, tl.int64) * HV + i_h) * T_max
                g_exp_base = g_exp
                g_ratio_base = g_ratio
                g_last_exp_base = g_last_exp
        else:
            g_base = g
            g_exp_base = g_exp
            g_ratio_base = g_ratio
            g_last_exp_base = g_last_exp

        for i_v in range(i_v_task, i_v_task + 1):
            v_start = i_v * BV

            if STATE_V_FIRST:
                b_dh1 = tl.zeros([BV, 64], dtype=tl.float32)
                if K > 64:
                    b_dh2 = tl.zeros([BV, 64], dtype=tl.float32)
                if K > 128:
                    b_dh3 = tl.zeros([BV, 64], dtype=tl.float32)
                if K > 192:
                    b_dh4 = tl.zeros([BV, 64], dtype=tl.float32)
            else:
                b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
                if K > 64:
                    b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
                if K > 128:
                    b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
                if K > 192:
                    b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

            if USE_FINAL_STATE_GRADIENT:
                dht_base = dht + i_nh * K * V
                if STATE_V_FIRST:
                    o_v = v_start + tl.arange(0, BV)
                    m_v = (o_v >= 0) & (o_v < V)
                    o_i = tl.arange(0, 64)
                    m_i = (o_i >= 0) & (o_i < K)
                    m_p_dht1 = m_v[:, None] & m_i[None, :]
                    p_dht1 = dht_base + o_v[:, None] * K + o_i[None, :]
                else:
                    o_i = tl.arange(0, 64)
                    m_i = (o_i >= 0) & (o_i < K)
                    o_v = v_start + tl.arange(0, BV)
                    m_v = (o_v >= 0) & (o_v < V)
                    m_p_dht1 = m_i[:, None] & m_v[None, :]
                    p_dht1 = dht_base + o_i[:, None] * V + o_v[None, :]
                b_dh1 += mload(p_dht1, m_p_dht1, MASK_KV)
                if K > 64:
                    if STATE_V_FIRST:
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        o_i = 64 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        m_p_dht2 = m_v[:, None] & m_i[None, :]
                        p_dht2 = dht_base + o_v[:, None] * K + o_i[None, :]
                    else:
                        o_i = 64 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_dht2 = m_i[:, None] & m_v[None, :]
                        p_dht2 = dht_base + o_i[:, None] * V + o_v[None, :]
                    b_dh2 += mload(p_dht2, m_p_dht2, MASK_KV)
                if K > 128:
                    if STATE_V_FIRST:
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        o_i = 128 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        m_p_dht3 = m_v[:, None] & m_i[None, :]
                        p_dht3 = dht_base + o_v[:, None] * K + o_i[None, :]
                    else:
                        o_i = 128 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_dht3 = m_i[:, None] & m_v[None, :]
                        p_dht3 = dht_base + o_i[:, None] * V + o_v[None, :]
                    b_dh3 += mload(p_dht3, m_p_dht3, MASK_KV)
                if K > 192:
                    if STATE_V_FIRST:
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        o_i = 192 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        m_p_dht4 = m_v[:, None] & m_i[None, :]
                        p_dht4 = dht_base + o_v[:, None] * K + o_i[None, :]
                    else:
                        o_i = 192 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_dht4 = m_i[:, None] & m_v[None, :]
                        p_dht4 = dht_base + o_i[:, None] * V + o_v[None, :]
                    b_dh4 += mload(p_dht4, m_p_dht4, MASK_KV)

            if USE_GK:
                o_k1 = tl.arange(0, 64)

            DH_CS: tl.constexpr = HV * K * V
            dh_chunk = dh_base + (NT - 1).to(tl.int64) * DH_CS

            for phase in tl.static_range(2):
                if phase == 0:
                    chunk_start, chunk_end = NT - 1, T // BT - 1
                else:
                    chunk_start, chunk_end = T // BT - 1, -1
                for i_t in range(chunk_start, chunk_end, -1):
                    if STATE_V_FIRST:
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        o_i = tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        m_p_dh1 = m_v[:, None] & m_i[None, :]
                        p_dh1 = dh_chunk + o_v[:, None] * K + o_i[None, :]
                    else:
                        o_i = tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_dh1 = m_i[:, None] & m_v[None, :]
                        p_dh1 = dh_chunk + o_i[:, None] * V + o_v[None, :]
                    mstore(p_dh1,  b_dh1.to(p_dh1.dtype.element_ty), m_p_dh1, MASK_KV)
                    if K > 64:
                        if STATE_V_FIRST:
                            o_v = v_start + tl.arange(0, BV)
                            m_v = (o_v >= 0) & (o_v < V)
                            o_i = 64 + tl.arange(0, 64)
                            m_i = (o_i >= 0) & (o_i < K)
                            m_p_dh2 = m_v[:, None] & m_i[None, :]
                            p_dh2 = dh_chunk + o_v[:, None] * K + o_i[None, :]
                        else:
                            o_i = 64 + tl.arange(0, 64)
                            m_i = (o_i >= 0) & (o_i < K)
                            o_v = v_start + tl.arange(0, BV)
                            m_v = (o_v >= 0) & (o_v < V)
                            m_p_dh2 = m_i[:, None] & m_v[None, :]
                            p_dh2 = dh_chunk + o_i[:, None] * V + o_v[None, :]
                        mstore(p_dh2,  b_dh2.to(p_dh2.dtype.element_ty), m_p_dh2, MASK_KV)
                    if K > 128:
                        if STATE_V_FIRST:
                            o_v = v_start + tl.arange(0, BV)
                            m_v = (o_v >= 0) & (o_v < V)
                            o_i = 128 + tl.arange(0, 64)
                            m_i = (o_i >= 0) & (o_i < K)
                            m_p_dh3 = m_v[:, None] & m_i[None, :]
                            p_dh3 = dh_chunk + o_v[:, None] * K + o_i[None, :]
                        else:
                            o_i = 128 + tl.arange(0, 64)
                            m_i = (o_i >= 0) & (o_i < K)
                            o_v = v_start + tl.arange(0, BV)
                            m_v = (o_v >= 0) & (o_v < V)
                            m_p_dh3 = m_i[:, None] & m_v[None, :]
                            p_dh3 = dh_chunk + o_i[:, None] * V + o_v[None, :]
                        mstore(p_dh3,  b_dh3.to(p_dh3.dtype.element_ty), m_p_dh3, MASK_KV)
                    if K > 192:
                        if STATE_V_FIRST:
                            o_v = v_start + tl.arange(0, BV)
                            m_v = (o_v >= 0) & (o_v < V)
                            o_i = 192 + tl.arange(0, 64)
                            m_i = (o_i >= 0) & (o_i < K)
                            m_p_dh4 = m_v[:, None] & m_i[None, :]
                            p_dh4 = dh_chunk + o_v[:, None] * K + o_i[None, :]
                        else:
                            o_i = 192 + tl.arange(0, 64)
                            m_i = (o_i >= 0) & (o_i < K)
                            o_v = v_start + tl.arange(0, BV)
                            m_v = (o_v >= 0) & (o_v < V)
                            m_p_dh4 = m_i[:, None] & m_v[None, :]
                            p_dh4 = dh_chunk + o_i[:, None] * V + o_v[None, :]
                        mstore(p_dh4,  b_dh4.to(p_dh4.dtype.element_ty), m_p_dh4, MASK_KV)

                    if not (MASK_DMA and phase == 0):
                        row_base = tl.cast(i_t, tl.int64) * BT
                        rows = tl.arange(0, BT)
                        m_rows = tl.full((BT,), True, tl.int1)
                    else:
                        row_base, rows_local, m_rows = chunk_token_rows(i_t, T, BT)
                        rows = clamp_token_rows(row_base + rows_local, T)
                        row_base = tl.cast(0, tl.int64)
                    if USE_G:
                        if USE_G_PRECOMP:
                            bg_last_exp = tl.load(g_last_exp_base + i_t).to(tl.float32)
                            p_g_exp = add_token_rows(g_exp_base, row_base, rows, 1, True)
                            p_g_ratio = add_token_rows(g_ratio_base, row_base, rows, 1, True)
                            b_g_exp = mload(p_g_exp, m_rows, MASK_DMA and phase == 0).to(tl.float32)
                            b_g_ratio = mload(p_g_ratio, m_rows, MASK_DMA and phase == 0).to(tl.float32)
                        else:
                            last_idx = min((i_t + 1) * BT, T) - 1
                            bg_last = tl.load(g_base + last_idx).to(tl.float32)
                            p_g = add_token_rows(g_base, row_base, rows, 1, True)
                            b_g = mload(p_g, m_rows, MASK_DMA and phase == 0).to(tl.float32)
                            bg_last_exp = exp2(bg_last)
                            b_g_exp = exp2(b_g)
                            b_g_ratio = exp2(bg_last - b_g)
                    o_v = v_start + tl.arange(0, BV)
                    m_v = (o_v >= 0) & (o_v < V)
                    m_p_dv = m_rows[:, None] & m_v[None, :]
                    p_dv = add_token_rows(dv_base, row_base, rows[:, None], stride_v, True) + o_v[None, :]
                    p_dv2 = add_token_rows(dv2_base, row_base, rows[:, None], stride_v, True) + o_v[None, :]
                    p_do = add_token_rows(do_base, row_base, rows[:, None], stride_v, True) + o_v[None, :]
                    b_do = mload(p_do, m_p_dv, (MASK_DMA and phase == 0) or K % BK != 0 or V % BV != 0)

                    o_i = tl.arange(0, 64)
                    m_i = (o_i >= 0) & (o_i < K)
                    m_p_k = m_rows[:, None] & m_i[None, :]
                    p_k = add_token_rows(k_base, row_base, rows[:, None], stride_k, True) + o_i[None, :]
                    b_k = mload(p_k, m_p_k, (MASK_DMA and phase == 0) or K % BK != 0 or V % BV != 0)
                    if USE_GK:
                        last_idx = min((i_t + 1) * BT, T) - 1
                        b_gk_last1 = mload(gk_base + last_idx * stride_w + o_k1, (o_k1 < K), K % BK != 0).to(tl.float32)
                    if STATE_V_FIRST:
                        b_dv = tl.dot(b_dh1.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)
                        b_dv = tl.trans(b_dv)
                    else:
                        b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype), allow_tf32=False)

                    if K > 64:
                        o_i_2 = 64 + tl.arange(0, 64)
                        m_i_2 = (o_i_2 >= 0) & (o_i_2 < K)
                        m_p_k = m_rows[:, None] & m_i_2[None, :]
                        p_k = add_token_rows(k_base, row_base, rows[:, None], stride_k, True) + o_i_2[None, :]
                        b_k = mload(p_k, m_p_k, (MASK_DMA and phase == 0) or K % BK != 0 or V % BV != 0)
                        if USE_GK:
                            o_k2 = 64 + o_k1
                            b_gk_last2 = mload(gk_base + last_idx * stride_w + o_k2, (o_k2 < K), K % BK != 0).to(tl.float32)
                        if STATE_V_FIRST:
                            b_dv_part = tl.dot(b_dh2.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)
                            b_dv += tl.trans(b_dv_part)
                        else:
                            b_dv = tl.dot(b_k, b_dh2.to(b_k.dtype), b_dv, allow_tf32=False)

                    if K > 128:
                        o_i_2 = 128 + tl.arange(0, 64)
                        m_i_2 = (o_i_2 >= 0) & (o_i_2 < K)
                        m_p_k = m_rows[:, None] & m_i_2[None, :]
                        p_k = add_token_rows(k_base, row_base, rows[:, None], stride_k, True) + o_i_2[None, :]
                        b_k = mload(p_k, m_p_k, (MASK_DMA and phase == 0) or K % BK != 0 or V % BV != 0)
                        if USE_GK:
                            o_k3 = 128 + o_k1
                            b_gk_last3 = mload(gk_base + last_idx * stride_w + o_k3, (o_k3 < K), K % BK != 0).to(tl.float32)
                        if STATE_V_FIRST:
                            b_dv_part = tl.dot(b_dh3.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)
                            b_dv += tl.trans(b_dv_part)
                        else:
                            b_dv = tl.dot(b_k, b_dh3.to(b_k.dtype), b_dv, allow_tf32=False)

                    if K > 192:
                        o_i_2 = 192 + tl.arange(0, 64)
                        m_i_2 = (o_i_2 >= 0) & (o_i_2 < K)
                        m_p_k = m_rows[:, None] & m_i_2[None, :]
                        p_k = add_token_rows(k_base, row_base, rows[:, None], stride_k, True) + o_i_2[None, :]
                        b_k = mload(p_k, m_p_k, (MASK_DMA and phase == 0) or K % BK != 0 or V % BV != 0)
                        if USE_GK:
                            o_k4 = 192 + o_k1
                            b_gk_last4 = mload(gk_base + last_idx * stride_w + o_k4, (o_k4 < K), K % BK != 0).to(tl.float32)
                        if STATE_V_FIRST:
                            b_dv_part = tl.dot(b_dh4.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)
                            b_dv += tl.trans(b_dv_part)
                        else:
                            b_dv = tl.dot(b_k, b_dh4.to(b_k.dtype), b_dv, allow_tf32=False)

                    # Cube dots can leave NaN in tail rows on the dh reduction axis.
                    if MASK_DMA and phase == 0:
                        if USE_G:
                            b_dv = tl.where(m_rows[:, None], b_dv * b_g_ratio[:, None], 0)
                        else:
                            b_dv = tl.where(m_rows[:, None], b_dv, 0)
                    elif USE_G:
                        b_dv *= b_g_ratio[:, None]
                    b_dv = b_dv.to(tl.float32) + mload(p_dv, m_p_dv, (MASK_DMA and phase == 0) or K %
                                                       BK != 0 or V % BV != 0).to(tl.float32)
                    mstore(p_dv2,  b_dv.to(p_dv2.dtype.element_ty), m_p_dv,
                           (MASK_DMA and phase == 0) or K % BK != 0 or V % BV != 0)
                    # v-first uses dv as lhs of the subtract dot, which clobbers it.
                    if STATE_V_FIRST:
                        b_dv_pristine = b_dv + 0.0

                    if USE_G:
                        b_dh1 *= bg_last_exp
                        if K > 64:
                            b_dh2 *= bg_last_exp
                        if K > 128:
                            b_dh3 *= bg_last_exp
                        if K > 192:
                            b_dh4 *= bg_last_exp
                        b_do = b_do * (b_g_exp * scale)[:, None]
                    else:
                        b_do = b_do * scale
                    # v-first feeds do as lhs, so each K-slab needs its own copy.
                    # Non-v-first uses do only as rhs; one buffer is enough.
                    if STATE_V_FIRST:
                        b_do_c = b_do + 0.0
                        if K > 128:
                            b_do_c2 = b_do + 0.0
                        if K > 192:
                            b_do_c3 = b_do + 0.0

                    m_p_w = m_i[:, None] & m_rows[None, :]
                    p_w = add_token_rows(w_base, row_base, rows[None, :], stride_w, True) + o_i[:, None]
                    p_q = add_token_rows(q_base, row_base, rows[None, :], stride_k, True) + o_i[:, None]
                    b_w = mload(p_w, m_p_w, (MASK_DMA and phase == 0) or K % BK != 0 or V % BV != 0)
                    b_q = mload(p_q, m_p_w, (MASK_DMA and phase == 0) or K % BK != 0 or V % BV != 0)
                    if USE_GK:
                        if STATE_V_FIRST:
                            b_dh1 *= exp2(b_gk_last1)[None, :]
                        else:
                            b_dh1 *= exp2(b_gk_last1[:, None])
                    if STATE_V_FIRST:
                        b_dh1 = tl.dot(tl.trans(b_do.to(b_q.dtype)), tl.trans(b_q), b_dh1, allow_tf32=False)
                        b_dv_i = b_dv_pristine + 0.0
                        b_dh1 -= tl.dot(tl.trans(b_dv_i.to(b_w.dtype)), tl.trans(b_w), allow_tf32=False)
                    else:
                        b_dh1 += (
                            tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), allow_tf32=False)
                            - tl.dot(b_w, b_dv.to(b_w.dtype), allow_tf32=False)
                        )

                    if K > 64:
                        o_i_2 = 64 + tl.arange(0, 64)
                        m_i_2 = (o_i_2 >= 0) & (o_i_2 < K)
                        m_p_q = m_i_2[:, None] & m_rows[None, :]
                        p_q = add_token_rows(q_base, row_base, rows[None, :], stride_k, True) + o_i_2[:, None]
                        p_w = add_token_rows(w_base, row_base, rows[None, :], stride_w, True) + o_i_2[:, None]
                        b_q = mload(p_q, m_p_q, (MASK_DMA and phase == 0) or K % BK != 0 or V % BV != 0)
                        b_w = mload(p_w, m_p_q, (MASK_DMA and phase == 0) or K % BK != 0 or V % BV != 0)
                        if USE_GK:
                            if STATE_V_FIRST:
                                b_dh2 *= exp2(b_gk_last2)[None, :]
                            else:
                                b_dh2 *= exp2(b_gk_last2[:, None])
                        if STATE_V_FIRST:
                            b_dh2 = tl.dot(tl.trans(b_do_c.to(b_q.dtype)), tl.trans(b_q), b_dh2, allow_tf32=False)
                            b_dv_i = b_dv_pristine + 0.0
                            b_dh2 -= tl.dot(tl.trans(b_dv_i.to(b_w.dtype)), tl.trans(b_w), allow_tf32=False)
                        else:
                            b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), allow_tf32=False) - tl.dot(
                                b_w, b_dv.to(b_w.dtype), allow_tf32=False
                            )

                    if K > 128:
                        o_i_2 = 128 + tl.arange(0, 64)
                        m_i_2 = (o_i_2 >= 0) & (o_i_2 < K)
                        m_p_q = m_i_2[:, None] & m_rows[None, :]
                        p_q = add_token_rows(q_base, row_base, rows[None, :], stride_k, True) + o_i_2[:, None]
                        p_w = add_token_rows(w_base, row_base, rows[None, :], stride_w, True) + o_i_2[:, None]
                        b_q = mload(p_q, m_p_q, (MASK_DMA and phase == 0) or K % BK != 0 or V % BV != 0)
                        b_w = mload(p_w, m_p_q, (MASK_DMA and phase == 0) or K % BK != 0 or V % BV != 0)
                        if USE_GK:
                            if STATE_V_FIRST:
                                b_dh3 *= exp2(b_gk_last3)[None, :]
                            else:
                                b_dh3 *= exp2(b_gk_last3[:, None])
                        if STATE_V_FIRST:
                            b_dh3 = tl.dot(tl.trans(b_do_c2.to(b_q.dtype)), tl.trans(b_q), b_dh3, allow_tf32=False)
                            b_dv_i = b_dv_pristine + 0.0
                            b_dh3 -= tl.dot(tl.trans(b_dv_i.to(b_w.dtype)), tl.trans(b_w), allow_tf32=False)
                        else:
                            b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), allow_tf32=False) - tl.dot(
                                b_w, b_dv.to(b_w.dtype), allow_tf32=False
                            )

                    if K > 192:
                        o_i_2 = 192 + tl.arange(0, 64)
                        m_i_2 = (o_i_2 >= 0) & (o_i_2 < K)
                        m_p_q = m_i_2[:, None] & m_rows[None, :]
                        p_q = add_token_rows(q_base, row_base, rows[None, :], stride_k, True) + o_i_2[:, None]
                        p_w = add_token_rows(w_base, row_base, rows[None, :], stride_w, True) + o_i_2[:, None]
                        b_q = mload(p_q, m_p_q, (MASK_DMA and phase == 0) or K % BK != 0 or V % BV != 0)
                        b_w = mload(p_w, m_p_q, (MASK_DMA and phase == 0) or K % BK != 0 or V % BV != 0)
                        if USE_GK:
                            if STATE_V_FIRST:
                                b_dh4 *= exp2(b_gk_last4)[None, :]
                            else:
                                b_dh4 *= exp2(b_gk_last4[:, None])
                        if STATE_V_FIRST:
                            b_dh4 = tl.dot(tl.trans(b_do_c3.to(b_q.dtype)), tl.trans(b_q), b_dh4, allow_tf32=False)
                            b_dv_i = b_dv_pristine + 0.0
                            b_dh4 -= tl.dot(tl.trans(b_dv_i.to(b_w.dtype)), tl.trans(b_w), allow_tf32=False)
                        else:
                            b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), allow_tf32=False) - tl.dot(
                                b_w, b_dv.to(b_w.dtype), allow_tf32=False
                            )

                    dh_chunk -= DH_CS

            if USE_INITIAL_STATE:
                dh0_base = dh0 + i_nh * K * V
                if STATE_V_FIRST:
                    o_v = v_start + tl.arange(0, BV)
                    m_v = (o_v >= 0) & (o_v < V)
                    o_i = tl.arange(0, 64)
                    m_i = (o_i >= 0) & (o_i < K)
                    m_p_dh0 = m_v[:, None] & m_i[None, :]
                    p_dh0 = dh0_base + o_v[:, None] * K + o_i[None, :]
                else:
                    o_i = tl.arange(0, 64)
                    m_i = (o_i >= 0) & (o_i < K)
                    o_v = v_start + tl.arange(0, BV)
                    m_v = (o_v >= 0) & (o_v < V)
                    m_p_dh0 = m_i[:, None] & m_v[None, :]
                    p_dh0 = dh0_base + o_i[:, None] * V + o_v[None, :]
                mstore(p_dh0,  b_dh1.to(p_dh0.dtype.element_ty), m_p_dh0, MASK_KV)
                if K > 64:
                    if STATE_V_FIRST:
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        o_i = 64 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        m_p_dh1 = m_v[:, None] & m_i[None, :]
                        p_dh1 = dh0_base + o_v[:, None] * K + o_i[None, :]
                    else:
                        o_i = 64 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_dh1 = m_i[:, None] & m_v[None, :]
                        p_dh1 = dh0_base + o_i[:, None] * V + o_v[None, :]
                    mstore(p_dh1,  b_dh2.to(p_dh1.dtype.element_ty), m_p_dh1, MASK_KV)
                if K > 128:
                    if STATE_V_FIRST:
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        o_i = 128 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        m_p_dh2 = m_v[:, None] & m_i[None, :]
                        p_dh2 = dh0_base + o_v[:, None] * K + o_i[None, :]
                    else:
                        o_i = 128 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_dh2 = m_i[:, None] & m_v[None, :]
                        p_dh2 = dh0_base + o_i[:, None] * V + o_v[None, :]
                    mstore(p_dh2,  b_dh3.to(p_dh2.dtype.element_ty), m_p_dh2, MASK_KV)
                if K > 192:
                    if STATE_V_FIRST:
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        o_i = 192 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        m_p_dh3 = m_v[:, None] & m_i[None, :]
                        p_dh3 = dh0_base + o_v[:, None] * K + o_i[None, :]
                    else:
                        o_i = 192 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_dh3 = m_i[:, None] & m_v[None, :]
                        p_dh3 = dh0_base + o_i[:, None] * V + o_v[None, :]
                    mstore(p_dh3,  b_dh4.to(p_dh3.dtype.element_ty), m_p_dh3, MASK_KV)


def _prepare_dhu_gate_tensors(
    g: torch.Tensor | None,
    *,
    B: int,
    T: int,
    HV: int,
    BT: int,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, bool]:
    """Prepare contiguous backward gates, precomputing fp32 scales for aligned dense chunks."""
    if g is None:
        return None, None, None, None, False
    g_log = g.transpose(1, 2).contiguous()
    if cu_seqlens is None and (T % BT == 0):
        NT = T // BT
        g_f = g_log.float()
        g_view = g_f.view(B, HV, NT, BT)
        g_last = g_view[:, :, :, -1:]
        # fp32 required: bf16 precomp drifts through the reverse recurrence at large NT.
        g_ratio = torch.exp2(g_last - g_view).reshape(B, HV, T)
        g_exp = torch.exp2(g_f)
        g_last_exp = torch.exp2(g_last).reshape(B, HV, NT)
        return None, g_exp, g_ratio, g_last_exp, True
    return g_log, None, None, None, False


@register(chunk_gated_delta_rule_bwd_dhu, backend=TritonAscendBackend)
@input_guard
def chunk_gated_delta_rule_bwd_dhu_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
    use_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    BT = chunk_size
    if use_graph and (cu_seqlens is None or chunk_indices is None or chunk_offsets is None):
        raise ValueError("GDN graph recurrence requires fixed chunk indices, offsets, and cu_seqlens.")
    cu_seqlens_cpu = None if use_graph else _recurrence_cu_seqlens_cpu(cu_seqlens, cu_seqlens_cpu)
    assert K <= 256, "current kernel does not support head dimension being larger than 256."
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens=cu_seqlens, chunk_size=BT)

    # Ragged K/V forces a partial slab. The selector then caps BV, and the
    # masked image still overflows (D=100 / two slabs). Pad onto the power-of-two
    # tile the aligned kernel already compiles and slice the gradients back.
    K_orig, V_orig = K, V
    if not (use_graph or time_axis_needs_mask(T, BT, cu_seqlens, cu_seqlens_cpu)):
        k_pad = triton.next_power_of_2(K)
        v_pad = triton.next_power_of_2(V)
        if k_pad <= 256 and v_pad <= 256 and (k_pad != K or v_pad != V):
            if k_pad != K:
                q = torch.nn.functional.pad(q, (0, k_pad - K))
                k = torch.nn.functional.pad(k, (0, k_pad - K))
                w = torch.nn.functional.pad(w, (0, k_pad - K))
                if gk is not None and gk.shape[-1] == K:
                    gk = torch.nn.functional.pad(gk, (0, k_pad - K))
            if v_pad != V:
                do = torch.nn.functional.pad(do, (0, v_pad - V))
                dv = torch.nn.functional.pad(dv, (0, v_pad - V))
            if h0 is not None:
                if state_v_first:
                    h0 = torch.nn.functional.pad(h0, (0, k_pad - K, 0, v_pad - V))
                else:
                    h0 = torch.nn.functional.pad(h0, (0, v_pad - V, 0, k_pad - K))
            if dht is not None:
                if state_v_first:
                    dht = torch.nn.functional.pad(dht, (0, k_pad - K, 0, v_pad - V))
                else:
                    dht = torch.nn.functional.pad(dht, (0, v_pad - V, 0, k_pad - K))
            K, V = k_pad, v_pad

    if state_v_first:
        dh = (q.new_zeros if use_graph else q.new_empty)(B, NT, HV, V, K)
    else:
        dh = (q.new_zeros if use_graph else q.new_empty)(B, NT, HV, K, V)
    dh0 = (torch.zeros_like if use_graph else torch.empty_like)(h0, dtype=torch.float32) if h0 is not None else None
    # keep the caller's dv unchanged while updating the value gradient.
    dv2 = (torch.zeros_like if use_graph else torch.empty_like)(dv)
    g_log, g_exp, g_ratio, g_last_exp, use_g_precomp = _prepare_dhu_gate_tensors(
        g=g,
        B=B,
        T=T,
        HV=HV,
        BT=BT,
        cu_seqlens=cu_seqlens,
    )
    gate_inline = g is not None and not use_g_precomp
    masked = use_graph or time_axis_needs_mask(T, BT, cu_seqlens, cu_seqlens_cpu)
    min_nv = triton.cdiv(get_npu_properties()["num_aicore"], N * HV)
    BK, BV = _select_bwd_dhu_tiles(
        K=K,
        V=V,
        state_v_first=state_v_first,
        min_nv=min_nv,
        gate_inline=gate_inline,
        masked=masked,
        use_gate=g is not None,
    )
    # unit_flag overlaps cube and fixpipe. On the v-first transpose path that
    # overlap deadlocks when the sequence is ragged or packed, and the device
    # sync then times out. Aligned dense launches keep the flag.
    compile_opts = dict(_DHU_COMPILE)
    if state_v_first and (masked or cu_seqlens is not None):
        compile_opts["unit_flag"] = False
    _launch_core_grid(
        kernel=chunk_gated_delta_rule_h_bwd_kernel,
        task_num=N * HV * triton.cdiv(V, BV),
        **compile_opts,
        kernel_kwargs={
            "q": q,
            "k": k,
            "w": w,
            "g": g_log,
            "g_exp": g_exp,
            "g_ratio": g_ratio,
            "g_last_exp": g_last_exp,
            "gk": gk,
            "dht": dht,
            "dh0": dh0,
            "do": do,
            "dh": dh,
            "dv": dv,
            "dv2": dv2,
            "cu_seqlens": cu_seqlens,
            "chunk_offsets": chunk_offsets,
            "scale": float(scale),
            "T": T,
            "H": H,
            "HV": HV,
            "K": K,
            "V": V,
            "BT": BT,
            "BK": BK,
            "BV": BV,
            "USE_G": g is not None,
            "USE_G_PRECOMP": use_g_precomp,
            "USE_GK": gk is not None,
            "USE_INITIAL_STATE": h0 is not None,
            "USE_FINAL_STATE_GRADIENT": dht is not None,
            "STATE_V_FIRST": state_v_first,
            "IS_VARLEN": cu_seqlens is not None,
            "MASK_DMA": use_graph or need_dma_mask(
                T, BT, K, BK, V, BV, cu_seqlens is not None, cu_seqlens, cu_seqlens_cpu,
            ),
        },
    )
    if K_orig != K or V_orig != V:
        if state_v_first:
            dh = dh[..., :V_orig, :K_orig].contiguous()
            if dh0 is not None:
                dh0 = dh0[..., :V_orig, :K_orig].contiguous()
        else:
            dh = dh[..., :K_orig, :V_orig].contiguous()
            if dh0 is not None:
                dh0 = dh0[..., :K_orig, :V_orig].contiguous()
        dv2 = dv2[..., :V_orig].contiguous()
    return dh, dh0, dv2
