# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.runtime import driver

from fla.ops.common.backends.triton_ascend.chunk_delta_h import _pad_recurrence_tokens, _restore_gate_gradient
from fla.ops.utils import prepare_chunk_offsets
from fla.ops.utils.triton_ascend.packing import pack_head_first


@triton.jit(do_not_specialize=['N'])
def _scatter_tail_rows_kernel(x, y, sources, destinations, N,
                              WIDTH: tl.constexpr, BLOCK: tl.constexpr, GATHER: tl.constexpr):
    blocks: tl.constexpr = tl.cdiv(WIDTH, BLOCK)
    lanes = tl.arange(0, BLOCK)
    for task in tl.range(tl.program_id(0), N * blocks, tl.num_programs(0)):
        row = task // blocks
        column = (task % blocks).to(tl.int64) * BLOCK
        source = tl.load(sources + row).to(tl.int64) if GATHER else row.to(tl.int64)
        destination = tl.load(destinations + row).to(tl.int64)
        values = tl.load((x + source * WIDTH + column) + lanes, column + lanes < WIDTH, other=0)
        tl.store((y + destination * WIDTH + column) + lanes, values, column + lanes < WIDTH)


def _scatter_tail_rows(x, y, sources, destinations):
    width = x.numel() // x.shape[1]
    properties = driver.active.utils.get_device_properties(x.device.index)
    _scatter_tail_rows_kernel[(properties['num_vectorcore'],)](
        x, y, sources, destinations, len(destinations),
        WIDTH=width,
        BLOCK=min(4096, triton.next_power_of_2(width)),
        GATHER=sources is not None,
    )


@lru_cache(maxsize=4)
def _tail_metadata(offsets: tuple[int, ...], chunk_size: int, device: torch.device, dtype: torch.dtype):
    if any(end <= start for start, end in zip(offsets[:-1], offsets[1:])):
        return None
    full, sources, valid, restore, destinations, sequences, chunks = [], [], [], [], [], [], []
    valid_offsets = [0]
    for sequence, (bos, eos) in enumerate(zip(offsets[:-1], offsets[1:])):
        length = eos - bos
        full_count, remainder = divmod(length, chunk_size)
        full.extend((sequence, chunk) for chunk in range(full_count))
        if not remainder:
            continue
        start = eos - remainder
        scratch_start = len(sources)
        sources.extend(range(start, eos))
        sources.extend([eos - 1] * (chunk_size - remainder))
        valid.extend([True] * remainder + [False] * (chunk_size - remainder))
        restore.extend(range(scratch_start, scratch_start + remainder))
        destinations.extend(range(start, eos))
        sequences.append(sequence)
        chunks.append(full_count)
        valid_offsets.append(valid_offsets[-1] + remainder)
    if not sequences or len(full) < 2 * len(sequences):
        return None

    def index(values):
        return torch.tensor(values, dtype=torch.int64, device=device)
    scratch_cpu = torch.arange(len(sequences) + 1, dtype=torch.int32) * chunk_size
    return (
        torch.tensor(full, dtype=dtype, device=device), index(sources),
        torch.tensor(valid, dtype=torch.bool, device=device), index(restore), index(destinations),
        index(sequences), index(chunks), scratch_cpu.to(device), scratch_cpu,
        torch.tensor(valid_offsets, dtype=torch.int32),
    )


def wy_tail_only(compute, tensors, *, scale, state_v_first, cu_seqlens, cu_seqlens_cpu,
                 chunk_size, chunk_indices, chunk_offsets):
    """Keep full chunks in their original layout and gather only partial chunks."""
    offsets = tuple(cu_seqlens_cpu.tolist())
    metadata = _tail_metadata(offsets, chunk_size, tensors['k'].device, cu_seqlens.dtype)
    if metadata is None:
        return None
    full, sources, valid, restore, destinations, sequences, chunks, scratch_cu, scratch_cpu, valid_cpu = metadata
    expected_chunks = sum(triton.cdiv(end - start, chunk_size) for start, end in zip(offsets[:-1], offsets[1:]))
    if chunk_indices is not None and len(chunk_indices) != expected_chunks:
        return None
    if chunk_offsets is None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
    outputs = compute(
        **tensors,
        scale=scale,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_size=chunk_size,
        chunk_indices=full,
        chunk_offsets=chunk_offsets,
        _full_chunks=True,
    )
    states = chunk_offsets.index_select(0, sequences).to(torch.int64) + chunks
    scratch = {}
    for name, tensor in tensors.items():
        if name in ('h', 'dh'):
            scratch[name] = tensor.index_select(1, states)
        elif name in ('v', 'dv', 'A'):
            scratch[name] = pack_head_first(tensor, indices=sources, valid=valid)
        else:
            scratch[name] = _pad_recurrence_tokens(tensor, sources, None if name == 'g' else valid)
    tail_outputs = compute(
        **scratch,
        scale=scale,
        state_v_first=state_v_first,
        cu_seqlens=scratch_cu,
        cu_seqlens_cpu=scratch_cpu,
        chunk_size=chunk_size,
        _full_chunks=True,
    )
    for index, (output, tail) in enumerate(zip(outputs, tail_outputs)):
        if index == 4:
            values = _restore_gate_gradient(tail, restore, valid_cpu, chunk_size)
            _scatter_tail_rows(values, output, None, destinations)
        else:
            _scatter_tail_rows(tail, output, restore, destinations)
    return outputs


def local_tail_only(compute, tensors, *, cu_seqlens, cu_seqlens_cpu, chunk_size, chunk_indices,
                    scale=None, backward=False):
    """Pack only tails for local WY construction or its dAv derivative."""
    metadata = _tail_metadata(tuple(cu_seqlens_cpu.tolist()), chunk_size,
                              tensors['k'].device, cu_seqlens.dtype)
    if metadata is None:
        return None
    full, sources, valid, restore, destinations, sequences, _, _, _, _ = metadata
    if chunk_indices is not None and len(chunk_indices) != len(full) + len(sequences):
        return None
    extra = dict(scale=scale, chunk_size=chunk_size, cu_seqlens_cpu=cu_seqlens_cpu) if backward else {}
    outputs = compute(
        **tensors,
        cu_seqlens=cu_seqlens,
        chunk_indices=full,
        _full_chunks=True,
        **extra,
    )
    scratch = {}
    for name, tensor in tensors.items():
        if tensor is None or (backward and name in ('q', 'k')):
            scratch[name] = tensor
        else:
            values = _pad_recurrence_tokens(tensor, sources, None if name == 'gk' else valid)
            scratch[name] = values.reshape(len(sequences), chunk_size, *tensor.shape[2:])
    extra = dict(scale=scale, chunk_size=chunk_size) if backward else {}
    tails = compute(**scratch, _full_chunks=True, **extra)
    for output, tail in zip(outputs, tails):
        if tail is not None:
            tail = tail.reshape(1, len(sources), *tail.shape[2:])
            _scatter_tail_rows(tail, output, restore, destinations)
    return outputs
