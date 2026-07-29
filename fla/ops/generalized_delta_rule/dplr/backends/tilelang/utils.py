# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from collections import deque
from dataclasses import dataclass

import torch

__all__ = [
    "ChunkLayout",
    "build_rect_chunk_layout",
    "build_varlen_chunk_layout",
]


@dataclass(frozen=True)
class ChunkLayout:
    """FLA-style chunk layout shared by DPLR stages."""

    cu_seqlens: torch.Tensor
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor


def varlen_chunk_count_upper(total_tokens: int, n_seqs: int, chunk_size: int) -> int:
    """Host-shape upper bound for packed-varlen chunk slots.

    The exact number of chunks is data-dependent:
        sum_i ceil((cu[i + 1] - cu[i]) / chunk_size)

    Reading that exact value on the host introduces a CUDA sync.  For varlen
    kernels we instead allocate a tight upper bound,
    ceil(total_tokens / chunk_size) + n_seqs - 1, and mark any extra slots as
    no-op sentinel rows in the CUDA-built layout.
    """
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    if n_seqs < 0:
        raise ValueError("n_seqs must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if total_tokens == 0 or n_seqs == 0:
        return 0
    return (total_tokens + chunk_size - 1) // chunk_size + max(n_seqs - 1, 0)


def _as_cuda_cu_seqlens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    if cu_seqlens.device.type != "cuda":
        raise ValueError("varlen DPLR requires CUDA cu_seqlens; CPU layout construction is not used")
    return cu_seqlens.to(dtype=torch.int32).contiguous()


# Bounded identity-keyed cache (same contract as fla.utils.tensor_cache):
# repeated calls with the same cu_seqlens object skip the layout build. The
# cached tensors are kept alive by the cache itself, so ids cannot recycle.
_VARLEN_LAYOUT_CACHE: deque = deque(maxlen=4)


def build_varlen_chunk_layout(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    total_tokens: int,
) -> ChunkLayout:
    """Build fixed-shape varlen chunk indices/offsets on CUDA."""
    for cu_cached, cs_cached, tt_cached, layout in _VARLEN_LAYOUT_CACHE:
        if cu_cached is cu_seqlens and cs_cached == chunk_size and tt_cached == total_tokens:
            return layout

    cu = _as_cuda_cu_seqlens(cu_seqlens)
    n_seqs = cu.shape[0] - 1
    nt_alloc = varlen_chunk_count_upper(total_tokens, n_seqs, chunk_size)

    lengths = cu[1:] - cu[:-1]
    chunks_per_seq = torch.div(lengths + chunk_size - 1, chunk_size, rounding_mode="floor")
    chunk_offsets = torch.cat(
        [
            torch.zeros((1,), device=cu.device, dtype=torch.int32),
            chunks_per_seq.cumsum(dim=0, dtype=torch.int32),
        ],
        dim=0,
    ).contiguous()

    rows = torch.arange(nt_alloc, device=cu.device, dtype=torch.int32)
    exact_total = chunk_offsets[-1]
    valid = rows < exact_total
    seq = torch.searchsorted(chunk_offsets, rows, right=True, out_int32=True) - 1
    seq = torch.where(valid, seq, torch.full_like(seq, -1))
    safe_seq = seq.clamp(min=0, max=max(n_seqs - 1, 0))
    local_chunk = rows - chunk_offsets[safe_seq]
    local_chunk = torch.where(valid, local_chunk, torch.zeros_like(local_chunk))

    chunk_indices = torch.stack([seq, local_chunk], dim=1).contiguous()
    layout = ChunkLayout(cu, chunk_indices, chunk_offsets)
    _VARLEN_LAYOUT_CACHE.append((cu_seqlens, chunk_size, total_tokens, layout))
    return layout


# Rect layouts are pure functions of (B, T, BT, device): every stage builds the
# same one within a call, so memoize it like the varlen layout above.
_RECT_LAYOUT_CACHE: dict = {}


def build_rect_chunk_layout(B: int, T_: int, BT: int, device: torch.device) -> ChunkLayout:
    """Build canonical chunk layout for a rectangular batch."""
    key = (B, T_, BT, device)
    layout = _RECT_LAYOUT_CACHE.get(key)
    if layout is not None:
        return layout
    n_chunks_per_seq = (T_ + BT - 1) // BT
    cu = torch.arange(B + 1, device=device, dtype=torch.int32) * T_
    seq = torch.arange(B, device=device, dtype=torch.int32).repeat_interleave(n_chunks_per_seq)
    local = torch.arange(n_chunks_per_seq, device=device, dtype=torch.int32).repeat(B)
    chunk_indices = torch.stack([seq, local], dim=1).contiguous()
    chunk_offsets = (torch.arange(B + 1, device=device, dtype=torch.int32) * n_chunks_per_seq).contiguous()
    layout = ChunkLayout(cu, chunk_indices, chunk_offsets)
    if len(_RECT_LAYOUT_CACHE) >= 8:
        _RECT_LAYOUT_CACHE.clear()
    _RECT_LAYOUT_CACHE[key] = layout
    return layout
