# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.utils import autotune_cache_kwargs, tensor_cache


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [4, 8, 16, 32]
    ],
    key=['B'],
    **autotune_cache_kwargs,
)
@triton.jit
def prepare_position_ids_kernel(
    y,
    cu_seqlens,
    B: tl.constexpr,
):
    i_n = tl.program_id(0)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    o = tl.arange(0, B)
    for i in range(0, tl.cdiv(T, B) * B, B):
        o_i = o + i
        tl.store(y + bos + o_i, o_i, o_i < T)


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_lens_from_mask(mask: torch.BoolTensor) -> torch.LongTensor:
    return mask.sum(dim=-1, dtype=torch.int32)


@tensor_cache
def prepare_cu_seqlens_from_lens(
    lens: torch.LongTensor,
    dtype: torch.dtype | None = torch.int32,
) -> torch.LongTensor:
    return F.pad(lens.cumsum(dim=0, dtype=dtype), (1, 0))


@tensor_cache
def prepare_cu_seqlens_from_mask(
    mask: torch.BoolTensor,
    dtype: torch.dtype | None = torch.int32,
) -> torch.LongTensor:
    return prepare_cu_seqlens_from_lens(prepare_lens_from_mask(mask), dtype)


@tensor_cache
def prepare_split_cu_seqlens(
    batch_size: int | None = None,
    seq_len: int | None = None,
    split_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    dtype: torch.dtype | None = torch.int32,
    device: torch.device | None = torch.device('cpu'),
) -> torch.LongTensor:
    """Sub-split a (optionally packed) batch along the token axis.

    Two calling modes:
      - **Rectangular batch**: pass `batch_size` and `seq_len`, leave
        `cu_seqlens=None`. Internally synthesizes `[0, L, 2L, ..., B*L]`.
      - **Packed varlen**: pass `cu_seqlens`. `batch_size` and `seq_len` are
        ignored (kept as optional kwargs for backward-compat with callers
        that used to pass dummies).

    `split_size` is always required.

    The legacy positional signature `(batch_size, seq_len, split_size, ...)`
    continues to work — the first two args retain their position but may now
    be omitted when `cu_seqlens` is supplied.
    """
    if split_size is None:
        raise TypeError("prepare_split_cu_seqlens() requires `split_size`")
    if cu_seqlens is None:
        if batch_size is None or seq_len is None:
            raise TypeError(
                "prepare_split_cu_seqlens() requires either `cu_seqlens`, "
                "or both `batch_size` and `seq_len`"
            )
        total_tokens = batch_size * seq_len
        cu_seqlens = list(range(0, total_tokens, seq_len)) + [total_tokens]
    else:
        cu_seqlens = cu_seqlens.tolist()
    return torch.tensor(
        [
            i
            for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)
            for i in range(bos, eos, split_size)
        ] + [cu_seqlens[-1]],
        dtype=dtype,
        device=device,
    )


def _segmented_arange(counts: torch.LongTensor) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Vectorised, ``torch.compile``-friendly per-segment index builder.

    Given per-segment element ``counts`` ``[c0, c1, ...]``, returns two flat
    1-D tensors, each of length ``counts.sum()``:

      - ``seg_id``    -- which segment each slot belongs to:
                         ``[0]*c0 + [1]*c1 + ...``
      - ``intra_idx`` -- within-segment running index: for segment ``i`` it is
                         ``[0, 1, ..., c_i - 1]``

    This replaces the historical ``.tolist()`` + per-segment ``torch.arange``
    list-comprehension used by the ``prepare_*`` helpers below, which both
    graph-broke under Dynamo (recompiling every batch as the segment count or
    per-segment lengths shifted) and dispatched ``O(segments)`` ops. Output is
    bit-identical for all non-empty segments.

    Note ``repeat_interleave`` below still reads ``counts.sum()`` to size its
    output; on a CUDA ``counts`` that is one device->host sync -- the same
    single sync the old ``.tolist()`` paid (not per-segment, not a
    regression). Pass host-side ``counts`` -- i.e. the ``prepare_*`` helpers'
    ``cu_seqlens_cpu`` argument -- to avoid it. Under ``torch.compile`` the
    size is instead an unbacked ``SymInt``: no graph break, no recompile.
    """
    n_segs = counts.numel()
    seg_idx = torch.arange(n_segs, device=counts.device, dtype=counts.dtype)
    # The one data-dependent-shape op: `seg_id` labels each output slot with
    # its segment. `repeat_interleave` is a prefix-sum + custom kernel (and,
    # under `torch.compile`, an unbacked-shape node), so we only pay for it
    # once and reuse the result below.
    seg_id = torch.repeat_interleave(seg_idx, counts)
    # Per-segment start offsets in the flat output: [0, c0, c0+c1, ...].
    seg_start = F.pad(counts.cumsum(0), (1, 0))[:-1]
    # `seg_start[seg_id]` is exactly `repeat_interleave(seg_start, counts)` --
    # gathering with the segment labels avoids a second `repeat_interleave`.
    total = seg_id.shape[0]
    flat_pos = torch.arange(total, device=counts.device, dtype=counts.dtype)
    intra_idx = flat_pos - seg_start[seg_id]
    return seg_id, intra_idx


@tensor_cache
def prepare_position_ids(cu_seqlens: torch.LongTensor, cu_seqlens_cpu: torch.LongTensor | None = None) -> torch.LongTensor:
    # Flat per-segment position ids: [0..len_0-1, 0..len_1-1, ...].
    # `cu_seqlens_cpu`, when supplied, runs the index math host-side to avoid
    # a device sync; the result is shipped back to `cu_seqlens` at the end.
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    _, position_ids = _segmented_arange(prepare_lens(src))
    return position_ids.to(cu_seqlens)


@tensor_cache
def prepare_sequence_ids(cu_seqlens: torch.LongTensor, cu_seqlens_cpu: torch.LongTensor | None = None) -> torch.LongTensor:
    return prepare_position_ids(cu_seqlens, cu_seqlens_cpu).eq(0).cumsum(0) - 1


@tensor_cache
def prepare_token_indices(cu_seqlens: torch.LongTensor, cu_seqlens_cpu: torch.LongTensor | None = None) -> torch.LongTensor:
    position_ids = prepare_position_ids(cu_seqlens, cu_seqlens_cpu)
    return torch.stack([prepare_sequence_ids(cu_seqlens, cu_seqlens_cpu), position_ids], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    # For each segment, emit one (segment_id, intra_chunk_idx) row per chunk.
    # `cu_seqlens_cpu`, when supplied, runs the index math host-side to avoid
    # a device sync; the result is shipped back to `cu_seqlens` at the end.
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    # ceil(len / chunk_size), written explicitly rather than via `triton.cdiv`
    # so the arithmetic is unambiguous under Dynamo tracing.
    chunk_counts = (prepare_lens(src) + (chunk_size - 1)).div(chunk_size, rounding_mode='floor')
    seg_id, intra_chunk_idx = _segmented_arange(chunk_counts)
    return torch.stack([seg_id, intra_chunk_idx], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    return F.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)


@tensor_cache
def get_max_num_splits(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None
) -> int:
    if cu_seqlens_cpu is not None:
        return triton.cdiv(int(max(prepare_lens(cu_seqlens_cpu))), chunk_size)
    return triton.cdiv(int(max(prepare_lens(cu_seqlens))), chunk_size)
