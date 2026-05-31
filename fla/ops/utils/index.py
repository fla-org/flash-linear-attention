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
    """Expand per-segment counts into flat per-slot index tensors.

    Given segment sizes ``counts = [c0, c1, ...]``, return two 1-D tensors of
    length ``counts.sum()`` that together label every slot with its segment and
    its position within that segment.

    Example -- ``counts = [2, 3]`` (segment 0 spans 2 slots, segment 1 spans 3)::

        seg_id    = [0, 0, 1, 1, 1]   # which segment each slot belongs to
        intra_idx = [0, 1, 0, 1, 2]   # running index within that segment

    With CUDA ``counts``, ``repeat_interleave`` reads ``counts.sum()`` on the
    host (one device sync). Pass host-side counts to avoid it.
    """
    seg_id = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device, dtype=counts.dtype),
        counts,
    )
    seg_start = F.pad(counts.cumsum(0), (1, 0))[:-1]
    intra_idx = torch.arange(seg_id.shape[0], device=counts.device, dtype=counts.dtype) - seg_start[seg_id]
    return seg_id, intra_idx


@tensor_cache
def prepare_position_ids(cu_seqlens: torch.LongTensor, cu_seqlens_cpu: torch.LongTensor | None = None) -> torch.LongTensor:
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
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
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
