# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Device sequence boundaries for Ascend zigzag graph replay."""

import torch
import triton
import triton.language as tl

from fla.utils import ascend_compile_kwargs


@triton.jit
def zigzag_graph_metadata_kernel(
    cu_seqlens,
    indices,
    windows,
    N: tl.constexpr,
    CU_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    total = tl.load(cu_seqlens + tl.cast(N, tl.int64) * CU_STRIDE).to(tl.int64)
    front_num_seqs = tl.full((), 0, tl.int64)
    num_seqs = tl.full((), 0, tl.int64)
    for block in range(triton.cdiv(N, BLOCK)):
        offsets = tl.cast(block, tl.int64) * BLOCK + tl.arange(0, BLOCK)
        starts = tl.load(cu_seqlens + offsets * CU_STRIDE, offsets < N, other=0).to(tl.int64)
        ends = tl.load(cu_seqlens + (offsets + 1) * CU_STRIDE, offsets < N, other=0).to(tl.int64)
        front_num_seqs += tl.sum(((offsets < N) & (starts < total // 2)).to(tl.int32), 0).to(tl.int64)
        num_seqs += tl.sum(((offsets < N) & (ends > starts)).to(tl.int32), 0).to(tl.int64)

    parts = tl.arange(0, 2)
    first = tl.where(parts == 0, 0, front_num_seqs)
    last = tl.where(parts == 0, front_num_seqs - 1, num_seqs - 1)
    tl.store(indices + parts, first)
    tl.store(indices + 2 + parts, last)

    lanes = tl.arange(0, 8)
    row = lanes // 2
    seq = tl.where(row == 0, 0, tl.where(row == 1, front_num_seqs, tl.where(row == 2, front_num_seqs - 1, num_seqs - 1)))
    # empty metadata has no final segment; do not read before the sequence buffer.
    bounds = tl.load(cu_seqlens + (seq + lanes % 2) * CU_STRIDE, (seq >= 0) & (seq < N), other=0)
    tl.store(windows + lanes, bounds)


def prepare_zigzag_graph_metadata(cu_seqlens: torch.Tensor):
    """Find both parts' first/last sequences and their boundary windows on device."""
    num_seqs = cu_seqlens.numel() - 1
    # four int64 indices precede four two-element windows in the same allocation.
    index_bytes = 4 * 8
    storage = torch.empty(index_bytes + 8 * cu_seqlens.element_size(), device=cu_seqlens.device, dtype=torch.uint8)
    indices = storage[:index_bytes].view(torch.int64)
    windows = storage[index_bytes:].view(cu_seqlens.dtype).view(4, 2)
    # bound the reduction tile so large sequence capacities do not enlarge the live UB footprint.
    block = min(1024, triton.next_power_of_2(num_seqs))
    zigzag_graph_metadata_kernel[(1,)](
        cu_seqlens=cu_seqlens,
        indices=indices,
        windows=windows,
        N=num_seqs,
        CU_STRIDE=cu_seqlens.stride(0),
        BLOCK=block,
        **ascend_compile_kwargs(),
    )
    return indices[:2], indices[2:], (windows[0], windows[1]), (windows[2], windows[3])
