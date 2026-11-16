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
    first_front = tl.full((), N, tl.int64)
    first_back = tl.full((), N, tl.int64)
    last_front = tl.full((), -1, tl.int64)
    last_back = tl.full((), -1, tl.int64)
    for block in range(triton.cdiv(N, BLOCK)):
        base = tl.cast(block, tl.int64) * BLOCK
        lanes = tl.arange(0, BLOCK)
        offsets = base + lanes
        starts = tl.load(cu_seqlens + offsets * CU_STRIDE, offsets < N, other=0).to(tl.int64)
        ends = tl.load(cu_seqlens + (offsets + 1) * CU_STRIDE, offsets < N, other=0).to(tl.int64)
        active = (offsets < N) & (ends > starts)
        front = active & (starts < total // 2)
        back = active & (starts >= total // 2)
        # empty sequences retain their row IDs, so counts cannot locate part boundaries.
        local_first_front = tl.min(tl.where(front, lanes, BLOCK), 0)
        local_first_back = tl.min(tl.where(back, lanes, BLOCK), 0)
        local_last_front = tl.max(tl.where(front, lanes, -1), 0)
        local_last_back = tl.max(tl.where(back, lanes, -1), 0)
        first_front = tl.minimum(first_front, tl.where(local_first_front < BLOCK, base + local_first_front, N))
        first_back = tl.minimum(first_back, tl.where(local_first_back < BLOCK, base + local_first_back, N))
        last_front = tl.maximum(last_front, tl.where(local_last_front >= 0, base + local_last_front, -1))
        last_back = tl.maximum(last_back, tl.where(local_last_back >= 0, base + local_last_back, -1))

    first_front = tl.where(first_front < N, first_front, 0)
    first_back = tl.where(first_back < N, first_back, 0)
    parts = tl.arange(0, 2)
    first = tl.where(parts == 0, first_front, first_back)
    last = tl.where(parts == 0, last_front, last_back)
    tl.store(indices + parts, first)
    tl.store(indices + 2 + parts, last)

    lanes = tl.arange(0, 8)
    row = lanes // 2
    seq = tl.where(row == 0, first_front, tl.where(row == 1, first_back, tl.where(row == 2, last_front, last_back)))
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
