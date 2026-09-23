# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from fla.utils import tensor_cache

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


@dataclass
class FLACPContext:
    """FLA Context Parallel Context - Operator-level context management."""
    group: ProcessGroup | None = None
    cu_seqlens: torch.Tensor | None = None
    cu_seqlens_cpu: torch.Tensor | None = None
    is_last_rank: bool | None = None
    pre_num_ranks: int | None = None
    is_first_rank: bool | None = None
    post_num_ranks: int | None = None
    conv1d_kernel_size: int | None = None
    pre_num_conv_tokens: int | None = None
    use_tf32x3_affine_chain: bool = False
    # Device-resident copies of pre/post_num_ranks for graph replay: kernel scalar
    # arguments are frozen at capture, so graph mode reads them via tl.load instead.
    # Persistent int32 tensors the caller refreshes in place before each replay.
    pre_num_ranks_dev: torch.Tensor | None = None
    post_num_ranks_dev: torch.Tensor | None = None
    # zigzag layout: rank r holds chain slots r (front part) and 2 * world_size - 1 - r
    # (back part); the fields below are per part, the scalar fields above contiguous only.
    layout: str = 'contiguous'
    part_len: int | None = None
    front_num_seqs: int | None = None
    pre_num_ranks_by_part: tuple[int, int] | None = None
    post_num_ranks_by_part: tuple[int, int] | None = None
    is_first_by_part: tuple[bool, bool] | None = None
    is_last_by_part: tuple[bool, bool] | None = None
    pre_num_conv_tokens_by_part: tuple[int, int] | None = None

    def copy_for_backward(self) -> FLACPContext:
        """Create a copy for backward pass (useful when PP_SIZE > 1)."""
        return FLACPContext(
            group=self.group,
            cu_seqlens=self.cu_seqlens.clone() if self.cu_seqlens is not None else None,
            cu_seqlens_cpu=self.cu_seqlens_cpu.clone() if self.cu_seqlens_cpu is not None else None,
            is_last_rank=self.is_last_rank,
            pre_num_ranks=self.pre_num_ranks,
            is_first_rank=self.is_first_rank,
            post_num_ranks=self.post_num_ranks,
            conv1d_kernel_size=self.conv1d_kernel_size,
            pre_num_conv_tokens=self.pre_num_conv_tokens,
            use_tf32x3_affine_chain=self.use_tf32x3_affine_chain,
            pre_num_ranks_dev=self.pre_num_ranks_dev,
            post_num_ranks_dev=self.post_num_ranks_dev,
            layout=self.layout,
            part_len=self.part_len,
            front_num_seqs=self.front_num_seqs,
            pre_num_ranks_by_part=self.pre_num_ranks_by_part,
            post_num_ranks_by_part=self.post_num_ranks_by_part,
            is_first_by_part=self.is_first_by_part,
            is_last_by_part=self.is_last_by_part,
            pre_num_conv_tokens_by_part=self.pre_num_conv_tokens_by_part,
        )

    @property
    def num_seqs(self) -> int:
        """Number of sequences in this rank."""
        return 0 if self.cu_seqlens is None else len(self.cu_seqlens) - 1

    @property
    def is_cp_enabled(self) -> bool:
        """Whether context parallel is enabled."""
        return self.group is not None


def _interval_cp_meta(
    cu_seqlens_cpu: torch.LongTensor,
    start: int,
    end: int,
    chain_pos: int,
    part_len: int,
) -> tuple[torch.Tensor, int, int, bool, int, bool]:
    """Local cu_seqlens and chain metadata for one token interval [start, end).

    `chain_pos` is the interval's position in the global processing chain and
    `part_len` the chain's interval size; both reduce to (rank, rank length)
    for the contiguous layout and to (slot, total/(2W)) for zigzag.
    """
    # Find sequences overlapping with [start, end)
    start_seq_idx = torch.searchsorted(cu_seqlens_cpu[1:], start, side='right')
    end_seq_idx = torch.searchsorted(cu_seqlens_cpu[:-1], end, side='left')

    # Clamp global coordinates to [start, end] and shift to local coordinates;
    # unique_consecutive removes duplicates from clamping
    subset = cu_seqlens_cpu[start_seq_idx: end_seq_idx + 1]
    local = (subset.clamp(min=start, max=end) - start).unique_consecutive().to(torch.int32)

    first_seq_global_start = cu_seqlens_cpu[start_seq_idx].item()
    last_seq_global_end = cu_seqlens_cpu[end_seq_idx].item()

    # Tokens the interval's first sequence extends before the interval (conv tail depth)
    pre_num_conv_tokens = max(0, start - first_seq_global_start)

    first_pos = first_seq_global_start // part_len
    # (last_seq_global_end - 1) is the index of the last token
    last_pos = (last_seq_global_end - 1) // part_len
    return (
        local,
        pre_num_conv_tokens,
        chain_pos - first_pos,
        chain_pos == first_pos,
        last_pos - chain_pos,
        last_pos == chain_pos,
    )


@tensor_cache
def get_cp_cu_seqlens(
    cu_seqlens: torch.LongTensor,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    world_size: int | None = None,
    rank: int | None = None,
    group: dist.ProcessGroup | None = None,
    conv1d_kernel_size: int | None = None,
    use_tf32x3_affine_chain: bool = False,
    layout: str = 'contiguous',
) -> FLACPContext:
    # 1. Initialize environment info
    if world_size is None:
        assert group is not None
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
    if layout not in ('contiguous', 'zigzag'):
        raise ValueError(f"`layout` must be 'contiguous' or 'zigzag', got {layout!r}.")

    # 2. Operate on CPU to avoid D2H sync and leverage vectorization (int64/long)
    if cu_seqlens_cpu is None:
        cu_seqlens_cpu = cu_seqlens.cpu()
    cu_seqlens_cpu = cu_seqlens_cpu.to(dtype=torch.long)

    # Get total tokens and the chain interval size. Assume cu_seqlens is [0, s1, s1+s2, ..., total]
    # zigzag doubles the number of chain intervals: rank r holds intervals r and 2W-1-r
    total_tokens = cu_seqlens_cpu[-1].item()
    num_parts = world_size if layout == 'contiguous' else 2 * world_size
    divisor = "`world_size`" if layout == 'contiguous' else "`2 * world_size`"
    if total_tokens < num_parts:
        raise ValueError(
            f"`total_tokens` ({total_tokens}) must be at least {divisor} ({num_parts}) for context parallelism."
        )
    if total_tokens % num_parts != 0:
        raise ValueError(
            f"`total_tokens` ({total_tokens}) must be divisible by {divisor} ({num_parts}) for context parallelism."
        )
    part_len = total_tokens // num_parts

    if layout == 'contiguous':
        local, pre_conv, pre_ranks, is_first, post_ranks, is_last = _interval_cp_meta(
            cu_seqlens_cpu, part_len * rank, part_len * (rank + 1), rank, part_len,
        )
        return FLACPContext(
            group=group,
            cu_seqlens=local.to(device=cu_seqlens.device, non_blocking=True),
            cu_seqlens_cpu=local,
            is_last_rank=is_last,
            pre_num_ranks=pre_ranks,
            is_first_rank=is_first,
            post_num_ranks=post_ranks,
            conv1d_kernel_size=conv1d_kernel_size,
            pre_num_conv_tokens=pre_conv,
            use_tf32x3_affine_chain=use_tf32x3_affine_chain,
        )

    # zigzag: local buffer = [front part; back part], each part_len tokens
    front_cu, front_conv, front_pre, front_first, front_post, front_last = _interval_cp_meta(
        cu_seqlens_cpu, part_len * rank, part_len * (rank + 1), rank, part_len,
    )
    back_pos = 2 * world_size - 1 - rank
    back_cu, back_conv, back_pre, back_first, back_post, back_last = _interval_cp_meta(
        cu_seqlens_cpu, part_len * back_pos, part_len * (back_pos + 1), back_pos, part_len,
    )
    local = torch.cat([front_cu, back_cu[1:] + part_len])
    return FLACPContext(
        group=group,
        cu_seqlens=local.to(device=cu_seqlens.device, non_blocking=True),
        cu_seqlens_cpu=local,
        conv1d_kernel_size=conv1d_kernel_size,
        use_tf32x3_affine_chain=use_tf32x3_affine_chain,
        layout='zigzag',
        part_len=part_len,
        front_num_seqs=len(front_cu) - 1,
        pre_num_ranks_by_part=(front_pre, back_pre),
        post_num_ranks_by_part=(front_post, back_post),
        is_first_by_part=(front_first, back_first),
        is_last_by_part=(front_last, back_last),
        pre_num_conv_tokens_by_part=(front_conv, back_conv),
    )


def build_cp_context(
    cu_seqlens: torch.Tensor,
    group: ProcessGroup,
    conv1d_kernel_size: int | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    use_tf32x3_affine_chain: bool = False,
    layout: str = 'contiguous',
) -> FLACPContext:
    """Build a CP context for the given cu_seqlens and process group.

    Args:
        cu_seqlens: Cumulative sequence lengths tensor (before partition).
        group: Process group for CP communication.
        conv1d_kernel_size: Kernel size for convolution (optional).
        cu_seqlens_cpu: CPU version of cu_seqlens to avoid d2h transfer (optional).
        use_tf32x3_affine_chain: Use tf32x3 for the affine-chain dots in the CP pre-process and merge kernels (NVIDIA only).
        layout: 'contiguous' (default) assigns rank r the token interval
            [r * T/W, (r + 1) * T/W); 'zigzag' assigns rank r two intervals of
            T/(2W) tokens, [r * T/2W, (r + 1) * T/2W) and
            [(2W - 1 - r) * T/2W, (2W - r) * T/2W), and the caller feeds the
            concatenation [front; back] as the local input. Zigzag balances
            causal-attention CP layouts without a re-shard at layer boundaries.

    Returns:
        FLACPContext with computed cu_seqlens and rank information.
    """
    return get_cp_cu_seqlens(
        cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, group=group,
        conv1d_kernel_size=conv1d_kernel_size, use_tf32x3_affine_chain=use_tf32x3_affine_chain,
        layout=layout,
    )
