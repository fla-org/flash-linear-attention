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
        )

    @property
    def num_seqs(self) -> int:
        """Number of sequences in this rank."""
        return 0 if self.cu_seqlens is None else len(self.cu_seqlens) - 1

    @property
    def is_cp_enabled(self) -> bool:
        """Whether context parallel is enabled."""
        return self.group is not None


@tensor_cache
def get_cp_cu_seqlens(
    cu_seqlens: torch.LongTensor,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    world_size: int | None = None,
    rank: int | None = None,
    group: dist.ProcessGroup | None = None,
    conv1d_kernel_size: int | None = None
) -> FLACPContext:
    # 1. Initialize environment info
    if world_size is None:
        assert group is not None
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)

    # 2. Operate on CPU to avoid D2H sync and leverage vectorization (int64/long)
    if cu_seqlens_cpu is None:
        cu_seqlens_cpu = cu_seqlens.cpu()
    cu_seqlens_cpu = cu_seqlens_cpu.to(dtype=torch.long)

    # Get total tokens and current rank's responsible range
    # Assume cu_seqlens is [0, s1, s1+s2, ..., total]
    total_tokens = cu_seqlens_cpu[-1].item()
    part_len = total_tokens // world_size
    rank_start = part_len * rank
    rank_end = rank_start + part_len

    # 3. Vectorized search: find sequences overlapping with current rank's interval [rank_start, rank_end)
    # We need to find idx such that: global_ends[idx] > rank_start AND global_starts[idx] < rank_end

    # Optimization: cu_seqlens is sorted, use searchsorted to quickly locate boundaries
    # Find first sequence whose end > rank_start
    # cu_seqlens_cpu[1:] contains all sequence end points
    start_seq_idx = torch.searchsorted(cu_seqlens_cpu[1:], rank_start, side='right')

    # Find first sequence whose start >= rank_end, sequences before this may overlap
    # cu_seqlens_cpu[:-1] contains all sequence start points
    end_seq_idx = torch.searchsorted(cu_seqlens_cpu[:-1], rank_end, side='left')

    # Slice cu_seqlens_cpu[start_seq_idx : end_seq_idx + 1] to get relevant global cu_seqlens nodes
    # +1 because end_seq_idx is an open boundary, and cu_seqlens length is num_seqs + 1
    subset_cu_seqlens = cu_seqlens_cpu[start_seq_idx: end_seq_idx + 1]

    # 4. Compute local cu_seqlens on CPU (int32)
    # Clamp global coordinates to [rank_start, rank_end], subtract rank_start to get local coordinates
    # unique_consecutive removes duplicates from clamping (e.g., sequences entirely outside this rank)
    local_cu_seqlens_cpu = (
        subset_cu_seqlens.clamp(min=rank_start, max=rank_end) - rank_start
    ).unique_consecutive().to(torch.int32)

    # Transfer to GPU (int32, small tensor, fast transfer)
    # non_blocking=True can further hide latency in CUDA streams
    local_cu_seqlens_gpu = local_cu_seqlens_cpu.to(
        device=cu_seqlens.device, non_blocking=True
    )

    # 5. Compute Context Parallel metadata (first/last rank info)
    # Use slice endpoints directly, avoiding loops

    # Get global info for the first sequence that has data on current rank
    first_seq_global_start = cu_seqlens_cpu[start_seq_idx].item()
    # Get global info for the last sequence that has data on current rank
    last_seq_global_end = cu_seqlens_cpu[end_seq_idx].item()

    # Number of tokens current rank needs from previous ranks for conv
    pre_num_conv_tokens = max(0, rank_start - first_seq_global_start)

    # Compute first sequence's starting rank
    first_rank_of_first_seq = first_seq_global_start // part_len
    # Number of previous ranks current rank needs to receive state from
    pre_num_ranks = rank - first_rank_of_first_seq
    # Whether current rank is the first in the sequence's processing chain
    is_first_rank = (rank == first_rank_of_first_seq)

    # Compute last sequence's ending rank
    # (last_seq_global_end - 1) is the index of the last token
    last_rank_of_last_seq = (last_seq_global_end - 1) // part_len
    # Number of subsequent ranks current rank needs to send state to
    post_num_ranks = last_rank_of_last_seq - rank
    # Whether current rank is the last in the sequence's processing chain
    is_last_rank = (rank == last_rank_of_last_seq)

    return FLACPContext(
        group=group,
        cu_seqlens=local_cu_seqlens_gpu,
        cu_seqlens_cpu=local_cu_seqlens_cpu,
        is_last_rank=is_last_rank,
        pre_num_ranks=pre_num_ranks,
        is_first_rank=is_first_rank,
        post_num_ranks=post_num_ranks,
        conv1d_kernel_size=conv1d_kernel_size,
        pre_num_conv_tokens=pre_num_conv_tokens
    )


def build_cp_context(
    cu_seqlens: torch.Tensor,
    group: ProcessGroup,
    conv1d_kernel_size: int | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> FLACPContext:
    """Build a CP context for the given cu_seqlens and process group.

    Args:
        cu_seqlens: Cumulative sequence lengths tensor (before partition).
        group: Process group for CP communication.
        conv1d_kernel_size: Kernel size for convolution (optional).
        cu_seqlens_cpu: CPU version of cu_seqlens to avoid d2h transfer (optional).

    Returns:
        FLACPContext with computed cu_seqlens and rank information.
    """
    return get_cp_cu_seqlens(cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, group=group, conv1d_kernel_size=conv1d_kernel_size)
