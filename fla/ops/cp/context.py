from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


@dataclass
class FLACPContext:
    """FLA Context Parallel Context - Operator-level context management."""
    group: ProcessGroup | None = None
    cu_seqlens: torch.Tensor | None = None
    is_last_rank: bool | None = None
    pre_num_ranks: int | None = None
    is_first_rank: bool | None = None
    post_num_ranks: int | None = None
    kernel_size: int | None = None
    pre_num_conv_tokens: int | None = None

    def copy_for_backward(self) -> FLACPContext:
        """Create a copy for backward pass (useful when PP_SIZE > 1)."""
        return FLACPContext(
            group=self.group,
            cu_seqlens=self.cu_seqlens.clone() if self.cu_seqlens is not None else None,
            is_last_rank=self.is_last_rank,
            pre_num_ranks=self.pre_num_ranks,
            is_first_rank=self.is_first_rank,
            post_num_ranks=self.post_num_ranks,
            kernel_size=self.kernel_size,
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


def build_cp_context(
    cu_seqlens: torch.Tensor,
    group: ProcessGroup,
    kernel_size: int | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> FLACPContext:
    """Build a CP context for the given cu_seqlens and process group.

    Args:
        cu_seqlens: Cumulative sequence lengths tensor (before partition).
        group: Process group for CP communication.
        kernel_size: Kernel size for convolution (optional).
        cu_seqlens_cpu: CPU version of cu_seqlens to avoid d2h transfer (optional).

    Returns:
        FLACPContext with computed cu_seqlens and rank information.
    """
    from fla.ops.cp.chunk_delta_h import get_cp_cu_seqlens
    return get_cp_cu_seqlens(cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, group=group, kernel_size=kernel_size)
