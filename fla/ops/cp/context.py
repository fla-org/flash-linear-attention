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
    cu_seqlens_conv1d: torch.Tensor | None = None

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
            cu_seqlens_conv1d=self.cu_seqlens_conv1d.clone() if self.cu_seqlens_conv1d is not None else None,
        )

    @property
    def num_seqs(self) -> int:
        """Number of sequences in this rank."""
        return 0 if self.cu_seqlens is None else len(self.cu_seqlens) - 1

    @property
    def is_cp_enabled(self) -> bool:
        """Whether context parallel is enabled."""
        return self.group is not None


# Global context instance
_CP_CONTEXT = FLACPContext()


def set_cp_context(cu_seqlens=None, group=None, kernel_size=None):
    """Set the global CP context."""
    global _CP_CONTEXT
    if group is None:
        _CP_CONTEXT = FLACPContext()
    else:
        from fla.ops.cp.chunk_delta_h import get_cp_cu_seqlens
        _CP_CONTEXT = get_cp_cu_seqlens(cu_seqlens, group=group, kernel_size=kernel_size)


def get_cp_context() -> FLACPContext:
    """Get the global CP context."""
    return _CP_CONTEXT


# Aliases for backward compatibility
set_gdn_cp_context = set_cp_context
get_gdn_cp_context = get_cp_context
