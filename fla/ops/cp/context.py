# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from fla.utils import IS_NVIDIA, get_device_capability, tensor_cache

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


def _validate_cp_kwargs(**kwargs) -> None:
    if kwargs.pop('use_tf32x3_affine_chain', True) is False:
        raise ValueError("use_tf32x3_affine_chain=False is no longer supported; CP requires TF32x3.")
    if kwargs:
        raise TypeError(f"Unexpected CP keyword arguments: {', '.join(kwargs)}")


def _check_cp_device(device: torch.device) -> None:
    if not IS_NVIDIA or device.type != 'cuda':
        raise RuntimeError("CP affine chains require an NVIDIA Ampere or newer GPU (compute capability >= 8.0).")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    if get_device_capability(device_index)[0] < 8:
        raise RuntimeError("CP affine chains require an NVIDIA Ampere or newer GPU (compute capability >= 8.0).")


@dataclass(init=False)
class FLACPContext:
    """Sequence partition and communication metadata for context parallelism.

    Contiguous layouts use scalar rank-boundary metadata. Zigzag layouts use front/back tuples in local-buffer order.
    The optional device rank counts are persistent int32 tensors refreshed in place for CUDA Graph replay.
    """
    group: ProcessGroup | None = None
    cu_seqlens: torch.Tensor | None = None
    cu_seqlens_cpu: torch.Tensor | None = None
    is_last_rank: bool | None = None
    pre_num_ranks: int | None = None
    is_first_rank: bool | None = None
    post_num_ranks: int | None = None
    conv1d_kernel_size: int | None = None
    pre_num_conv_tokens: int | None = None
    # graph capture freezes scalar arguments; callers refresh these device tensors before replay
    pre_num_ranks_dev: torch.Tensor | None = None
    post_num_ranks_dev: torch.Tensor | None = None
    # zigzag rank r holds chain slots r and 2 * world_size - 1 - r; metadata is per part
    layout: str = 'contiguous'
    part_len: int | None = None
    front_num_seqs: int | None = None
    pre_num_ranks_by_part: tuple[int, int] | None = None
    post_num_ranks_by_part: tuple[int, int] | None = None
    is_first_by_part: tuple[bool, bool] | None = None
    is_last_by_part: tuple[bool, bool] | None = None
    pre_num_conv_tokens_by_part: tuple[int, int] | None = None

    def __init__(
        self,
        group: ProcessGroup | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        is_last_rank: bool | None = None,
        pre_num_ranks: int | None = None,
        is_first_rank: bool | None = None,
        post_num_ranks: int | None = None,
        conv1d_kernel_size: int | None = None,
        pre_num_conv_tokens: int | None = None,
        *,
        pre_num_ranks_dev: torch.Tensor | None = None,
        post_num_ranks_dev: torch.Tensor | None = None,
        layout: str = 'contiguous',
        part_len: int | None = None,
        front_num_seqs: int | None = None,
        pre_num_ranks_by_part: tuple[int, int] | None = None,
        post_num_ranks_by_part: tuple[int, int] | None = None,
        is_first_by_part: tuple[bool, bool] | None = None,
        is_last_by_part: tuple[bool, bool] | None = None,
        pre_num_conv_tokens_by_part: tuple[int, int] | None = None,
        **kwargs,
    ):
        _validate_cp_kwargs(**kwargs)
        self.group = group
        self.cu_seqlens = cu_seqlens
        self.cu_seqlens_cpu = cu_seqlens_cpu
        self.is_last_rank = is_last_rank
        self.pre_num_ranks = pre_num_ranks
        self.is_first_rank = is_first_rank
        self.post_num_ranks = post_num_ranks
        self.conv1d_kernel_size = conv1d_kernel_size
        self.pre_num_conv_tokens = pre_num_conv_tokens
        self.pre_num_ranks_dev = pre_num_ranks_dev
        self.post_num_ranks_dev = post_num_ranks_dev
        self.layout = layout
        self.part_len = part_len
        self.front_num_seqs = front_num_seqs
        self.pre_num_ranks_by_part = pre_num_ranks_by_part
        self.post_num_ranks_by_part = post_num_ranks_by_part
        self.is_first_by_part = is_first_by_part
        self.is_last_by_part = is_last_by_part
        self.pre_num_conv_tokens_by_part = pre_num_conv_tokens_by_part

    def copy_for_backward(self) -> FLACPContext:
        """Create a copy for backward pass (useful when PP_SIZE > 1)."""
        return replace(
            self,
            cu_seqlens=self.cu_seqlens.clone() if self.cu_seqlens is not None else None,
            cu_seqlens_cpu=self.cu_seqlens_cpu.clone() if self.cu_seqlens_cpu is not None else None,
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
    """Compute local sequence boundaries and chain metadata for [start, end).

    `chain_pos` identifies a rank in contiguous layouts or a slot in zigzag layouts.
    `part_len` is the token length of one such interval.
    The first and last overlapping sequences determine how many neighboring intervals share their recurrent state.
    """
    start_seq_idx = torch.searchsorted(cu_seqlens_cpu[1:], start, side='right')
    end_seq_idx = torch.searchsorted(cu_seqlens_cpu[:-1], end, side='left')

    # clamping can collapse multiple sequence boundaries
    subset = cu_seqlens_cpu[start_seq_idx: end_seq_idx + 1]
    local = (subset.clamp(min=start, max=end) - start).unique_consecutive().to(torch.int32)

    first_seq_global_start = cu_seqlens_cpu[start_seq_idx].item()
    last_seq_global_end = cu_seqlens_cpu[end_seq_idx].item()

    # convolution needs the preceding tokens from the same sequence
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
    *,
    layout: str = 'contiguous',
    **kwargs,
) -> FLACPContext:
    _validate_cp_kwargs(**kwargs)
    if world_size is None:
        assert group is not None
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
    if layout not in ('contiguous', 'zigzag'):
        raise ValueError(f"`layout` must be 'contiguous' or 'zigzag', got {layout!r}.")

    if cu_seqlens_cpu is None:
        cu_seqlens_cpu = cu_seqlens.cpu()
    cu_seqlens_cpu = cu_seqlens_cpu.to(dtype=torch.long)

    # zigzag doubles the number of chain intervals: rank r holds intervals r and 2W-1-r
    total_tokens = cu_seqlens_cpu[-1].item()
    num_parts = world_size if layout == 'contiguous' else 2 * world_size
    divisor = "`world_size`" if layout == 'contiguous' else "`2 * world_size`"
    if total_tokens < num_parts:
        raise ValueError(f"`total_tokens` ({total_tokens}) must be at least {divisor} ({num_parts}) for context parallelism.")
    if total_tokens % num_parts != 0:
        raise ValueError(
            f"`total_tokens` ({total_tokens}) must be divisible by {divisor} ({num_parts}) for context parallelism.",
        )
    part_len = total_tokens // num_parts

    if layout == 'contiguous':
        local, pre_conv, pre_ranks, is_first, post_ranks, is_last = _interval_cp_meta(
            cu_seqlens_cpu=cu_seqlens_cpu,
            start=part_len * rank,
            end=part_len * (rank + 1),
            chain_pos=rank,
            part_len=part_len,
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
        )

    # zigzag: local buffer = [front part; back part], each part_len tokens
    front_cu, front_conv, front_pre, front_first, front_post, front_last = _interval_cp_meta(
        cu_seqlens_cpu=cu_seqlens_cpu,
        start=part_len * rank,
        end=part_len * (rank + 1),
        chain_pos=rank,
        part_len=part_len,
    )
    back_pos = 2 * world_size - 1 - rank
    back_cu, back_conv, back_pre, back_first, back_post, back_last = _interval_cp_meta(
        cu_seqlens_cpu=cu_seqlens_cpu,
        start=part_len * back_pos,
        end=part_len * (back_pos + 1),
        chain_pos=back_pos,
        part_len=part_len,
    )
    local = torch.cat([front_cu, back_cu[1:] + part_len])
    return FLACPContext(
        group=group,
        cu_seqlens=local.to(device=cu_seqlens.device, non_blocking=True),
        cu_seqlens_cpu=local,
        conv1d_kernel_size=conv1d_kernel_size,
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
    *,
    layout: str = 'contiguous',
    **kwargs,
) -> FLACPContext:
    """Build sequence partitions and rank metadata for a CP process group.

    Args:
        cu_seqlens (torch.Tensor):
            Global cumulative sequence lengths before partitioning.
        group (ProcessGroup):
            Process group for CP communication.
        conv1d_kernel_size (int, Optional):
            Convolution kernel size. Default: `None`.
        cu_seqlens_cpu (torch.Tensor, Optional):
            CPU sequence boundaries to avoid a device-to-host transfer. Default: `None`.
        layout (str, Optional):
            Partition layout. Default: 'contiguous'. Each rank receives one interval in contiguous mode.
            In 'zigzag' mode, rank r receives intervals r and 2 * world_size - 1 - r,
            concatenated as [front; back]. Each interval contains T / (2 * world_size) tokens.
        **kwargs:
            Accepts legacy `use_tf32x3_affine_chain=True`; `False` raises `ValueError`.

    Returns:
        FLACPContext:
            Local sequence boundaries and communication metadata.
    """
    return get_cp_cu_seqlens(
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        group=group,
        conv1d_kernel_size=conv1d_kernel_size,
        layout=layout,
        **kwargs,
    )
