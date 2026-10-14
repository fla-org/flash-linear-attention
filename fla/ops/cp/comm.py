# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


def all_gather_into_tensor(
    inp: torch.Tensor,
    out: torch.Tensor | None = None,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> tuple[torch.Tensor, dist.Work | None]:
    """Gather into [world_size, *inp.shape] and return the output and optional async handle."""
    world_size = dist.get_world_size(group=group)
    if out is None:
        out = torch.empty(world_size, *inp.shape, device=inp.device, dtype=inp.dtype)
    handle = dist.all_gather_into_tensor(out, inp, group=group, async_op=async_op)
    return out, handle


def all_reduce_sum(
    inp: torch.Tensor,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> tuple[torch.Tensor, dist.Work | None]:
    """Sum inp in place across ranks and return it with the optional async handle."""
    handle = dist.all_reduce(inp, op=dist.ReduceOp.SUM, group=group, async_op=async_op)
    return inp, handle


def _gather_neighbor(send_tensor: torch.Tensor, group: ProcessGroup, offset: int) -> torch.Tensor:
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    # every rank must participate even when its neighbor is outside the group
    gathered, _ = all_gather_into_tensor(inp=send_tensor, group=group)
    source = rank + offset
    if source < 0 or source >= world_size:
        return torch.zeros_like(send_tensor)
    return gathered[source].clone()


def send_recv_fwd(
    send_tensor: torch.Tensor,
    group: ProcessGroup,
    recv_from_prev: bool = True,
) -> torch.Tensor:
    """Gather the previous rank's tensor, or the next rank's when recv_from_prev is False.

    All ranks participate in the collective; group boundaries receive zeros.
    """
    return _gather_neighbor(send_tensor=send_tensor, group=group, offset=-1 if recv_from_prev else 1)


def send_recv_bwd(
    send_tensor: torch.Tensor,
    group: ProcessGroup,
    recv_from_next: bool = True,
) -> torch.Tensor:
    """Gather the next rank's gradient, or the previous rank's when recv_from_next is False.

    All ranks participate in the collective; group boundaries receive zeros.
    """
    return _gather_neighbor(send_tensor=send_tensor, group=group, offset=1 if recv_from_next else -1)


def conv_cp_send_recv_fwd(tails: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    """Receive the previous rank's convolution tails, with zeros at the group boundary."""
    return send_recv_fwd(send_tensor=tails, group=group, recv_from_prev=True)


def conv_cp_send_recv_bwd(d_initial_state: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    """Receive the next rank's initial-state gradient to add to local tail gradients."""
    return send_recv_bwd(send_tensor=d_initial_state, group=group, recv_from_next=True)
