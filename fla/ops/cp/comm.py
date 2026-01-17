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
    async_op: bool = False
) -> tuple[torch.Tensor, dist.Work | None]:
    """
    All-gather a tensor across ranks.

    Args:
        inp: Input tensor to gather
        out: Optional output tensor of shape [world_size, *inp.shape]
        group: Process group
        async_op: Whether to perform async operation

    Returns:
        Tuple of (output tensor, handle if async_op else None)
    """
    world_size = dist.get_world_size(group=group)
    if out is None:
        out = torch.empty(world_size, *inp.shape, device=inp.device, dtype=inp.dtype)
    handle = dist.all_gather_into_tensor(out, inp, group=group, async_op=async_op)
    return out, handle


def all_reduce_sum(
    inp: torch.Tensor,
    group: ProcessGroup | None = None,
    async_op: bool = False
) -> tuple[torch.Tensor, dist.Work | None]:
    """
    All-reduce sum a tensor across ranks.

    Args:
        inp: Input tensor to reduce (modified in-place)
        group: Process group
        async_op: Whether to perform async operation

    Returns:
        Tuple of (reduced tensor, handle if async_op else None)
    """
    handle = dist.all_reduce(inp, op=dist.ReduceOp.SUM, group=group, async_op=async_op)
    return inp, handle


def send_recv_fwd(
    send_tensor: torch.Tensor,
    group: ProcessGroup,
    recv_from_prev: bool = True
) -> torch.Tensor:
    """
    Forward pass communication: send tensor to next rank, receive from previous rank.

    Uses all_gather for simplicity and to ensure all ranks participate.

    Args:
        send_tensor: Tensor to send (e.g., tails for conv1d)
        group: Process group
        recv_from_prev: If True, receive from previous rank; if False, receive from next rank

    Returns:
        Received tensor from the specified rank (zeros if no valid source)
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    # All-gather to ensure all ranks participate
    gathered, _ = all_gather_into_tensor(send_tensor, group=group, async_op=False)

    if recv_from_prev:
        # Receive from previous rank
        if rank == 0:
            return torch.zeros_like(send_tensor)
        else:
            return gathered[rank - 1].clone()
    else:
        # Receive from next rank
        if rank == world_size - 1:
            return torch.zeros_like(send_tensor)
        else:
            return gathered[rank + 1].clone()


def send_recv_bwd(
    send_tensor: torch.Tensor,
    group: ProcessGroup,
    recv_from_next: bool = True
) -> torch.Tensor:
    """
    Backward pass communication: send gradient to previous rank, receive from next rank.

    Uses all_gather for simplicity and to ensure all ranks participate.

    Args:
        send_tensor: Gradient tensor to send
        group: Process group
        recv_from_next: If True, receive from next rank; if False, receive from previous rank

    Returns:
        Received gradient tensor from the specified rank (zeros if no valid source)
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    # All-gather to ensure all ranks participate
    gathered, _ = all_gather_into_tensor(send_tensor, group=group, async_op=False)

    if recv_from_next:
        # Receive from next rank
        if rank == world_size - 1:
            return torch.zeros_like(send_tensor)
        else:
            return gathered[rank + 1].clone()
    else:
        # Receive from previous rank
        if rank == 0:
            return torch.zeros_like(send_tensor)
        else:
            return gathered[rank - 1].clone()


# ============ Convenience aliases for conv1d CP ============

def conv_cp_send_recv_fwd(tails: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    """
    Conv1d CP forward: each rank sends its tails, receives previous rank's tails as heads.

    Args:
        tails: [W-1, D] or [N, D, W-1] - tail tokens from current rank
        group: Process group

    Returns:
        heads: Same shape as tails - head tokens from previous rank (zeros for rank 0)
    """
    return send_recv_fwd(tails, group, recv_from_prev=True)


def conv_cp_send_recv_bwd(d_initial_state: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    """
    Conv1d CP backward: each rank sends d_initial_state, receives from next rank.

    The received gradient should be added to the last W-1 tokens' gradient.

    Args:
        d_initial_state: [W-1, D] or [N, D, W-1] - gradient w.r.t. initial state
        group: Process group

    Returns:
        recv_grad: Same shape - gradient from next rank (zeros for last rank)
    """
    return send_recv_bwd(d_initial_state, group, recv_from_next=True)
