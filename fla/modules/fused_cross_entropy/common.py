# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


def _cross_entropy_reduce(
    loss: torch.Tensor,
    lse: torch.Tensor,
    target: torch.Tensor,
    lse_square_scale: float,
    ignore_index: int,
    world_size: int,
    process_group: ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if lse.ndim == 2:
        lse = torch.logsumexp(lse, dim=0)
        loss = loss.sum(dim=0)
    if world_size > 1:
        gathered_lse = lse.new_empty(world_size, lse.shape[0])
        torch.distributed.all_gather_into_tensor(gathered_lse, lse, group=process_group)
        work = torch.distributed.all_reduce(loss, group=process_group, async_op=True)
        lse = torch.logsumexp(gathered_lse, dim=0)
        work.wait()
    loss += lse
    if lse_square_scale != 0:
        z_loss = lse_square_scale * lse.square()
        z_loss.masked_fill_(target == ignore_index, 0.0)
        loss += z_loss
    else:
        z_loss = torch.zeros_like(loss)
    loss.masked_fill_(target == ignore_index, 0.0)
    return loss, z_loss, lse


def _cross_entropy_fwd(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    logit_softcapping: float | None = None,
    ignore_index: int = -100,
    process_group: ProcessGroup | None = None,
    *,
    kernel,
    block_size: int,
    num_warps: int,
    reduce_fn=_cross_entropy_reduce,
):
    N, V = logits.shape
    assert target.shape == (N,)
    world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
    total_classes = world_size * V
    rank = 0 if process_group is None else torch.distributed.get_rank(process_group)
    class_start_idx = rank * V

    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    BV = min(triton.next_power_of_2(V), block_size)
    # vocab partitions contribute partial loss until the global LSE is available
    NV = triton.cdiv(V, BV)
    split = world_size > 1 or NV > 1
    shape = (NV, N) if NV > 1 else (N,)
    loss = logits.new_empty(shape, dtype=torch.float)
    lse = torch.empty_like(loss)
    z_loss = logits.new_empty(N, dtype=torch.float)

    kernel[(N, NV)](
        logits=logits,
        target=target,
        loss=loss,
        lse=lse,
        z_loss=z_loss,
        s_logits=logits.stride(0),
        scale=logit_scale,
        softcap=logit_softcapping,
        smoothing=label_smoothing,
        z_scale=lse_square_scale,
        ignore_index=ignore_index,
        N=N,
        V=V,
        V_TOTAL=total_classes,
        V_START=class_start_idx,
        BV=BV,
        SPLIT=split,
        num_warps=num_warps,
    )

    if split:
        loss, z_loss, lse = reduce_fn(
            loss=loss,
            lse=lse,
            target=target,
            lse_square_scale=lse_square_scale,
            ignore_index=ignore_index,
            world_size=world_size,
            process_group=process_group,
        )

    return loss, z_loss, lse, total_classes, class_start_idx


def _cross_entropy_bwd(
    logits: torch.Tensor,
    target: torch.Tensor,
    lse: torch.Tensor,
    dloss: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    logit_softcapping: float | None = None,
    ignore_index: int = -100,
    total_classes: int | None = None,
    class_start_idx: int = 0,
    inplace_backward: bool = False,
    *,
    kernel,
    block_size: int,
    num_warps: int,
):
    N, V = logits.shape
    BV = min(triton.next_power_of_2(V), block_size)
    dlogits = logits if inplace_backward else torch.empty_like(logits)
    kernel[(N, triton.cdiv(V, BV))](
        logits=logits,
        target=target,
        lse=lse,
        dloss=dloss,
        dlogits=dlogits,
        s_logits=logits.stride(0),
        s_dloss=dloss.stride(0),
        s_dlogits=dlogits.stride(0),
        scale=logit_scale,
        softcap=logit_softcapping,
        smoothing=label_smoothing,
        z_scale=lse_square_scale,
        ignore_index=ignore_index,
        V=V,
        V_TOTAL=V if total_classes is None else total_classes,
        V_START=class_start_idx,
        BV=BV,
        num_warps=num_warps,
    )
    return dlogits
