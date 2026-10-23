# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import logging
import os
from collections.abc import Callable
from datetime import timedelta

import torch
import torch.distributed as dist

from fla.ops.cp import build_cp_context
from fla.utils import assert_close, device_torch_lib


def shard_tensor(tensor: torch.Tensor, rank: int, world_size: int, layout: str = 'contiguous') -> torch.Tensor:
    """Select one rank's tokens without changing their dtype or gradient ownership."""
    parts = world_size * (2 if layout == 'zigzag' else 1)
    assert tensor.shape[1] % parts == 0
    part_len = tensor.shape[1] // parts
    front = tensor[:, rank * part_len:(rank + 1) * part_len]
    if layout == 'contiguous':
        return front
    back_start = (parts - 1 - rank) * part_len
    return torch.cat((front, tensor[:, back_start:back_start + part_len]), dim=1)


def gather_global(tensor: torch.Tensor, world_size: int, layout: str = 'contiguous') -> torch.Tensor:
    tensor = tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    if layout == 'zigzag':
        part_len = tensor.shape[1] // 2
        # undo the zigzag permutation: fronts in rank order, then backs reversed
        gathered = [part[:, :part_len] for part in gathered] + [part[:, part_len:] for part in reversed(gathered)]
    return torch.cat(gathered, dim=1)


def _run_worker(rank, world_size, init_method, backend, worker, kwargs, destroy_process_group):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
    device_torch_lib.set_device(rank)
    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10),
        )
        worker(rank=rank, world_size=world_size, **kwargs)
    finally:
        # graph workers keep captured communicators alive until process exit
        if destroy_process_group and dist.is_initialized():
            dist.destroy_process_group()


def assert_chunk_matches_cp(
    op: Callable,
    inputs: dict[str, torch.Tensor],
    do: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rank: int,
    world_size: int,
    layout: str = 'contiguous',
    ratio: float = 0.006,
    **kwargs,
) -> None:
    """Compare CP outputs and every input gradient with the full-sequence chunk operator."""
    def shard(tensor):
        return shard_tensor(tensor=tensor, rank=rank, world_size=world_size, layout=layout).contiguous()

    def run(tensors, do, cp_context=None):
        tensors = {name: tensor.detach().clone().requires_grad_() for name, tensor in tensors.items()}
        out, _ = op(
            **tensors,
            cu_seqlens=cu_seqlens if cp_context is None else None,
            cp_context=cp_context,
            **kwargs,
        )
        gradients = torch.autograd.grad(outputs=out, inputs=tuple(tensors.values()), grad_outputs=do)
        return (out.detach(), *gradients)

    ref = run(tensors=inputs, do=do)
    context = build_cp_context(cu_seqlens=cu_seqlens, group=dist.group.WORLD, layout=layout)
    tri = run(tensors={name: shard(tensor) for name, tensor in inputs.items()}, do=shard(do), cp_context=context)
    names = ('o', *('d' + name for name in inputs))
    for name, expected, actual in zip(names, ref, tri):
        expected = shard(expected)
        assert torch.isfinite(expected).all(), name
        assert torch.isfinite(actual).all(), name
        assert_close(name, expected, actual, ratio=ratio)
