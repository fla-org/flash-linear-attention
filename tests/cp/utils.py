# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from fla.utils import device_torch_lib


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
