# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import logging
import os
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fla.utils import device_platform, device_torch_lib


def gather_global(tensor, world_size, layout='contiguous'):
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
        if destroy_process_group and dist.is_initialized():
            dist.destroy_process_group()


def run_distributed(worker, world_size, *, backend=None, destroy_process_group=True, **kwargs):
    """Run one single-host CP case with an isolated rendezvous and propagate worker failures."""
    if not dist.is_available():
        pytest.skip('requires torch.distributed')
    if device_torch_lib.device_count() < world_size:
        pytest.skip(f'requires {world_size} devices')
    if backend is None:
        backend = 'hccl' if device_platform == 'npu' else 'nccl'
    with TemporaryDirectory(prefix='fla-cp-') as directory:
        init_method = (Path(directory) / 'rendezvous').as_uri()
        mp.spawn(
            fn=_run_worker,
            args=(world_size, init_method, backend, worker, kwargs, destroy_process_group),
            nprocs=world_size,
            join=True,
        )
