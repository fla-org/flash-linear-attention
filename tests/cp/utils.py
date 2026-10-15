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

from fla.ops.cp import build_cp_context
from fla.utils import assert_close, device, device_torch_lib


def distributed_worker(rank, world_size, rendezvous, worker, kwargs, teardown):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    device_torch_lib.set_device(rank)
    dist.init_process_group(
        backend=dist.get_default_backend_for_device(device),
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=10),
    )
    try:
        worker(rank=rank, world_size=world_size, **kwargs)
    finally:
        # graph tests keep captured communicators alive until process exit
        if teardown and dist.is_initialized():
            dist.destroy_process_group()


def slice_local(tensor, rank, world_size, layout='contiguous'):
    parts = tensor.chunk(world_size * (2 if layout == 'zigzag' else 1), dim=1)
    if layout == 'zigzag':
        return torch.cat([parts[rank], parts[2 * world_size - 1 - rank]], dim=1)
    return parts[rank]


def gather_global(tensor, world_size, layout='contiguous'):
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    if layout == 'zigzag':
        part_len = tensor.shape[1] // 2
        # fronts follow rank order; backs follow the reverse order
        gathered = [part[:, :part_len] for part in gathered] + [part[:, part_len:] for part in reversed(gathered)]
    return torch.cat(gathered, dim=1)


def assert_chunk_matches_cp(op, inputs, do, cu_seqlens, rank, world_size, layout='contiguous', ratio=0.006, **kwargs):
    def shard(tensor):
        return slice_local(tensor, rank=rank, world_size=world_size, layout=layout).contiguous()

    expected, actual = [], []
    for cp in [False, True]:
        leaves = {
            name: (shard(tensor) if cp else tensor).detach().clone().requires_grad_()
            for name, tensor in inputs.items()
        }
        context = build_cp_context(cu_seqlens=cu_seqlens, group=dist.group.WORLD, layout=layout) if cp else None
        out, _ = op(
            **leaves,
            cu_seqlens=None if cp else cu_seqlens,
            cp_context=context,
            **kwargs,
        )
        gradients = torch.autograd.grad(outputs=out, inputs=tuple(leaves.values()), grad_outputs=shard(do) if cp else do)
        values = [out.detach(), *gradients]
        if cp:
            actual = values
        else:
            expected = [shard(value) for value in values]
    for name, ref, result in zip(('o', *('d' + name for name in inputs)), expected, actual):
        assert torch.isfinite(result).all(), name
        assert_close(name, ref, result, ratio=ratio)
