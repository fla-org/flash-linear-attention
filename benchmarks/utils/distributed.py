# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Process-group lifecycle and tensor gathering for distributed benchmarks."""

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist


@contextmanager
def distributed_context(enabled: bool = True, backend: str | None = None):
    if not enabled:
        yield
        return
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group(backend=backend)
    try:
        yield
    finally:
        dist.destroy_process_group()


def all_gather(x, group=None) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    y = torch.empty(world_size * x.size(0), *x.shape[1:], device=x.device, dtype=x.dtype)
    dist.all_gather_into_tensor(y, x, group=group)
    return y
