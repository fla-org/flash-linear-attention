# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from fla.utils import device, device_platform, device_torch_lib
from tests.cp.utils import _run_worker


@pytest.fixture
def run_distributed(tmp_path):
    """Give each launch its own rendezvous and propagate worker failures to pytest."""
    def run(worker, world_size, *, backend=None, destroy_process_group=True, **kwargs):
        if not dist.is_available():
            pytest.skip('requires torch.distributed')
        if device_torch_lib.device_count() < world_size:
            pytest.skip(f'requires {world_size} devices')
        if backend is None:
            backend = 'hccl' if device_platform == 'npu' else 'nccl'
        with TemporaryDirectory(prefix='fla-cp-', dir=tmp_path) as directory:
            init_method = (Path(directory) / 'rendezvous').as_uri()
            mp.spawn(
                fn=_run_worker,
                args=(world_size, init_method, backend, worker, kwargs, destroy_process_group),
                nprocs=world_size,
                join=True,
            )
    return run


@pytest.fixture
def single_rank_group(tmp_path):
    if not dist.is_available() or not device_torch_lib.is_available():
        pytest.skip('requires a distributed accelerator backend')
    device_torch_lib.set_device(0)
    dist.init_process_group(
        backend=dist.get_default_backend_for_device(device),
        init_method=(tmp_path / 'rendezvous').as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()
