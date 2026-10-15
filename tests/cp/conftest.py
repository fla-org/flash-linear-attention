# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch.multiprocessing as mp

from cp.utils import distributed_worker
from fla.utils import device_torch_lib


@pytest.fixture
def run_distributed(tmp_path):
    def run(worker, world_size, teardown=True, **kwargs):
        if device_torch_lib.device_count() < world_size:
            pytest.skip(f'requires {world_size} GPUs')
        rendezvous = (tmp_path / 'rendezvous').as_uri()
        mp.spawn(
            distributed_worker,
            args=(world_size, rendezvous, worker, kwargs, teardown),
            nprocs=world_size,
            join=True,
        )
    return run
