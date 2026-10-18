# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch.distributed as dist

from fla.utils import device, device_torch_lib


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
