# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard
from transformers import AutoModelForCausalLM

from fla.models import RWKV7Config


@pytest.fixture
def make_model():
    """Construct a small model for API tests."""
    def create(config_class, **kwargs):
        if config_class is RWKV7Config and not torch._C.has_lapack:
            pytest.skip('RWKV7 initialization requires LAPACK.')
        config = config_class(**{
            'hidden_size': 64,
            'num_heads': 2,
            'num_hidden_layers': 0,
            'num_self_decoder_layers': 0,
            'intermediate_size': 128,
            'vocab_size': 64,
            'fuse_norm': False,
            'fuse_cross_entropy': False,
            **kwargs,
        })
        return AutoModelForCausalLM.from_config(config)

    return create


@pytest.fixture(scope='session')
def cpu_mesh(tmp_path_factory):
    if not dist.is_gloo_available():
        pytest.skip('CPU DTensor initialization requires Gloo.')
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    dist.init_process_group(
        backend='gloo',
        init_method='env://' if world_size > 1 else f'file://{tmp_path_factory.mktemp("gloo") / "store"}',
        rank=int(os.environ.get('RANK', '0')),
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        yield init_device_mesh('cpu', (world_size,))
    finally:
        dist.destroy_process_group()


@pytest.fixture(params=[None, Replicate(), Shard(0)], ids=['tensor', 'replicate', 'shard'])
def init_layout(request):
    if request.param is None:
        return None
    return request.getfixturevalue('cpu_mesh'), (request.param,)
