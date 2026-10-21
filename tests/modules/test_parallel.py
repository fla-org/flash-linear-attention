# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.parallel import parallelize_module

from fla.modules.parallel import PrepareModuleWeight
from fla.utils import assert_close


@pytest.mark.parametrize(
    ('layout', 'nested', 'frozen'),
    [(None, False, False), (Replicate(), False, False), (Replicate(), True, False), (Replicate(), False, True)],
    ids=['default', 'replicate', 'nested', 'frozen'],
)
@pytest.mark.skipif(not dist.is_gloo_available(), reason='CPU DTensor tests require Gloo')
def test_prepare_module_weight(layout, nested, frozen, tmp_path):
    torch.manual_seed(42)
    dist.init_process_group(
        backend='gloo',
        init_method=f'file://{tmp_path / "store"}',
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh = init_device_mesh('cpu', (1,))
        reference = nn.Linear(3, 4, device='cpu')
        reference.weight.requires_grad_(not frozen)
        if nested:
            reference = nn.Sequential(reference)
        module = parallelize_module(deepcopy(reference), mesh, PrepareModuleWeight(layouts=layout))
        x = torch.randn(2, 3, device='cpu', requires_grad=True)
        local_x = x.detach().clone().requires_grad_()
        ref = reference(x)
        out = module(DTensor.from_local(local_x, mesh, [Replicate()]))
        assert_close('output', ref, out.to_local(), 1e-6)
        ref.sum().backward()
        out.sum().backward()
        assert_close('dx', x.grad, local_x.grad, 1e-6)
        for expected, actual in zip(reference.parameters(), module.parameters()):
            assert actual.placements == (Replicate(),)
            assert actual.requires_grad == expected.requires_grad
            if expected.grad is None:
                assert actual.grad is None
            else:
                assert_close('dparam', expected.grad, actual.grad.to_local(), 1e-6)
    finally:
        dist.destroy_process_group()
