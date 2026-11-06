# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.modules import GroupLinear
from fla.utils import assert_close, device, device_torch_lib


@pytest.mark.parametrize(
    ('shape', 'out_features', 'groups', 'bias', 'strided', 'dtype'),
    [
        ((2, 7), 5, 1, True, False, torch.float32),
        ((2, 3, 12), 20, 4, False, True, torch.float32),
        ((2, 3, 16), 16, 4, True, False, torch.float64),
        ((5,), 10, 5, True, False, torch.float32),
        ((2, 3, 12), 8, 4, True, False, torch.bfloat16),
    ],
    ids=['single-group', 'strided-expansion', 'square', 'vector-depthwise', 'bf16-contraction'],
)
def test_grouplinear(shape, out_features, groups, bias, strided, dtype):
    torch.manual_seed(42)
    test_device = device if device_torch_lib.is_available() else 'cpu'
    layer = GroupLinear(in_features=shape[-1], out_features=out_features, groups=groups, bias=bias)
    layer = layer.to(device=test_device, dtype=dtype)
    if strided:
        x = torch.randn(*shape[:-1], shape[-1] * 2, device=test_device, dtype=dtype)[..., ::2]
    else:
        x = torch.randn(shape, device=test_device, dtype=dtype)
    x.requires_grad_()
    dy = torch.randn(*shape[:-1], out_features, device=test_device, dtype=dtype)

    weight = torch.block_diag(*layer.weight.chunk(groups, dim=0))
    ref = F.linear(x, weight, layer.bias)
    out = layer(x)
    parameters = (x, layer.weight, layer.bias) if bias else (x, layer.weight)
    ref_grads = torch.autograd.grad(ref, parameters, dy)
    out_grads = torch.autograd.grad(out, parameters, dy)

    assert_close('y', ref, out, 0.005)
    for name, ref_grad, out_grad in zip(('dx', 'dw', 'db'), ref_grads, out_grads, strict=False):
        assert_close(name, ref_grad, out_grad, 0.005)


@pytest.mark.parametrize(
    ('in_features', 'out_features', 'groups', 'match'),
    [(8, 8, 0, 'positive'), (7, 8, 2, 'in_features'), (8, 7, 2, 'out_features')],
    ids=['zero-groups', 'input-not-divisible', 'output-not-divisible'],
)
def test_grouplinear_invalid_groups(in_features, out_features, groups, match):
    with pytest.raises(ValueError, match=match):
        GroupLinear(in_features=in_features, out_features=out_features, groups=groups)
