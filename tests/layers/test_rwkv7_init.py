# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn as nn

from fla.layers.rwkv7 import RWKV7Attention


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape', [(16, 8), (8, 16)])
@pytest.mark.parametrize('gain', [1.0, 0.1])
@torch.no_grad()
def test_orthogonal_init(dtype, shape, gain):
    weight = nn.Parameter(torch.zeros(shape, dtype=dtype))
    data_ptr = weight.data_ptr()
    reference = torch.empty(shape, dtype=torch.float32)
    torch.manual_seed(42)
    nn.init.orthogonal_(reference, gain=gain)
    torch.manual_seed(42)
    RWKV7Attention._orthogonal_init(weight, gain=gain)

    assert weight.data_ptr() == data_ptr
    assert weight.dtype == dtype
    torch.testing.assert_close(weight, reference.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('layer_idx', [0, 1])
def test_attention_initialization(dtype, layer_idx):
    torch.manual_seed(42)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        layer = RWKV7Attention(hidden_size=64, head_dim=16, layer_idx=layer_idx, num_hidden_layers=2, fuse_norm=False)
    finally:
        torch.set_default_dtype(original_dtype)

    for projection, gain in [(layer.r_proj, 1.0), (layer.k_proj, 0.1), (layer.v_proj, 1.0)]:
        assert projection.weight.dtype == dtype
        weight = projection.weight.float()
        torch.testing.assert_close(weight @ weight.T, torch.eye(64) * gain**2, rtol=0.01, atol=0.005 * gain**2)

    weights = {name: weight.detach().clone() for name, weight in layer.named_parameters()}
    layer.apply(layer._initialize_weights)
    for name, weight in layer.named_parameters():
        torch.testing.assert_close(weight, weights[name], rtol=0, atol=0)
