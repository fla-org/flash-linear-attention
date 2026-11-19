# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.layers import PHBA
from fla.models.phba import PHBAConfig, PHBAModel
from fla.modules.activations import swiglu_linear
from fla.utils import IS_NVIDIA, device

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='PHBA CUDA projections')


@pytest.mark.parametrize('bias', [False, True])
def test_attention_projections(bias):
    torch.manual_seed(42)
    layer = PHBA(hidden_size=128, num_heads=4, gsa_num_slots=16, qkv_bias=bias).to(device)
    x = torch.randn(2, 33, 128, device=device, requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    for projection in (layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj, layer.gsa_gate_proj):
        weight = projection.weight.detach().clone().requires_grad_()
        params = [weight]
        actual_params = [projection.weight]
        bias_ref = None
        if projection.bias is not None:
            bias_ref = projection.bias.detach().clone().requires_grad_()
            params.append(bias_ref)
            actual_params.append(projection.bias)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actual = projection(x)
            expected = F.linear(reference, weight, bias_ref)
        dy = torch.randn_like(actual)
        assert torch.equal(actual, expected)
        grads = torch.autograd.grad(actual, [x, *actual_params], dy)
        ref_grads = torch.autograd.grad(expected, [reference, *params], dy)
        for grad, ref in zip(grads, ref_grads):
            assert torch.equal(grad.view(torch.uint8), ref.view(torch.uint8))


def test_mlp_projections():
    torch.manual_seed(42)
    config = PHBAConfig(hidden_size=128, num_heads=4, num_hidden_layers=1, intermediate_size=192, vocab_size=256)
    mlp = PHBAModel(config).layers[0].mlp.to(device)
    weights = [mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight]
    refs = [p.detach().clone().requires_grad_() for p in weights]
    x = torch.randn(2, 33, 128, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actual = mlp(x)
        expected = swiglu_linear(F.linear(y, refs[0]), F.linear(y, refs[1]), refs[2], None)
    dy = torch.randn_like(actual)
    assert torch.equal(actual, expected)
    for grad, ref in zip(torch.autograd.grad(actual, [x, *weights], dy), torch.autograd.grad(expected, [y, *refs], dy)):
        assert torch.equal(grad.view(torch.uint8), ref.view(torch.uint8))
