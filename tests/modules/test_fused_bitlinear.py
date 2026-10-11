# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.modules.fused_bitlinear import FusedBitLinear, activation_quant, bit_linear, weight_quant
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'O'),
    [
        pytest.param(*test, id="B{}-T{}-D{}-O{}".format(*test))
        for test in [(1, 1, 64, 32), (2, 33, 63, 16)]
    ],
)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16], ids=['fp32', 'fp16', 'bf16'])
@pytest.mark.parametrize('eps', [None, 1e-6, 1e-4], ids=['default-eps', 'eps1e-6', 'eps1e-4'])
@pytest.mark.parametrize('bias', [False, True], ids=['no-bias', 'bias'])
@pytest.mark.parametrize('fused_module', [False, True], ids=['function', 'module'])
def test_bit_linear(B: int, T: int, D: int, O: int, dtype: torch.dtype, eps: float | None, bias: bool, fused_module: bool):
    torch.manual_seed(42)
    layer = FusedBitLinear(in_features=D, out_features=O, bias=bias).to(device=device, dtype=dtype)
    if eps is not None:
        layer.norm.eps = eps
    # small inputs make the normalization sensitive to epsilon
    x = (torch.randn(B, T, D, device=device, dtype=dtype) * 1e-3).requires_grad_()
    w = layer.norm.weight
    linear_weight, linear_bias = layer.weight, layer.bias
    inputs = (x, w, linear_weight) + ((linear_bias,) if bias else ())

    ref_norm = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + (1e-8 if eps is None else eps))
    ref_norm = ref_norm * w.float()
    ref_quant = ref_norm + (activation_quant(ref_norm) - ref_norm).detach()
    ref_weight = linear_weight + (weight_quant(linear_weight) - linear_weight).detach()
    ref = F.linear(input=ref_quant.to(dtype), weight=ref_weight, bias=linear_bias)

    if fused_module:
        tri = layer(x)
    else:
        kwargs = {} if eps is None else {'eps': eps}
        tri = bit_linear(x=x, weight=linear_weight, bias=linear_bias, norm_weight=w, **kwargs)
    do = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(outputs=ref, inputs=inputs, grad_outputs=do)
    tri_grads = torch.autograd.grad(outputs=tri, inputs=inputs, grad_outputs=do)

    assert torch.isfinite(tri).all()
    assert_close('o', ref, tri, 0.01)
    for name, ref_grad, tri_grad in zip(('dx', 'dw', 'dlw', 'dlb'), ref_grads, tri_grads):
        assert torch.isfinite(tri_grad).all()
        assert_close(name, ref_grad, tri_grad, 0.01)
