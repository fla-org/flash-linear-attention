# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.gdn2.bound import bound_symmetric_key


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return request.param


def test_bf16_rounding_makes_symmetric_erase_expansive(device):
    k = torch.tensor([1.0, 2.0], dtype=torch.float64, device=device)
    u = (2**0.5 * k / k.norm()).bfloat16()
    assert u.double().square().sum() > 2
    bounded = bound_symmetric_key(u)
    assert bounded.double().square().sum() <= 1.99
    transition = torch.eye(2, dtype=torch.float64, device=device) - torch.outer(bounded.double(), bounded.double())
    assert torch.linalg.matrix_norm(transition, ord=2) <= 1 + 1e-14


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("width", [64, 128, 256])
def test_rounded_bound(dtype, width, device):
    generator = torch.Generator(device=device).manual_seed(42)
    k = torch.randn(512, 2, width, generator=generator, device=device).to(dtype)
    acc = k.to(torch.promote_types(dtype, torch.float32))
    u = (2**0.5 * acc / (acc.square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(dtype)
    bounded = bound_symmetric_key(u)
    assert bounded.dtype == dtype
    assert bounded.shape == u.shape
    assert bounded.double().square().sum(-1).max() <= 1.99
    if dtype == torch.bfloat16:
        assert u.double().square().sum(-1).max() > 2
        norm2 = u.float().square().sum(-1, keepdim=True)
        naive = (u.float() * (norm2 / 1.99).clamp_min(1).rsqrt()).to(dtype)
        assert naive.double().square().sum(-1).max() > 1.99


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_zero_interior_and_tiny_key_gradients(dtype, device):
    u = torch.tensor([[0.0, 0.0], [0.1, -0.2], [1e-15, -1e-15]], dtype=dtype, device=device, requires_grad=True)
    bounded = bound_symmetric_key(u)
    assert torch.equal(bounded, u)
    bounded.float().sum().backward()
    torch.testing.assert_close(u.grad, torch.ones_like(u))


def test_active_cap_gradient(device):
    u = torch.tensor([[1.2, -0.8, 0.4]], dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(bound_symmetric_key, (u,))


def test_fullgraph_forward_and_backward(device):
    u = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.bfloat16, device=device, requires_grad=True)
    compiled = torch.compile(bound_symmetric_key, backend="eager", fullgraph=True)
    bounded = compiled(u)
    torch.testing.assert_close(bounded, bound_symmetric_key(u))
    bounded.float().sum().backward()
    assert torch.isfinite(u.grad).all()


def test_reject_integer_keys():
    with pytest.raises(TypeError, match="Expected"):
        bound_symmetric_key(torch.ones(2, dtype=torch.int64))
