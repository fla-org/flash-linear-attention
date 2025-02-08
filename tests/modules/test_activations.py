import pytest
import torch
import torch.nn.functional as F
from fla.modules.activations import swiglu, swiglu_linear, logsigmoid
from fla.utils import device


@pytest.mark.parametrize("shape", [(16, 32), (32, 64)])
def test_swiglu_forward(shape):
    x = torch.randn(shape, dtype=torch.float32, requires_grad=True).to(device)
    y = torch.randn(shape, dtype=torch.float32, requires_grad=True).to(device)
    torch_output = x * y * torch.sigmoid(x)
    triton_output = swiglu(x, y)

    assert torch.allclose(torch_output, triton_output, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("shape", [(16, 32), (32, 64)])
def test_swiglu_backward(shape):
    x = torch.randn(shape, dtype=torch.float32, requires_grad=True).to(device)
    y = torch.randn(shape, dtype=torch.float32, requires_grad=True).to(device)
    x.retain_grad()
    y.retain_grad()
    torch_output = x * y * torch.sigmoid(x)
    grad_output = torch.randn_like(torch_output)
    torch_output.backward(grad_output, retain_graph=True)
    torch_dx = x.grad.clone()
    torch_dy = y.grad.clone()

    x.grad = None
    y.grad = None

    triton_output = swiglu(x, y)
    triton_output.backward(grad_output)
    triton_dx = x.grad.clone()
    triton_dy = y.grad.clone()

    assert torch.allclose(torch_dx, triton_dx, atol=1e-5, rtol=1e-3)
    assert torch.allclose(torch_dy, triton_dy, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("shape", [(16, 32), (32, 64)])
@pytest.mark.parametrize("out_features", [16, 32])
def test_swiglu_linear_forward(shape, out_features):
    x = torch.randn(shape, dtype=torch.float32, requires_grad=True).to(device)
    y = torch.randn(shape, dtype=torch.float32, requires_grad=True).to(device)
    weight = torch.randn(out_features, shape[-1], dtype=torch.float32, requires_grad=True).to(device)
    bias = torch.randn(out_features, dtype=torch.float32, requires_grad=True).to(device)
    swiglu_output = x * y * torch.sigmoid(x)
    torch_output = F.linear(swiglu_output, weight, bias)
    triton_output = swiglu_linear(x, y, weight, bias)

    assert torch.allclose(torch_output, triton_output, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("shape", [(16, 32), (32, 64)])
@pytest.mark.parametrize("out_features", [16, 32])
def test_swiglu_linear_backward(shape, out_features):
    x = torch.randn(shape, dtype=torch.float32, requires_grad=True).to(device)
    y = torch.randn(shape, dtype=torch.float32, requires_grad=True).to(device)
    weight = torch.randn(out_features, shape[-1], dtype=torch.float32, requires_grad=True).to(device)
    bias = torch.randn(out_features, dtype=torch.float32, requires_grad=True).to(device)
    x.retain_grad()
    y.retain_grad()
    weight.retain_grad()
    bias.retain_grad()

    swiglu_output = x * y * torch.sigmoid(x)
    torch_output = F.linear(swiglu_output, weight, bias)
    grad_output = torch.randn_like(torch_output)

    torch_output.backward(grad_output, retain_graph=True)
    torch_dx = x.grad.clone()
    torch_dy = y.grad.clone()
    torch_dweight = weight.grad.clone()
    torch_dbias = bias.grad.clone()

    x.grad = None
    y.grad = None
    weight.grad = None
    bias.grad = None

    triton_output = swiglu_linear(x, y, weight, bias)
    triton_output.backward(grad_output)
    triton_dx = x.grad.clone()
    triton_dy = y.grad.clone()
    triton_dweight = weight.grad.clone()
    triton_dbias = bias.grad.clone()

    assert torch.allclose(torch_dx, triton_dx, atol=1e-5, rtol=1e-3)
    assert torch.allclose(torch_dy, triton_dy, atol=1e-5, rtol=1e-3)
    assert torch.allclose(torch_dweight, triton_dweight, atol=1e-5, rtol=1e-3)
    assert torch.allclose(torch_dbias, triton_dbias, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("shape", [(16, 32), (32, 64)])
@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_logsigmoid_forward(shape, temperature):
    x = torch.randn(shape, dtype=torch.float32, requires_grad=True).to(device)
    torch_output = F.logsigmoid(x) / temperature
    triton_output = logsigmoid(x, temperature)
    assert torch.allclose(torch_output, triton_output, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("shape", [(16, 32), (32, 64)])
@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_logsigmoid_backward(shape, temperature):
    x = torch.randn(shape, dtype=torch.float32, requires_grad=True).to(device)
    x.retain_grad()
    torch_output = F.logsigmoid(x) / temperature
    torch_output.sum().backward()
    torch_dx = x.grad.clone()
    x.grad = None
    triton_output = logsigmoid(x, temperature)
    triton_output.sum().backward()
    triton_dx = x.grad.clone()

    assert torch.allclose(torch_dx, triton_dx, atol=1e-5, rtol=1e-3)
