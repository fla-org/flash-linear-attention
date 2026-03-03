import pytest
import torch
import torch.nn.functional as F

from fla.modules.activations import logsigmoid, sigmoid, swiglu, swish
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'compile'),
    [
        (2, 500, 128, False),
        (2, 512, 128, True),
        (3, 2048, 1200, False),
    ],
)
def test_sigmoid(B: int, T: int, D: int, compile: bool):
    torch.manual_seed(42)
    data = torch.randn(B, T, D * 2, device=device)
    x, _ = data.chunk(2, dim=-1)
    x.requires_grad_()

    y_ref = torch.sigmoid(x)
    y_tri = sigmoid(x) if not compile else torch.compile(sigmoid)(x)

    g = torch.randn_like(y_ref)
    dx_ref, = torch.autograd.grad(y_ref, (x,), g)
    dx_tri, = torch.autograd.grad(y_tri, (x,), g)

    assert_close('sigmoid fwd', y_ref, y_tri, 1e-3)
    assert_close('sigmoid dx', dx_ref, dx_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'compile'),
    [
        (2, 500, 128, False),
        (2, 512, 128, True),
        (3, 2048, 1200, False),
    ],
)
def test_logsigmoid(B: int, T: int, D: int, compile: bool):
    torch.manual_seed(42)
    data = torch.randn(B, T, D * 2, device=device)
    x, _ = data.chunk(2, dim=-1)
    x.requires_grad_()

    y_ref = F.logsigmoid(x)
    y_tri = logsigmoid(x) if not compile else torch.compile(logsigmoid)(x)

    g = torch.randn_like(y_ref)
    dx_ref, = torch.autograd.grad(y_ref, (x,), g)
    dx_tri, = torch.autograd.grad(y_tri, (x,), g)

    assert_close('logsigmoid fwd', y_ref, y_tri, 1e-3)
    assert_close('logsigmoid dx', dx_ref, dx_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'compile'),
    [
        (2, 500, 128, False),
        (2, 512, 128, True),
        (3, 2048, 1200, False),
    ],
)
def test_swish(B: int, T: int, D: int, compile: bool):
    torch.manual_seed(42)
    data = torch.randn(B, T, D * 2, device=device)
    x, _ = data.chunk(2, dim=-1)
    x.requires_grad_()

    y_ref = F.silu(x)
    y_tri = swish(x) if not compile else torch.compile(swish)(x)

    g = torch.randn_like(y_ref)
    dx_ref, = torch.autograd.grad(y_ref, (x,), g)
    dx_tri, = torch.autograd.grad(y_tri, (x,), g)

    assert_close('swish fwd', y_ref, y_tri, 1e-3)
    assert_close('swish dx', dx_ref, dx_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'compile'),
    [
        (2, 500, 128, True),
        (2, 512, 128, False),
        (3, 2048, 1200, False),
    ],
)
def test_swiglu(B: int, T: int, D: int, compile: bool):
    torch.manual_seed(42)

    data = torch.randn(B, T, D * 2, device=device)
    x, y = data.chunk(2, dim=-1)
    x.requires_grad_()
    y.requires_grad_()

    y_ref = F.silu(x) * y
    y_tri = swiglu(x, y) if not compile else torch.compile(swiglu)(x, y)

    g = torch.randn_like(y_ref)
    dx_ref, dy_ref = torch.autograd.grad(y_ref, (x, y), g)
    dx_tri, dy_tri = torch.autograd.grad(y_tri, (x, y), g)

    assert_close('swiglu fwd', y_ref, y_tri, 1e-3)
    assert_close('swiglu dx', dx_ref, dx_tri, 1e-3)
    assert_close('swiglu dy', dy_ref, dy_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'compile'),
    [
        (2, 500, 128, False),
        (2, 512, 128, False),
    ],
)
def test_swiglu_contiguous(B: int, T: int, D: int, compile: bool):
    """Test that contiguous inputs still work correctly."""
    torch.manual_seed(42)

    x = torch.randn(B, T, D, device=device, requires_grad=True)
    y = torch.randn(B, T, D, device=device, requires_grad=True)

    y_ref = F.silu(x) * y
    y_tri = swiglu(x, y)

    g = torch.randn_like(y_ref)
    dx_ref, dy_ref = torch.autograd.grad(y_ref, (x, y), g)
    dx_tri, dy_tri = torch.autograd.grad(y_tri, (x, y), g)

    assert_close('swiglu_cont fwd', y_ref, y_tri, 1e-3)
    assert_close('swiglu_cont dx', dx_ref, dx_tri, 1e-3)
    assert_close('swiglu_cont dy', dy_ref, dy_tri, 1e-3)
