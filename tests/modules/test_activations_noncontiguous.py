import pytest
import torch
import torch.nn.functional as F

from fla.modules.activations import _is_inner_contiguous, logsigmoid, sigmoid, swiglu, swish
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
    ('B', 'T', 'D'),
    [
        (2, 500, 128),
        (2, 512, 128),
    ],
)
def test_swiglu_contiguous(B: int, T: int, D: int):
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


def test_is_inner_contiguous():
    """Test _is_inner_contiguous correctly classifies tensor layouts."""
    # 0D and 1D should always be True (no inner dimensions to check)
    assert _is_inner_contiguous(torch.randn(())) is True
    assert _is_inner_contiguous(torch.randn(10)) is True

    # 2D contiguous - should be True
    x2d = torch.randn(10, 20)
    assert _is_inner_contiguous(x2d) is True

    # 2D with non-unit stride in last dim - should be False
    x2d_transposed = x2d.t()
    assert _is_inner_contiguous(x2d_transposed) is False

    # 2D with strided last dim - should be False (stride(-1) != 1)
    x2d_strided = x2d[:, ::2]  # shape (10, 10), stride (20, 2)
    assert _is_inner_contiguous(x2d_strided) is False

    # 3D contiguous - should be True
    x3d = torch.randn(5, 10, 20)
    assert _is_inner_contiguous(x3d) is True

    # 3D inner-contiguous via chunk (simulates real use case)
    data3d = torch.randn(5, 10, 40)
    x3d_chunk, _ = data3d.chunk(2, dim=-1)  # shape (5, 10, 20), stride (400, 40, 1)
    assert _is_inner_contiguous(x3d_chunk) is True

    # 3D transposed - should be False
    x3d_transposed = x3d.transpose(-1, -2)  # stride not contiguous
    assert _is_inner_contiguous(x3d_transposed) is False

    # 4D contiguous - should be True
    x4d = torch.randn(3, 5, 10, 20)
    assert _is_inner_contiguous(x4d) is True

    # 4D inner-contiguous via chunk
    data4d = torch.randn(3, 5, 10, 40)
    x4d_chunk, _ = data4d.chunk(2, dim=-1)  # shape (3, 5, 10, 20)
    assert _is_inner_contiguous(x4d_chunk) is True

    # 5D contiguous - should be True
    x5d = torch.randn(2, 3, 5, 10, 20)
    assert _is_inner_contiguous(x5d) is True

    # Outer dim strided - NOT inner-contiguous because 2D view offset is wrong
    x3d_outer_nc = x3d[::2]  # shape (3, 10, 20), stride (400, 20, 1)
    assert _is_inner_contiguous(x3d_outer_nc) is False

    # Regression test: previously buggy 3D check
    # Standard 3D (B, T, D) should be inner-contiguous
    x_btd = torch.randn(2, 500, 128)
    assert _is_inner_contiguous(x_btd) is True

    # Regression test: previously buggy 4D check
    x_bhtd = torch.randn(2, 8, 500, 128)
    assert _is_inner_contiguous(x_bhtd) is True
