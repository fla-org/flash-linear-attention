# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.modules.activations import (
    _is_inner_contiguous,
    logsigmoid,
    powglu,
    powglu_linear,
    sigmoid,
    sigmoidglu,
    sigmoidglu_linear,
    sqrelu,
    swiglu,
    swiglu_linear,
    swish,
)
from fla.modules.activations import fast_gelu_impl as gelu
from fla.utils import assert_close, device


def powglu_ref(x: torch.Tensor, y: torch.Tensor, power: float = 3.0) -> torch.Tensor:
    s = torch.sigmoid(x)
    # guard log/sqrt to the x>0 branch so autograd through the masked lanes stays finite
    xpos = torch.where(x > 0, x, torch.ones_like(x))
    g = torch.where(x > 0, xpos ** (power / (xpos.sqrt() + 1)) * s, x * s)
    return g * y


def make_inputs(B: int, T: int, D: int, n: int, noncontiguous: bool) -> tuple[torch.Tensor, ...]:
    """Return ``n`` grad-requiring (B, T, D) tensors; non-contiguous ones are halves of a chunked buffer."""
    if noncontiguous:
        xs = torch.randn(B, T, D * 2, device=device).chunk(2, dim=-1)[:n]
    else:
        xs = tuple(torch.randn(B, T, D, device=device) for _ in range(n))
    return tuple(x.requires_grad_() for x in xs)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'noncontiguous', 'compile'),
    [
        (1, 1, 64, False, False),
        (2, 500, 128, False, False),
        (2, 512, 128, False, True),
        (3, 2048, 1200, False, True),
        (2, 500, 128, True, False),
        (2, 512, 128, True, True),
        (3, 2048, 1200, True, False),
    ],
)
def test_sigmoid(B: int, T: int, D: int, noncontiguous: bool, compile: bool):
    torch.manual_seed(42)
    (x,) = make_inputs(B, T, D, 1, noncontiguous)
    y_ref = torch.sigmoid(x)
    y_tri = sigmoid(x) if not compile else torch.compile(sigmoid)(x)

    g = torch.randn_like(y_ref)
    dx_ref, = torch.autograd.grad(y_ref, (x,), g)
    dx_tri, = torch.autograd.grad(y_tri, (x,), g)

    assert_close('sigmoid fwd', y_ref, y_tri, 1e-3)
    assert_close('sigmoid dx ', dx_ref, dx_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'temperature', 'noncontiguous', 'compile'),
    [
        (1, 1, 64, 1.0, False, False),
        (2, 500, 128, 0.5, False, False),
        (2, 512, 128, 0.5, False, True),
        (3, 2048, 1200, 2.0, False, True),
        (2, 500, 128, 0.5, True, False),
        (2, 512, 128, 1.0, True, True),
        (3, 2048, 1200, 2.0, True, False),
    ],
)
def test_logsigmoid(B: int, T: int, D: int, temperature: float, noncontiguous: bool, compile: bool):
    torch.manual_seed(42)
    (x,) = make_inputs(B, T, D, 1, noncontiguous)
    y_ref = F.logsigmoid(x) / temperature
    y_tri = logsigmoid(x, temperature) if not compile else torch.compile(logsigmoid)(x, temperature)

    g = torch.randn_like(y_ref)
    dx_ref, = torch.autograd.grad(y_ref, (x,), g)
    dx_tri, = torch.autograd.grad(y_tri, (x,), g)

    assert_close('logsigmoid fwd', y_ref, y_tri, 1e-3)
    assert_close('logsigmoid dx ', dx_ref, dx_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'noncontiguous', 'compile'),
    [
        (1, 1, 64, False, True),
        (2, 500, 128, False, True),
        (2, 512, 128, False, False),
        (3, 2048, 1200, False, False),
        (2, 500, 128, True, False),
        (2, 512, 128, True, True),
        (3, 2048, 1200, True, False),
    ],
)
def test_swish(B: int, T: int, D: int, noncontiguous: bool, compile: bool):
    torch.manual_seed(42)
    (x,) = make_inputs(B, T, D, 1, noncontiguous)
    y_ref = F.silu(x)
    y_tri = swish(x) if not compile else torch.compile(swish)(x)

    g = torch.randn_like(y_ref)
    dx_ref, = torch.autograd.grad(y_ref, (x,), g)
    dx_tri, = torch.autograd.grad(y_tri, (x,), g)

    assert_close('swish fwd', y_ref, y_tri, 1e-3)
    assert_close('swish dx ', dx_ref, dx_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'noncontiguous', 'compile'),
    [
        (1, 1, 64, False, True),
        (2, 500, 128, False, True),
        (2, 512, 128, False, False),
        (3, 2048, 1200, False, False),
        (2, 500, 128, True, False),
        (2, 512, 128, True, True),
        (3, 2048, 1200, True, False),
    ],
)
def test_gelu(B: int, T: int, D: int, noncontiguous: bool, compile: bool):
    torch.manual_seed(42)
    (x,) = make_inputs(B, T, D, 1, noncontiguous)
    y_ref = F.gelu(x, approximate='tanh')
    y_tri = gelu(x) if not compile else torch.compile(gelu)(x)

    g = torch.randn_like(y_ref)
    dx_ref, = torch.autograd.grad(y_ref, (x,), g)
    dx_tri, = torch.autograd.grad(y_tri, (x,), g)

    assert_close('gelu fwd', y_ref, y_tri, 1e-3)
    assert_close('gelu dx ', dx_ref, dx_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'noncontiguous', 'compile'),
    [
        (1, 1, 64, False, True),
        (2, 500, 128, False, True),
        (2, 512, 128, False, False),
        (3, 2048, 1200, False, False),
        (2, 500, 128, True, False),
        (2, 512, 128, True, True),
        (3, 2048, 1200, True, False),
    ],
)
def test_sqrelu(B: int, T: int, D: int, noncontiguous: bool, compile: bool):
    torch.manual_seed(42)
    (x,) = make_inputs(B, T, D, 1, noncontiguous)
    y_ref = F.relu(x) ** 2
    y_tri = sqrelu(x) if not compile else torch.compile(sqrelu)(x)

    g = torch.randn_like(y_ref)
    dx_ref, = torch.autograd.grad(y_ref, (x,), g)
    dx_tri, = torch.autograd.grad(y_tri, (x,), g)

    assert_close('sqrelu fwd', y_ref, y_tri, 1e-3)
    assert_close('sqrelu dx ', dx_ref, dx_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'noncontiguous', 'compile'),
    [
        (1, 1, 64, False, True),
        (2, 500, 128, False, True),
        (2, 512, 128, False, False),
        (3, 2048, 1200, False, False),
        (2, 500, 128, True, True),
        (2, 512, 128, True, False),
        (3, 2048, 1200, True, False),
    ],
)
def test_swiglu(B: int, T: int, D: int, noncontiguous: bool, compile: bool):
    torch.manual_seed(42)
    x, y = make_inputs(B, T, D, 2, noncontiguous)

    y_ref = F.silu(x) * y
    y_tri = swiglu(x, y) if not compile else torch.compile(swiglu)(x, y)

    g = torch.randn_like(y_ref)
    dx_ref, dy_ref = torch.autograd.grad(y_ref, (x, y), g)
    dx_tri, dy_tri = torch.autograd.grad(y_tri, (x, y), g)

    assert_close('swiglu fwd', y_ref, y_tri, 1e-3)
    assert_close('swiglu dx ',  dx_ref, dx_tri, 1e-3)
    assert_close('swiglu dy ',  dy_ref, dy_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'O', 'noncontiguous', 'compile'),
    [
        (2, 512, 128, 256, False, True),
        (1, 1, 64, 32, False, False),
        (2, 500, 128, 64, False, True),
        (3, 2048, 1200, 600, False, False),
        (2, 500, 128, 64, True, False),
        (2, 512, 128, 256, True, True),
        (3, 2048, 1200, 600, True, False),
    ],
)
def test_swiglu_linear(B: int, T: int, D: int, O: int, noncontiguous: bool, compile: bool):  # noqa: E741
    torch.manual_seed(42)
    x, y = make_inputs(B, T, D, 2, noncontiguous)
    w = torch.randn(O, D, device=device, requires_grad=True)
    b = torch.randn(O, device=device, requires_grad=True)

    z_ref = F.silu(x) * y
    out_ref = F.linear(z_ref, w, b)
    out_tri = swiglu_linear(x, y, w, b) if not compile else torch.compile(swiglu_linear)(x, y, w, b)

    g = torch.randn_like(out_ref)
    dx_ref, dy_ref, dw_ref, db_ref = torch.autograd.grad(out_ref, (x, y, w, b), g)
    dx_tri, dy_tri, dw_tri, db_tri = torch.autograd.grad(out_tri, (x, y, w, b), g)

    assert_close('swiglu_linear out', out_ref, out_tri, 1e-3)
    assert_close('swiglu_linear dx ',  dx_ref,  dx_tri,  1e-3)
    assert_close('swiglu_linear dy ',  dy_ref,  dy_tri,  1e-3)
    assert_close('swiglu_linear dw ',  dw_ref,  dw_tri,  1e-3)
    assert_close('swiglu_linear db ',  db_ref,  db_tri,  1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'noncontiguous', 'compile'),
    [
        (1, 1, 64, False, True),
        (2, 500, 128, False, True),
        (2, 512, 128, False, False),
        (3, 2048, 1200, False, False),
        (2, 500, 128, True, True),
        (2, 512, 128, True, False),
        (3, 2048, 1200, True, False),
    ],
)
def test_sigmoidglu(B: int, T: int, D: int, noncontiguous: bool, compile: bool):
    torch.manual_seed(42)
    x, y = make_inputs(B, T, D, 2, noncontiguous)

    y_ref = torch.sigmoid(x) * y
    y_tri = sigmoidglu(x, y) if not compile else torch.compile(sigmoidglu)(x, y)

    g = torch.randn_like(y_ref)
    dx_ref, dy_ref = torch.autograd.grad(y_ref, (x, y), g)
    dx_tri, dy_tri = torch.autograd.grad(y_tri, (x, y), g)

    assert_close('sigmoidglu fwd', y_ref, y_tri, 1e-3)
    assert_close('sigmoidglu dx ',  dx_ref, dx_tri, 1e-3)
    assert_close('sigmoidglu dy ',  dy_ref, dy_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'O', 'noncontiguous', 'compile'),
    [
        (2, 512, 128, 256, False, True),
        (1, 1, 64, 32, False, False),
        (2, 500, 128, 64, False, True),
        (3, 2048, 1200, 600, False, False),
        (2, 500, 128, 64, True, False),
        (2, 512, 128, 256, True, True),
        (3, 2048, 1200, 600, True, False),
    ],
)
def test_sigmoidglu_linear(B: int, T: int, D: int, O: int, noncontiguous: bool, compile: bool):  # noqa: E741
    torch.manual_seed(42)
    x, y = make_inputs(B, T, D, 2, noncontiguous)
    w = torch.randn(O, D, device=device, requires_grad=True)
    b = torch.randn(O, device=device, requires_grad=True)

    z_ref = torch.sigmoid(x) * y
    out_ref = F.linear(z_ref, w, b)
    out_tri = sigmoidglu_linear(x, y, w, b) if not compile else torch.compile(sigmoidglu_linear)(x, y, w, b)

    g = torch.randn_like(out_ref)
    dx_ref, dy_ref, dw_ref, db_ref = torch.autograd.grad(out_ref, (x, y, w, b), g)
    dx_tri, dy_tri, dw_tri, db_tri = torch.autograd.grad(out_tri, (x, y, w, b), g)

    assert_close('sigmoidglu_linear out', out_ref, out_tri, 1e-3)
    assert_close('sigmoidglu_linear dx ',  dx_ref,  dx_tri,  1e-3)
    assert_close('sigmoidglu_linear dy ',  dy_ref,  dy_tri,  1e-3)
    assert_close('sigmoidglu_linear dw ',  dw_ref,  dw_tri,  1e-3)
    assert_close('sigmoidglu_linear db ',  db_ref,  db_tri,  1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'power', 'noncontiguous', 'compile'),
    [
        (1, 1, 64, 3.0, False, True),
        (2, 500, 128, 2.0, False, True),
        (2, 512, 128, 3.0, False, False),
        (3, 2048, 1200, 4.0, False, False),
        (2, 500, 128, 3.0, True, True),
        (2, 512, 128, 2.0, True, False),
        (3, 2048, 1200, 3.0, True, False),
    ],
)
def test_powglu(B: int, T: int, D: int, power: float, noncontiguous: bool, compile: bool):
    torch.manual_seed(42)
    x, y = make_inputs(B, T, D, 2, noncontiguous)

    y_ref = powglu_ref(x, y, power)
    y_tri = powglu(x, y, power) if not compile else torch.compile(powglu)(x, y, power)

    g = torch.randn_like(y_ref)
    dx_ref, dy_ref = torch.autograd.grad(y_ref, (x, y), g)
    dx_tri, dy_tri = torch.autograd.grad(y_tri, (x, y), g)

    assert_close('powglu fwd', y_ref, y_tri, 1e-3)
    assert_close('powglu dx ',  dx_ref, dx_tri, 1e-3)
    assert_close('powglu dy ',  dy_ref, dy_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'O', 'power', 'noncontiguous', 'compile'),
    [
        (2, 512, 128, 256, 3.0, False, True),
        (1, 1, 64, 32, 2.0, False, False),
        (2, 500, 128, 64, 3.0, False, True),
        (3, 2048, 1200, 600, 4.0, False, False),
        (2, 500, 128, 64, 3.0, True, False),
        (2, 512, 128, 256, 2.0, True, True),
        (3, 2048, 1200, 600, 3.0, True, False),
    ],
)
def test_powglu_linear(B: int, T: int, D: int, O: int, power: float, noncontiguous: bool, compile: bool):  # noqa: E741
    torch.manual_seed(42)
    x, y = make_inputs(B, T, D, 2, noncontiguous)
    w = torch.randn(O, D, device=device, requires_grad=True)
    b = torch.randn(O, device=device, requires_grad=True)

    z_ref = powglu_ref(x, y, power)
    out_ref = F.linear(z_ref, w, b)
    out_tri = powglu_linear(x, y, w, b, power) if not compile else torch.compile(powglu_linear)(x, y, w, b, power)

    g = torch.randn_like(out_ref)
    dx_ref, dy_ref, dw_ref, db_ref = torch.autograd.grad(out_ref, (x, y, w, b), g)
    dx_tri, dy_tri, dw_tri, db_tri = torch.autograd.grad(out_tri, (x, y, w, b), g)

    assert_close('powglu_linear out', out_ref, out_tri, 1e-3)
    assert_close('powglu_linear dx ',  dx_ref,  dx_tri,  1e-3)
    assert_close('powglu_linear dy ',  dy_ref,  dy_tri,  1e-3)
    assert_close('powglu_linear dw ',  dw_ref,  dw_tri,  1e-3)
    assert_close('powglu_linear db ',  db_ref,  db_tri,  1e-3)


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
