# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.modules.activations import logsigmoid, sigmoid, swiglu, swiglu_linear, swish
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'D'),
    [
        (1, 1, 64),
        (2, 512, 128),
        (3, 2048, 1200),
    ]
)
def test_sigmoid(B: int, T: int, D: int):
    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, requires_grad=True)
    y_ref = torch.sigmoid(x)
    y_tri = sigmoid(x)

    g = torch.randn_like(y_ref)
    dx_ref = torch.autograd.grad(y_ref, x, g)[0]
    dx_tri = torch.autograd.grad(y_tri, x, g)[0]

    assert_close('sigmoid fwd', y_ref, y_tri, 1e-3)
    assert_close('sigmoid bwd', dx_ref, dx_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'temperature'),
    [
        (1, 1, 64, 1.0),
        (2, 512, 128, 0.5),
        (3, 2048, 1200, 2.0),
    ]
)
def test_logsigmoid(B: int, T: int, D: int, temperature: float):
    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, requires_grad=True)
    y_ref = F.logsigmoid(x) / temperature
    y_tri = logsigmoid(x, temperature)

    g = torch.randn_like(y_ref)
    dx_ref = torch.autograd.grad(y_ref, x, g)[0]
    dx_tri = torch.autograd.grad(y_tri, x, g)[0]

    assert_close('logsigmoid fwd', y_ref, y_tri, 1e-3)
    assert_close('logsigmoid bwd', dx_ref, dx_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D'),
    [
        (1, 1, 64),
        (2, 512, 128),
        (3, 2048, 1200),
    ]
)
def test_swish(B: int, T: int, D: int):
    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, requires_grad=True)
    y_ref = F.silu(x)
    y_tri = swish(x)

    g = torch.randn_like(y_ref)
    dx_ref = torch.autograd.grad(y_ref, x, g)[0]
    dx_tri = torch.autograd.grad(y_tri, x, g)[0]

    assert_close('swish fwd', y_ref, y_tri, 1e-3)
    assert_close('swish bwd', dx_ref, dx_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D'),
    [
        (1, 1, 64),
        (2, 512, 128),
        (3, 2048, 1200),
    ]
)
def test_swiglu(B: int, T: int, D: int):
    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, requires_grad=True)
    y = torch.randn(B, T, D, device=device, requires_grad=True)

    y_ref = F.silu(x) * y
    y_tri = swiglu(x, y)

    g = torch.randn_like(y_ref)
    dx_ref, dy_ref = torch.autograd.grad(y_ref, (x, y), g)
    dx_tri, dy_tri = torch.autograd.grad(y_tri, (x, y), g)

    assert_close('swiglu fwd', y_ref, y_tri, 1e-3)
    assert_close('swiglu dx',  dx_ref, dx_tri, 1e-3)
    assert_close('swiglu dy',  dy_ref, dy_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'O'),
    [
        (2, 512, 128, 256),
        (1, 1, 64, 32),
    ]
)
def test_swiglu_linear(B: int, T: int, D: int, O: int):  # noqa: E741
    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, requires_grad=True)
    y = torch.randn(B, T, D, device=device, requires_grad=True)
    w = torch.randn(O, D, device=device, requires_grad=True)
    b = torch.randn(O, device=device, requires_grad=True)

    z_ref = F.silu(x) * y
    out_ref = F.linear(z_ref, w, b)
    out_tri = swiglu_linear(x, y, w, b)

    g = torch.randn_like(out_ref)
    dx_ref, dy_ref, dw_ref, db_ref = torch.autograd.grad(out_ref, (x, y, w, b), g)
    dx_tri, dy_tri, dw_tri, db_tri = torch.autograd.grad(out_tri, (x, y, w, b), g)

    assert_close('swiglu_linear out', out_ref, out_tri, 1e-3)
    assert_close('swiglu_linear dx',  dx_ref,  dx_tri,  1e-3)
    assert_close('swiglu_linear dy',  dy_ref,  dy_tri,  1e-3)
    assert_close('swiglu_linear dw',  dw_ref,  dw_tri,  1e-3)
    assert_close('swiglu_linear db',  db_ref,  db_tri,  1e-3)
