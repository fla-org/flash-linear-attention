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
    elu_p1,
    fast_gelu_impl,
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
from fla.modules.activations.ops import flatten_input, get_row_stride
from fla.utils import assert_close, device


def powglu_ref(x: torch.Tensor, y: torch.Tensor, power: float = 3.0) -> torch.Tensor:
    s = torch.sigmoid(x)
    # guard log/sqrt to the x>0 branch so autograd through the masked lanes stays finite
    xpos = torch.where(x > 0, x, torch.ones_like(x))
    g = torch.where(x > 0, xpos ** (power / (xpos.sqrt() + 1)) * s, x * s)
    return g * y


def make_inputs(shape: tuple[int, ...], n: int, noncontiguous: bool) -> tuple[torch.Tensor, ...]:
    """Return grad-requiring tensors with contiguous, strided-vector, or chunked-row layouts."""
    if noncontiguous and len(shape) == 1:
        xs = tuple(torch.randn(shape[0] * 3, device=device)[::3] for _ in range(n))
    elif noncontiguous:
        xs = torch.randn(*shape[:-1], shape[-1] * 2, device=device).chunk(2, dim=-1)[:n]
    else:
        xs = tuple(torch.randn(shape, device=device) for _ in range(n))
    return tuple(x.requires_grad_() for x in xs)


def make_singleton_input(shape: tuple[int, ...], strides: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    x = torch.randn(size=shape, device=device, dtype=dtype).as_strided(size=shape, stride=strides)
    assert x.is_contiguous() and x.contiguous() is x
    return x


@pytest.mark.parametrize('shape', [(), (1,), (0,), (2, 0)])
def test_elu_p1(shape: tuple[int, ...]):
    # bf16 rounds ELU(-8) to -1, so adding 1 loses the positive feature
    x = torch.full(size=shape, fill_value=-8., device=device, dtype=torch.bfloat16, requires_grad=True)
    y = elu_p1(x)
    dx, = torch.autograd.grad(y.sum(), x)
    expected = x.detach().float().exp().to(x.dtype)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    torch.testing.assert_close(dx, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ('shape', 'noncontiguous', 'grad_stride', 'compile'),
    [
        pytest.param(*test, id="shape{}-nc{}-grad_stride{}-compile{}".format("x".join(map(str, test[0])), *test[1:]))
        for test in [
            ((1, 1, 64), False, 1, False),
            ((2, 500, 128), False, 1, False),
            ((2, 512, 128), False, 1, True),
            ((3, 2048, 1200), False, 1, True),
            ((2, 500, 128), True, 1, False),
            ((2, 512, 128), True, 1, True),
            ((3, 2048, 1200), True, 1, False),
            ((63,), False, 1, False),
            ((63,), False, 3, False),
            ((63,), True, 1, False),
            ((63,), True, 3, False),
        ]
    ],
)
def test_sigmoid(shape: tuple[int, ...], noncontiguous: bool, grad_stride: int, compile: bool):
    torch.manual_seed(42)
    (x,) = make_inputs(shape=shape, n=1, noncontiguous=noncontiguous)
    y_ref = torch.sigmoid(x)
    y_tri = sigmoid(x) if not compile else torch.compile(sigmoid)(x)

    g = torch.randn(*y_ref.shape, grad_stride, device=device)[..., 0]
    dx_ref, = torch.autograd.grad(y_ref, (x,), g)
    dx_tri, = torch.autograd.grad(y_tri, (x,), g)

    assert_close('sigmoid fwd', y_ref, y_tri, 1e-3)
    assert_close('sigmoid dx ', dx_ref, dx_tri, 1e-3)


@pytest.mark.parametrize(
    ('shape', 'temperature', 'noncontiguous', 'grad_stride', 'compile'),
    [
        pytest.param(*test, id="shape{}-temp{}-nc{}-grad{}-compile{}".format("x".join(map(str, test[0])), *test[1:]))
        for test in [
            ((1, 1, 64), 1.0, False, 1, False),
            ((2, 500, 128), 0.5, False, 1, False),
            ((2, 512, 128), 0.5, False, 1, True),
            ((3, 2048, 1200), 2.0, False, 1, True),
            ((2, 500, 128), 0.5, True, 1, False),
            ((2, 512, 128), 1.0, True, 1, True),
            ((3, 2048, 1200), 2.0, True, 1, False),
            ((63,), 1.0, False, 1, False),
            ((63,), 1.0, False, 3, False),
            ((63,), 1.0, True, 1, False),
            ((63,), 1.0, True, 3, False),
        ]
    ],
)
def test_logsigmoid(shape: tuple[int, ...], temperature: float, noncontiguous: bool, grad_stride: int, compile: bool):
    torch.manual_seed(42)
    (x,) = make_inputs(shape=shape, n=1, noncontiguous=noncontiguous)
    y_ref = F.logsigmoid(x) / temperature
    y_tri = logsigmoid(x, temperature) if not compile else torch.compile(logsigmoid)(x, temperature)

    g = torch.randn(*y_ref.shape, grad_stride, device=device)[..., 0]
    dx_ref, = torch.autograd.grad(y_ref, (x,), g)
    dx_tri, = torch.autograd.grad(y_tri, (x,), g)

    assert_close('logsigmoid fwd', y_ref, y_tri, 1e-3)
    assert_close('logsigmoid dx ', dx_ref, dx_tri, 1e-3)


@pytest.mark.parametrize(
    ('shape', 'noncontiguous', 'grad_stride', 'compile'),
    [
        pytest.param(*test, id="shape{}-nc{}-grad_stride{}-compile{}".format("x".join(map(str, test[0])), *test[1:]))
        for test in [
            ((1, 1, 64), False, 1, True),
            ((2, 500, 128), False, 1, True),
            ((2, 512, 128), False, 1, False),
            ((3, 2048, 1200), False, 1, False),
            ((2, 500, 128), True, 1, False),
            ((2, 512, 128), True, 1, True),
            ((3, 2048, 1200), True, 1, False),
            ((63,), False, 1, False),
            ((63,), False, 3, False),
            ((63,), True, 1, False),
            ((63,), True, 3, False),
        ]
    ],
)
def test_swish(shape: tuple[int, ...], noncontiguous: bool, grad_stride: int, compile: bool):
    torch.manual_seed(42)
    (x,) = make_inputs(shape=shape, n=1, noncontiguous=noncontiguous)
    y_ref = F.silu(x)
    y_tri = swish(x) if not compile else torch.compile(swish)(x)

    g = torch.randn(*y_ref.shape, grad_stride, device=device)[..., 0]
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
    (x,) = make_inputs(shape=(B, T, D), n=1, noncontiguous=noncontiguous)
    y_ref = F.gelu(x, approximate='tanh')
    y_tri = fast_gelu_impl(x) if not compile else torch.compile(fast_gelu_impl)(x)

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
    (x,) = make_inputs(shape=(B, T, D), n=1, noncontiguous=noncontiguous)
    y_ref = F.relu(x) ** 2
    y_tri = sqrelu(x) if not compile else torch.compile(sqrelu)(x)

    g = torch.randn_like(y_ref)
    dx_ref, = torch.autograd.grad(y_ref, (x,), g)
    dx_tri, = torch.autograd.grad(y_tri, (x,), g)

    assert_close('sqrelu fwd', y_ref, y_tri, 1e-3)
    assert_close('sqrelu dx ', dx_ref, dx_tri, 1e-3)


@pytest.mark.parametrize(
    ('shape', 'noncontiguous', 'grad_stride', 'compile'),
    [
        pytest.param(*test, id="shape{}-nc{}-grad_stride{}-compile{}".format("x".join(map(str, test[0])), *test[1:]))
        for test in [
            ((1, 1, 64), False, 1, True),
            ((2, 500, 128), False, 1, True),
            ((2, 512, 128), False, 1, False),
            ((3, 2048, 1200), False, 1, False),
            ((2, 500, 128), True, 1, True),
            ((2, 512, 128), True, 1, False),
            ((3, 2048, 1200), True, 1, False),
            ((63,), False, 1, False),
            ((63,), False, 3, False),
            ((63,), True, 1, False),
            ((63,), True, 3, False),
        ]
    ],
)
def test_swiglu(shape: tuple[int, ...], noncontiguous: bool, grad_stride: int, compile: bool):
    torch.manual_seed(42)
    x, y = make_inputs(shape=shape, n=2, noncontiguous=noncontiguous)

    y_ref = F.silu(x) * y
    y_tri = swiglu(x, y) if not compile else torch.compile(swiglu)(x, y)

    g = torch.randn(*y_ref.shape, grad_stride, device=device)[..., 0]
    dx_ref, dy_ref = torch.autograd.grad(y_ref, (x, y), g)
    dx_tri, dy_tri = torch.autograd.grad(y_tri, (x, y), g)

    assert_close('swiglu fwd', y_ref, y_tri, 1e-3)
    assert_close('swiglu dx ', dx_ref, dx_tri, 1e-3)
    assert_close('swiglu dy ', dy_ref, dy_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'O', 'noncontiguous', 'compile'),
    [
        pytest.param(*test, id="B{}-T{}-D{}-O{}-noncontiguous{}-compile{}".format(*test))
        for test in [
            (2, 512, 128, 256, False, True),
            (1, 1, 64, 32, False, False),
            (2, 500, 128, 64, False, True),
            (3, 2048, 1200, 600, False, False),
            (2, 500, 128, 64, True, False),
            (2, 512, 128, 256, True, True),
            (3, 2048, 1200, 600, True, False),
        ]
    ],
)
@pytest.mark.parametrize(
    ('requires_grad', 'use_bias'),
    [
        pytest.param((True, True, True, True), True, id='all'),
        pytest.param((True, True, False, False), True, id='inputs'),
        pytest.param((True, False, False, False), True, id='x'),
        pytest.param((False, True, False, False), True, id='y'),
        pytest.param((False, False, True, False), True, id='weight'),
        pytest.param((False, False, False, True), True, id='bias'),
        pytest.param((False, False, True, True), True, id='weight-bias'),
        pytest.param((True, True, True, False), False, id='all-no-bias'),
        pytest.param((True, True, False, False), False, id='inputs-no-bias'),
        pytest.param((False, False, True, False), False, id='weight-no-bias'),
    ],
)
def test_swiglu_linear(
    B: int,
    T: int,
    D: int,
    O: int,
    noncontiguous: bool,
    compile: bool,
    requires_grad: tuple[bool, bool, bool, bool],
    use_bias: bool,
):
    torch.manual_seed(42)
    x, y = make_inputs(shape=(B, T, D), n=2, noncontiguous=noncontiguous)
    x.requires_grad_(requires_grad[0])
    y.requires_grad_(requires_grad[1])
    w = torch.randn(O, D, device=device, requires_grad=requires_grad[2])
    b = torch.randn(O, device=device, requires_grad=requires_grad[3]) if use_bias else None
    inputs = (x, y, w, b)
    grad_inputs = [tensor for tensor in inputs if tensor is not None and tensor.requires_grad]
    grad_names = [
        name for name, tensor in zip(('dx', 'dy', 'dw', 'db'), inputs)
        if tensor is not None and tensor.requires_grad
    ]

    ref_z = F.silu(x) * y
    ref = F.linear(input=ref_z, weight=w, bias=b)
    fn = torch.compile(swiglu_linear) if compile else swiglu_linear
    tri = fn(x=x, y=y, weight=w, bias=b)

    do = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(outputs=ref, inputs=grad_inputs, grad_outputs=do)
    tri_grads = torch.autograd.grad(outputs=tri, inputs=grad_inputs, grad_outputs=do)

    assert_close('swiglu_linear out', ref, tri, 1e-3)
    for name, ref_grad, tri_grad in zip(grad_names, ref_grads, tri_grads):
        assert_close(f'swiglu_linear {name}', ref_grad, tri_grad, 1e-3)


@pytest.mark.parametrize(
    ('shape', 'noncontiguous', 'grad_stride', 'compile'),
    [
        pytest.param(*test, id="shape{}-nc{}-grad_stride{}-compile{}".format("x".join(map(str, test[0])), *test[1:]))
        for test in [
            ((1, 1, 64), False, 1, True),
            ((2, 500, 128), False, 1, True),
            ((2, 512, 128), False, 1, False),
            ((3, 2048, 1200), False, 1, False),
            ((2, 500, 128), True, 1, True),
            ((2, 512, 128), True, 1, False),
            ((3, 2048, 1200), True, 1, False),
            ((63,), False, 1, False),
            ((63,), False, 3, False),
            ((63,), True, 1, False),
            ((63,), True, 3, False),
        ]
    ],
)
def test_sigmoidglu(shape: tuple[int, ...], noncontiguous: bool, grad_stride: int, compile: bool):
    torch.manual_seed(42)
    x, y = make_inputs(shape=shape, n=2, noncontiguous=noncontiguous)

    y_ref = torch.sigmoid(x) * y
    y_tri = sigmoidglu(x, y) if not compile else torch.compile(sigmoidglu)(x, y)

    g = torch.randn(*y_ref.shape, grad_stride, device=device)[..., 0]
    dx_ref, dy_ref = torch.autograd.grad(y_ref, (x, y), g)
    dx_tri, dy_tri = torch.autograd.grad(y_tri, (x, y), g)

    assert_close('sigmoidglu fwd', y_ref, y_tri, 1e-3)
    assert_close('sigmoidglu dx ', dx_ref, dx_tri, 1e-3)
    assert_close('sigmoidglu dy ', dy_ref, dy_tri, 1e-3)


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
    x, y = make_inputs(shape=(B, T, D), n=2, noncontiguous=noncontiguous)
    w = torch.randn(O, D, device=device, requires_grad=True)
    b = torch.randn(O, device=device, requires_grad=True)

    z_ref = torch.sigmoid(x) * y
    out_ref = F.linear(z_ref, w, b)
    out_tri = sigmoidglu_linear(x, y, w, b) if not compile else torch.compile(sigmoidglu_linear)(x, y, w, b)

    g = torch.randn_like(out_ref)
    dx_ref, dy_ref, dw_ref, db_ref = torch.autograd.grad(out_ref, (x, y, w, b), g)
    dx_tri, dy_tri, dw_tri, db_tri = torch.autograd.grad(out_tri, (x, y, w, b), g)

    assert_close('sigmoidglu_linear out', out_ref, out_tri, 1e-3)
    assert_close('sigmoidglu_linear dx ', dx_ref, dx_tri, 1e-3)
    assert_close('sigmoidglu_linear dy ', dy_ref, dy_tri, 1e-3)
    assert_close('sigmoidglu_linear dw ', dw_ref, dw_tri, 1e-3)
    assert_close('sigmoidglu_linear db ', db_ref, db_tri, 1e-3)


@pytest.mark.parametrize(
    ('shape', 'power', 'noncontiguous', 'grad_stride', 'compile'),
    [
        pytest.param(*test, id="shape{}-power{}-nc{}-grad{}-compile{}".format("x".join(map(str, test[0])), *test[1:]))
        for test in [
            ((1, 1, 64), 3.0, False, 1, True),
            ((2, 500, 128), 2.0, False, 1, True),
            ((2, 512, 128), 3.0, False, 1, False),
            ((3, 2048, 1200), 4.0, False, 1, False),
            ((2, 500, 128), 3.0, True, 1, True),
            ((2, 512, 128), 2.0, True, 1, False),
            ((3, 2048, 1200), 3.0, True, 1, False),
            ((63,), 3.0, False, 1, False),
            ((63,), 3.0, False, 3, False),
            ((63,), 3.0, True, 1, False),
            ((63,), 3.0, True, 3, False),
        ]
    ],
)
def test_powglu(shape: tuple[int, ...], power: float, noncontiguous: bool, grad_stride: int, compile: bool):
    torch.manual_seed(42)
    x, y = make_inputs(shape=shape, n=2, noncontiguous=noncontiguous)

    y_ref = powglu_ref(x, y, power)
    y_tri = powglu(x, y, power) if not compile else torch.compile(powglu)(x, y, power)

    g = torch.randn(*y_ref.shape, grad_stride, device=device)[..., 0]
    dx_ref, dy_ref = torch.autograd.grad(y_ref, (x, y), g)
    dx_tri, dy_tri = torch.autograd.grad(y_tri, (x, y), g)

    assert_close('powglu fwd', y_ref, y_tri, 1e-3)
    assert_close('powglu dx ', dx_ref, dx_tri, 1e-3)
    assert_close('powglu dy ', dy_ref, dy_tri, 1e-3)


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
    x, y = make_inputs(shape=(B, T, D), n=2, noncontiguous=noncontiguous)
    w = torch.randn(O, D, device=device, requires_grad=True)
    b = torch.randn(O, device=device, requires_grad=True)

    z_ref = powglu_ref(x, y, power)
    out_ref = F.linear(z_ref, w, b)
    out_tri = powglu_linear(x, y, w, b, power) if not compile else torch.compile(powglu_linear)(x, y, w, b, power)

    g = torch.randn_like(out_ref)
    dx_ref, dy_ref, dw_ref, db_ref = torch.autograd.grad(out_ref, (x, y, w, b), g)
    dx_tri, dy_tri, dw_tri, db_tri = torch.autograd.grad(out_tri, (x, y, w, b), g)

    assert_close('powglu_linear out', out_ref, out_tri, 1e-3)
    assert_close('powglu_linear dx ', dx_ref, dx_tri, 1e-3)
    assert_close('powglu_linear dy ', dy_ref, dy_tri, 1e-3)
    assert_close('powglu_linear dw ', dw_ref, dw_tri, 1e-3)
    assert_close('powglu_linear db ', db_ref, db_tri, 1e-3)


@pytest.mark.parametrize('compile', [False, True], ids=['eager', 'compile'])
@pytest.mark.parametrize('linear', [False, True], ids=['activation', 'linear'])
def test_powglu_singleton_strides(compile: bool, linear: bool):
    torch.manual_seed(42)
    shape, strides = (3, 1, 128), (128, 1536, 1)
    inputs = tuple(make_singleton_input(shape=shape, strides=strides, dtype=torch.float32).requires_grad_() for _ in range(2))
    if linear:
        inputs += (torch.randn(64, 128, device=device, requires_grad=True), torch.randn(64, device=device, requires_grad=True))
    ref_inputs = tuple(x.detach().requires_grad_() for x in inputs)
    ref = powglu_ref(*ref_inputs[:2])
    if linear:
        ref = F.linear(input=ref, weight=ref_inputs[2], bias=ref_inputs[3])
    fn = powglu_linear if linear else powglu
    if compile:
        torch.compiler.reset()
    tri = torch.compile(fn)(*inputs) if compile else fn(*inputs)
    if not linear:
        # custom activation outputs must allow in-place use
        tri.add_(0)
    g = make_singleton_input(shape=ref.shape, strides=(ref.shape[-1], 1536, 1), dtype=torch.float32)
    ref_grads = torch.autograd.grad(outputs=ref, inputs=ref_inputs, grad_outputs=g)
    tri_grads = torch.autograd.grad(outputs=tri, inputs=inputs, grad_outputs=g)

    assert tri.shape == ref.shape
    assert torch.isfinite(tri).all()
    assert_close('singleton-stride output', ref, tri, 1e-3)
    for i, (ref_grad, tri_grad) in enumerate(zip(ref_grads, tri_grads)):
        assert tri_grad.shape == inputs[i].shape
        assert torch.isfinite(tri_grad).all()
        assert_close(f'singleton-stride gradient {i}', ref_grad, tri_grad, 1e-3)


@pytest.mark.parametrize(
    ('shape', 'strides', 'needs_copy'),
    [
        pytest.param((), (), False, id='scalar'),
        pytest.param((10,), (1,), False, id='vector'),
        pytest.param((10, 20), (20, 1), False, id='dense-2d'),
        pytest.param((5, 10, 20), (200, 20, 1), False, id='dense-3d'),
        pytest.param((3, 5, 10, 20), (1000, 200, 20, 1), False, id='dense-4d'),
        pytest.param((2, 3, 5, 10, 20), (3000, 1000, 200, 20, 1), False, id='dense-5d'),
        pytest.param((3, 1, 128), (128, 1536, 1), False, id='singleton-3d'),
        pytest.param((2, 1, 1, 63), (63, 999, 756, 1), False, id='singleton-4d'),
        pytest.param((2, 1, 3, 1, 63), (189, 777, 63, 756, 1), False, id='singleton-5d'),
        pytest.param((3, 1, 1), (1, 12, 7), False, id='singleton-feature'),
        pytest.param((1, 1, 128), (999, 756, 1), False, id='single-row'),
        pytest.param((3, 1), (1, 7), False, id='singleton-2d'),
        pytest.param((1,), (7,), False, id='singleton-vector'),
        pytest.param((5, 10, 20), (400, 40, 1), False, id='chunked-3d'),
        pytest.param((3, 5, 10, 20), (2000, 400, 40, 1), False, id='chunked-4d'),
        pytest.param((3, 1, 128), (256, 1536, 1), False, id='chunked-singleton'),
        pytest.param((3, 4, 5), (20, 1, 4), True, id='transposed'),
        pytest.param((3, 10, 20), (400, 20, 1), True, id='outer-strided'),
        pytest.param((10, 10), (20, 2), True, id='strided-columns'),
        pytest.param((10,), (3,), True, id='strided-vector'),
        pytest.param((0, 3), (3, 1), False, id='empty-rows'),
        pytest.param((2, 0), (1, 1), False, id='empty-columns'),
    ],
)
def test_flatten_input(shape: tuple[int, ...], strides: tuple[int, ...], needs_copy: bool):
    torch.manual_seed(42)
    x = torch.empty_strided(size=shape, stride=strides).copy_(torch.randn(size=shape))
    flat = flatten_input(x=x)
    assert flat.ndim == 2
    assert flat.shape[-1] == (shape[-1] if shape else 1)
    assert (flat.data_ptr() != x.data_ptr()) == needs_copy
    addressed = flat.as_strided(size=flat.shape, stride=(get_row_stride(x=flat), 1))
    torch.testing.assert_close(addressed.view(shape), x, rtol=0, atol=0)


@pytest.mark.parametrize(
    ('fn', 'reference', 'n'),
    [
        pytest.param(sigmoid, torch.sigmoid, 1, id='sigmoid'),
        pytest.param(elu_p1, lambda x: F.elu(x) + 1, 1, id='elu_p1'),
        pytest.param(logsigmoid, F.logsigmoid, 1, id='logsigmoid'),
        pytest.param(swish, F.silu, 1, id='swish'),
        pytest.param(swiglu, lambda x, y: F.silu(x) * y, 2, id='swiglu'),
        pytest.param(sigmoidglu, lambda x, y: torch.sigmoid(x) * y, 2, id='sigmoidglu'),
        pytest.param(powglu, powglu_ref, 2, id='powglu'),
    ],
)
@pytest.mark.parametrize(
    ('shape', 'strides'),
    [
        pytest.param((3, 1, 128), (128, 1536, 1), id='3d'),
        pytest.param((2, 1, 1, 63), (63, 999, 756, 1), id='4d'),
        pytest.param((2, 1, 3, 1, 63), (189, 777, 63, 756, 1), id='5d'),
        pytest.param((3, 1, 1), (1, 12, 7), id='singleton-feature'),
    ],
)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16], ids=['fp32', 'fp16', 'bf16'])
@pytest.mark.parametrize('compile', [False, True], ids=['eager', 'compile'])
def test_activation_singleton_strides(
    fn,
    reference,
    n: int,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    dtype: torch.dtype,
    compile: bool,
):
    torch.manual_seed(42)
    inputs = tuple(make_singleton_input(shape=shape, strides=strides, dtype=dtype).requires_grad_() for _ in range(n))
    ref_inputs = tuple(x.detach().float().requires_grad_() for x in inputs)
    ref = reference(*ref_inputs)
    if compile:
        torch.compiler.reset()
    tri = torch.compile(fn)(*inputs) if compile else fn(*inputs)
    tri.add_(0)
    g = make_singleton_input(shape=shape, strides=strides, dtype=dtype)
    ref_grads = torch.autograd.grad(outputs=ref, inputs=ref_inputs, grad_outputs=g.float())
    tri_grads = torch.autograd.grad(outputs=tri, inputs=inputs, grad_outputs=g)

    assert tri.shape == shape
    assert torch.isfinite(tri).all()
    assert_close('singleton-stride output', ref.to(dtype), tri, 1e-3)
    for i, (ref_grad, tri_grad) in enumerate(zip(ref_grads, tri_grads)):
        assert tri_grad.shape == shape
        assert torch.isfinite(tri_grad).all()
        assert_close(f'singleton-stride gradient {i}', ref_grad.to(dtype), tri_grad, 1e-3)
