# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from unittest.mock import Mock

import pytest
import torch

from fla.backends import _DISPATCH_DISABLED, GluonBackend, TileLangBackend
from fla.ops.utils.sinkhorn import ops, sinkhorn
from fla.ops.utils.sinkhorn.naive import naive_sinkhorn
from fla.utils import IS_NVIDIA, assert_close, device

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='NVIDIA CUDA required')


@pytest.fixture(params=['triton', 'gluon', 'tilelang'])
def backend(request, monkeypatch):
    for name in ('GLUON', 'TILELANG'):
        monkeypatch.setenv(f'FLA_{name}', '0')
    name = request.param
    if name == 'triton':
        return name
    if _DISPATCH_DISABLED:
        pytest.skip('Backend dispatch disabled before import')
    backend_type = GluonBackend if name == 'gluon' else TileLangBackend
    if not backend_type.is_available():
        pytest.skip(f'{name} is unavailable')
    monkeypatch.setenv(f'FLA_{name.upper()}', '1')
    return name


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'N', 'num_iters', 'scale'),
    [
        pytest.param(*test, id='B{}-T{}-H{}-N{}-iters{}-scale{}'.format(*test))
        for test in [
            (2, 7, 1, 1, 20, 1),
            (2, 7, 1, 3, 1, 1),
            (2, 7, 1, 4, 20, 1),
            (2, 7, 1, 8, 7, 1),
            (2, 7, 1, 4, 20, 1000),
            (1, 127, 1, 4, 20, 1),
            (1, 128, 1, 4, 20, 1),
            (1, 129, 1, 1, 20, 1),
            (1, 129, 1, 3, 20, 1),
            (1, 129, 1, 8, 20, 1),
            (1, 8192, 1, 4, 20, 1),
            (1, 8192, 1, 8, 20, 1),
            (1, 32768, 1, 4, 20, 1),
            (1, 32768, 1, 8, 20, 1),
        ]
    ],
)
def test_sinkhorn(B: int, T: int, H: int, N: int, num_iters: int, scale: float, dtype: torch.dtype, backend):
    torch.manual_seed(42)
    x = (torch.randn(B, T, H, N, N, dtype=dtype, device=device) * scale).requires_grad_(True)
    ref = naive_sinkhorn(logits=x, num_iters=num_iters)
    do = torch.randn_like(ref)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None

    tri = sinkhorn(logits=x, num_iters=num_iters)
    tri.backward(do)
    tri_dx = x.grad

    assert tri.shape == x.shape
    assert tri.dtype == torch.float32
    assert tri_dx.dtype == dtype
    assert torch.isfinite(tri).all()
    assert torch.isfinite(tri_dx).all()
    assert_close('o', ref, tri, 1e-4)
    assert_close('dx', ref_dx, tri_dx, 1e-4)
    torch.testing.assert_close(tri.sum(-1), torch.ones_like(tri[..., 0]), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize(('shape', 'strided'), [
    pytest.param((4, 4), False, id='unbatched'),
    pytest.param((2, 4, 4), True, id='strided'),
    pytest.param((2, 0, 4, 4), False, id='empty'),
])
def test_sinkhorn_layout(shape: tuple[int, ...], strided: bool, dtype: torch.dtype, backend):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, device=device)
    if strided:
        x = x.transpose(-1, -2)
    x.requires_grad_()
    ref = naive_sinkhorn(logits=x)
    do = torch.randn_like(ref)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None

    tri = sinkhorn(logits=x)
    tri.backward(do)
    tri_dx = x.grad

    assert tri.shape == x.shape
    assert tri.dtype == torch.float32
    assert tri_dx.dtype == dtype
    if x.numel() == 0:
        torch.testing.assert_close(ref, tri)
        torch.testing.assert_close(ref_dx, tri_dx)
    else:
        assert_close('o', ref, tri, 1e-4)
        assert_close('dx', ref_dx, tri_dx, 1e-4)


def test_sinkhorn_packed(backend):
    torch.manual_seed(42)
    logits = torch.randn(2, 3, 4, 4, device=device)
    with torch.no_grad():
        dense = sinkhorn(logits=logits)
        packed = sinkhorn(logits=logits.flatten(0, 1))
    torch.testing.assert_close(dense.flatten(0, 1), packed, rtol=0, atol=0)


def test_sinkhorn_backend(backend, monkeypatch):
    if backend == 'gluon':
        from fla.ops.utils.sinkhorn import gluon as module
    elif backend == 'tilelang':
        from fla.ops.utils.sinkhorn import tilelang as module
    else:
        module = ops
    name = 'sinkhorn_bwd' if backend == 'triton' else f'sinkhorn_bwd_{backend}'
    backward = Mock(wraps=getattr(module, name))
    monkeypatch.setattr(module, name, backward)

    x = torch.randn(3, 4, 4, device=device, requires_grad=True)
    out = sinkhorn(logits=x)
    for flag in ('FLA_GLUON', 'FLA_TILELANG'):
        monkeypatch.setenv(flag, '0')
    out.backward(torch.randn_like(out))
    backward.assert_called_once()


@pytest.mark.parametrize('backend_type', [GluonBackend, TileLangBackend])
def test_sinkhorn_unavailable_backend(monkeypatch, backend_type):
    if _DISPATCH_DISABLED:
        pytest.skip('Backend dispatch disabled before import')
    for name in ('GLUON', 'TILELANG'):
        monkeypatch.setenv(f'FLA_{name}', '0')
    monkeypatch.setenv(f'FLA_{backend_type.backend_type.upper()}', '1')
    monkeypatch.setattr(backend_type, 'is_available', classmethod(lambda cls: False))

    implementation = Mock(side_effect=AssertionError('Unavailable backend was selected'))
    monkeypatch.setattr(backend_type, 'get_implementation', implementation)
    logits = torch.zeros(1, 1, device=device)
    torch.testing.assert_close(sinkhorn(logits=logits), torch.ones_like(logits))


@pytest.mark.parametrize(('shape', 'dtype', 'num_iters', 'error'), [
    pytest.param((), torch.float32, 20, 'shape', id='scalar'),
    pytest.param((4,), torch.float32, 20, 'shape', id='vector'),
    pytest.param((4, 3), torch.float32, 20, 'shape', id='rectangular'),
    pytest.param((0, 0), torch.float32, 20, 'shape', id='N0'),
    pytest.param((9, 9), torch.float32, 20, 'shape', id='N9'),
    pytest.param((4, 4), torch.float64, 20, 'FP16, BF16, or FP32', id='fp64'),
    pytest.param((4, 4), torch.int32, 20, 'FP16, BF16, or FP32', id='int32'),
    pytest.param((4, 4), torch.float32, 0, 'positive integer', id='iter0'),
    pytest.param((4, 4), torch.float32, -1, 'positive integer', id='negative-iters'),
    pytest.param((4, 4), torch.float32, 1.5, 'positive integer', id='fractional-iters'),
    pytest.param((4, 4), torch.float32, True, 'positive integer', id='bool-iters'),
])
def test_sinkhorn_invalid_inputs(monkeypatch, shape, dtype, num_iters, error):
    forward = Mock(side_effect=AssertionError('Invalid inputs reached backend dispatch'))
    monkeypatch.setattr(ops, 'sinkhorn_fwd', forward)
    with pytest.raises(ValueError, match=error):
        sinkhorn(logits=torch.empty(shape, device=device, dtype=dtype), num_iters=num_iters)
