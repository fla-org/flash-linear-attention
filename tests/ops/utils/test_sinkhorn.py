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
    ('shape', 'num_iters', 'scale', 'noncontiguous'),
    [
        pytest.param(*test, id='shape{}-iters{}-scale{}-nc{}'.format('x'.join(map(str, test[0])), *test[1:]))
        for test in [
            ((2, 7, 1, 1, 1), 20, 1, False),
            ((2, 7, 1, 3, 3), 1, 1, False),
            ((2, 7, 1, 4, 4), 20, 1, False),
            ((2, 7, 1, 8, 8), 7, 1, False),
            ((2, 7, 1, 4, 4), 20, 1000, False),
            ((1, 127, 1, 4, 4), 20, 1, False),
            ((1, 128, 1, 4, 4), 20, 1, False),
            ((1, 129, 1, 1, 1), 20, 1, False),
            ((1, 129, 1, 3, 3), 20, 1, False),
            ((1, 129, 1, 8, 8), 20, 1, False),
            ((1, 129, 1, 8, 8), 21, 1, False),
            ((2, 7, 1, 3, 3), 40, 1, False),
            ((2, 7, 1, 4, 4), 41, 1, False),
            ((1, 8192, 1, 4, 4), 20, 1, False),
            ((1, 8192, 1, 8, 8), 20, 1, False),
            ((1, 32768, 1, 4, 4), 20, 1, False),
            ((1, 32768, 1, 8, 8), 20, 1, False),
            ((4, 4), 20, 1, False),
            ((2, 4, 4), 20, 1, True),
            ((2, 0, 4, 4), 20, 1, False),
        ]
    ],
)
def test_sinkhorn(shape: tuple[int, ...], num_iters: int, scale: float, noncontiguous: bool, dtype: torch.dtype, backend):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, device=device) * scale
    if noncontiguous:
        x = x.transpose(-1, -2)
    x.requires_grad_()
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
    if x.numel() == 0:
        torch.testing.assert_close(ref, tri)
        torch.testing.assert_close(ref_dx, tri_dx)
    else:
        assert_close('o', ref, tri, 1e-4)
        assert_close('dx', ref_dx, tri_dx, 1e-4)
    torch.testing.assert_close(tri.sum(-1), torch.ones_like(tri[..., 0]), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16], ids=str)
def test_sinkhorn_inference(dtype: torch.dtype, backend):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 4, 4, dtype=dtype, device=device, requires_grad=True)
    with torch.no_grad():
        ref = naive_sinkhorn(logits=x)
        tri = sinkhorn(logits=x)
        flat = sinkhorn(logits=x.flatten(0, 1))
    assert tri.dtype == torch.float32
    assert not tri.requires_grad
    assert not flat.requires_grad
    assert_close('o', ref, tri, 1e-4)
    torch.testing.assert_close(tri.flatten(0, 1), flat, rtol=0, atol=0)


def test_sinkhorn_backward_backend(backend, monkeypatch):
    torch.manual_seed(42)
    if backend == 'gluon':
        from fla.ops.utils.sinkhorn import gluon

        backward = Mock(wraps=gluon.sinkhorn_bwd_gluon)
        monkeypatch.setattr(gluon, 'sinkhorn_bwd_gluon', backward)
    elif backend == 'tilelang':
        from fla.ops.utils.sinkhorn import tilelang

        backward = Mock(wraps=tilelang.sinkhorn_bwd_tilelang)
        monkeypatch.setattr(tilelang, 'sinkhorn_bwd_tilelang', backward)
    else:
        backward = Mock(wraps=ops.sinkhorn_bwd)
        monkeypatch.setattr(ops, 'sinkhorn_bwd', backward)

    x = torch.randn(3, 4, 4, device=device, requires_grad=True)
    out = sinkhorn(logits=x)
    for flag in ('FLA_GLUON', 'FLA_TILELANG'):
        monkeypatch.setenv(flag, '0')
    out.backward(torch.randn_like(out))
    backward.assert_called_once()


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
def test_sinkhorn_invalid_inputs(shape, dtype, num_iters, error):
    with pytest.raises(ValueError, match=error):
        sinkhorn(logits=torch.empty(shape, device=device, dtype=dtype), num_iters=num_iters)
