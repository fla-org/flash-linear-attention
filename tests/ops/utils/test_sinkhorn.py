# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib

import pytest
import torch

from fla.backends import _DISPATCH_DISABLED, GluonBackend, TileLangBackend
from fla.ops.utils.sinkhorn import naive_sinkhorn, sinkhorn
from fla.utils import IS_NVIDIA, assert_close, device


@pytest.fixture(params=['triton', 'gluon', 'tilelang'])
def backend(request, monkeypatch):
    if not IS_NVIDIA:
        pytest.skip('NVIDIA CUDA required')
    for name in ('GLUON', 'TILELANG'):
        monkeypatch.setenv(f'FLA_{name}', '0')
    name = request.param
    if name == 'triton':
        return importlib.import_module('fla.ops.utils.sinkhorn')
    if _DISPATCH_DISABLED:
        pytest.skip('Backend dispatch disabled before import')
    backend_type = GluonBackend if name == 'gluon' else TileLangBackend
    if not backend_type.is_available():
        pytest.skip(f'{name} is unavailable')
    monkeypatch.setenv(f'FLA_{name.upper()}', '1')
    return importlib.import_module(f'fla.ops.utils.{name}.sinkhorn')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(('shape', 'num_iters', 'scale', 'strided'), [
    pytest.param((2, 7, 1, 1), 20, 1, False, id='N1'),
    pytest.param((2, 7, 3, 3), 1, 1, False, id='N3-iter1'),
    pytest.param((2, 7, 4, 4), 20, 1, False, id='N4'),
    pytest.param((2, 7, 8, 8), 7, 1, False, id='N8-iter7'),
    pytest.param((2, 7, 4, 4), 20, 1000, False, id='large-logits'),
    pytest.param((127, 4, 4), 20, 1, False, id='T127-N4'),
    pytest.param((128, 4, 4), 20, 1, False, id='T128-N4'),
    pytest.param((129, 1, 1), 20, 1, False, id='T129-N1'),
    pytest.param((129, 3, 3), 20, 1, False, id='T129-N3'),
    pytest.param((129, 8, 8), 20, 1, False, id='T129-N8'),
    pytest.param((4, 4), 20, 1, False, id='unbatched'),
    pytest.param((2, 4, 4), 20, 1, True, id='strided'),
    pytest.param((2, 0, 4, 4), 20, 1, False, id='empty'),
])
def test_sinkhorn(backend, dtype, shape, num_iters, scale, strided):
    torch.manual_seed(42)
    logits = torch.randn(shape, device=device, dtype=dtype) * scale
    if strided:
        logits = logits.transpose(-1, -2)
    logits.requires_grad_()
    ref = naive_sinkhorn(logits=logits, num_iters=num_iters)
    tri = sinkhorn(logits=logits, num_iters=num_iters)
    do = torch.randn_like(ref)
    ref_dx, = torch.autograd.grad(outputs=ref, inputs=logits, grad_outputs=do)
    tri_dx, = torch.autograd.grad(outputs=tri, inputs=logits, grad_outputs=do)
    assert tri.shape == ref.shape and tri.dtype == torch.float32
    assert tri_dx.dtype == dtype
    if logits.numel() == 0:
        torch.testing.assert_close(ref, tri)
        torch.testing.assert_close(ref_dx, tri_dx)
        return
    assert_close('o', ref, tri, 1e-4)
    assert_close('dx', ref_dx, tri_dx, 1e-4)
    torch.testing.assert_close(tri.sum(-1), torch.ones_like(tri[..., 0]), rtol=1e-5, atol=1e-5)


def test_sinkhorn_packed(backend):
    torch.manual_seed(42)
    logits = torch.randn(2, 3, 4, 4, device=device)
    with torch.no_grad():
        dense = sinkhorn(logits=logits)
        packed = sinkhorn(logits=logits.flatten(0, 1))
    torch.testing.assert_close(dense.flatten(0, 1), packed, rtol=0, atol=0)


def test_sinkhorn_backend(backend, monkeypatch):
    module = importlib.import_module('fla.ops.utils.sinkhorn')
    calls = []
    forward = module.sinkhorn_fwd

    def fwd(*args, **kwargs):
        calls.append('fwd')
        out, history, backward = forward(*args, **kwargs)
        assert backward.__module__ == backend.__name__

        def bwd(*args, **kwargs):
            calls.append('bwd')
            return backward(*args, **kwargs)

        return out, history, bwd

    monkeypatch.setattr(module, 'sinkhorn_fwd', fwd)
    logits = torch.randn(3, 4, 4, device=device, requires_grad=True)
    out = sinkhorn(logits=logits)
    for name in ('GLUON', 'TILELANG'):
        monkeypatch.setenv(f'FLA_{name}', '0')
    out.sum().backward()
    assert calls == ['fwd', 'bwd']


@pytest.mark.parametrize('backend_type', [GluonBackend, TileLangBackend])
def test_sinkhorn_unavailable_backend(monkeypatch, backend_type):
    if not IS_NVIDIA:
        pytest.skip('NVIDIA CUDA required')
    if _DISPATCH_DISABLED:
        pytest.skip('Backend dispatch disabled before import')
    for name in ('GLUON', 'TILELANG'):
        monkeypatch.setenv(f'FLA_{name}', '0')
    monkeypatch.setenv(f'FLA_{backend_type.backend_type.upper()}', '1')
    monkeypatch.setattr(backend_type, 'is_available', classmethod(lambda cls: False))

    def implementation(*args, **kwargs):
        pytest.fail('Unavailable backend was selected')

    monkeypatch.setattr(backend_type, 'get_implementation', implementation)
    logits = torch.zeros(1, 1, device=device)
    torch.testing.assert_close(sinkhorn(logits=logits), torch.ones_like(logits))


@pytest.mark.skipif(not IS_NVIDIA, reason='NVIDIA CUDA required')
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
    module = importlib.import_module('fla.ops.utils.sinkhorn')

    def forward(*args, **kwargs):
        pytest.fail('Invalid inputs reached backend dispatch')

    monkeypatch.setattr(module, 'sinkhorn_fwd', forward)
    with pytest.raises(ValueError, match=error):
        sinkhorn(logits=torch.empty(shape, device=device, dtype=dtype), num_iters=num_iters)
