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
from fla.ops.hc import fused, fused_hc_post, fused_hc_pre, naive_hc_post, naive_hc_pre
from fla.ops.utils.sinkhorn import naive_sinkhorn, sinkhorn
from fla.utils import IS_NVIDIA, assert_close, device

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='NVIDIA CUDA required')

TOLERANCES = {torch.float32: 5e-5, torch.float16: 2e-3, torch.bfloat16: 6e-3}


@pytest.fixture(autouse=True)
def hc_environment(monkeypatch):
    # keep the FP32 reference's einsum backward at full precision
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    for name in ('GLUON', 'TILELANG'):
        monkeypatch.setenv(f'FLA_{name}', '0')


@pytest.fixture
def backend(request, monkeypatch):
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


def _assert_outputs_and_grads(ref, tri, inputs, ratio):
    do = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(outputs=ref, inputs=tuple(inputs.values()), grad_outputs=do)
    tri_grads = torch.autograd.grad(outputs=tri, inputs=tuple(inputs.values()), grad_outputs=do)
    assert tri.shape == ref.shape and tri.dtype == inputs['x'].dtype
    assert_close('o', ref, tri, ratio)
    for (name, tensor), ref_grad, tri_grad in zip(inputs.items(), ref_grads, tri_grads):
        assert tri_grad.dtype == tensor.dtype
        assert_close(f'd{name}', ref_grad, tri_grad, ratio)


def _hc_inputs(kind, shape, dtype):
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(shape[:-1], device=device, dtype=torch.float32, requires_grad=True)
    if kind == 'pre':
        return {'x': x, 'pre': weight}
    inputs = {
        'x': x,
        'branch': torch.randn((*shape[:-2], shape[-1]), device=device, dtype=dtype, requires_grad=True),
        'post': weight,
    }
    if kind == 'mhc':
        inputs['mixing'] = torch.randn((*shape[:-1], shape[-2]), device=device, dtype=torch.float32, requires_grad=True)
    return inputs


@pytest.mark.parametrize(
    ('shape', 'dtype', 'backend'),
    [
        pytest.param((2, 3, 1, 1), torch.float32, 'triton', id='triton-single-stream'),
        pytest.param((2, 3, 3, 65), torch.float16, 'triton', id='triton-fp16-partial-tile'),
        pytest.param((2, 3, 5, 65), torch.bfloat16, 'gluon', id='gluon-bf16-padded-streams'),
        pytest.param((2, 3, 4, 256), torch.float32, 'tilelang', id='tilelang-exact-tile'),
        pytest.param((2, 3, 4, 1025), torch.bfloat16, 'triton', id='triton-bf16-one-feature-tail'),
        pytest.param((2, 3, 8, 4096), torch.float16, 'gluon', id='gluon-fp16-max-streams'),
        pytest.param((2, 3, 4, 8193), torch.bfloat16, 'tilelang', id='tilelang-bf16-many-tiles'),
        pytest.param((4, 65), torch.float16, 'tilelang', id='tilelang-fp16-unbatched'),
        pytest.param((2, 3, 1, 1), torch.float32, 'tilelang', id='tilelang-single-element-tile'),
        pytest.param((2, 3, 3, 65), torch.float16, 'tilelang', id='tilelang-padded-streams-and-features'),
    ],
    indirect=['backend'],
)
def test_fused_hc_pre(shape: tuple[int, ...], dtype: torch.dtype, backend):
    torch.manual_seed(42)
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    pre = torch.randn(shape[:-1], device=device, dtype=torch.float32, requires_grad=True)
    ref = naive_hc_pre(x=x, pre=pre)
    tri = fused_hc_pre(x=x, pre=pre)
    _assert_outputs_and_grads(ref, tri, {'x': x, 'pre': pre}, TOLERANCES[dtype])


@pytest.mark.parametrize(
    ('shape', 'dtype', 'use_mixing', 'backend'),
    [
        pytest.param((2, 3, 1, 1), torch.float32, False, 'triton', id='triton-ihc-single-stream'),
        pytest.param((2, 3, 3, 65), torch.float16, True, 'triton', id='triton-mhc-fp16-partial-tile'),
        pytest.param((2, 3, 5, 65), torch.bfloat16, False, 'gluon', id='gluon-ihc-bf16-padded-streams'),
        pytest.param((2, 3, 4, 256), torch.float32, True, 'tilelang', id='tilelang-mhc-exact-tile'),
        pytest.param((2, 3, 4, 1025), torch.bfloat16, False, 'triton', id='triton-ihc-bf16-one-feature-tail'),
        pytest.param((2, 3, 8, 4096), torch.float16, True, 'gluon', id='gluon-mhc-fp16-max-streams'),
        pytest.param((2, 3, 4, 8193), torch.bfloat16, True, 'tilelang', id='tilelang-mhc-bf16-many-tiles'),
        pytest.param((4, 65), torch.float16, False, 'tilelang', id='tilelang-ihc-fp16-unbatched'),
        pytest.param((2, 3, 5, 65), torch.float32, True, 'triton', id='triton-mhc-padded-streams'),
        pytest.param((2, 3, 4, 8193), torch.float32, False, 'gluon', id='gluon-ihc-many-tiles'),
        pytest.param((2, 3, 8, 4096), torch.bfloat16, False, 'tilelang', id='tilelang-ihc-bf16-max-streams'),
        pytest.param((2, 3, 8, 1), torch.float32, True, 'tilelang', id='tilelang-mhc-max-streams-one-feature'),
        pytest.param((2, 3, 1, 1), torch.float32, True, 'tilelang', id='tilelang-mhc-single-element-tile'),
        pytest.param((2, 3, 5, 65), torch.bfloat16, True, 'tilelang', id='tilelang-mhc-padded-streams-and-features'),
    ],
    indirect=['backend'],
)
def test_fused_hc_post(shape: tuple[int, ...], dtype: torch.dtype, use_mixing: bool, backend):
    torch.manual_seed(42)
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    branch = torch.randn((*shape[:-2], shape[-1]), device=device, dtype=dtype, requires_grad=True)
    post = torch.randn(shape[:-1], device=device, dtype=torch.float32, requires_grad=True)
    inputs = {'x': x, 'branch': branch, 'post': post}
    if use_mixing:
        inputs['mixing'] = torch.randn((*shape[:-1], shape[-2]), device=device, dtype=torch.float32, requires_grad=True)
    ref = naive_hc_post(**inputs)
    tri = fused_hc_post(**inputs)
    _assert_outputs_and_grads(ref, tri, inputs, TOLERANCES[dtype])


@pytest.mark.parametrize(
    ('T', 'N', 'D', 'dtype', 'kind', 'backend'),
    [
        pytest.param(127, 4, 1025, torch.float32, 'pre', 'gluon', id='gluon-pre-127-tokens'),
        pytest.param(128, 4, 1025, torch.float32, 'pre', 'triton', id='triton-pre-128-tokens'),
        pytest.param(127, 4, 1025, torch.float32, 'mhc', 'gluon', id='gluon-mhc-127-tokens'),
        pytest.param(128, 4, 1025, torch.float32, 'ihc', 'triton', id='triton-ihc-128-tokens'),
        pytest.param(127, 8, 1025, torch.bfloat16, 'mhc', 'tilelang', id='tilelang-mhc-bf16-max-streams'),
    ],
    indirect=['backend'],
)
def test_fused_hc_packed(T: int, N: int, D: int, dtype: torch.dtype, kind: str, backend):
    torch.manual_seed(42)
    inputs = _hc_inputs(kind=kind, shape=(T, N, D), dtype=dtype)
    op, reference = (fused_hc_pre, naive_hc_pre) if kind == 'pre' else (fused_hc_post, naive_hc_post)
    ref = reference(**inputs)
    tri = op(**inputs)
    _assert_outputs_and_grads(ref, tri, inputs, TOLERANCES[dtype])


@pytest.mark.parametrize('backend', ['triton', 'gluon', 'tilelang'], indirect=True)
def test_fused_hc_pre_noncontiguous(backend):
    torch.manual_seed(42)
    x = torch.randn(2, 65, 4, device=device).transpose(-1, -2).requires_grad_()
    pre = torch.randn(4, 2, device=device, dtype=torch.bfloat16).T.requires_grad_()
    ref = naive_hc_pre(x=x, pre=pre)
    tri = fused_hc_pre(x=x, pre=pre)
    _assert_outputs_and_grads(ref, tri, {'x': x, 'pre': pre}, 6e-3)


@pytest.mark.parametrize(
    ('use_mixing', 'backend'),
    [
        pytest.param(False, 'triton', id='triton-ihc'),
        pytest.param(True, 'triton', id='triton-mhc'),
        pytest.param(True, 'gluon', id='gluon-mhc'),
        pytest.param(True, 'tilelang', id='tilelang-mhc'),
    ],
    indirect=['backend'],
)
def test_fused_hc_post_noncontiguous(use_mixing: bool, backend):
    torch.manual_seed(42)
    x = torch.randn(2, 65, 4, device=device).transpose(-1, -2).requires_grad_()
    branch = torch.randn(65, 2, device=device, dtype=torch.float16).T.requires_grad_()
    post = torch.randn(4, 2, device=device, dtype=torch.bfloat16).T.requires_grad_()
    inputs = {'x': x, 'branch': branch, 'post': post}
    if use_mixing:
        inputs['mixing'] = torch.randn(2, 4, 4, device=device).transpose(-1, -2).requires_grad_()
    ref = naive_hc_post(**inputs)
    tri = fused_hc_post(**inputs)
    _assert_outputs_and_grads(ref, tri, inputs, 6e-3)


@pytest.mark.parametrize(
    ('use_mixing', 'backend'),
    [
        pytest.param(False, 'triton', id='triton-ihc'),
        pytest.param(True, 'triton', id='triton-mhc'),
        pytest.param(False, 'gluon', id='gluon-ihc'),
        pytest.param(True, 'gluon', id='gluon-mhc'),
        pytest.param(True, 'tilelang', id='tilelang-mhc'),
    ],
    indirect=['backend'],
)
def test_fused_hc_stack(use_mixing: bool, backend):
    torch.manual_seed(42)
    x = torch.randn(2, 4, 127, device=device, requires_grad=True)
    pre = torch.randn(3, 2, 4, device=device, requires_grad=True)
    post = torch.randn(3, 2, 4, device=device, requires_grad=True)
    logits = torch.randn(3, 2, 4, 4, device=device, requires_grad=True) if use_mixing else None
    ref, tri = x, x
    for step in range(3):
        ref_branch = naive_hc_pre(x=ref, pre=pre[step].sigmoid()).tanh()
        tri_branch = fused_hc_pre(x=tri, pre=pre[step].sigmoid()).tanh()
        ref_mixing = naive_sinkhorn(logits=logits[step]) if use_mixing else None
        tri_mixing = sinkhorn(logits=logits[step]) if use_mixing else None
        ref = naive_hc_post(x=ref, branch=ref_branch, post=2 * post[step].sigmoid(), mixing=ref_mixing)
        tri = fused_hc_post(x=tri, branch=tri_branch, post=2 * post[step].sigmoid(), mixing=tri_mixing)
    inputs = {'x': x, 'pre': pre, 'post': post}
    if use_mixing:
        inputs['logits'] = logits
    _assert_outputs_and_grads(ref, tri, inputs, 1e-4)


@pytest.mark.parametrize(
    ('backend', 'kind'),
    [('triton', 'pre'), ('gluon', 'ihc'), ('tilelang', 'mhc')],
    indirect=['backend'],
    ids=['triton-pre', 'gluon-ihc', 'tilelang-mhc'],
)
def test_fused_hc_packed_layout(backend, kind: str):
    torch.manual_seed(42)
    inputs = _hc_inputs(kind=kind, shape=(2, 3, 4, 65), dtype=torch.float32)
    op = fused_hc_pre if kind == 'pre' else fused_hc_post
    with torch.no_grad():
        dense = op(**inputs)
        packed = op(**{name: tensor.flatten(0, 1) for name, tensor in inputs.items()})
    torch.testing.assert_close(dense.flatten(0, 1), packed, rtol=0, atol=0)


@pytest.mark.parametrize(
    ('use_mixing', 'backend'),
    [
        pytest.param(False, 'triton', id='triton-ihc'),
        pytest.param(True, 'triton', id='triton-mhc'),
        pytest.param(True, 'gluon', id='gluon-mhc'),
        pytest.param(False, 'tilelang', id='tilelang-ihc'),
        pytest.param(True, 'tilelang', id='tilelang-mhc'),
    ],
    indirect=['backend'],
)
def test_fused_hc_backward_backend(use_mixing: bool, backend, monkeypatch):
    torch.manual_seed(42)
    if backend == 'gluon':
        from fla.ops.hc import gluon

        backward = Mock(wraps=gluon.hc_bwd_gluon)
        monkeypatch.setattr(gluon, 'hc_bwd_gluon', backward)
    elif backend == 'tilelang':
        from fla.ops.hc import tilelang

        backward = Mock(wraps=tilelang.hc_bwd_tilelang)
        monkeypatch.setattr(tilelang, 'hc_bwd_tilelang', backward)
    else:
        backward = Mock(wraps=fused.hc_bwd)
        monkeypatch.setattr(fused, 'hc_bwd', backward)

    x = torch.randn(3, 4, 65, device=device, requires_grad=True)
    pre = torch.randn(3, 4, device=device, requires_grad=True)
    post = torch.randn_like(pre, requires_grad=True)
    mixing = torch.randn(3, 4, 4, device=device, requires_grad=True) if use_mixing else None
    branch = fused_hc_pre(x=x, pre=pre)
    out = fused_hc_post(x=x, branch=branch, post=post, mixing=mixing)
    monkeypatch.setenv('FLA_GLUON', '0' if backend == 'gluon' else '1')
    monkeypatch.setenv('FLA_TILELANG', '0')
    out.backward(torch.randn_like(out))
    assert [call.kwargs['is_post'] for call in backward.call_args_list] == [True, False]


@pytest.mark.parametrize(
    ('backend_type', 'kind'),
    [
        pytest.param(GluonBackend, 'pre', id='gluon-pre'),
        pytest.param(GluonBackend, 'ihc', id='gluon-ihc'),
        pytest.param(TileLangBackend, 'pre', id='tilelang-pre'),
        pytest.param(TileLangBackend, 'mhc', id='tilelang-mhc'),
    ],
)
def test_fused_hc_unavailable_backend(backend_type, kind: str, monkeypatch):
    if _DISPATCH_DISABLED:
        pytest.skip('Backend dispatch disabled before import')
    torch.manual_seed(42)
    inputs = _hc_inputs(kind=kind, shape=(2, 3, 4, 65), dtype=torch.float32)
    op = fused_hc_pre if kind == 'pre' else fused_hc_post
    ref = op(**inputs)
    do = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(outputs=ref, inputs=tuple(inputs.values()), grad_outputs=do)

    monkeypatch.setenv(f'FLA_{backend_type.backend_type.upper()}', '1')
    monkeypatch.setattr(backend_type, 'is_available', classmethod(lambda cls: False))
    backward = Mock(wraps=fused.hc_bwd)
    monkeypatch.setattr(fused, 'hc_bwd', backward)
    tri = op(**inputs)
    tri_grads = torch.autograd.grad(outputs=tri, inputs=tuple(inputs.values()), grad_outputs=do)
    backward.assert_called_once()
    torch.testing.assert_close(ref, tri, rtol=0, atol=0)
    for ref_grad, tri_grad in zip(ref_grads, tri_grads):
        torch.testing.assert_close(ref_grad, tri_grad, rtol=0, atol=0)


def test_fused_hc_invalid_inputs():
    torch.manual_seed(42)
    x = torch.randn(2, 4, 65, device=device)
    with pytest.raises(ValueError, match='pre must'):
        fused_hc_pre(x=x, pre=torch.randn(4, device=device))
    with pytest.raises(ValueError, match='branch and post'):
        fused_hc_post(x=x, branch=torch.randn(65, device=device), post=torch.randn(2, 4, device=device))
    with pytest.raises(ValueError, match='mixing must'):
        fused_hc_post(x=x, branch=x[:, 0], post=x[:, :, 0], mixing=torch.randn(4, 4, device=device))
