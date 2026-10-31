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


@pytest.fixture
def hc(request):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 4, 65, device=device, requires_grad=True)
    weight = torch.randn(2, 3, 4, device=device, requires_grad=True)
    if request.param == 'pre':
        return fused_hc_pre, {'x': x, 'pre': weight}
    inputs = {
        'x': x,
        'branch': torch.randn(2, 3, 65, device=device, requires_grad=True),
        'post': weight,
    }
    if request.param == 'mhc':
        inputs['mixing'] = torch.randn(2, 3, 4, 4, device=device, requires_grad=True)
    return fused_hc_post, inputs


@pytest.mark.parametrize(
    ('shape', 'dtype', 'ratio', 'backend'),
    [
        pytest.param((2, 3, 1, 1), torch.float32, 5e-5, 'triton', id='triton-single-stream'),
        pytest.param((2, 3, 3, 65), torch.float16, 2e-3, 'triton', id='triton-fp16-partial-tile'),
        pytest.param((2, 3, 5, 65), torch.bfloat16, 6e-3, 'gluon', id='gluon-bf16-padded-streams'),
        pytest.param((2, 3, 4, 256), torch.float32, 5e-5, 'tilelang', id='tilelang-exact-tile'),
        pytest.param((2, 3, 4, 1025), torch.bfloat16, 6e-3, 'triton', id='triton-bf16-one-feature-tail'),
        pytest.param((2, 3, 8, 4096), torch.float16, 2e-3, 'gluon', id='gluon-fp16-max-streams'),
        pytest.param((2, 3, 4, 8193), torch.bfloat16, 6e-3, 'tilelang', id='tilelang-bf16-many-tiles'),
        pytest.param((127, 4, 1025), torch.float32, 5e-5, 'gluon', id='gluon-packed-127-tokens'),
        pytest.param((128, 4, 1025), torch.float32, 5e-5, 'triton', id='triton-packed-128-tokens'),
        pytest.param((4, 65), torch.float16, 2e-3, 'tilelang', id='tilelang-fp16-unbatched'),
    ],
    indirect=['backend'],
)
def test_fused_hc_pre(shape: tuple[int, ...], dtype: torch.dtype, ratio: float, backend):
    torch.manual_seed(42)
    x = torch.randn(shape, device=device, dtype=dtype)
    pre = torch.randn(shape[:-1], device=device, dtype=torch.float32)
    x.requires_grad_()
    pre.requires_grad_()
    ref = naive_hc_pre(x=x, pre=pre)
    tri = fused_hc_pre(x=x, pre=pre)
    do = torch.randn_like(ref)
    ref_dx, ref_dpre = torch.autograd.grad(outputs=ref, inputs=(x, pre), grad_outputs=do)
    tri_dx, tri_dpre = torch.autograd.grad(outputs=tri, inputs=(x, pre), grad_outputs=do)
    assert tri.shape == ref.shape and tri.dtype == dtype
    assert tri_dx.dtype == dtype and tri_dpre.dtype == pre.dtype
    assert_close('o', ref, tri, ratio)
    assert_close('dx', ref_dx, tri_dx, ratio)
    assert_close('dpre', ref_dpre, tri_dpre, ratio)


@pytest.mark.parametrize(
    ('shape', 'dtype', 'ratio', 'use_mixing', 'backend'),
    [
        pytest.param((2, 3, 1, 1), torch.float32, 5e-5, False, 'triton', id='triton-ihc-single-stream'),
        pytest.param((2, 3, 3, 65), torch.float16, 2e-3, True, 'triton', id='triton-mhc-fp16-partial-tile'),
        pytest.param((2, 3, 5, 65), torch.bfloat16, 6e-3, False, 'gluon', id='gluon-ihc-bf16-padded-streams'),
        pytest.param((2, 3, 4, 256), torch.float32, 5e-5, True, 'tilelang', id='tilelang-mhc-exact-tile'),
        pytest.param((2, 3, 4, 1025), torch.bfloat16, 6e-3, False, 'triton', id='triton-ihc-bf16-one-feature-tail'),
        pytest.param((2, 3, 8, 4096), torch.float16, 2e-3, True, 'gluon', id='gluon-mhc-fp16-max-streams'),
        pytest.param((2, 3, 4, 8193), torch.bfloat16, 6e-3, True, 'tilelang', id='tilelang-mhc-bf16-many-tiles'),
        pytest.param((127, 4, 1025), torch.float32, 5e-5, True, 'gluon', id='gluon-mhc-packed-127-tokens'),
        pytest.param((128, 4, 1025), torch.float32, 5e-5, False, 'triton', id='triton-ihc-packed-128-tokens'),
        pytest.param((4, 65), torch.float16, 2e-3, False, 'tilelang', id='tilelang-ihc-fp16-unbatched'),
        pytest.param((2, 3, 5, 65), torch.float32, 5e-5, True, 'triton', id='triton-mhc-padded-streams'),
        pytest.param((2, 3, 4, 8193), torch.float32, 5e-5, False, 'gluon', id='gluon-ihc-many-tiles'),
        pytest.param((2, 3, 8, 4096), torch.bfloat16, 6e-3, False, 'tilelang', id='tilelang-ihc-bf16-max-streams'),
    ],
    indirect=['backend'],
)
def test_fused_hc_post(shape: tuple[int, ...], dtype: torch.dtype, ratio: float, use_mixing: bool, backend):
    torch.manual_seed(42)
    x = torch.randn(shape, device=device, dtype=dtype)
    branch = torch.randn((*shape[:-2], shape[-1]), device=device, dtype=dtype)
    post = torch.randn(shape[:-1], device=device, dtype=torch.float32)
    mixing = torch.randn((*shape[:-1], shape[-2]), device=device, dtype=torch.float32) if use_mixing else None
    inputs = (x, branch, post, mixing) if use_mixing else (x, branch, post)
    for tensor in inputs:
        tensor.requires_grad_()
    ref = naive_hc_post(x=x, branch=branch, post=post, mixing=mixing)
    tri = fused_hc_post(x=x, branch=branch, post=post, mixing=mixing)
    do = torch.randn_like(ref)
    ref_dx, ref_dbranch, ref_dpost, *ref_dmixing = torch.autograd.grad(outputs=ref, inputs=inputs, grad_outputs=do)
    tri_dx, tri_dbranch, tri_dpost, *tri_dmixing = torch.autograd.grad(outputs=tri, inputs=inputs, grad_outputs=do)
    assert tri.shape == ref.shape and tri.dtype == dtype
    assert tri_dx.dtype == dtype and tri_dbranch.dtype == branch.dtype and tri_dpost.dtype == post.dtype
    assert_close('o', ref, tri, ratio)
    assert_close('dx', ref_dx, tri_dx, ratio)
    assert_close('dbranch', ref_dbranch, tri_dbranch, ratio)
    assert_close('dpost', ref_dpost, tri_dpost, ratio)
    if use_mixing:
        assert tri_dmixing[0].dtype == mixing.dtype
        assert_close('dmixing', ref_dmixing[0], tri_dmixing[0], ratio)


@pytest.mark.parametrize('backend', ['triton', 'gluon', 'tilelang'], indirect=True)
def test_fused_hc_pre_noncontiguous(backend):
    torch.manual_seed(42)
    x = torch.randn(2, 65, 4, device=device).transpose(-1, -2).requires_grad_()
    pre = torch.randn(4, 2, device=device, dtype=torch.bfloat16).T.requires_grad_()
    ref = naive_hc_pre(x=x, pre=pre)
    tri = fused_hc_pre(x=x, pre=pre)
    do = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(outputs=ref, inputs=(x, pre), grad_outputs=do)
    tri_grads = torch.autograd.grad(outputs=tri, inputs=(x, pre), grad_outputs=do)
    assert tri.shape == ref.shape and tri.dtype == x.dtype
    assert_close('o', ref, tri, 6e-3)
    for name, tensor, ref_grad, tri_grad in zip(('dx', 'dpre'), (x, pre), ref_grads, tri_grads):
        assert tri_grad.dtype == tensor.dtype
        assert_close(name, ref_grad, tri_grad, 6e-3)


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
    mixing = torch.randn(2, 4, 4, device=device).transpose(-1, -2).requires_grad_() if use_mixing else None
    inputs = (x, branch, post, mixing) if use_mixing else (x, branch, post)
    ref = naive_hc_post(x=x, branch=branch, post=post, mixing=mixing)
    tri = fused_hc_post(x=x, branch=branch, post=post, mixing=mixing)
    do = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(outputs=ref, inputs=inputs, grad_outputs=do)
    tri_grads = torch.autograd.grad(outputs=tri, inputs=inputs, grad_outputs=do)
    assert tri.shape == ref.shape and tri.dtype == x.dtype
    assert_close('o', ref, tri, 6e-3)
    for name, tensor, ref_grad, tri_grad in zip(('dx', 'dbranch', 'dpost', 'dmixing'), inputs, ref_grads, tri_grads):
        assert tri_grad.dtype == tensor.dtype
        assert_close(name, ref_grad, tri_grad, 6e-3)


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
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for step in range(3):
            ref_branch = naive_hc_pre(x=ref, pre=pre[step].sigmoid()).tanh()
            tri_branch = fused_hc_pre(x=tri, pre=pre[step].sigmoid()).tanh()
            ref_mixing = naive_sinkhorn(logits=logits[step]) if use_mixing else None
            tri_mixing = sinkhorn(logits=logits[step]) if use_mixing else None
            ref = naive_hc_post(x=ref, branch=ref_branch, post=2 * post[step].sigmoid(), mixing=ref_mixing)
            tri = fused_hc_post(x=tri, branch=tri_branch, post=2 * post[step].sigmoid(), mixing=tri_mixing)
    inputs = (x, pre, post, logits) if use_mixing else (x, pre, post)
    do = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(outputs=ref, inputs=inputs, grad_outputs=do)
    tri_grads = torch.autograd.grad(outputs=tri, inputs=inputs, grad_outputs=do)
    assert_close('o', ref, tri, 1e-4)
    for name, ref_grad, tri_grad in zip(('dx', 'dpre', 'dpost', 'dlogits'), ref_grads, tri_grads):
        assert_close(name, ref_grad, tri_grad, 1e-4)


@pytest.mark.parametrize(
    ('backend', 'hc'),
    [('triton', 'pre'), ('gluon', 'ihc'), ('tilelang', 'mhc')],
    indirect=True,
    ids=['triton-pre', 'gluon-ihc', 'tilelang-mhc'],
)
def test_fused_hc_packed(backend, hc):
    op, inputs = hc
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
    ('backend_type', 'hc'),
    [
        pytest.param(GluonBackend, 'pre', id='gluon-pre'),
        pytest.param(GluonBackend, 'ihc', id='gluon-ihc'),
        pytest.param(TileLangBackend, 'pre', id='tilelang-pre'),
        pytest.param(TileLangBackend, 'mhc', id='tilelang-mhc'),
    ],
    indirect=['hc'],
)
def test_fused_hc_unavailable_backend(backend_type, hc, monkeypatch):
    if _DISPATCH_DISABLED:
        pytest.skip('Backend dispatch disabled before import')
    op, inputs = hc
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
