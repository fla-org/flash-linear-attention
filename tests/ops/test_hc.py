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
from fla.ops.hc import fused, fused_hc_post, fused_hc_pre, naive_hc_post, naive_hc_pre, naive_sinkhorn, sinkhorn
from fla.utils import IS_NVIDIA, assert_close, device


@pytest.fixture(params=['triton', 'gluon', 'tilelang'])
def backend(request, monkeypatch):
    if not IS_NVIDIA:
        pytest.skip('NVIDIA CUDA required')
    for name in ('GLUON', 'TILELANG'):
        monkeypatch.setenv(f'FLA_{name}', '0')
        monkeypatch.setenv(f'FLA_HC_{name}', '0')
    name = request.param
    if name == 'triton':
        return fused
    if _DISPATCH_DISABLED:
        pytest.skip('Backend dispatch disabled before import')
    backend_type = GluonBackend if name == 'gluon' else TileLangBackend
    if not backend_type.is_available():
        pytest.skip(f'{name} is unavailable')
    monkeypatch.setenv(f'FLA_HC_{name.upper()}', '1')
    return importlib.import_module(f'fla.ops.hc.backends.{name}')


_HC_CASES = [
    pytest.param((2, 3, N, D), dtype, False, ratio, id=f'N{N}-D{D}-{dtype}')
    for N, D in [(1, 1), (3, 65), (5, 65), (4, 256), (4, 1025), (8, 4096), (4, 8193)]
    for dtype, ratio in [(torch.float32, 5e-5), (torch.float16, 2e-3), (torch.bfloat16, 6e-3)]
] + [
    pytest.param((T, 4, 1025), torch.float32, False, 5e-5, id=f'T{T}') for T in (127, 128)
] + [
    pytest.param((4, 65), torch.float32, False, 5e-5, id='unbatched'),
    pytest.param((2, 4, 65), torch.float32, True, 6e-3, id='strided-mixed-dtypes'),
    pytest.param((2, 0, 4, 65), torch.float32, False, 5e-5, id='empty'),
]


@pytest.mark.parametrize('shape,dtype,strided,ratio', _HC_CASES)
def test_hc_pre(backend, shape, dtype, strided, ratio):
    torch.manual_seed(42)
    x = torch.randn(shape, device=device, dtype=dtype)
    weight_dtype = torch.bfloat16 if strided else torch.float32
    pre = torch.randn(shape[:-1], device=device, dtype=weight_dtype)
    if strided:
        x = x.transpose(-1, -2).contiguous().transpose(-1, -2)
        pre = pre.T.contiguous().T
    x.requires_grad_()
    pre.requires_grad_()
    ref = naive_hc_pre(x=x, pre=pre)
    tri = fused_hc_pre(x=x, pre=pre)
    do = torch.randn_like(ref)
    ref_dx, ref_dpre = torch.autograd.grad(outputs=ref, inputs=(x, pre), grad_outputs=do)
    tri_dx, tri_dpre = torch.autograd.grad(outputs=tri, inputs=(x, pre), grad_outputs=do)
    assert tri.shape == ref.shape and tri.dtype == dtype
    assert tri_dx.dtype == dtype and tri_dpre.dtype == weight_dtype
    if x.numel() == 0:
        torch.testing.assert_close(ref, tri)
        torch.testing.assert_close(ref_dx, tri_dx)
        torch.testing.assert_close(ref_dpre, tri_dpre)
        return
    assert_close('o', ref, tri, ratio)
    assert_close('dx', ref_dx, tri_dx, ratio)
    assert_close('dpre', ref_dpre, tri_dpre, ratio)


@pytest.mark.parametrize('shape,dtype,strided,ratio', _HC_CASES)
@pytest.mark.parametrize('identity', [True, False], ids=['ihc', 'mhc'])
def test_hc_post(backend, shape, dtype, strided, ratio, identity):
    torch.manual_seed(42)
    x = torch.randn(shape, device=device, dtype=dtype)
    weight_dtype = torch.bfloat16 if strided else torch.float32
    branch_dtype = torch.float16 if strided else dtype
    post = torch.randn(shape[:-1], device=device, dtype=weight_dtype)
    branch = torch.randn((*shape[:-2], shape[-1]), device=device, dtype=branch_dtype)
    mixing = None if identity else torch.randn((*shape[:-1], shape[-2]), device=device, dtype=torch.float32)
    if strided:
        x = x.transpose(-1, -2).contiguous().transpose(-1, -2)
        post = post.T.contiguous().T
        branch = branch.T.contiguous().T
        if mixing is not None:
            mixing = mixing.transpose(-1, -2)
    inputs = (x, branch, post) if identity else (x, branch, post, mixing)
    for tensor in inputs:
        tensor.requires_grad_()
    ref = naive_hc_post(x=x, branch=branch, post=post, mixing=mixing)
    tri = fused_hc_post(x=x, branch=branch, post=post, mixing=mixing)
    do = torch.randn_like(ref)
    ref_dx, ref_dbranch, ref_dpost, *ref_dmixing = torch.autograd.grad(outputs=ref, inputs=inputs, grad_outputs=do)
    tri_dx, tri_dbranch, tri_dpost, *tri_dmixing = torch.autograd.grad(outputs=tri, inputs=inputs, grad_outputs=do)
    assert tri.shape == ref.shape and tri.dtype == dtype
    assert tri_dx.dtype == dtype and tri_dbranch.dtype == branch_dtype and tri_dpost.dtype == weight_dtype
    if x.numel() == 0:
        for expected, actual in zip((ref, ref_dx, ref_dbranch, ref_dpost, *ref_dmixing),
                                    (tri, tri_dx, tri_dbranch, tri_dpost, *tri_dmixing)):
            torch.testing.assert_close(expected, actual)
        return
    assert_close('o', ref, tri, ratio)
    assert_close('dx', ref_dx, tri_dx, ratio)
    assert_close('dbranch', ref_dbranch, tri_dbranch, ratio)
    assert_close('dpost', ref_dpost, tri_dpost, ratio)
    if not identity:
        assert tri_dmixing[0].dtype == mixing.dtype
        assert_close('dmixing', ref_dmixing[0], tri_dmixing[0], ratio)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape,num_iters,scale,strided', [
    ((2, 7, 1, 1), 20, 1, False),
    ((2, 7, 3, 3), 1, 1, False),
    ((2, 7, 4, 4), 20, 1, False),
    ((2, 7, 8, 8), 7, 1, False),
    ((2, 7, 4, 4), 20, 1000, False),
    ((127, 4, 4), 20, 1, False),
    ((128, 4, 4), 20, 1, False),
    ((129, 1, 1), 20, 1, False),
    ((129, 3, 3), 20, 1, False),
    ((129, 8, 8), 20, 1, False),
    ((4, 4), 20, 1, False),
    ((2, 4, 4), 20, 1, True),
    ((2, 0, 4, 4), 20, 1, False),
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


@pytest.mark.parametrize('identity', [True, False], ids=['ihc', 'mhc'])
def test_hc_stack(backend, identity):
    torch.manual_seed(42)
    x = torch.randn(2, 4, 127, device=device, requires_grad=True)
    pre = torch.randn(3, 2, 4, device=device, requires_grad=True)
    post = torch.randn(3, 2, 4, device=device, requires_grad=True)
    logits = None if identity else torch.randn(3, 2, 4, 4, device=device, requires_grad=True)
    ref, tri = x, x
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for step in range(3):
            ref_branch = naive_hc_pre(x=ref, pre=pre[step].sigmoid()).tanh()
            tri_branch = fused_hc_pre(x=tri, pre=pre[step].sigmoid()).tanh()
            ref_mixing = None if identity else naive_sinkhorn(logits=logits[step])
            tri_mixing = None if identity else sinkhorn(logits=logits[step])
            ref = naive_hc_post(x=ref, branch=ref_branch, post=2 * post[step].sigmoid(), mixing=ref_mixing)
            tri = fused_hc_post(x=tri, branch=tri_branch, post=2 * post[step].sigmoid(), mixing=tri_mixing)
    inputs = (x, pre, post) if identity else (x, pre, post, logits)
    do = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(outputs=ref, inputs=inputs, grad_outputs=do)
    tri_grads = torch.autograd.grad(outputs=tri, inputs=inputs, grad_outputs=do)
    assert_close('o', ref, tri, 1e-4)
    for name, ref_grad, tri_grad in zip(('dx', 'dpre', 'dpost', 'dlogits'), ref_grads, tri_grads):
        assert_close(name, ref_grad, tri_grad, 1e-4)


@pytest.mark.parametrize('op,identity', [(fused_hc_pre, True), (fused_hc_post, True), (fused_hc_post, False), (sinkhorn, True)])
def test_hc_packed(backend, op, identity):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 4, 65, device=device)
    if op is fused_hc_pre:
        inputs = {'x': x, 'pre': torch.randn(2, 3, 4, device=device)}
    elif op is fused_hc_post:
        inputs = {'x': x, 'branch': torch.randn(2, 3, 65, device=device), 'post': torch.randn(2, 3, 4, device=device)}
        if not identity:
            inputs['mixing'] = torch.randn(2, 3, 4, 4, device=device)
    else:
        inputs = {'logits': torch.randn(2, 3, 4, 4, device=device)}
    with torch.no_grad():
        dense = op(**inputs)
        packed = op(**{name: tensor.flatten(0, 1) for name, tensor in inputs.items()})
    torch.testing.assert_close(dense.flatten(0, 1), packed, rtol=0, atol=0)


@pytest.mark.parametrize('op', [fused_hc_pre, fused_hc_post, sinkhorn])
def test_hc_backend(backend, monkeypatch, op):
    calls = []
    prefix = 'sinkhorn' if op is sinkhorn else 'hc'
    forward, backward = getattr(backend, f'{prefix}_fwd'), getattr(backend, f'{prefix}_bwd')

    def fwd(*args, **kwargs):
        calls.append('fwd')
        return forward(*args, **kwargs)

    def bwd(*args, **kwargs):
        calls.append('bwd')
        return backward(*args, **kwargs)

    monkeypatch.setattr(backend, f'{prefix}_fwd', fwd)
    monkeypatch.setattr(backend, f'{prefix}_bwd', bwd)
    x = torch.randn(3, 4, 65, device=device, requires_grad=True)
    weight = torch.randn(3, 4, device=device, requires_grad=True)
    if op is fused_hc_pre:
        out = op(x=x, pre=weight)
    elif op is fused_hc_post:
        out = op(x=x, branch=x[:, 0], post=weight)
    else:
        out = op(logits=torch.randn(3, 4, 4, device=device, requires_grad=True))
    for name in ('GLUON', 'TILELANG'):
        monkeypatch.setenv(f'FLA_HC_{name}', '0')
    out.sum().backward()
    assert calls == ['fwd', 'bwd']


@pytest.mark.parametrize('backend_type', [GluonBackend, TileLangBackend])
def test_hc_unavailable_backend(monkeypatch, backend_type):
    if _DISPATCH_DISABLED:
        pytest.skip('Backend dispatch disabled before import')
    for name in ('GLUON', 'TILELANG'):
        monkeypatch.setenv(f'FLA_{name}', '0')
        monkeypatch.setenv(f'FLA_HC_{name}', '0')
    monkeypatch.setenv(f'FLA_HC_{backend_type.backend_type.upper()}', '1')
    monkeypatch.setattr(backend_type, 'is_available', classmethod(lambda cls: False))
    sentinel = object()
    monkeypatch.setattr(fused, '_validate_hc_pre', lambda **kwargs: None)
    monkeypatch.setattr(fused.FusedHCFunction, 'apply', lambda *args: sentinel)
    assert fused_hc_pre(x=None, pre=None) is sentinel


def test_hc_invalid_inputs(backend):
    x = torch.randn(2, 4, 65, device=device)
    with pytest.raises(ValueError, match='pre must'):
        fused_hc_pre(x=x, pre=torch.randn(4, device=device))
    with pytest.raises(ValueError, match='branch and post'):
        fused_hc_post(x=x, branch=torch.randn(65, device=device), post=torch.randn(2, 4, device=device))
    with pytest.raises(ValueError, match='mixing must'):
        fused_hc_post(x=x, branch=x[:, 0], post=x[:, :, 0], mixing=torch.randn(4, 4, device=device))
    with pytest.raises(ValueError, match='positive integer'):
        sinkhorn(logits=torch.randn(4, 4, device=device), num_iters=0)
    with pytest.raises(ValueError, match='shape'):
        sinkhorn(logits=torch.randn(4, 3, device=device))
    with pytest.raises(ValueError, match='FP16, BF16, or FP32'):
        sinkhorn(logits=torch.randn(4, 4, device=device, dtype=torch.float64))
