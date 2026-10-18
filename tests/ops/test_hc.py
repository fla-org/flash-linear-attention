# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.backends import _DISPATCH_DISABLED
from fla.ops.hc import (
    fused,
    hc_post,
    hc_pre,
    naive_hc_post,
    naive_hc_pre,
    naive_sinkhorn,
    sinkhorn,
)
from fla.ops.hc.backends import GluonBackend, HCTileLangBackend
from fla.utils import IS_NVIDIA, assert_close


@pytest.fixture(params=['triton', 'gluon', 'tilelang'])
def backend(request, monkeypatch):
    if not IS_NVIDIA:
        pytest.skip('NVIDIA CUDA required')
    for name in ('GLUON', 'TILELANG'):
        monkeypatch.setenv(f'FLA_{name}', '0')
        monkeypatch.setenv(f'FLA_HC_{name}', '0')
    name = request.param
    if name != 'triton':
        if _DISPATCH_DISABLED:
            pytest.skip('Backend dispatch disabled before import')
        cls = GluonBackend if name == 'gluon' else HCTileLangBackend
        if not cls.is_available():
            pytest.skip(f'{name} is unavailable')
        monkeypatch.setenv(f'FLA_HC_{name.upper()}', '1')
    original = fused._launch_kernel
    calls = []

    def launch(backend, **kwargs):
        assert backend == name
        calls.append(backend)
        return original(backend=backend, **kwargs)

    monkeypatch.setattr(fused, '_launch_kernel', launch)
    return calls


def _check(fn, reference, inputs, ratio):
    expected = reference(*inputs)
    actual = fn(*inputs)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert_close(prefix='output', ref=expected, tri=actual, ratio=ratio)
    dout = torch.randn_like(expected)
    ref_grads = torch.autograd.grad(outputs=expected, inputs=inputs, grad_outputs=dout)
    grads = torch.autograd.grad(outputs=actual, inputs=inputs, grad_outputs=dout)
    for index, (ref, grad) in enumerate(zip(ref_grads, grads)):
        assert ref.dtype == grad.dtype
        assert_close(prefix=f'gradient {index}', ref=ref, tri=grad, ratio=ratio)


@pytest.mark.parametrize('shape', [(1, 1), (3, 65), (4, 256), (4, 1025), (8, 4096), (4, 8193)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('kind', ['pre', 'ihc', 'mhc'])
def test_hc(backend, shape, dtype, kind):
    torch.manual_seed(42)
    N, D = shape
    x = torch.randn(2, 3, N, D, device='cuda', dtype=dtype, requires_grad=True)
    weight = torch.randn(2, 3, N, device='cuda', dtype=torch.float32, requires_grad=True)
    ratio = {torch.float32: 5e-5, torch.float16: 2e-3, torch.bfloat16: 6e-3}[dtype]
    if kind == 'pre':
        _check(fn=hc_pre, reference=naive_hc_pre, inputs=(x, weight), ratio=ratio)
    else:
        branch = torch.randn(2, 3, D, device='cuda', dtype=dtype, requires_grad=True)
        args = (x, branch, weight)
        if kind == 'mhc':
            mixing = torch.randn(2, 3, N, N, device='cuda', dtype=torch.float32, requires_grad=True)
            args += (mixing,)
        _check(fn=hc_post, reference=naive_hc_post, inputs=args, ratio=ratio)
    assert backend


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('N,num_iters,scale', [(1, 20, 1), (3, 1, 1), (4, 20, 1), (8, 7, 1), (4, 20, 1000)])
def test_sinkhorn(backend, dtype, N, num_iters, scale):
    torch.manual_seed(9)
    x = (torch.randn(2, 7, N, N, device='cuda', dtype=dtype) * scale).requires_grad_()
    _check(
        fn=lambda x: sinkhorn(logits=x, num_iters=num_iters),
        reference=lambda x: naive_sinkhorn(logits=x, num_iters=num_iters),
        inputs=(x,),
        ratio=1e-4,
    )
    out = sinkhorn(logits=x, num_iters=num_iters)
    torch.testing.assert_close(out.sum(-1), torch.ones_like(out[..., 0]), rtol=1e-5, atol=1e-5)
    assert backend


@pytest.mark.parametrize('kind', ['pre', 'ihc', 'mhc', 'sinkhorn'])
def test_noncontiguous(backend, kind):
    torch.manual_seed(5)
    x = torch.randn(2, 65, 4, device='cuda', requires_grad=True).transpose(-1, -2)
    weight = torch.randn(4, 2, device='cuda', dtype=torch.bfloat16, requires_grad=True).T
    branch = torch.randn(65, 2, device='cuda', dtype=torch.float16, requires_grad=True).T
    mixing = torch.randn(2, 4, 4, device='cuda', requires_grad=True).transpose(-1, -2)
    if kind == 'pre':
        _check(fn=hc_pre, reference=naive_hc_pre, inputs=(x, weight), ratio=6e-3)
    elif kind == 'sinkhorn':
        _check(fn=sinkhorn, reference=naive_sinkhorn, inputs=(mixing,), ratio=1e-4)
    else:
        args = (x, branch, weight, mixing) if kind == 'mhc' else (x, branch, weight)
        _check(fn=hc_post, reference=naive_hc_post, inputs=args, ratio=6e-3)
    assert backend


@pytest.mark.parametrize('kind', ['pre', 'ihc', 'mhc', 'sinkhorn'])
def test_empty(backend, kind):
    x = torch.empty(2, 0, 4, 65, device='cuda', requires_grad=True)
    weight = torch.empty(2, 0, 4, device='cuda', requires_grad=True)
    branch = torch.empty(2, 0, 65, device='cuda', requires_grad=True)
    mixing = torch.empty(2, 0, 4, 4, device='cuda', requires_grad=True)
    if kind == 'pre':
        out, args = hc_pre(x=x, pre=weight), (x, weight)
    elif kind == 'sinkhorn':
        out, args = sinkhorn(logits=mixing), (mixing,)
    else:
        args = (x, branch, weight, mixing) if kind == 'mhc' else (x, branch, weight)
        out = hc_post(*args)
    assert out.numel() == 0
    assert all(grad.numel() == 0 for grad in torch.autograd.grad(outputs=out.sum(), inputs=args))
    assert not backend


@pytest.mark.parametrize('identity', [False, True])
def test_chained(backend, identity):
    torch.manual_seed(7)
    inputs = (
        torch.randn(2, 4, 127, device='cuda', requires_grad=True),
        torch.randn(3, 2, 4, device='cuda', requires_grad=True),
        torch.randn(3, 2, 4, device='cuda', requires_grad=True),
    )
    if not identity:
        inputs += (torch.randn(3, 2, 4, 4, device='cuda', requires_grad=True),)

    def chain(x, pre, post, logits=None, naive=False):
        read, write, normalize = (
            (naive_hc_pre, naive_hc_post, naive_sinkhorn) if naive else
            (hc_pre, hc_post, sinkhorn)
        )
        for step in range(3):
            branch = torch.tanh(read(x, pre[step].sigmoid()))
            mixing = None if logits is None else normalize(logits[step])
            x = write(x, branch, post[step].sigmoid() * 2, mixing)
        return x

    with torch.autocast('cuda', dtype=torch.bfloat16):
        _check(fn=chain, reference=lambda *args: chain(*args, naive=True), inputs=inputs, ratio=1e-4)
    assert backend


def test_naive_gradcheck():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    branch = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    logits = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(func=naive_hc_pre, inputs=(x, weight))
    assert torch.autograd.gradcheck(func=naive_hc_post, inputs=(x, branch, weight))
    assert torch.autograd.gradcheck(func=naive_hc_post, inputs=(x, branch, weight, logits))
    assert torch.autograd.gradcheck(func=naive_sinkhorn, inputs=(logits,))


@pytest.mark.parametrize('backend_type', [GluonBackend, HCTileLangBackend])
def test_unavailable_backend(monkeypatch, backend_type):
    if _DISPATCH_DISABLED:
        pytest.skip('Backend dispatch disabled before import')
    monkeypatch.setenv('FLA_GLUON', '0')
    monkeypatch.setenv('FLA_TILELANG', '0')
    monkeypatch.setenv('FLA_HC_GLUON', '0')
    monkeypatch.setenv('FLA_HC_TILELANG', '0')
    monkeypatch.setenv(f'FLA_HC_{backend_type.backend_type.upper()}', '1')
    monkeypatch.setattr(backend_type, 'is_available', classmethod(lambda cls: False))
    sentinel = object()
    monkeypatch.setattr(fused, '_hc_pre', lambda **kwargs: sentinel)
    assert hc_pre(x=None, pre=None) is sentinel


def test_invalid_inputs(backend):
    x = torch.randn(2, 4, 65, device='cuda')
    with pytest.raises(ValueError, match='pre must'):
        hc_pre(x=x, pre=torch.randn(4, device='cuda'))
    with pytest.raises(ValueError, match='branch and post'):
        hc_post(x=x, branch=torch.randn(65, device='cuda'), post=torch.randn(2, 4, device='cuda'))
    with pytest.raises(ValueError, match='mixing must'):
        hc_post(x=x, branch=x[:, 0], post=x[:, :, 0], mixing=torch.randn(4, 4, device='cuda'))
    with pytest.raises(ValueError, match='positive integer'):
        sinkhorn(logits=torch.randn(4, 4, device='cuda'), num_iters=0)
    with pytest.raises(ValueError, match='shape'):
        sinkhorn(logits=torch.randn(4, 3, device='cuda'))
    with pytest.raises(ValueError, match='FP16, BF16, or FP32'):
        sinkhorn(logits=torch.randn(4, 4, device='cuda', dtype=torch.float64))
    assert not backend


@pytest.mark.parametrize('kind', ['pre', 'ihc', 'mhc', 'sinkhorn'])
def test_no_grad_and_packed(backend, kind):
    x = torch.randn(2, 3, 4, 65, device='cuda')
    weight = torch.randn(2, 3, 4, device='cuda')
    branch = torch.randn(2, 3, 65, device='cuda')
    mixing = torch.randn(2, 3, 4, 4, device='cuda')
    if kind == 'pre':
        fn, args = hc_pre, (x, weight)
    elif kind == 'sinkhorn':
        fn, args = sinkhorn, (mixing,)
    else:
        fn = hc_post
        args = (x, branch, weight, mixing) if kind == 'mhc' else (x, branch, weight)
    with torch.no_grad():
        dense = fn(*args)
        packed = fn(*(arg.flatten(0, 1) for arg in args))
    torch.testing.assert_close(dense.flatten(0, 1), packed, rtol=0, atol=0)
    assert backend


def test_backward_keeps_backend(backend, monkeypatch):
    x = torch.randn(3, 4, 65, device='cuda', requires_grad=True)
    pre = torch.randn(3, 4, device='cuda', requires_grad=True)
    out = hc_pre(x=x, pre=pre)
    for name in ('GLUON', 'TILELANG'):
        monkeypatch.setenv(f'FLA_HC_{name}', '0')
    out.sum().backward()
    assert len(backend) == 3


@pytest.mark.parametrize('kind', ['pre', 'ihc', 'mhc', 'sinkhorn'])
def test_unbatched(backend, kind):
    torch.manual_seed(17)
    x = torch.randn(4, 65, device='cuda', requires_grad=True)
    weight = torch.randn(4, device='cuda', requires_grad=True)
    branch = torch.randn(65, device='cuda', requires_grad=True)
    mixing = torch.randn(4, 4, device='cuda', requires_grad=True)
    if kind == 'pre':
        _check(fn=hc_pre, reference=naive_hc_pre, inputs=(x, weight), ratio=5e-5)
    elif kind == 'sinkhorn':
        _check(fn=sinkhorn, reference=naive_sinkhorn, inputs=(mixing,), ratio=1e-4)
    else:
        args = (x, branch, weight, mixing) if kind == 'mhc' else (x, branch, weight)
        _check(fn=hc_post, reference=naive_hc_post, inputs=args, ratio=5e-5)
    assert backend
