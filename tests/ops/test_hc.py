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
from fla.utils import IS_NVIDIA, assert_close, device

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='NVIDIA CUDA required')


@pytest.fixture
def backend(request, monkeypatch):
    # keep the FP32 reference's einsum backward at full precision
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
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


@pytest.mark.parametrize(
    ('B', 'T', 'N', 'D', 'dtype', 'use_mixing', 'noncontiguous', 'backend'),
    [
        pytest.param(*test, id='B{}-T{}-N{}-D{}-{}-mix{}-nc{}-{}'.format(*test))
        for test in [
            (2, 3, 1, 1, torch.float32, False, False, 'triton'),
            (2, 3, 3, 65, torch.float16, True, False, 'triton'),
            (2, 3, 4, 1025, torch.bfloat16, False, False, 'triton'),
            (2, 3, 4, 65, torch.bfloat16, True, True, 'triton'),
            (2, 3, 5, 65, torch.bfloat16, False, False, 'gluon'),
            (2, 3, 8, 4096, torch.float16, True, False, 'gluon'),
            (1, 127, 4, 1025, torch.float32, True, False, 'gluon'),
            (2, 3, 1, 1, torch.float32, True, False, 'tilelang'),
            (2, 3, 4, 256, torch.float32, True, False, 'tilelang'),
            (2, 3, 5, 65, torch.float16, True, False, 'tilelang'),
            (2, 64, 8, 4096, torch.bfloat16, False, False, 'tilelang'),
            (1, 128, 4, 1025, torch.float32, True, False, 'tilelang'),
        ]
    ],
    indirect=['backend'],
)
def test_fused_hc(
    B: int,
    T: int,
    N: int,
    D: int,
    dtype: torch.dtype,
    use_mixing: bool,
    noncontiguous: bool,
    backend,
):
    torch.manual_seed(42)
    x = torch.randn(B, T, N, D, device=device, dtype=dtype, requires_grad=True)
    pre = torch.randn(B, T, N, device=device, dtype=torch.float32, requires_grad=True)
    branch = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    post = torch.randn(B, T, N, device=device, dtype=torch.float32, requires_grad=True)
    mixing = torch.randn(B, T, N, N, device=device, dtype=torch.float32, requires_grad=True) if use_mixing else None
    if noncontiguous:
        x = x.transpose(0, 1)
        pre = pre.transpose(0, 1)
        branch = branch.transpose(0, 1)
        post = post.transpose(0, 1)
        if mixing is not None:
            mixing = mixing.transpose(0, 1)
    ratio = {torch.float32: 5e-5, torch.float16: 2e-3, torch.bfloat16: 6e-3}[dtype]

    ref = naive_hc_pre(x=x, pre=pre)
    tri = fused_hc_pre(x=x, pre=pre)
    do = torch.randn_like(ref)
    ref_dx, ref_dpre = torch.autograd.grad(outputs=ref, inputs=(x, pre), grad_outputs=do)
    tri_dx, tri_dpre = torch.autograd.grad(outputs=tri, inputs=(x, pre), grad_outputs=do)
    assert tri.shape == ref.shape and tri.dtype == dtype
    assert_close('pre', ref, tri, ratio)
    assert_close('dx', ref_dx, tri_dx, ratio)
    assert_close('dpre', ref_dpre, tri_dpre, ratio)

    ref = naive_hc_post(x=x, branch=branch, post=post, mixing=mixing)
    tri = fused_hc_post(x=x, branch=branch, post=post, mixing=mixing)
    do = torch.randn_like(ref)
    inputs = (x, branch, post, mixing) if use_mixing else (x, branch, post)
    ref_grads = torch.autograd.grad(outputs=ref, inputs=inputs, grad_outputs=do)
    tri_grads = torch.autograd.grad(outputs=tri, inputs=inputs, grad_outputs=do)
    assert tri.shape == ref.shape and tri.dtype == dtype
    assert_close('post', ref, tri, ratio)
    for name, ref_grad, tri_grad in zip(('dx', 'dbranch', 'dpost', 'dmixing'), ref_grads, tri_grads):
        assert_close(name, ref_grad, tri_grad, ratio)


@pytest.mark.parametrize('backend', ['triton', 'gluon', 'tilelang'], indirect=True)
def test_fused_hc_backend(backend, monkeypatch):
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

    x = torch.randn(2, 3, 4, 65, device=device, requires_grad=True)
    pre = torch.randn(2, 3, 4, device=device, requires_grad=True)
    post = torch.randn_like(pre, requires_grad=True)
    mixing = torch.randn(2, 3, 4, 4, device=device, requires_grad=True)
    branch = fused_hc_pre(x=x, pre=pre)
    out = fused_hc_post(x=x, branch=branch, post=post, mixing=mixing)
    monkeypatch.setenv('FLA_GLUON', '0' if backend == 'gluon' else '1')
    monkeypatch.setenv('FLA_TILELANG', '0')
    out.backward(torch.randn_like(out))
    assert [call.kwargs['is_post'] for call in backward.call_args_list] == [True, False]
