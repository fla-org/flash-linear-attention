# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.utils import addmm
from fla.utils import assert_close, device


def test_addmm_unit_scalars():
    x = torch.ones(1, 1, device=device)
    a = torch.ones(1, 1, device=device)
    b = torch.ones(1, 1, device=device)
    for kwargs in ({}, {'alpha': 2.0}, {'beta': 0.5}, {'alpha': 2.0, 'beta': 0.5}):
        expected = torch.addmm(x, a, b, **kwargs)
        got = addmm(x, a, b, **kwargs)
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    ('alpha', 'beta'),
    [
        (None, None),
        (2.0, None),
        (None, 0.5),
        (1.5, -0.5),
        (-2.0, 3.0),
        (0.0, 0.5),
        (2.0, 0.0),
        (2, 3),
    ],
)
@pytest.mark.parametrize('x_ndim', [1, 2])
def test_addmm_2d(alpha: float | None, beta: float | None, x_ndim: int, dtype: torch.dtype):
    torch.manual_seed(42)
    M, K, N = 37, 53, 41
    x = torch.randn((N,) if x_ndim == 1 else (M, N), device=device, dtype=dtype)
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)

    kwargs = {}
    if alpha is not None:
        kwargs['alpha'] = alpha
    if beta is not None:
        kwargs['beta'] = beta

    ref = torch.addmm(x.float(), a.float(), b.float(), **kwargs).to(dtype)
    got = addmm(x, a, b, alpha=alpha, beta=beta)
    assert got.shape == (M, N)
    assert got.dtype == dtype
    assert_close('addmm_2d', ref, got, 1e-2 if dtype != torch.float32 else 1e-4)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize('x_ndim', [1, 2, 3])
def test_addmm_3d_and_noncontiguous(x_ndim: int, dtype: torch.dtype):
    torch.manual_seed(42)
    B, M, K, N = 3, 35, 49, 39
    if x_ndim == 1:
        x = torch.randn(2 * N, device=device, dtype=dtype)[::2]
    elif x_ndim == 2:
        x = torch.randn(N, M, device=device, dtype=dtype).t()
    else:
        x = torch.randn(M, B, N, device=device, dtype=dtype).transpose(0, 1)
    a = torch.randn(M, B, K, device=device, dtype=dtype).transpose(0, 1)
    b = torch.randn(N, K, device=device, dtype=dtype).t()

    alpha, beta = 1.25, -0.75
    ref = (alpha * torch.matmul(a.float(), b.float()) + beta * x.float()).to(dtype)
    got = addmm(x, a, b, alpha=alpha, beta=beta)
    assert got.shape == (B, M, N)
    assert got.dtype == dtype
    assert_close('addmm_3d', ref, got, 1e-2 if dtype != torch.float32 else 1e-4)


def test_addmm_zero_scale_and_tensor_scalars():
    o = torch.ones(4, 4, device=device)
    nan_buf = torch.full((4, 4), float('nan'), device=device)
    inf_buf = torch.full((4, 4), float('inf'), device=device)

    torch.testing.assert_close(addmm(nan_buf, o, o, alpha=2.0, beta=0.0), torch.addmm(nan_buf, o, o, alpha=2.0, beta=0.0))
    torch.testing.assert_close(addmm(inf_buf, o, o, alpha=2.0, beta=0.0), torch.addmm(inf_buf, o, o, alpha=2.0, beta=0.0))
    torch.testing.assert_close(addmm(o, nan_buf, o, alpha=0.0, beta=0.5), torch.addmm(o, nan_buf, o, alpha=0.0, beta=0.5))
    torch.testing.assert_close(addmm(o, inf_buf, o, alpha=0.0, beta=0.5), torch.addmm(o, inf_buf, o, alpha=0.0, beta=0.5))

    alpha_t = torch.tensor(2.0, device=device)
    beta_t = torch.tensor(0.5, device=device)
    torch.testing.assert_close(addmm(o, o, o, alpha=alpha_t, beta=beta_t), torch.addmm(o, o, o, alpha=2.0, beta=0.5))
