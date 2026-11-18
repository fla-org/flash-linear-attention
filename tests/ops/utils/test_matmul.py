# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.utils import addmm, matmul
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('shape', 'n', 'dtype', 'noncontiguous'),
    [
        pytest.param((16, 1), 1, torch.float32, False, id='fp32-saturation'),
        pytest.param((65, 33), 17, torch.float16, False, id='fp16-partial-tiles'),
        pytest.param((2, 63, 65), 33, torch.bfloat16, False, id='bf16-batched'),
        pytest.param((129, 17), 65, torch.float32, True, id='fp32-strided'),
    ],
)
def test_matmul_tanh(shape, n, dtype, noncontiguous):
    torch.manual_seed(42)
    values = torch.tensor(
        [-1000, -100, -90, -89, -20, -1, -1e-4, 0, 1e-4, 1, 20, 89, 90, 100, 1000, 0.5],
        dtype=dtype,
        device=device,
    )
    a = torch.zeros(shape, dtype=dtype, device=device)
    a[..., 0] = values[torch.arange(shape[-2], device=device) % len(values)]
    b = torch.ones((n, shape[-1]) if noncontiguous else (shape[-1], n), dtype=dtype, device=device)
    if noncontiguous:
        b = b.t()
    ref = torch.tanh(a.float() @ b.float()).to(dtype)
    tri = matmul(a=a, b=b, activation='tanh')
    assert torch.isfinite(tri).all()
    assert_close('tanh', ref, tri, 1e-5)


@pytest.mark.parametrize(
    ('shape', 'n', 'dtype', 'activation'),
    [
        pytest.param((17, 33), 65, torch.float32, '', id='identity'),
        pytest.param((2, 65, 17), 33, torch.bfloat16, 'relu', id='relu-batched'),
        pytest.param((63, 65), 17, torch.float16, 'sigmoid', id='sigmoid'),
        pytest.param((129, 17), 65, torch.float32, 'leaky_relu', id='leaky-relu'),
    ],
)
def test_matmul_activation(shape, n, dtype, activation):
    torch.manual_seed(42)
    a = torch.randn(shape, dtype=dtype, device=device)
    b = torch.randn((shape[-1], n), dtype=dtype, device=device)
    ref = a.float() @ b.float()
    if activation:
        ref = {'relu': torch.relu, 'sigmoid': torch.sigmoid, 'leaky_relu': F.leaky_relu}[activation](ref)
    tri = matmul(a=a, b=b, activation=activation)
    assert_close('activation', ref.to(dtype), tri, 0.006)


@pytest.mark.parametrize(
    ('shape', 'n', 'dtype', 'bias_shape'),
    [
        pytest.param((65, 33), 17, torch.float32, (17,), id='vector-bias'),
        pytest.param((2, 63, 65), 33, torch.bfloat16, (63, 33), id='batched-matrix-bias'),
    ],
)
def test_addmm(shape, n, dtype, bias_shape):
    torch.manual_seed(42)
    a = torch.randn(shape, dtype=dtype, device=device)
    b = torch.randn((shape[-1], n), dtype=dtype, device=device)
    x = torch.randn(bias_shape, dtype=dtype, device=device)
    ref = (a.float() @ b.float() + x.float()).to(dtype)
    tri = addmm(x=x, a=a, b=b)
    assert_close('addmm', ref, tri, 0.006)
