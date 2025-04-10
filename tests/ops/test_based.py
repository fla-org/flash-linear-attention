# -*- coding: utf-8 -*-

import os

import pytest
import torch

from fla.ops.based import fused_chunk_based, parallel_based
from fla.ops.based.naive import naive_parallel_based
from fla.ops.utils.testing import COMPILER_MODE
from fla.utils import device

if COMPILER_MODE:
    test_b_list = [1]
    test_t_list = [4096]
    test_d_list = [64, 128, 256]
else:
    test_b_list = [2]
    test_t_list = [1, 15, 63, 300]
    test_d_list = [64, 32, 100, 256]
test_h_list = [2]


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
def test_based(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    q = torch.randn((B, H, T, 16), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, H, T, 16), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn_like(v)
    ref = naive_parallel_based(q, k, v, use_norm=True)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_based(q, k, v, use_norm=True)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    if dtype == torch.float32:
        assert ref.allclose(tri, 0, 1e-4)
        assert ref_dq.allclose(tri_dq, 0, 1e-4)
        assert ref_dk.allclose(tri_dk, 0, 1e-4)
        assert ref_dv.allclose(tri_dv, 0, 1e-4)

    tri = fused_chunk_based(q, k, v, use_norm=True)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    if dtype == torch.float32:
        assert ref.allclose(tri, 0, 1e-4)
        assert ref_dq.allclose(tri_dq, 0, 1e-4)
        assert ref_dk.allclose(tri_dk, 0, 1e-4)
        assert ref_dv.allclose(tri_dv, 0, 1e-4)
