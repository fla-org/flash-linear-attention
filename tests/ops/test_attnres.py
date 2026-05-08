# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.attnres import fused_attnres, naive_attnres
from fla.utils import assert_close, device, device_platform

TEST_CASES = [
    (1, 1, 128, 1024, torch.float32, 1, 1.0),
    (3, 1, 128, 1024, torch.float32, 1, 1.0),
    (7, 1, 128, 1024, torch.float32, 1, 1024 ** -0.5),
    (3, 1, 1024, 1024, torch.float32, 2, 1024 ** -0.5),
    (7, 1, 1024, 1024, torch.float32, 1, 1024 ** -0.5),
    (7, 1, 1024, 1024, torch.bfloat16, 1, 1024 ** -0.5),
]


@pytest.mark.skipif(device_platform == 'cpu', reason='Triton attnres requires a GPU backend')
@pytest.mark.parametrize(
    ('L', 'B', 'T', 'D', 'dtype', 'query_ndim', 'scale'),
    [
        pytest.param(
            L,
            B,
            T,
            D,
            dtype,
            query_ndim,
            scale,
            id=f"L{L}-B{B}-T{T}-D{D}-{dtype}-Q{query_ndim}-scale{scale}",
        )
        for L, B, T, D, dtype, query_ndim, scale in TEST_CASES
    ],
)
def test_fused_attnres(
    L: int,
    B: int,
    T: int,
    D: int,
    dtype: torch.dtype,
    query_ndim: int,
    scale: float,
):
    torch.manual_seed(42)
    rms_eps = 1e-6

    residuals = torch.randn(L, B, T, D, dtype=dtype, device=device).requires_grad_(True)
    query = torch.randn(D, dtype=dtype, device=device).requires_grad_(True)
    if query_ndim == 2:
        query = query.detach().clone().unsqueeze(-1).requires_grad_(True)
    rms_weight = torch.randn(D, dtype=dtype, device=device).requires_grad_(True)

    tri, tri_p = fused_attnres(
        query,
        residuals,
        rms_weight,
        rms_eps,
        scale,
        return_weights=True,
    )
    do = torch.randn_like(tri)
    (tri * do).sum().backward()
    tri_dv = residuals.grad
    tri_dq, tri_dw = query.grad, rms_weight.grad

    residuals_ref = residuals.detach().clone().requires_grad_(True)
    query_ref = query.detach().clone().requires_grad_(True)
    rms_weight_ref = rms_weight.detach().clone().requires_grad_(True)

    ref, ref_p = naive_attnres(
        query_ref,
        residuals_ref,
        rms_weight_ref,
        rms_eps,
        scale,
        return_weights=True,
    )
    (ref * do).sum().backward()

    tol = 0.02 if dtype is torch.bfloat16 else 0.005
    assert_close('o', ref, tri, tol)
    assert_close('p', ref_p, tri_p, tol)
    assert_close('dv', residuals_ref.grad, tri_dv, tol)
    assert_close('dq', query_ref.grad, tri_dq, tol)
    assert_close('dw', rms_weight_ref.grad, tri_dw, tol)
