# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.attnres import fused_attnres, naive_attnres
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('L', 'B', 'T', 'D', 'scale', 'dtype'),
    [
        pytest.param(*test, id="L{}-B{}-T{}-D{}-scale{}-{}".format(*test))
        for test in [
            # single-axis stress
            (1,  1, 1000, 4096, 1.0,             torch.float16),  # L=1
            (3,  1, 1000, 4096, 4096 ** -0.5,    torch.float16),  # L=3
            (15, 1, 15,   4096, 1.0,             torch.float16),  # T=15
            (7,  1, 1000, 1000, 1000 ** -0.5,    torch.float16),  # D=1000
            (7,  1, 1000, 2000, 2000 ** -0.5,    torch.float16),  # D=2000
            # multi-axis stress (extremes stacked)
            (29, 5, 1000, 4096, 4096 ** -0.5,    torch.float16),  # L=29 + B=5
            (29, 1, 8000, 4096, 4096 ** -0.5,    torch.float16),  # L=29 + T=8000
            (15, 5, 1000, 7186, 7186 ** -0.5,    torch.float16),  # B=5  + D=7186
            (15, 1, 8000, 7186, 1.0,             torch.float16),  # T=8000 + D=7186
            (29, 3, 63,   7186, 7186 ** -0.5,    torch.float16),  # L=29 + D=7186 + T=63
            # fp32 sanity at a larger size
            (10, 2, 8000, 4096, 4096 ** -0.5,    torch.float32),
        ]
    ],
)
def test_attnres(
    L: int,
    B: int,
    T: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    rms_eps = 1e-6

    residuals = torch.randn(L, B, T, D, dtype=dtype, device=device).requires_grad_(True)
    query = torch.randn(D, dtype=dtype, device=device).requires_grad_(True)
    rms_weight = torch.randn(D, dtype=dtype, device=device).requires_grad_(True)

    tri, tri_p = fused_attnres(
        query=query,
        residuals=residuals,
        rms_weight=rms_weight,
        rms_eps=rms_eps,
        scale=scale,
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
        query=query_ref,
        residuals=residuals_ref,
        rms_weight=rms_weight_ref,
        rms_eps=rms_eps,
        scale=scale,
        return_weights=True,
    )
    (ref * do).sum().backward()

    assert_close(' o', ref, tri, 0.005)
    assert_close(' p', ref_p, tri_p, 0.005)
    assert_close('dq', query_ref.grad, tri_dq, 0.005)
    assert_close('dv', residuals_ref.grad, tri_dv, 0.005)
    assert_close('dw', rms_weight_ref.grad, tri_dw, 0.005)
