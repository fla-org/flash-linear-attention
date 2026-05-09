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
    ('L', 'B', 'T', 'D', 'scale', 'with_output_norm', 'dtype'),
    [
        pytest.param(*test, id="L{}-B{}-T{}-D{}-scale{}-onorm{}-{}".format(*test))
        for test in [
            # single-axis stress (no output norm)
            (1,  1, 1000, 4096, 1.0,             False, torch.float16),  # L=1
            (3,  1, 1000, 4096, 4096 ** -0.5,    False, torch.float16),  # L=3
            (15, 1, 15,   4096, 1.0,             False, torch.float16),  # T=15
            (7,  1, 1000, 1000, 1000 ** -0.5,    False, torch.float16),  # D=1000
            (7,  1, 1000, 2000, 2000 ** -0.5,    False, torch.float16),  # D=2000
            # multi-axis stress (extremes stacked, no output norm)
            (29, 5, 1000, 4096, 4096 ** -0.5,    False, torch.float16),  # L=29 + B=5
            (29, 1, 8000, 4096, 4096 ** -0.5,    False, torch.float16),  # L=29 + T=8000
            (15, 5, 1000, 7186, 7186 ** -0.5,    False, torch.float16),  # B=5  + D=7186
            (15, 1, 8000, 7186, 1.0,             False, torch.float16),  # T=8000 + D=7186
            (29, 3, 63,   7186, 7186 ** -0.5,    False, torch.float16),  # L=29 + D=7186 + T=63
            # fp32 sanity at a larger size
            (10, 2, 8000, 4096, 4096 ** -0.5,    False, torch.float32),
            # output_rms_weight on: fold-in path (fwd + bwd dw_out)
            (3,  1, 1000, 4096, 4096 ** -0.5,    True,  torch.float16),  # L=3
            (29, 5, 1000, 4096, 4096 ** -0.5,    True,  torch.float16),  # L=29 + B=5
            (15, 1, 8000, 7186, 1.0,             True,  torch.float16),  # T=8000 + D=7186
            (10, 2, 8000, 4096, 4096 ** -0.5,    True,  torch.float32),  # fp32 sanity
        ]
    ],
)
def test_attnres(
    L: int,
    B: int,
    T: int,
    D: int,
    scale: float,
    with_output_norm: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    # disable TF32 in the PyTorch reference path so the fp32 sanity case
    # actually compares fp32 vs fp32 (otherwise einsum bwd uses cuBLAS TF32
    # which only has 10-bit mantissa and inflates the diff)
    torch.backends.cuda.matmul.allow_tf32 = False
    rms_eps = 1e-6

    residuals = torch.randn(L, B, T, D, dtype=dtype, device=device).requires_grad_(True)
    query = torch.randn(D, dtype=dtype, device=device).requires_grad_(True)
    rms_weight = torch.randn(D, dtype=dtype, device=device).requires_grad_(True)
    output_rms_weight = (
        torch.randn(D, dtype=dtype, device=device).requires_grad_(True)
        if with_output_norm else None
    )

    tri, tri_p = fused_attnres(
        query=query,
        residuals=residuals,
        rms_weight=rms_weight,
        output_rms_weight=output_rms_weight,
        rms_eps=rms_eps,
        scale=scale,
        return_weights=True,
    )
    do = torch.randn_like(tri)
    (tri * do).sum().backward()
    tri_dv = residuals.grad
    tri_dq, tri_dw = query.grad, rms_weight.grad
    tri_dw_out = output_rms_weight.grad if with_output_norm else None

    residuals_ref = residuals.detach().clone().requires_grad_(True)
    query_ref = query.detach().clone().requires_grad_(True)
    rms_weight_ref = rms_weight.detach().clone().requires_grad_(True)
    output_rms_weight_ref = (
        output_rms_weight.detach().clone().requires_grad_(True)
        if with_output_norm else None
    )

    ref, ref_p = naive_attnres(
        query=query_ref,
        residuals=residuals_ref,
        rms_weight=rms_weight_ref,
        output_rms_weight=output_rms_weight_ref,
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
    if with_output_norm:
        assert_close('dw_out', output_rms_weight_ref.grad, tri_dw_out, 0.005)
