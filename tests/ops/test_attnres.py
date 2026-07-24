# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.attnres import fused_attnres, naive_attnres
from fla.utils import IS_NPU, assert_close, device


@pytest.mark.parametrize(
    ('L', 'B', 'T', 'D', 'scale', 'fuse_output_norm', 'dtype', 'checkpoint_level'),
    [
        pytest.param(*test, id="L{}-B{}-T{}-D{}-scale{}-onorm{}-{}-ckpt{}".format(*test))
        for test in [
            # single-axis stress (no output norm); checkpoint_level spread across shapes
            (1,  1, 1000, 4096, 1.0,             False, torch.float16, 1),  # L=1
            (3,  1, 1000, 4096, 4096 ** -0.5,    False, torch.float16, 0),  # L=3
            (15, 1, 15,   4096, 1.0,             False, torch.float16, 1),  # T=15
            (7,  1, 1000, 1000, 1000 ** -0.5,    False, torch.float16, 0),  # D=1000
            (7,  1, 1000, 2000, 2000 ** -0.5,    False, torch.float16, 1),  # D=2000
            # multi-axis stress (extremes stacked, no output norm)
            (29, 5, 1000, 4096, 4096 ** -0.5,    False, torch.float16, 1),  # L=29 + B=5
            (29, 1, 8000, 4096, 4096 ** -0.5,    False, torch.float16, 0),  # L=29 + T=8000
            (15, 5, 1000, 7186, 7186 ** -0.5,    False, torch.float16, 1),  # B=5  + D=7186
            (15, 1, 8000, 7186, 1.0,             False, torch.float16, 0),  # T=8000 + D=7186
            (29, 3, 63,   7186, 7186 ** -0.5,    False, torch.float16, 1),  # L=29 + D=7186 + T=63
            # fp32 sanity at a larger size
            (10, 2, 8000, 4096, 4096 ** -0.5,    False, torch.float32, 1),
            # output_rms_weight on: fold-in path (fwd + bwd dow)
            (3,  1, 1000, 4096, 4096 ** -0.5,    True,  torch.float16, 1),  # L=3
            (29, 5, 1000, 4096, 4096 ** -0.5,    True,  torch.float16, 0),  # L=29 + B=5
            (15, 1, 8000, 7186, 1.0,             True,  torch.float16, 1),  # T=8000 + D=7186
            (10, 2, 8000, 4096, 4096 ** -0.5,    True,  torch.float32, 0),  # fp32 sanity
        ]
    ],
)
def test_attnres(
    L: int,
    B: int,
    T: int,
    D: int,
    scale: float,
    fuse_output_norm: bool,
    dtype: torch.dtype,
    checkpoint_level: int,
):
    torch.manual_seed(42)
    # disable TF32 in the PyTorch reference path so the fp32 sanity case
    # actually compares fp32 vs fp32 (otherwise einsum bwd uses cuBLAS TF32
    # which only has 10-bit mantissa and inflates the diff)
    torch.backends.cuda.matmul.allow_tf32 = False
    rms_eps = 1e-6

    # list of L independently allocated `[B, T, D]` tensors — true zero-cat input.
    residuals = [
        torch.randn(B, T, D, dtype=dtype, device=device).requires_grad_(True)
        for _ in range(L)
    ]
    query = torch.randn(D, dtype=dtype, device=device).requires_grad_(True)
    rms_weight = torch.randn(D, dtype=dtype, device=device).requires_grad_(True)
    output_rms_weight = (
        torch.randn(D, dtype=dtype, device=device).requires_grad_(True)
        if fuse_output_norm else None
    )

    tri, tri_p = fused_attnres(
        query=query,
        residuals=residuals,
        rms_weight=rms_weight,
        output_rms_weight=output_rms_weight,
        rms_eps=rms_eps,
        scale=scale,
        return_weights=True,
        checkpoint_level=checkpoint_level,
    )
    do = torch.randn_like(tri)
    (tri * do).sum().backward()
    tri_dvs = [r.grad for r in residuals]
    tri_dq, tri_dw = query.grad, rms_weight.grad
    tri_dow = output_rms_weight.grad if fuse_output_norm else None

    residuals_ref = [r.detach().clone().requires_grad_(True) for r in residuals]
    query_ref = query.detach().clone().requires_grad_(True)
    rms_weight_ref = rms_weight.detach().clone().requires_grad_(True)
    output_rms_weight_ref = (
        output_rms_weight.detach().clone().requires_grad_(True)
        if fuse_output_norm else None
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
    assert_close('dw', rms_weight_ref.grad, tri_dw, 0.005)
    assert_close('dv', torch.stack([r.grad for r in residuals_ref]), torch.stack(tri_dvs), 0.005)
    if fuse_output_norm:
        assert_close('dow', output_rms_weight_ref.grad, tri_dow, 0.005)


_TRITON_ASCEND_ATTNRES_OPS = ('fused_attnres',)


def _spy_on_triton_ascend_attnres_backend():
    """Patch every op of the Triton-Ascend AttnRes backend to record dispatched calls."""
    from fla.ops.backends import BackendRegistry

    BackendRegistry.ensure_initialized('attnres')
    backend = BackendRegistry._registries['attnres']._backends.get('triton_ascend')
    assert backend is not None, 'Triton-Ascend AttnRes backend is not registered'

    calls = []
    for name in _TRITON_ASCEND_ATTNRES_OPS:
        original = getattr(backend, name)

        def make_spy(name, original):
            def spy(*args, **kwargs):
                calls.append(name)
                return original(*args, **kwargs)
            return spy

        setattr(backend, name, make_spy(name, original))
    return backend, calls


@pytest.mark.skipif(not IS_NPU, reason='Triton-Ascend AttnRes backend routing is only exercised on NPU')
def test_triton_ascend_backend_routing():
    """fused_attnres must actually dispatch to the Triton-Ascend backend on NPU.

    Numerical parity tests alone cannot catch silently-failing verifiers: if
    every verifier rejected, the call would fall back to the default CUDA Triton
    path and parity tests would still pass, leaving the NPU kernels dead.
    """
    backend, calls = _spy_on_triton_ascend_attnres_backend()
    try:
        L, B, T, D = 3, 1, 64, 128
        dtype = torch.float16
        residuals = [
            torch.randn(B, T, D, dtype=dtype, device=device).requires_grad_(True)
            for _ in range(L)
        ]
        query = torch.randn(D, dtype=dtype, device=device).requires_grad_(True)
        rms_weight = torch.randn(D, dtype=dtype, device=device).requires_grad_(True)

        calls.clear()
        o = fused_attnres(
            query=query,
            residuals=residuals,
            rms_weight=rms_weight,
            scale=D ** -0.5,
            checkpoint_level=1,
        )
        (o * torch.randn_like(o)).sum().backward()
        assert calls == ['fused_attnres'], (
            f'fused_attnres not routed to the Triton-Ascend backend (dispatched: {calls})'
        )
    finally:
        for name in _TRITON_ASCEND_ATTNRES_OPS:
            delattr(backend, name)
