# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.attnres import fused_attnres, naive_attnres
from fla.utils import IS_NPU, IS_NVIDIA, assert_close, device


@pytest.mark.parametrize(
    ('L', 'B', 'T', 'D', 'scale', 'fuse_output_norm', 'dtype', 'checkpoint_level', 'use_rms_weight'),
    [
        pytest.param(*test, True, id="L{}-B{}-T{}-D{}-scale{}-onorm{}-{}-ckpt{}".format(*test))
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
            (10, 2, 8000, 4096, 4096 ** -0.5,    True,  torch.float32, 0),
        ]
    ] + [
        pytest.param(
            3, 1, 63, 1000, 1000 ** -0.5, True, torch.float16, 1, False,
            id="no-rms-weight",
        ),
    ],
)
@pytest.mark.smoke
def test_attnres(
    L: int,
    B: int,
    T: int,
    D: int,
    scale: float,
    fuse_output_norm: bool,
    dtype: torch.dtype,
    checkpoint_level: int,
    use_rms_weight: bool,
):
    torch.manual_seed(42)
    # disable TF32 so the fp32 reference retains its full mantissa in einsum backward.
    torch.backends.cuda.matmul.allow_tf32 = False
    rms_eps = 1e-6

    # list of L independently allocated `[B, T, D]` tensors — true zero-cat input.
    residuals = [torch.randn(B, T, D, dtype=dtype, device=device).requires_grad_(True) for _ in range(L)]
    query = torch.randn(D, dtype=dtype, device=device).requires_grad_(True)
    rms_weight = torch.randn(D, dtype=dtype, device=device).requires_grad_(True) if use_rms_weight else None
    output_rms_weight = (torch.randn(D, dtype=dtype, device=device).requires_grad_(True) if fuse_output_norm else None)

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
    tri_dq, tri_dw = query.grad, rms_weight.grad if use_rms_weight else None
    tri_dow = output_rms_weight.grad if fuse_output_norm else None

    residuals_ref = [r.detach().clone().requires_grad_(True) for r in residuals]
    query_ref = query.detach().clone().requires_grad_(True)
    rms_weight_ref = rms_weight.detach().clone().requires_grad_(True) if use_rms_weight else None
    output_rms_weight_ref = (output_rms_weight.detach().clone().requires_grad_(True) if fuse_output_norm else None)

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
    if use_rms_weight:
        assert_close('dw', rms_weight_ref.grad, tri_dw, 0.005)
    assert_close('dv', torch.stack([r.grad for r in residuals_ref]), torch.stack(tri_dvs), 0.005)
    if fuse_output_norm:
        assert_close('dow', output_rms_weight_ref.grad, tri_dow, 0.005)


@pytest.mark.skipif(not IS_NVIDIA, reason='Gluon AttnRes requires NVIDIA CUDA')
@pytest.mark.parametrize(
    ('weight_mode', 'enabled', 'D', 'dtype', 'fuse_output_norm', 'checkpoint_level', 'expect_gluon'),
    [
        pytest.param('tensor', True, 128, torch.float16, False, 0, True, id='weighted'),
        pytest.param('none', True, 1000, torch.float16, True, 1, True, id='none'),
        pytest.param('omitted', True, 128, torch.float32, True, 0, True, id='omitted'),
        pytest.param('none', False, 128, torch.float16, False, 1, False, id='disabled'),
        pytest.param('none', True, 32, torch.float16, False, 1, False, id='small-D-fallback'),
    ],
)
def test_gluon_backend_routing(
    monkeypatch,
    weight_mode: str,
    enabled: bool,
    D: int,
    dtype: torch.dtype,
    fuse_output_norm: bool,
    checkpoint_level: int,
    expect_gluon: bool,
):
    from fla import backends

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip('Gluon AttnRes requires SM80+')
    if backends._DISPATCH_DISABLED:
        pytest.skip('backend dispatch is disabled in this process')
    backend = backends._load_operation_registry('attnres')._backends.get('gluon')
    if backend is None or not backend.is_available():
        pytest.skip('Gluon is not available')
    monkeypatch.setenv('FLA_GLUON', '0')
    monkeypatch.setenv('FLA_ATTNRES_GLUON', '1' if enabled else '0')
    calls = []
    original = backend.fused_attnres

    def spy(*args, **kwargs):
        calls.append('fused_attnres')
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, 'fused_attnres', spy)
    torch.manual_seed(42)
    query = torch.randn(D, 1, dtype=dtype, device=device).requires_grad_(True)
    residuals = [torch.randn(1, 63, D, dtype=dtype, device=device).requires_grad_(True) for _ in range(3)]
    weight = torch.randn(D, dtype=dtype, device=device).requires_grad_(True) if weight_mode == 'tensor' else None
    output_weight = torch.randn(D, dtype=dtype, device=device).requires_grad_(True) if fuse_output_norm else None
    kwargs = dict(query=query, residuals=residuals, output_rms_weight=output_weight, scale=D ** -0.5, return_weights=True)
    if weight_mode != 'omitted':
        kwargs['rms_weight'] = weight
    inputs = [query, *residuals]
    if weight is not None:
        inputs.append(weight)
    if output_weight is not None:
        inputs.append(output_weight)
    actual, actual_p = fused_attnres(**kwargs, checkpoint_level=checkpoint_level)
    do = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, do)
    reference, reference_p = naive_attnres(**kwargs)
    reference_grads = torch.autograd.grad(reference, inputs, do)
    assert calls == (['fused_attnres'] if expect_gluon else [])
    assert_close('o', reference, actual, 0.005)
    assert_close('p', reference_p, actual_p, 0.005)
    for expected, grad in zip(reference_grads, actual_grads):
        assert torch.isfinite(grad).all()
        assert_close('grad', expected, grad, 0.005)


_TRITON_ASCEND_ATTNRES_OPS = ('fused_attnres',)


def _spy_on_triton_ascend_attnres_backend():
    """Patch every op of the Triton-Ascend AttnRes backend to record dispatched calls."""
    from fla import backends

    backend = backends._load_operation_registry('attnres')._backends.get('triton_ascend')
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
@pytest.mark.parametrize(
    ('weight_mode', 'D', 'dtype', 'fuse_output_norm', 'checkpoint_level'),
    [
        pytest.param('tensor', 128, torch.float16, False, 1, id='weighted'),
        pytest.param('none', 128, torch.float16, False, 1, id='none'),
        pytest.param('none', 1000, torch.float16, True, 0, id='none-onorm'),
        pytest.param('omitted', 128, torch.float32, True, 1, id='omitted'),
        pytest.param('none', 8193, torch.float16, False, 0, id='none-D-loop'),
    ],
)
def test_triton_ascend_backend_routing(
    weight_mode: str,
    D: int,
    dtype: torch.dtype,
    fuse_output_norm: bool,
    checkpoint_level: int,
):
    """Verify fused_attnres dispatches to Triton-Ascend on NPU.

    Numerical parity can pass through the fallback, so record which implementation actually runs.
    """
    backend, calls = _spy_on_triton_ascend_attnres_backend()
    try:
        torch.manual_seed(42)
        L, B, T = 3, 1, 64
        residuals = [torch.randn(B, T, D, dtype=dtype, device=device).requires_grad_(True) for _ in range(L)]
        query = torch.randn(D, dtype=dtype, device=device).requires_grad_(True)
        rms_weight = torch.randn(D, dtype=dtype, device=device).requires_grad_(True) if weight_mode == 'tensor' else None
        output_weight = torch.randn(D, dtype=dtype, device=device).requires_grad_(True) if fuse_output_norm else None
        kwargs = dict(query=query, residuals=residuals, output_rms_weight=output_weight, scale=D ** -0.5, return_weights=True)
        if weight_mode != 'omitted':
            kwargs['rms_weight'] = rms_weight
        inputs = [query, *residuals]
        if rms_weight is not None:
            inputs.append(rms_weight)
        if output_weight is not None:
            inputs.append(output_weight)

        calls.clear()
        o, p = fused_attnres(**kwargs, checkpoint_level=checkpoint_level)
        do = torch.randn_like(o)
        grads = torch.autograd.grad(o, inputs, do)
        assert calls == ['fused_attnres'], (f'fused_attnres not routed to the Triton-Ascend backend (dispatched: {calls})')
        ref, ref_p = naive_attnres(**kwargs)
        ref_grads = torch.autograd.grad(ref, inputs, do)
        assert_close('o', ref, o, 0.005)
        assert_close('p', ref_p, p, 0.005)
        for expected, grad in zip(ref_grads, grads):
            assert torch.isfinite(grad).all()
            assert_close('grad', expected, grad, 0.005)
    finally:
        for name in _TRITON_ASCEND_ATTNRES_OPS:
            delattr(backend, name)
