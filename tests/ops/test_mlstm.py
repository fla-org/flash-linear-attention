# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch

from fla.ops.mlstm import chunk_mlstm, fused_recurrent_mlstm
from fla.ops.mlstm.naive import naive_chunk_mlstm, naive_recurrent_mlstm
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'scale', 'max_norm', 'dtype', 'use_initial_state', 'output_final_state'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-V{}-scale{}-max_norm={}-{}-initial_state-{}-final_state-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 64, 1, None, torch.float, True, True),
            (1, 63, 1, 64, 64, 1, False, torch.float, True, True),
            (1, 63, 1, 64, 64, 1, True, torch.float, True, True),
            (1, 63, 1, 64, 64, 1, None, torch.float16, True, True),
            (1, 63, 1, 64, 64, 1, False, torch.float16, True, True),
            (1, 63, 1, 64, 64, 1, True, torch.float16, True, True),
            (2, 7, 2, 16, 16, 1, True, torch.float16, False, True),
            (2, 7, 2, 16, 16, 1, True, torch.float16, False, False),
            (2, 63, 2, 16, 16, 1, True, torch.float16, True, False),
            (2, 500, 4, 60, 60, 1, None, torch.float, True, True),
            (2, 1024, 8, 128, 128, 1, None, torch.float, True, True),
            (2, 1024, 8, 128, 128, 0.1, None, torch.float, True, True),
            (4, 2048, 8, 64, 64, 0.1, None, torch.float, True, True),
            (2, 7, 2, 16, 16, 1, False, torch.float32, False, True),
            (2, 7, 2, 16, 16, 1, True, torch.float32, False, True),
            (2, 7, 2, 5, 7, 1, True, torch.float32, True, True),
            (2, 7, 2, 16, 32, 1, True, torch.float32, True, True),
            (2, 7, 2, 32, 16, 1, True, torch.float32, True, True),
            (2, 63, 2, 16, 16, 1, False, torch.float32, True, False),
            (2, 63, 2, 16, 16, 1, True, torch.float32, True, False),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    scale: float,
    max_norm: bool | None,
    dtype: torch.dtype,
    use_initial_state: bool,
    output_final_state: bool,
):
    torch.manual_seed(42)

    q = torch.randn((B, T, H, K), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, K), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, V), dtype=dtype, device=device).requires_grad_()
    i = torch.randn((B, T, H), dtype=dtype, device=device).requires_grad_()
    f = torch.randn((B, T, H), dtype=dtype, device=device).requires_grad_()
    state_factory = torch.randn if use_initial_state else torch.zeros
    c0 = state_factory(B, H, K, V, device=device).requires_grad_()
    n0 = state_factory(B, H, K, device=device).requires_grad_()
    m0 = state_factory(B, H, device=device)
    dct = torch.randn_like(c0)
    dnt = torch.randn_like(n0)
    dh = torch.randn_like(v)
    ref, ref_state = naive_recurrent_mlstm(
        q=q,
        k=k,
        v=v,
        i=i,
        f=f,
        scale=scale,
        initial_state=(c0, n0, m0) if use_initial_state else None,
        output_final_state=output_final_state,
        max_normalisation=max_norm,
    )
    ref_loss = (ref * dh).sum()
    if output_final_state:
        ref_ct, ref_nt, ref_mt = ref_state
        ref_loss += (ref_ct * dct).sum() + (ref_nt * dnt).sum()
    else:
        assert ref_state is None
    ref_loss.backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_di, i.grad = i.grad.clone(), None
    ref_df, f.grad = f.grad.clone(), None
    if use_initial_state:
        ref_dc0, c0.grad = c0.grad.clone(), None
        ref_dn0, n0.grad = n0.grad.clone(), None

    tri, tri_state = fused_recurrent_mlstm(
        q=q,
        k=k,
        v=v,
        i=i,
        f=f,
        scale=scale,
        initial_state=(c0, n0, m0) if use_initial_state else None,
        output_final_state=output_final_state,
        max_normalisation=max_norm,
    )
    tri_loss = (tri * dh).sum()
    if output_final_state:
        tri_ct, tri_nt, tri_mt = tri_state
        assert tri_ct.dtype == torch.float32
        assert tri_mt.dtype == torch.float32
        if max_norm is not None:
            assert tri_nt.dtype == torch.float32
            assert tri_nt.shape == (B, H, K)
        tri_loss += (tri_ct * dct).sum() + (tri_nt * dnt).sum()
    else:
        assert tri_state is None
    tri_loss.backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_di, i.grad = i.grad.clone(), None
    tri_df, f.grad = f.grad.clone(), None
    if use_initial_state:
        assert c0.grad is not None, "missing gradient for initial matrix state"
        assert n0.grad is not None, "missing gradient for initial normalizer state"
        tri_dc0, c0.grad = c0.grad.clone(), None
        tri_dn0, n0.grad = n0.grad.clone(), None

    assert_close('h', ref.float(), tri.float(), 0.005)
    if output_final_state:
        assert_close('ct', ref_ct, tri_ct, 0.005)
        assert_close('nt', ref_nt, tri_nt, 0.005)
        assert_close('mt', ref_mt, tri_mt, 0.005)
    assert_close('dq', ref_dq.float(), tri_dq.float(), 0.005)
    assert_close('dk', ref_dk.float(), tri_dk.float(), 0.005)
    assert_close('dv', ref_dv.float(), tri_dv.float(), 0.005)
    assert_close('di', ref_di.float(), tri_di.float(), 0.005)
    assert_close('df', ref_df.float(), tri_df.float(), 0.005, err_atol=2e-4)
    if use_initial_state:
        assert_close('dc0', ref_dc0, tri_dc0, 0.005)
        assert_close('dn0', ref_dn0, tri_dn0, 0.005)


@pytest.mark.parametrize(
    ('H', 'D', 'scale', 'cu_seqlens', 'max_norm', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-scale{}-cu_seqlens{}-max_norm={}-{}".format(*test))
        for test in [
            (4, 64, 1, [0, 15], None, torch.float),
            (4, 64, 1, [0, 15], False, torch.float),
            (4, 64, 1, [0, 15], True, torch.float),
            (4, 64, 1, [0, 256, 500, 1000], None, torch.float),
            (4, 100, 0.1, [0, 15, 100, 300, 1200, 2000], None, torch.float),
            (4, 100, 1, [0, 15, 100, 300, 1200, 2000], None, torch.float),
            (4, 64, 1, [0, 1, 100, 300, 1200, 2048], None, torch.float16),
            (4, 128, 1, [0, 200, 512, 1200, 2048], None, torch.float16),
        ]
    ],
)
def test_fused_recurrent_varlen(
    H: int,
    D: int,
    scale: float,
    cu_seqlens: list[int],
    max_norm: bool | None,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    i = torch.randn((1, T, H), dtype=dtype, device=device).requires_grad_()
    f = torch.randn((1, T, H), dtype=dtype, device=device).requires_grad_()
    c0 = torch.randn(N, H, D, D, device=device).requires_grad_()
    n0 = torch.randn(N, H, D, device=device).requires_grad_()
    m0 = torch.randn(N, H, device=device)
    dct = torch.randn_like(c0)
    dnt = torch.randn_like(n0)
    dh = torch.randn_like(v)

    refs, ref_cts, ref_nts, ref_mts = [], [], [], []
    for n, (bos, eos) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)):
        ref, (ref_ct, ref_nt, ref_mt) = naive_recurrent_mlstm(
            q=q[:, bos:eos],
            k=k[:, bos:eos],
            v=v[:, bos:eos],
            i=i[:, bos:eos],
            f=f[:, bos:eos],
            scale=scale,
            initial_state=(c0[n:n+1], n0[n:n+1], m0[n:n+1]),
            output_final_state=True,
            max_normalisation=max_norm,
        )
        refs.append(ref)
        ref_cts.append(ref_ct)
        ref_nts.append(ref_nt)
        ref_mts.append(ref_mt)
    ref = torch.cat(refs, 1)
    ref_ct = torch.cat(ref_cts, 0)
    ref_nt = torch.cat(ref_nts, 0)
    ref_mt = torch.cat(ref_mts, 0)
    ((ref * dh).sum() + (ref_ct * dct).sum() + (ref_nt * dnt).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_di, i.grad = i.grad.clone(), None
    ref_df, f.grad = f.grad.clone(), None
    ref_dc0, c0.grad = c0.grad.clone(), None
    ref_dn0, n0.grad = n0.grad.clone(), None

    tri, (tri_ct, tri_nt, tri_mt) = fused_recurrent_mlstm(
        q=q,
        k=k,
        v=v,
        i=i,
        f=f,
        scale=scale,
        initial_state=(c0, n0, m0),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        max_normalisation=max_norm,
    )
    ((tri * dh).sum() + (tri_ct * dct).sum() + (tri_nt * dnt).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_di, i.grad = i.grad.clone(), None
    tri_df, f.grad = f.grad.clone(), None
    tri_dc0, c0.grad = c0.grad.clone(), None
    tri_dn0, n0.grad = n0.grad.clone(), None

    assert_close('h', ref, tri, 0.005)
    assert_close('ct', ref_ct, tri_ct, 0.005)
    assert_close('nt', ref_nt, tri_nt, 0.005)
    assert_close('mt', ref_mt, tri_mt, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('di', ref_di, tri_di, 0.005)
    assert_close('df', ref_df, tri_df, 0.005, err_atol=2e-4)
    assert_close('dc0', ref_dc0, tri_dc0, 0.005)
    assert_close('dn0', ref_dn0, tri_dn0, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'scale', 'max_norm', 'dtype', 'use_initial_state', 'output_final_state'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-V{}-scale{}-max_norm={}-{}-initial_state-{}-final_state-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 64, 1, None, torch.float16, True, True),
            (1, 63, 1, 64, 64, 1, None, torch.float32, True, True),
            (1, 63, 1, 64, 64, 1, False, torch.float32, True, True),
            (1, 63, 1, 64, 64, 1, True, torch.float32, True, True),
            (2, 500, 3, 60, 60, 1, None, torch.float16, True, True),
            (1, 1000, 4, 128, 128, 1, None, torch.float16, True, True),
            (2, 1000, 4, 128, 128, 0.1, None, torch.float16, True, True),
            (3, 1000, 4, 128, 128, 0.1, None, torch.float16, True, True),
            (4, 2048, 8, 64, 64, 0.1, None, torch.float16, True, True),
            (2, 7, 2, 16, 16, 1, False, torch.float32, False, True),
            (2, 7, 2, 16, 16, 1, True, torch.float32, False, True),
            (2, 7, 2, 5, 7, 1, True, torch.float32, True, True),
            (2, 7, 2, 16, 32, 1, True, torch.float32, True, True),
            (2, 7, 2, 32, 16, 1, True, torch.float32, True, True),
            (2, 63, 2, 16, 16, 1, False, torch.float32, True, False),
            (2, 63, 2, 16, 16, 1, True, torch.float32, True, False),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    scale: float,
    max_norm: bool | None,
    dtype: torch.dtype,
    use_initial_state: bool,
    output_final_state: bool,
):
    torch.manual_seed(42)
    q = torch.randn((B, T, H, K), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, K), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, V), dtype=dtype, device=device).requires_grad_(True)
    i = torch.randn((B, T, H), dtype=dtype, device=device).requires_grad_(True)
    f = torch.randn((B, T, H), dtype=dtype, device=device).requires_grad_(True)
    c0 = torch.rand((B, H, K, V), dtype=dtype, device=device).requires_grad_(True)
    n0 = torch.rand((B, H, K), dtype=dtype, device=device).requires_grad_(True)
    m0 = torch.rand((B, H), dtype=dtype, device=device)
    dct = torch.randn_like(c0)
    dnt = torch.randn_like(n0)
    dh = torch.randn_like(v)

    ref, ref_state = naive_chunk_mlstm(
        q=q,
        k=k,
        v=v,
        i=i,
        f=f,
        scale=scale,
        initial_state=(c0, n0, m0) if use_initial_state else None,
        output_final_state=output_final_state,
        max_normalisation=max_norm,
    )
    ref_loss = (ref * dh).sum()
    if output_final_state:
        ref_ct, ref_nt, ref_mt = ref_state
        ref_loss += (ref_ct * dct).sum() + (ref_nt * dnt).sum()
    else:
        assert ref_state is None
    ref_loss.backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_di, i.grad = i.grad.clone(), None
    ref_df, f.grad = f.grad.clone(), None
    if use_initial_state:
        ref_dc0, c0.grad = c0.grad.clone(), None
        ref_dn0, n0.grad = n0.grad.clone(), None

    tri, tri_state = chunk_mlstm(
        q=q,
        k=k,
        v=v,
        i=i,
        f=f,
        scale=scale,
        initial_state=(c0, n0, m0) if use_initial_state else None,
        output_final_state=output_final_state,
        max_normalisation=max_norm,
    )
    tri_loss = (tri * dh).sum()
    if output_final_state:
        tri_ct, tri_nt, tri_mt = tri_state
        tri_loss += (tri_ct * dct).sum() + (tri_nt * dnt).sum()
    else:
        assert tri_state is None
    tri_loss.backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_di, i.grad = i.grad.clone(), None
    tri_df, f.grad = f.grad.clone(), None
    if use_initial_state:
        assert c0.grad is not None, "missing gradient for initial matrix state"
        assert n0.grad is not None, "missing gradient for initial normalizer state"
        tri_dc0, c0.grad = c0.grad.clone(), None
        tri_dn0, n0.grad = n0.grad.clone(), None

    assert_close('h', ref, tri, 0.004)
    if output_final_state:
        assert_close('ct', ref_ct, tri_ct, 0.005)
        assert_close('nt', ref_nt, tri_nt, 0.005)
        assert_close('mt', ref_mt, tri_mt, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('di', ref_di, tri_di, 0.005)
    assert_close('df', ref_df, tri_df, 0.005)
    if use_initial_state:
        assert_close('dc0', ref_dc0, tri_dc0, 0.005)
        assert_close('dn0', ref_dn0, tri_dn0, 0.005)


@pytest.mark.parametrize(
    ('implementation', 'state_dtype'),
    [
        pytest.param(fused_recurrent_mlstm, torch.float32, id='fused_recurrent'),
        pytest.param(chunk_mlstm, torch.float16, id='chunk'),
    ],
)
@pytest.mark.parametrize('output_final_state', [False, True])
def test_fp16_state(implementation, state_dtype: torch.dtype, output_final_state: bool):
    torch.manual_seed(42)
    B, T, H, K, V = 2, 7, 2, 16, 16
    q = torch.randn(B, T, H, K, dtype=torch.float16, device=device).requires_grad_()
    k = torch.randn_like(q).requires_grad_()
    v = torch.randn(B, T, H, V, dtype=torch.float16, device=device).requires_grad_()
    i = torch.randn(B, T, H, dtype=torch.float16, device=device).requires_grad_()
    f = torch.randn_like(i).requires_grad_()
    c0 = torch.randn(B, H, K, V, dtype=torch.float16, device=device).requires_grad_()
    n0 = torch.randn(B, H, K, dtype=torch.float16, device=device).requires_grad_()
    m0 = torch.randn(B, H, dtype=torch.float16, device=device)
    dh, dc, dn = torch.randn_like(v), torch.randn_like(c0), torch.randn_like(n0)
    inputs = (q, k, v, i, f, c0, n0)
    ref, ref_state = naive_recurrent_mlstm(
        q=q.float(),
        k=k.float(),
        v=v.float(),
        i=i.float(),
        f=f.float(),
        scale=1,
        initial_state=(c0.float(), n0.float(), m0.float()),
        output_final_state=output_final_state,
    )
    ref_loss = (ref * dh).sum()
    if output_final_state:
        ref_loss += (ref_state[0] * dc).sum() + (ref_state[1] * dn).sum()
    ref_grads = torch.autograd.grad(ref_loss, inputs)
    tri, tri_state = implementation(
        q=q,
        k=k,
        v=v,
        i=i,
        f=f,
        scale=1,
        initial_state=(c0, n0, m0),
        output_final_state=output_final_state,
    )
    assert tri.dtype == torch.float16
    tri_loss = (tri * dh).sum()
    if output_final_state:
        for name, ref_value, tri_value in zip(('ct', 'nt', 'mt'), ref_state, tri_state):
            assert tri_value.dtype == state_dtype
            assert_close(name, ref_value, tri_value.float(), 0.005)
        tri_loss += (tri_state[0] * dc).sum() + (tri_state[1] * dn).sum()
    else:
        assert tri_state is None
    tri_grads = torch.autograd.grad(tri_loss, inputs)
    assert_close('h', ref, tri.float(), 0.005)
    for name, ref_grad, tri_grad in zip(('dq', 'dk', 'dv', 'di', 'df', 'dc0', 'dn0'), ref_grads, tri_grads):
        assert tri_grad.dtype == torch.float16
        assert_close(name, ref_grad.float(), tri_grad.float(), 0.005)


@pytest.mark.parametrize('output_final_state', [False, True])
def test_chunk_fp16_accumulation(fp16_accumulation, output_final_state: bool):
    torch.manual_seed(42)
    B, T, H, K, V = 2, 7, 2, 16, 16
    inputs = tuple(
        torch.randn(shape, dtype=torch.float16, device=device).requires_grad_()
        for shape in ((B, T, H, K), (B, T, H, K), (B, T, H, V), (B, T, H), (B, T, H))
    )
    ref_inputs = tuple(x.detach().float().requires_grad_() for x in inputs)
    dh = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    dc = torch.randn(B, H, K, V, dtype=torch.float32, device=device)
    dn = torch.randn(B, H, K, dtype=torch.float32, device=device)
    ref, ref_state = naive_chunk_mlstm(
        *ref_inputs, scale=1, output_final_state=output_final_state, max_normalisation=True,
    )
    tri, tri_state = chunk_mlstm(
        *inputs, scale=1, output_final_state=output_final_state, max_normalisation=True,
    )
    assert tri.dtype == torch.float16
    assert_close('h', ref, tri.float(), 0.004)
    ref_loss, tri_loss = (ref * dh).sum(), (tri.float() * dh).sum()
    if output_final_state:
        for name, ref_value, tri_value in zip(('ct', 'nt', 'mt'), ref_state, tri_state):
            assert tri_value.dtype == torch.float16
            assert_close(name, ref_value, tri_value.float(), 0.005)
        ref_loss += (ref_state[0] * dc).sum() + (ref_state[1] * dn).sum()
        tri_loss += (tri_state[0].float() * dc).sum() + (tri_state[1].float() * dn).sum()
    else:
        assert ref_state is None
        assert tri_state is None
    ref_grads = torch.autograd.grad(ref_loss, ref_inputs)
    tri_grads = torch.autograd.grad(tri_loss, inputs)
    for name, ref_grad, tri_grad in zip(('dq', 'dk', 'dv', 'di', 'df'), ref_grads, tri_grads):
        assert tri_grad.dtype == torch.float16
        assert_close(name, ref_grad, tri_grad.float(), 0.005)


@pytest.mark.parametrize(
    ('H', 'D', 'cu_seqlens', 'max_norm', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-cu_seqlens{}-max_norm={}-{}".format(*test))
        for test in [
            (4, 64, [0, 15], None, torch.float16),
            (4, 64, [0, 15], False, torch.float32),
            (4, 64, [0, 15], True, torch.float32),
            (4, 64, [0, 256, 500, 1000], None, torch.float16),
            (4, 100, [0, 15, 100, 300, 1200, 2000], None, torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_chunk_varlen(
    H: int,
    D: int,
    cu_seqlens: list[int],
    max_norm: bool | None,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    i = torch.randn((1, T, H), dtype=dtype, device=device).requires_grad_()
    f = torch.randn((1, T, H), dtype=dtype, device=device).requires_grad_()
    c0 = torch.randn((N, H, D, D), dtype=torch.float32, device=device).requires_grad_()
    n0 = torch.randn((N, H, D), dtype=torch.float32, device=device).requires_grad_()
    m0 = torch.randn((N, H), dtype=torch.float32, device=device).requires_grad_()
    dct = torch.randn_like(c0)
    dnt = torch.randn_like(n0)
    dh = torch.randn_like(v)

    refs, refs_c, refs_n, refs_m = [], [], [], []
    for n in range(N):
        ref, (ref_ct, ref_nt, ref_mt) = naive_chunk_mlstm(
            q=q[:, cu_seqlens[n]:cu_seqlens[n + 1]],
            k=k[:, cu_seqlens[n]:cu_seqlens[n + 1]],
            v=v[:, cu_seqlens[n]:cu_seqlens[n + 1]],
            i=i[:, cu_seqlens[n]:cu_seqlens[n + 1]],
            f=f[:, cu_seqlens[n]:cu_seqlens[n + 1]],
            initial_state=(c0[n:n+1], n0[n:n+1], m0[n:n+1]),
            output_final_state=True,
            max_normalisation=max_norm,
        )
        refs.append(ref)
        refs_c.append(ref_ct)
        refs_n.append(ref_nt)
        refs_m.append(ref_mt)
    ref = torch.cat(refs, dim=1)
    ref_ct = torch.cat(refs_c, dim=0)
    ref_nt = torch.cat(refs_n, dim=0)
    ref_mt = torch.cat(refs_m, dim=0)
    ((ref * dh).sum() + (dct * ref_ct).sum() + (dnt * ref_nt).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_di, i.grad = i.grad.clone(), None
    ref_df, f.grad = f.grad.clone(), None
    ref_dc0, c0.grad = c0.grad.clone(), None
    ref_dn0, n0.grad = n0.grad.clone(), None

    tri, (tri_ct, tri_nt, tri_mt) = chunk_mlstm(
        q=q,
        k=k,
        v=v,
        i=i,
        f=f,
        initial_state=(c0, n0, m0),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        max_normalisation=max_norm,
    )
    ((tri * dh).sum() + (dct * tri_ct).sum() + (dnt * tri_nt).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_di, i.grad = i.grad.clone(), None
    tri_df, f.grad = f.grad.clone(), None
    tri_dc0, c0.grad = c0.grad.clone(), None
    tri_dn0, n0.grad = n0.grad.clone(), None

    assert_close('h', ref, tri, 0.004)
    assert_close('ct', ref_ct, tri_ct, 0.005)
    assert_close('nt', ref_nt, tri_nt, 0.005)
    assert_close('mt', ref_mt, tri_mt, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('di', ref_di, tri_di, 0.005)
    assert_close('df', ref_df, tri_df, 0.005)
    assert_close('dc0', ref_dc0, tri_dc0, 0.005)
    assert_close('dn0', ref_dn0, tri_dn0, 0.005)
