# -*- coding: utf-8 -*-

import os
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from fla.ops.simple_gla import chunk_simple_gla
from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
from fla.ops.simple_gla.parallel import parallel_simple_gla
from fla.ops.utils.testing import COMPILER_MODE, assert_close
from fla.utils import device

if COMPILER_MODE:
    test_b_list = [1]
    test_t_list = [4096]
    test_t_varlen_list = test_t_list
    test_d_list = [64, 128, 256]
    test_gate_list = [1.0]
else:
    test_b_list = [2]
    test_t_list = [1, 15, 63, 300]
    test_t_varlen_list = [63, 286, 300, 512]
    test_d_list = [64, 32, 100, 256]
    test_gate_list = [1, 0.1, 10]
test_h_list = [2]


def chunk_simple_gla_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    scale: Optional[float] = None,
):
    q, k, v = map(lambda x: rearrange(x, 'b t h ... -> b h t ...'), [q, k, v])
    if g is not None:
        g = rearrange(g, 'b t h ... -> b h t ...')
    if scale is None:
        scale = 1.0 / q.shape[-1] ** 0.5

    T = q.shape[-2]
    BT = chunk_size
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        # Pad all tensors
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        g = F.pad(g, (0, pad_len))
    q, k, v, g = map(lambda x: x.to(torch.float32), [q, k, v, g])
    decay = g
    b, h, t, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    q, k, v, decay = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, decay.unsqueeze(-1)])
    decay = decay.squeeze(-1).cumsum(-1)
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S = initial_state
    o = torch.zeros_like(v)
    for i in range(0, t // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i])
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn @ v_i
        S = S * decay[:, :, i, -1, None, None].exp() + \
            (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_i
    if not output_final_state:
        S = None
    # unpad
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T]
    o = o.transpose(1, 2)
    return o, S


def parallel_simple_gla_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None
):
    q, k, v = map(lambda x: rearrange(x, 'b t h ... -> b h t ...'), [q, k, v])
    if g is not None:
        g = rearrange(g, 'b t h ... -> b h t ...')
    if scale is None:
        scale = 1.0 / q.shape[-1] ** 0.5
    original_dtype = q.dtype
    q, k, v, g = map(lambda x: x.float() if x is not None else None, [q, k, v, g])
    A = (q @ k.transpose(-1, -2) * scale)
    if g is not None:
        g = g.cumsum(-1)
        D = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
        A = A * D
    else:
        A = A.tril()
    o = A @ v
    o = o.transpose(1, 2)
    return o.to(original_dtype), A


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('gate_logit_normalizer', test_gate_list)
@pytest.mark.parametrize('dtype', [torch.float])
@pytest.mark.parametrize('scale', [1, 0.1])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    scale: float,
    gate_logit_normalizer: float
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    g = torch.randn((B, T, H), dtype=torch.float32, device=device)
    h0 = torch.rand((B, H, D, D), dtype=torch.float32, device=device).requires_grad_(True)
    dht = torch.randn_like(h0)
    g = (F.logsigmoid(g) / gate_logit_normalizer).requires_grad_(True)
    do = torch.randn_like(v)

    ref, ref_ht = chunk_simple_gla_ref(
        q, k, v, g,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    ((ref * do).sum() + (dht * ref_ht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = chunk_simple_gla(
        q, k, v, g,
        scale=scale,
        initial_state=h0,
        output_final_state=True
    )
    ((tri * do).sum() + (dht * tri_ht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('  o', ref, tri, 0.004)
    assert_close(' ht', ref_ht, tri_ht, 0.005)
    assert_close(' dq', ref_dq, tri_dq, 0.005)
    assert_close(' dk', ref_dk, tri_dk, 0.005)
    assert_close(' dv', ref_dv, tri_dv, 0.005)
    assert_close(' dg', ref_dg, tri_dg, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize('N', test_b_list)
@pytest.mark.parametrize('T', test_t_varlen_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set'
)
def test_chunk_varlen(
    N: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    # randomly split the sequence into N segments
    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 16)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]
    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    g = F.logsigmoid(torch.randn((1, T, H), dtype=dtype, device=device)).requires_grad_()
    h0 = torch.randn((N, H, D, D), dtype=torch.float32, device=device).requires_grad_()
    do = torch.randn_like(v)

    ref, ref_ht = fused_recurrent_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((ref * do).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = chunk_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('  o', ref, tri, 0.004)
    assert_close(' ht', ref_ht, tri_ht, 0.005)
    assert_close(' dq', ref_dq, tri_dq, 0.005)
    assert_close(' dk', ref_dk, tri_dk, 0.005)
    assert_close(' dv', ref_dv, tri_dv, 0.005)
    assert_close(' dg', ref_dg, tri_dg, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize('N', test_b_list)
@pytest.mark.parametrize('T', test_t_varlen_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set'
)
def test_parallel_varlen(
    N: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    # randomly split the sequence into N segments
    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 16)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]
    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    g = F.logsigmoid(torch.randn((1, T, H), dtype=dtype, device=device)).requires_grad_()
    do = torch.randn_like(v)

    ref, _ = fused_recurrent_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    ((ref * do).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    tri, _ = parallel_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert_close('  o', ref, tri, 0.004)
    assert_close(' dq', ref_dq, tri_dq, 0.005)
    assert_close(' dk', ref_dk, tri_dk, 0.005)
    assert_close(' dv', ref_dv, tri_dv, 0.005)
    assert_close(' dg', ref_dg, tri_dg, 0.005)


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('gate_logit_normalizer', test_gate_list)
@pytest.mark.parametrize('scale', [0.1])
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
def test_parallel(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype,
    scale: float,
    gate_logit_normalizer: float,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    USE_G = gate_logit_normalizer > 0
    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    g = F.logsigmoid(torch.randn((B, T, H), dtype=dtype, device=device)) if USE_G else None
    g = (g / gate_logit_normalizer).requires_grad_(True) if USE_G else None
    do = torch.randn_like(v)

    ref, ref_A = parallel_simple_gla_ref(q=q, k=k, v=v, g=g, scale=scale)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    if USE_G:
        ref_dg, g.grad = g.grad.clone(), None

    tri, tri_A = parallel_simple_gla(q=q, k=k, v=v, g=g, scale=scale, output_attentions=True)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    if USE_G:
        tri_dg, g.grad = g.grad.clone(), None
    assert_close(' o', ref, tri, 0.005)
    assert_close(' A', ref_A, tri_A, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    if USE_G:
        assert_close('dg', ref_dg, tri_dg, 0.015)


@pytest.mark.parametrize('vary_A', [True, False])
@pytest.mark.parametrize('dtype', [torch.float, torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
def test_simple_gla_to_mamba2(vary_A, dtype):
    try:
        from mamba_ssm.modules.ssd_minimal import ssd_minimal_discrete
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except ImportError:
        pytest.skip('mamba_ssm is not installed.')
    torch.manual_seed(42)

    # Dimensions, Denoted (B, T, Q, D, P) in Mamba2 paper
    batch, seq_len, chunk_size, dim, headdim = 2, 512, 8, 64, 16
    n_heads = dim // headdim  # (H) in the paper
    ngroups = n_heads  # (G) in the paper; NOTE: do not use group-query here
    dstate = 64  # (N) in the paper
    atol = 5e-4 if dtype == torch.float else 1e-2

    x = 0.1 * torch.randn(batch, seq_len, n_heads, headdim, dtype=dtype, device=device)
    dt = torch.ones(batch, seq_len, n_heads, dtype=dtype, device=device)  # dt=1 can be ignored

    if vary_A:
        A = -0.1 * torch.rand(1, seq_len, n_heads, dtype=dtype, device=device)
    else:  # constant A for all position
        A = -0.1 * torch.rand(n_heads, dtype=dtype, device=device)

    B = 0.1 * torch.randn(batch, seq_len, ngroups, dstate, dtype=dtype, device=device)
    C = 0.1 * torch.randn(batch, seq_len, ngroups, dstate, dtype=dtype, device=device)

    y_ssd, final_ssd = ssd_minimal_discrete(x * dt.unsqueeze(-1), A * dt, B, C, chunk_size)

    if not vary_A:
        # NOTE: fused kernel does not support varying A with time
        y_fuse, final_fuse = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, return_final_states=True)
        assert y_ssd.allclose(y_fuse, 0, atol), f'y diff: {torch.abs(y_ssd - y_fuse).max()}'
        # fused kernel upcasts state to float32
        # https://github.com/state-spaces/mamba/blob/v2.2.2/mamba_ssm/ops/triton/ssd_combined.py#L650
        final_fuse = final_fuse.to(dtype)
        assert final_ssd.allclose(final_fuse, 0, atol), f'final diff: {torch.abs(final_ssd - final_fuse).max()}'

    # mapping inputs Mamba2 -> FLA
    # C, B, X: [batch, seq, head, hidden] -> [batch, head, seq, hidden]
    # g: [batch, seq, head] -> [batch, head, seq]
    q = C.transpose(1, 2)
    k = B.transpose(1, 2)
    v = x.transpose(1, 2)
    g = (A * dt).transpose(1, 2)

    # mapping outputs Mamba2 -> FLA
    y_rearrange = y_ssd.transpose(1, 2)
    final_rearrange = final_ssd.transpose(2, 3)

    # comparing output results between FLA kernel and Mamba2 kernel
    outputs_gla_fuse, final_gla_fuse = chunk_simple_gla(q, k, v, g, scale=1.0, output_final_state=True)
    assert y_rearrange.allclose(outputs_gla_fuse, 0, atol), f'y diff: {torch.abs(y_rearrange - outputs_gla_fuse).max()}'
    final_gla_fuse = final_gla_fuse.to(dtype)  # states hard-coded to float32 in FLA kernel
    assert final_rearrange.allclose(final_gla_fuse, 0, atol), f'final diff: {torch.abs(final_ssd - final_gla_fuse).max()}'
