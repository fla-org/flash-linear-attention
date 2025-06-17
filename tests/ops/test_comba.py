# -*- coding: utf-8 -*-

import os

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from fla.ops.comba import chunk_comba, fused_recurrent_comba
from fla.ops.comba.utils import chunk_comba_cumsum_scalar_fwd
from fla.utils import COMPILER_MODE, assert_close, device, is_intel_alchemist

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
test_hv_list = [4]


def cumsum_comba_local_fwd_reference(s, reverse=False, chunk_size=128):
    o_0 = torch.zeros_like(s)
    o_1 = torch.zeros_like(s)
    T = s.size(1)
    fn = torch.cumsum
    for i in range(0, T, chunk_size):
        s_chunk = s[:, i:i+chunk_size]
        o_1[:, i:i+chunk_size] = fn(s_chunk.float(), dim=1).to(o_1)
        o_0[:, i:i+chunk_size] = o_1[:, i:i+chunk_size] - s_chunk

    return o_0, o_1


@pytest.mark.parametrize("B", [32])
@pytest.mark.parametrize("T", [256, 1024, 2048])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("dtype", [torch.float, torch.float16])
@pytest.mark.parametrize("chunk_size", [32, 64])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
def test_cumsum_local_scalar_fwd(B, T, H, dtype, chunk_size):
    s = torch.randn((B, T, H), dtype=dtype, device=device).requires_grad_()
    ref_0, ref_1 = cumsum_comba_local_fwd_reference(s, chunk_size=chunk_size)
    tri_0, tri_1 = chunk_comba_cumsum_scalar_fwd(s, chunk_size=chunk_size)
    assert_close("local cumsum scalar", ref_0, tri_0, 0.001 if dtype == torch.float else 0.003)
    assert_close("local cumsum scalar", ref_1, tri_1, 0.001 if dtype == torch.float else 0.003)


def chunk_comba_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    BT = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    # Calculate padding needed to make T a multiple of BT
    q, k, v, p, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, p, beta, g])

    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        # Pad all tensors
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        p = F.pad(p, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))
    q, k, v, p, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, p, beta, g])
    decay = g
    chunk_size = BT
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    v = v * beta[..., None]
    p_beta = p * beta[..., None]
    assert l % chunk_size == 0
    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, p_beta, decay, g = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size),
        [q, k, v, p_beta, decay.unsqueeze(-1), g.unsqueeze(-1)]
    )
    decay = decay.squeeze(-1).cumsum(-1)  # [B, H, n, c]
    decay_0 = decay - g.squeeze(-1)  # [B, H, n, c]
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
    L_mask_0 = ((decay_0.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
    # [B, H, n, c, d] @ [B, H, n, d, c] -> [B, H, n, c, c]
    attn = -((p_beta @ k.transpose(-1, -2)) * L_mask_0).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    # for U
    k_cumsum = attn @ v
    # for W
    k_cumdecay = attn @ (p_beta * decay_0[..., None].exp())
    v = k_cumsum
    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S = initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, l // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn @ v_new
        S = S * decay[:, :, i, -1, None, None].exp() + (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()
                                                        [..., None]).transpose(-1, -2) @ v_new
    if not output_final_state:
        S = None
    # unpad
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T]
    o = o.transpose(1, 2)
    return o, S


@pytest.mark.parametrize('B', [4, 8, 16])
@pytest.mark.parametrize('T', [1024, 1314, 2048])
@pytest.mark.parametrize('H', [2, 4])
@pytest.mark.parametrize('D', [128, 256])
@pytest.mark.parametrize('gate_logit_normalizer', [1])
@pytest.mark.parametrize('scale', [1, 0.1])
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
def test_chunk_dplr(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    scale: float,
    gate_logit_normalizer: float
):
    if is_intel_alchemist and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')

    q = torch.randn(B, T, H, D, dtype=dtype)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    p = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    g = g / gate_logit_normalizer
    q, k, v, p, beta, g, h0 = map(lambda x: x.cuda().requires_grad_(True), (q, k, v, p, beta, g, h0))

    tri, tri_ht = chunk_comba(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        p.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dp, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, p.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = p.grad = beta.grad = g.grad = h0.grad = None

    ref, ref_ht = chunk_comba_ref(
        q.clone(),
        k.clone(),
        v.clone(),
        p.clone(),
        g.clone(),
        beta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dp, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, p.grad, beta.grad, g.grad, h0.grad

    assert_close("  o", ref, tri, 0.005)
    assert_close(" ht", ref_ht, tri_ht, 0.005)
    assert_close(" dq", ref_dq, tri_dq, 0.005)
    assert_close(" dk", ref_dk, tri_dk, 0.008)
    assert_close(" dv", ref_dv, tri_dv, 0.005)
    assert_close(" dp", ref_dp, tri_dp, 0.008)
    assert_close(" db", ref_dbeta, tri_dbeta, 0.005)
    assert_close("dh0", ref_dh0, tri_dh0, 0.008)
    if gate_logit_normalizer >= 1 and ref_dg.norm() > 0.01:
        assert_close("dg", ref_dg, tri_dg, 0.02)


@pytest.mark.parametrize('B', [4, 8, 16])
@pytest.mark.parametrize('T', [1024, 1314, 2048])
@pytest.mark.parametrize('H', [2, 4])
@pytest.mark.parametrize('D', [64, 128, 256])
@pytest.mark.parametrize('gate_logit_normalizer', [1])
@pytest.mark.parametrize('scale', [1, 0.1])
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
def test_forward(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    scale: float,
    gate_logit_normalizer: float
):
    if is_intel_alchemist and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')

    q = torch.randn(B, T, H, D, dtype=dtype)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    p = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    g = g / gate_logit_normalizer
    q, k, v, p, beta, g, h0 = map(lambda x: x.cuda().requires_grad_(True), (q, k, v, p, beta, g, h0))

    tri, tri_ht = chunk_comba(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        p.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )

    ref, ref_ht = fused_recurrent_comba(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        p.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )

    assert_close("  o", ref, tri, 0.005)
    assert_close(" ht", ref_ht, tri_ht, 0.005)
