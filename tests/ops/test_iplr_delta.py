# -*- coding: utf-8 -*-

import os
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from fla.ops.generalized_delta_rule.iplr.chunk import chunk_iplr_delta_rule
from fla.ops.generalized_delta_rule.iplr.fused_recurrent import fused_recurrent_iplr_delta_rule
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
    test_d_list = [32, 64, 100, 256]
    test_gate_list = [1, 0.1, 10]
test_h_list = [2]


def chunk_iplr_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    scale: float = None,
    chunk_size: int = 64,
):
    BT = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)

    q, k, v, a, b = map(lambda x: x.transpose(1, 2), (q, k, v, a, b))
    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        a = F.pad(a, (0, 0, 0, pad_len))
        b = F.pad(b, (0, 0, 0, pad_len))
    q, k, v, a, b = map(lambda x: x.to(torch.float32), [q, k, v, a, b])

    B, H, L, DK = q.shape
    DV = v.shape[-1]
    q = q * scale

    S = k.new_zeros(B, H, DK, DV)
    if initial_state is not None:
        S += initial_state

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, a, b = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, a, b])

    v2 = (a @ k.transpose(-1, -2)).masked_fill_(mask, 0) @ v
    attn = (a @ b.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    u = attn @ v2
    w = attn @ a
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, L // chunk_size):
        current_chunk_size = min(chunk_size, L - i * chunk_size)  # to handle the last chunk with possibly padding
        q_i = q[:, :, i, :current_chunk_size]
        k_i = k[:, :, i, :current_chunk_size]
        v_i = v[:, :, i, :current_chunk_size]
        u_i = u[:, :, i, :current_chunk_size]
        w_i = w[:, :, i, :current_chunk_size]
        b_i = b[:, :, i, :current_chunk_size]
        o_1 = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask, 0) @ v_i
        v2_i = u_i + w_i @ S
        o_2 = (q_i @ b_i.transpose(-1, -2)).masked_fill_(mask, 0) @ v2_i
        o_3 = q_i @ S
        o[:, :, i, :current_chunk_size] = o_1 + o_2 + o_3
        S = S + k_i.transpose(-1, -2) @ v_i + b_i.transpose(-1, -2) @ v2_i
    S = None if output_final_state is False else S
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T]
    o = o.transpose(1, 2)
    return o, S


def recurrence_iplr_delta_rule_ref(
    q,
    k,
    v,
    a,
    b,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    scale: Optional[float] = None
):
    orig_dtype = q.dtype
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q, k, v, a, b = map(lambda x: x.transpose(1, 2).to(torch.float32), [q, k, v, a, b])
    q = q * scale
    B, H, L, DK = q.shape
    DV = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(B, H, DK, DV).to(v)
    if initial_state is not None:
        S += initial_state

    for i in range(q.shape[-2]):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i]
        _a = a[:, :, i]
        _b = b[:, :, i]
        _kv = _k[..., None] * _v[..., None, :] + (S.clone() * _a[..., None]).sum(-2, keepdim=True) * _b[..., None]
        S = S + _kv
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    S = None if output_final_state is False else S
    o = o.transpose(1, 2)
    return o.to(orig_dtype), S


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('scale', [0.25])
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
):
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    a = torch.rand(B, T, H, D, dtype=dtype)

    a = F.normalize(a, p=2, dim=-1)
    b = -a
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    q, k, v, a, b, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, a, b, h0))
    ref, ref_ht = recurrence_iplr_delta_rule_ref(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    tri, tri_ht = chunk_iplr_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    assert_close(' o', ref, tri, 0.007)
    assert_close('ht', ref_ht, tri_ht, 0.008)


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('scale', [0.25])
@pytest.mark.parametrize('dtype', [torch.float16])
def test_recurrent(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
):
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    a = torch.rand(B, T, H, D, dtype=dtype)

    a = F.normalize(a, p=2, dim=-1)
    b = -a
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    q, k, v, a, b, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, a, b, h0))
    ref, ref_ht = recurrence_iplr_delta_rule_ref(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    dht = torch.rand_like(h0)
    do = torch.rand_like(ref)
    ((dht * ref_ht).sum() + (do * ref).sum()).backward()
    dq, dk, dv, da, db, dh0 = map(lambda x: x.grad, (q, k, v, a, b, h0))
    q.grad, k.grad, v.grad, a.grad, b.grad, h0.grad = None, None, None, None, None, None
    tri, tri_ht = fused_recurrent_iplr_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    ((dht * tri_ht).sum() + (do * tri).sum()).backward()
    assert_close('  o', ref, tri, 0.003)
    assert_close(' ht', ref_ht, tri_ht, 0.003)
    assert_close(' dq', dq, q.grad, 0.003)
    assert_close(' dk', dk, k.grad, 0.003)
    assert_close(' dv', dv, v.grad, 0.003)
    assert_close(' da', da, a.grad, 0.003)
    assert_close(' db', db, b.grad, 0.003)
    assert_close('dh0', dh0, h0.grad, 0.003)
