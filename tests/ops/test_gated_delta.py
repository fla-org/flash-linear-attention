# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch
import torch.nn.functional as F
from einops import repeat

from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.gated_delta_rule.gate import fused_gdn_gate, naive_gdn_gate
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'scale', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HV{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 1, 1, torch.float),
            (2, 500, 4, 4, 60, 1, 1, torch.float),
            (2, 1000, 2, 8, 128, 1, 0.1, torch.float),
            (3, 1024, 2, 2, 128, 0.1, 1, torch.float),
            (4, 1024, 3, 3, 128, 1, 10, torch.float),
            (4, 2048, 4, 4, 64, 0.1, 1, torch.float),
            (2, 1024, 4, 4, 128, 1, 0.1, torch.float16),
            (2, 1024, 4, 8, 128, 1, 10, torch.float16),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=torch.float32)
    k = torch.randn(B, T, H, D, dtype=torch.float32)
    v = torch.randn(B, T, HV, D, dtype=dtype)
    beta = torch.rand(B, T, HV, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32))
    g = g / gate_logit_normalizer
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0))
    ref, ref_ht = naive_recurrent_gated_delta_rule(
        q=F.normalize(repeat(q.clone(), 'b t h d -> b t (h g) d', g=HV // H), p=2, dim=-1).to(dtype),
        k=F.normalize(repeat(k.clone(), 'b t h d -> b t (h g) d', g=HV // H), p=2, dim=-1).to(dtype),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    tri, tri_ht = fused_recurrent_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
    )
    assert_close('o', ref, tri, 0.002)
    assert_close('ht', ref_ht, tri_ht, 0.002)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'scale', 'gate_logit_normalizer', 'mask_p', 'use_qk_l2norm_in_kernel', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-HV{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-use_qk_l2norm_in_kernel{}-{}".format(*test),
        )
        for test in [
            # non-GVA (HV == H)
            (2, 75, 4, 4, 64, 1, 0.01, 0, False, torch.float16),
            (2, 500, 3, 3, 60, 1, 1, 0, False, torch.float16),
            (2, 1000, 3, 3, 64, 0.1, 1, 0.5, False, torch.float16),
            (3, 1024, 4, 4, 100, 1, 0.1, 0, False, torch.float16),
            (4, 1024, 4, 4, 128, 0.1, 1, 0, True, torch.float16),
            (2, 1500, 4, 4, 128, 0.1, 10, 0, False, torch.float16),
            (4, 2048, 8, 8, 64, 0.1, 1, 0, False, torch.float16),
            # GVA (HV > H)
            (2, 256, 2, 4, 64, 1, 1, 0, False, torch.float16),
            (2, 512, 2, 8, 64, 1, 0.1, 0, True, torch.float16),
            (2, 1024, 4, 8, 128, 0.1, 1, 0, False, torch.float16),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')
    assert HV % H == 0
    G = HV // H

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, HV, D, dtype=dtype)
    beta = torch.rand(B, T, HV, dtype=torch.float).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32))
    g = g / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.zeros(B, HV, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, g, h0))

    tri, tri_ht = chunk_gated_delta_rule(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref, ref_ht = naive_recurrent_gated_delta_rule(
        q=F.normalize(repeat(q.clone(), 'b t h d -> b t (h g) d', g=G), p=2, dim=-1),
        k=F.normalize(repeat(k.clone(), 'b t h d -> b t (h g) d', g=G), p=2, dim=-1),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    assert_close('dg', ref_dg, tri_dg, 0.02)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, 1, torch.float16),
            (2, 500, 3, 60, 1, 1, torch.float16),
            (3, 1024, 4, 128, 0.1, 1, torch.float16),
            (4, 2048, 8, 64, 0.1, 1, torch.float16),
        ]
    ],
)
def test_chunk_transpose_state(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    g = g / gate_logit_normalizer
    # Non-zero initial state so transpose load path is actually exercised
    h0_kv = torch.randn(B, H, D, D, dtype=torch.float32)
    h0_vk = h0_kv.transpose(-1, -2).contiguous()
    q, k, v, beta, g, h0_kv, h0_vk = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, g, h0_kv, h0_vk))

    tri, tri_ht = chunk_gated_delta_rule(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0_vk.clone(),
        output_final_state=True,
        transpose_state_layout=True,
    )
    do = torch.randn_like(v)
    dht_vk = torch.randn(B, H, D, D, dtype=torch.float32, device=device)
    dht_kv = dht_vk.transpose(-1, -2).contiguous()
    ((tri * do).sum() + (tri_ht * dht_vk).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0_vk.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0_vk.grad = None

    ref, ref_ht = chunk_gated_delta_rule(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0_kv.clone(),
        output_final_state=True,
        transpose_state_layout=False,
    )
    ((ref * do).sum() + (ref_ht * dht_kv).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0_kv.grad

    assert_close('o', ref, tri, 1e-4)
    assert_close('ht', ref_ht, tri_ht.transpose(-1, -2), 1e-4)
    assert_close('dq', ref_dq, tri_dq, 1e-4)
    assert_close('dk', ref_dk, tri_dk, 1e-4)
    assert_close('dv', ref_dv, tri_dv, 1e-4)
    assert_close('db', ref_dbeta, tri_dbeta, 1e-4)
    assert_close('dg', ref_dg, tri_dg, 1e-4)
    assert_close('dh0', ref_dh0, tri_dh0.transpose(-1, -2), 1e-4)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'scale', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HV{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 1, 1, torch.float),
            (2, 500, 4, 4, 60, 1, 1, torch.float),
            (2, 1000, 2, 8, 128, 1, 0.1, torch.float),
            (3, 1024, 2, 2, 128, 0.1, 1, torch.float),
            (4, 2048, 4, 4, 64, 0.1, 1, torch.float),
        ]
    ],
)
def test_fused_recurrent_transpose_state(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=torch.float32)
    k = torch.randn(B, T, H, D, dtype=torch.float32)
    v = torch.randn(B, T, HV, D, dtype=dtype)
    beta = torch.rand(B, T, HV, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32))
    g = g / gate_logit_normalizer
    h0_kv = torch.randn(B, HV, D, D, dtype=torch.float32)
    h0_vk = h0_kv.transpose(-1, -2).contiguous()
    q, k, v, beta, g, h0_kv, h0_vk = map(lambda x: x.to(device), (q, k, v, beta, g, h0_kv, h0_vk))

    ref, ref_ht = fused_recurrent_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0_kv.clone(),
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
        transpose_state_layout=False,
    )
    tri, tri_ht = fused_recurrent_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0_vk.clone(),
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
        transpose_state_layout=True,
    )
    assert_close('o', ref, tri, 1e-4)
    assert_close('ht', ref_ht, tri_ht.transpose(-1, -2), 1e-4)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'scale', 'has_dt_bias', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-HV{}-D{}-scale{}-has_dt_bias{}-{}".format(*test),
        )
        for test in [
            (1, 64, 1, 1, 64, 1, False, torch.float),
            (2, 256, 2, 2, 64, 1, True, torch.float),
            (2, 512, 2, 4, 64, 0.1, True, torch.float16),
            (3, 1000, 2, 8, 128, 1, False, torch.float16),
            (4, 1024, 4, 4, 128, 0.1, True, torch.float16),
        ]
    ],
)
def test_fused_recurrent_gate_in_kernel(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    scale: float,
    has_dt_bias: bool,
    dtype: torch.dtype,
):
    """fused_recurrent_gated_delta_rule with use_gate_in_kernel=True matches manual gate."""
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    beta = torch.rand(B, T, HV, dtype=dtype, device=device).sigmoid()
    g_raw = torch.randn(B, T, HV, dtype=torch.float32, device=device)
    A_log = torch.log(torch.empty(HV, dtype=torch.float32, device=device).uniform_(1, 16))
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) if has_dt_bias else None
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32, device=device)

    g_ref = naive_gdn_gate(g_raw, A_log, dt_bias)
    ref, ref_ht = fused_recurrent_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g_ref,
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    tri, tri_ht = fused_recurrent_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g_raw.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=A_log.clone(),
        dt_bias=dt_bias.clone() if dt_bias is not None else None,
    )
    assert_close('o', ref, tri, 0.002)
    assert_close('ht', ref_ht, tri_ht, 0.002)


@pytest.mark.parametrize(
    ('H', 'HV', 'D', 'has_dt_bias', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-HV{}-D{}-has_dt_bias{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 2, 64, True, [0, 15, 100, 300], torch.float16),
            (2, 4, 64, False, [0, 256, 500, 1000], torch.float16),
            (4, 4, 128, True, [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ],
)
def test_fused_recurrent_gate_in_kernel_varlen(
    H: int,
    HV: int,
    D: int,
    has_dt_bias: bool,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    """Varlen fused_recurrent_gated_delta_rule with use_gate_in_kernel=True."""
    torch.manual_seed(42)
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1].item()
    N = len(cu_seqlens) - 1

    q = torch.randn(1, T, H, D, dtype=dtype, device=device)
    k = torch.randn(1, T, H, D, dtype=dtype, device=device)
    v = torch.randn(1, T, HV, D, dtype=dtype, device=device)
    beta = torch.rand(1, T, HV, dtype=dtype, device=device).sigmoid()
    g_raw = torch.randn(1, T, HV, dtype=torch.float32, device=device)
    A_log = torch.log(torch.empty(HV, dtype=torch.float32, device=device).uniform_(1, 16))
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) if has_dt_bias else None
    h0 = torch.randn(N, HV, D, D, dtype=torch.float32, device=device)

    g_ref = naive_gdn_gate(g_raw, A_log, dt_bias)
    ref, ref_ht = fused_recurrent_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g_ref,
        beta=beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    tri, tri_ht = fused_recurrent_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g_raw.clone(),
        beta=beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=A_log.clone(),
        dt_bias=dt_bias.clone() if dt_bias is not None else None,
    )
    assert_close('o', ref, tri, 0.002)
    assert_close('ht', ref_ht, tri_ht, 0.002)


@pytest.mark.parametrize(
    ('H', 'D', 'mask_p', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 60, 0, [0, 15], torch.float16),
            (4, 64, 0, [0, 256, 500, 1000], torch.float16),
            (4, 64, 0.5, [0, 256, 500, 1000], torch.float16),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.float16),
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
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    # randomly split the sequence into N segments
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=dtype))
    g = g * (torch.rand_like(g) > mask_p)
    beta = torch.rand(1, T, H, dtype=torch.float).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=dtype)

    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0))
    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    tri, tri_ht = chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = naive_recurrent_gated_delta_rule(
            q=q[:, cu_seqlens[i]:cu_seqlens[i+1]],
            k=k[:, cu_seqlens[i]:cu_seqlens[i+1]],
            v=v[:, cu_seqlens[i]:cu_seqlens[i+1]],
            beta=beta[:, cu_seqlens[i]:cu_seqlens[i+1]],
            g=g[:, cu_seqlens[i]:cu_seqlens[i+1]],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.007)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.007)
    assert_close('db', ref_dbeta, tri_dbeta, 0.015)
    assert_close('dg', ref_dg, tri_dg, 0.015)
    assert_close('dh0', ref_dh0, tri_dh0, 0.007)


@pytest.mark.parametrize(
    ('H', 'D', 'mask_p', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 60, 0, [0, 8192], torch.float16),
            (4, 60, 0, [0, 15], torch.float16),
            (4, 64, 0, [0, 256, 500, 1000], torch.float16),
            (4, 64, 0.5, [0, 256, 500, 1000], torch.float16),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
@torch.inference_mode()
def test_chunk_varlen_prefill(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    # randomly split the sequence into N segments
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype).to(device)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype).to(device)
    v = torch.randn((1, T, H, D), dtype=dtype).to(device)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=dtype)).to(device)
    g = g * (torch.rand_like(g) > mask_p)
    beta = torch.rand(1, T, H, dtype=dtype).sigmoid().to(device)
    h0 = torch.randn((N, H, D, D), dtype=dtype).to(device)

    tri, tri_ht = chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = naive_recurrent_gated_delta_rule(
            q=q[:, cu_seqlens[i]:cu_seqlens[i+1]],
            k=k[:, cu_seqlens[i]:cu_seqlens[i+1]],
            v=v[:, cu_seqlens[i]:cu_seqlens[i+1]],
            beta=beta[:, cu_seqlens[i]:cu_seqlens[i+1]],
            g=g[:, cu_seqlens[i]:cu_seqlens[i+1]],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'has_dt_bias', 'use_qk_l2norm_in_kernel', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-has_dt_bias{}-use_qk_l2norm{}-{}".format(*test),
        )
        for test in [
            (2, 75, 4, 64, 1, True, True, torch.float16),
            (2, 500, 3, 60, 1, False, False, torch.float16),
            (2, 1000, 3, 64, 0.1, True, False, torch.float16),
            (3, 1024, 4, 100, 1, True, True, torch.float16),
            (4, 1024, 4, 128, 0.1, False, True, torch.float16),
            (4, 2048, 8, 64, 0.1, True, False, torch.float16),
        ]
    ],
)
def test_chunk_gate_in_kernel(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    has_dt_bias: bool,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    """Test use_gate_in_kernel=True path: fused gate activation + chunk cumsum inside kernel."""
    torch.manual_seed(42)
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=torch.float).sigmoid()
    # Raw gate input (before activation)
    g_raw = torch.randn(B, T, H, dtype=torch.float32)
    A_log = torch.randn(H, dtype=torch.float32)
    dt_bias = torch.randn(H, dtype=torch.float32) if has_dt_bias else None
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)

    q, k, v, beta, g_raw, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, g_raw, h0))
    A_log = A_log.to(device).requires_grad_(True)
    dt_bias = dt_bias.to(device).requires_grad_(True) if dt_bias is not None else None

    # === Triton path: use_gate_in_kernel=True ===
    tri, tri_ht = chunk_gated_delta_rule(
        q=q.clone() if use_qk_l2norm_in_kernel else F.normalize(q.clone(), p=2, dim=-1),
        k=k.clone() if use_qk_l2norm_in_kernel else F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g_raw.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=True,
        A_log=A_log.clone(),
        dt_bias=dt_bias.clone() if dt_bias is not None else None,
    )
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g_raw.grad, h0.grad
    tri_dA_log = A_log.grad
    tri_ddt_bias = dt_bias.grad if dt_bias is not None else None
    q.grad = k.grad = v.grad = beta.grad = g_raw.grad = h0.grad = None
    A_log.grad = None
    if dt_bias is not None:
        dt_bias.grad = None

    # === Reference path: manually compute gate, then use_gate_in_kernel=False ===
    g_ref = naive_gdn_gate(g_raw, A_log, dt_bias)
    ref, ref_ht = chunk_gated_delta_rule(
        q=q.clone() if use_qk_l2norm_in_kernel else F.normalize(q.clone(), p=2, dim=-1),
        k=k.clone() if use_qk_l2norm_in_kernel else F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g_ref,
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    ref_dg = g_raw.grad
    ref_dA_log = A_log.grad
    ref_ddt_bias = dt_bias.grad if dt_bias is not None else None

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    assert_close('dg', ref_dg, tri_dg, 0.02)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)
    assert_close('dA_log', ref_dA_log, tri_dA_log, 0.02)
    if dt_bias is not None:
        assert_close('ddt_bias', ref_ddt_bias, tri_ddt_bias, 0.02)


@pytest.mark.parametrize(
    ('B', 'T', 'Hq', 'H', 'D', 'scale', 'has_dt_bias', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-Hq{}-H{}-D{}-scale{}-has_dt_bias{}-{}".format(*test),
        )
        for test in [
            (2, 256, 2, 4, 64, 1, True, torch.float16),
            (2, 512, 1, 4, 64, 0.1, False, torch.float16),
            (2, 512, 2, 8, 64, 1, True, torch.float16),
            (2, 1024, 4, 8, 128, 0.1, True, torch.float16),
        ]
    ],
)
def test_chunk_gate_in_kernel_gqa(
    B: int,
    T: int,
    Hq: int,
    H: int,
    D: int,
    scale: float,
    has_dt_bias: bool,
    dtype: torch.dtype,
):
    """Test use_gate_in_kernel=True with grouped value attention (HV > H)."""
    torch.manual_seed(42)
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')
    assert H % Hq == 0

    q = torch.rand(B, T, Hq, D, dtype=dtype)
    k = torch.rand(B, T, Hq, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=torch.float).sigmoid()
    g_raw = torch.randn(B, T, H, dtype=torch.float32)
    A_log = torch.randn(H, dtype=torch.float32)
    dt_bias = torch.randn(H, dtype=torch.float32) if has_dt_bias else None
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)

    q, k, v, beta, g_raw, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, g_raw, h0))
    A_log = A_log.to(device).requires_grad_(True)
    dt_bias = dt_bias.to(device).requires_grad_(True) if dt_bias is not None else None

    tri, tri_ht = chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g_raw.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=A_log.clone(),
        dt_bias=dt_bias.clone() if dt_bias is not None else None,
    )
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g_raw.grad, h0.grad
    tri_dA_log = A_log.grad
    tri_ddt_bias = dt_bias.grad if dt_bias is not None else None
    q.grad = k.grad = v.grad = beta.grad = g_raw.grad = h0.grad = None
    A_log.grad = None
    if dt_bias is not None:
        dt_bias.grad = None

    g_ref = naive_gdn_gate(g_raw, A_log, dt_bias)
    ref, ref_ht = chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g_ref,
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    ref_dg = g_raw.grad
    ref_dA_log = A_log.grad
    ref_ddt_bias = dt_bias.grad if dt_bias is not None else None

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    assert_close('dg', ref_dg, tri_dg, 0.02)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)
    assert_close('dA_log', ref_dA_log, tri_dA_log, 0.02)
    if dt_bias is not None:
        assert_close('ddt_bias', ref_ddt_bias, tri_ddt_bias, 0.02)


@pytest.mark.parametrize(
    ('H', 'D', 'has_dt_bias', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-has_dt_bias{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 60, True, [0, 15], torch.float16),
            (4, 64, False, [0, 256, 500, 1000], torch.float16),
            (4, 64, True, [0, 256, 500, 1000], torch.float16),
            (4, 100, True, [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_chunk_gate_in_kernel_varlen(
    H: int,
    D: int,
    has_dt_bias: bool,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    """Test use_gate_in_kernel=True with variable-length sequences."""
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.randn((1, T, H, D), dtype=dtype)
    k = torch.randn((1, T, H, D), dtype=dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    beta = torch.rand(1, T, H, dtype=torch.float).sigmoid()
    g_raw = torch.randn(1, T, H, dtype=torch.float32)
    A_log = torch.randn(H, dtype=torch.float32)
    dt_bias = torch.randn(H, dtype=torch.float32) if has_dt_bias else None
    h0 = torch.randn((N, H, D, D), dtype=torch.float32)

    q, k, v, beta, g_raw, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, g_raw, h0))
    A_log = A_log.to(device).requires_grad_(True)
    dt_bias = dt_bias.to(device).requires_grad_(True) if dt_bias is not None else None

    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    tri, tri_ht = chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g_raw.clone(),
        beta=beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=A_log.clone(),
        dt_bias=dt_bias.clone() if dt_bias is not None else None,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g_raw.grad, h0.grad
    tri_dA_log = A_log.grad
    tri_ddt_bias = dt_bias.grad if dt_bias is not None else None
    q.grad = k.grad = v.grad = beta.grad = g_raw.grad = h0.grad = None
    A_log.grad = None
    if dt_bias is not None:
        dt_bias.grad = None

    g_ref = naive_gdn_gate(g_raw, A_log, dt_bias)
    ref, ref_ht = chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g_ref,
        beta=beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    ref_dg = g_raw.grad
    ref_dA_log = A_log.grad
    ref_ddt_bias = dt_bias.grad if dt_bias is not None else None

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    assert_close('dg', ref_dg, tri_dg, 0.02)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)
    assert_close('dA_log', ref_dA_log, tri_dA_log, 0.02)
    if dt_bias is not None:
        assert_close('ddt_bias', ref_ddt_bias, tri_ddt_bias, 0.02)


@pytest.mark.parametrize(
    ('B', 'T', 'HV', 'HAS_BIAS'),
    [
        pytest.param(*test, id="B{}-T{}-HV{}-bias{}".format(*test))
        for test in [
            (1, 32, 2, False),
            (2, 64, 4, True),
            (4, 128, 8, True),
            (4, 128, 16, False),
        ]
    ],
)
def test_gate(
    B: int,
    T: int,
    HV: int,
    HAS_BIAS: bool,
):
    torch.manual_seed(42)
    g = torch.randn(B, T, HV, dtype=torch.float32)
    A_log = torch.log(torch.randn(HV, dtype=torch.float32).uniform_(1, 16))
    dt_bias = torch.randn(HV, dtype=torch.float32) if HAS_BIAS else None
    g, A_log = map(lambda x: x.to(device).requires_grad_(True), (g, A_log))
    if dt_bias is not None:
        dt_bias = dt_bias.to(device).requires_grad_(True)
    do = torch.randn_like(g)

    ref = naive_gdn_gate(
        g.clone(), A_log.clone(), dt_bias.clone() if dt_bias is not None else None,
    )
    tri = fused_gdn_gate(
        g.clone(), A_log.clone(), dt_bias.clone() if dt_bias is not None else None,
    )
    (ref * do).sum().backward(retain_graph=True)

    ref_dg, ref_dA = g.grad, A_log.grad
    ref_dbias = dt_bias.grad if dt_bias is not None else None
    g.grad = A_log.grad = None
    if dt_bias is not None:
        dt_bias.grad = None

    (tri * do).sum().backward(retain_graph=True)
    tri_dg, tri_dA = g.grad, A_log.grad
    tri_dbias = dt_bias.grad if dt_bias is not None else None

    assert_close("o", ref, tri, 1e-4)
    assert_close("dg", ref_dg, tri_dg, 1e-4)
    assert_close("dA", ref_dA, tri_dA, 1e-4)
    if HAS_BIAS:
        assert_close("dbias", ref_dbias, tri_dbias, 1e-4)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HV{}-D{}".format(*test))
        for test in [
            (1, 128, 1, 1, 64),
            (2, 256, 2, 4, 64),
        ]
    ],
)
@pytest.mark.skipif(IS_INTEL_ALCHEMIST, reason='Skipped on Intel Alchemist')
def test_prepare_wy_repr_bwd_no_g(B: int, T: int, H: int, HV: int, D: int):
    """
    Regression for #862: prepare_wy_repr_bwd previously left b_dk uninitialized
    and missed the dk-side contribution to db when g is None. With g=zeros, the
    USE_G=True path is exp(0)=1 throughout, so the no-g path must produce the
    same dk/dv/db. The public chunk_gated_delta_rule does not currently accept
    g=None (its bwd assumes g is a tensor), so the bug is only reachable
    through the helper and is exercised here directly.
    """
    torch.manual_seed(0)
    BT = 64
    dtype = torch.float32

    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    beta = torch.rand(B, T, HV, dtype=dtype, device=device).sigmoid()
    g_zero = torch.zeros(B, T, HV, dtype=dtype, device=device)

    # Per-chunk unit-lower-triangular A — the bwd kernel just does linear algebra
    # against whatever A is supplied and does not require it to be the WY inverse.
    NT = T // BT
    A = torch.randn(B, NT, HV, BT, BT, dtype=dtype, device=device)
    A = torch.tril(A, diagonal=-1) + torch.eye(BT, dtype=dtype, device=device)
    A = A.permute(0, 1, 3, 2, 4).reshape(B, T, HV, BT).contiguous()
    # Forward smoke check that the no-g path also runs.
    _, _ = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=None)

    dw = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    du = torch.randn(B, T, HV, D, dtype=dtype, device=device)

    dk_no_g, dv_no_g, db_no_g, dg_no_g = prepare_wy_repr_bwd(
        k=k, v=v, beta=beta, A=A, dw=dw, du=du, g=None,
    )
    dk_zero_g, dv_zero_g, db_zero_g, dg_zero_g = prepare_wy_repr_bwd(
        k=k, v=v, beta=beta, A=A, dw=dw, du=du, g=g_zero,
    )

    assert dg_no_g is None
    assert_close('dk', dk_zero_g, dk_no_g, 1e-4)
    assert_close('dv', dv_zero_g, dv_no_g, 1e-4)
    assert_close('db', db_zero_g, db_no_g, 1e-4)
