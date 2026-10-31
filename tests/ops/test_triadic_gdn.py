# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.kda.naive import naive_recurrent_kda
from fla.ops.triadic_gdn import chunk_triadic_gdn, naive_chunk_triadic_gdn, naive_recurrent_triadic_gdn
from fla.utils import assert_close, device


def _make_inputs(B, T, H, K, V, E, gate_logit_normalizer, dtype):
    q = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, V, dtype=dtype)
    k2 = F.normalize(F.softplus(torch.randn(B, T, H, E, dtype=torch.float32)), p=2, dim=-1)
    q2 = F.normalize(F.softplus(torch.randn(B, T, H, E, dtype=torch.float32)), p=2, dim=-1)
    g = F.logsigmoid(torch.rand(B, T, H, E, dtype=torch.float32)) / gate_logit_normalizer
    beta = torch.rand(B, T, H, dtype=torch.float32).sigmoid()
    return [x.to(device) for x in (q, k, v, k2, q2, g, beta)]


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'E', 'gate_logit_normalizer'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-V{}-E{}-gate_logit_normalizer{}".format(*test))
        for test in [
            (2, 64, 2, 32, 32, 1, 1),
            (2, 100, 2, 32, 48, 4, 1),
            (1, 63, 3, 16, 16, 12, 1),
            (2, 200, 2, 32, 32, 8, 0.01),
        ]
    ],
)
def test_naive(B: int, T: int, H: int, K: int, V: int, E: int, gate_logit_normalizer: float):
    # the references must equal a KDA recurrence over the joint key k2 ⊗ k, with the gate shared within each slice
    torch.manual_seed(42)
    q, k, v, k2, q2, g, beta = _make_inputs(B, T, H, K, V, E, gate_logit_normalizer, torch.float32)
    h0 = torch.randn(B, H, E, K, V, dtype=torch.float32, device=device)
    scale = K ** -0.5

    ref, ref_ht = naive_recurrent_kda(
        q=(q2[..., None] * q[..., None, :]).flatten(-2),
        k=(k2[..., None] * k[..., None, :]).flatten(-2),
        v=v,
        g=g[..., None].expand(B, T, H, E, K).flatten(-2),
        beta=beta,
        scale=scale,
        initial_state=h0.flatten(2, 3),
        output_final_state=True,
    )
    tri, tri_ht = naive_recurrent_triadic_gdn(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    assert_close('o', ref, tri, 1e-5)
    assert_close('ht', ref_ht, tri_ht.flatten(2, 3), 1e-5)

    tri, tri_ht = naive_chunk_triadic_gdn(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    assert_close('chunk_o', ref, tri, 1e-5)
    assert_close('chunk_ht', ref_ht, tri_ht.flatten(2, 3), 1e-5)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'E', 'scale', 'gate_logit_normalizer', 'use_state', 'disable_recompute', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-K{}-V{}-E{}-scale{}-gate_logit_normalizer{}-use_state{}-disable_recompute{}-{}".format(*test),
        )
        for test in [
            (1, 63, 1, 64, 64, 1, 1, 1, True, False, torch.float16),
            (1, 200, 2, 64, 64, 4, 1, 1, True, False, torch.float),
            (2, 500, 3, 60, 64, 2, 1, 1, True, True, torch.float16),
            # strong decay, where factored exponents would overflow
            (2, 1000, 2, 128, 128, 4, 0.1, 0.1, False, False, torch.bfloat16),
            (2, 1024, 2, 128, 128, 8, 1, 1, False, True, torch.bfloat16),
            # E, K and V that are not powers of two
            (1, 777, 2, 100, 96, 12, 1, 10, True, False, torch.float16),
            (1, 256, 2, 128, 128, 16, 1, 1, True, False, torch.bfloat16),
            # head dims below the 16-wide tl.dot tiles
            (2, 100, 2, 8, 12, 2, 1, 1, True, False, torch.float16),
            # the largest supported key dim
            (1, 300, 2, 256, 128, 4, 1, 1, True, False, torch.bfloat16),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    E: int,
    scale: float,
    gate_logit_normalizer: float,
    use_state: bool,
    disable_recompute: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    h0 = torch.randn(B, H, E, K, V, dtype=torch.float32, device=device) if use_state else None
    inputs = [x.requires_grad_(True) for x in _make_inputs(B, T, H, K, V, E, gate_logit_normalizer, dtype)]
    if use_state:
        inputs.append(h0.requires_grad_(True))
    q, k, v, k2, q2, g, beta = inputs[:7]
    do = torch.randn_like(v)
    dht = torch.randn_like(h0) if use_state else None

    ref, ref_ht = naive_recurrent_triadic_gdn(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=use_state,
    )
    ((ref * do).sum() + ((ref_ht * dht).sum() if use_state else 0)).backward()
    ref_grads = [x.grad for x in inputs]
    for x in inputs:
        x.grad = None

    tri, tri_ht = chunk_triadic_gdn(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=use_state,
        disable_recompute=disable_recompute,
    )
    ((tri * do).sum() + ((tri_ht * dht).sum() if use_state else 0)).backward()
    tri_grads = [x.grad for x in inputs]

    assert_close('o', ref, tri, 0.005)
    if use_state:
        assert_close('ht', ref_ht, tri_ht, 0.005)
    for name, ref_grad, tri_grad, ratio in zip(
        ('dq', 'dk', 'dv', 'dk2', 'dq2', 'dg', 'db', 'dh0'),
        ref_grads,
        tri_grads,
        (0.008, 0.008, 0.008, 0.008, 0.008, 0.02, 0.02, 0.008),
    ):
        assert_close(name, ref_grad, tri_grad, ratio)


@pytest.mark.parametrize(
    ('H', 'K', 'V', 'E', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-K{}-V{}-E{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 64, 64, 4, [0, 15], torch.float16),
            (3, 128, 128, 8, [0, 256, 500, 1000], torch.bfloat16),
            (2, 128, 128, 4, [0, 15, 100, 300, 1200, 2000], torch.bfloat16),
        ]
    ],
)
def test_chunk_varlen(H: int, K: int, V: int, E: int, cu_seqlens: list[int], dtype: torch.dtype):
    torch.manual_seed(42)
    N, T = len(cu_seqlens) - 1, cu_seqlens[-1]
    inputs = _make_inputs(1, T, H, K, V, E, 1, dtype)
    h0 = torch.randn(N, H, E, K, V, dtype=torch.float32, device=device)
    q, k, v, k2, q2, g, beta, h0 = map(lambda x: x.requires_grad_(True), inputs + [h0])
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    refs, ref_hts = [], []
    for i in range(N):
        bos, eos = cu_seqlens[i], cu_seqlens[i + 1]
        ref, ref_ht = naive_recurrent_triadic_gdn(
            q=q[:, bos:eos],
            k=k[:, bos:eos],
            v=v[:, bos:eos],
            k2=k2[:, bos:eos],
            q2=q2[:, bos:eos],
            g=g[:, bos:eos],
            beta=beta[:, bos:eos],
            initial_state=h0[i:i+1],
            output_final_state=True,
        )
        refs.append(ref)
        ref_hts.append(ref_ht)
    ref, ref_ht = torch.cat(refs, 1), torch.cat(ref_hts, 0)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_grads = [x.grad for x in (q, k, v, k2, q2, g, beta, h0)]
    q.grad = k.grad = v.grad = k2.grad = q2.grad = g.grad = beta.grad = h0.grad = None

    tri, tri_ht = chunk_triadic_gdn(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_grads = [x.grad for x in (q, k, v, k2, q2, g, beta, h0)]

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    for name, ref_grad, tri_grad, ratio in zip(
        ('dq', 'dk', 'dv', 'dk2', 'dq2', 'dg', 'db', 'dh0'),
        ref_grads,
        tri_grads,
        (0.008, 0.008, 0.008, 0.008, 0.008, 0.02, 0.02, 0.008),
    ):
        assert_close(name, ref_grad, tri_grad, ratio)
