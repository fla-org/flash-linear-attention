# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from itertools import accumulate

import pytest
import torch
import torch.nn.functional as F

from fla.ops.kda.naive import naive_recurrent_kda
from fla.ops.triadic_gdn import naive_recurrent_triadic_gdn
from fla.ops.triadic_kda import naive_chunk_triadic_kda, naive_recurrent_triadic_kda
from fla.utils import assert_close, device_name


def _make_inputs(B: int, T: int, H: int, K: int, V: int, E: int, gate_scale: float, dtype: torch.dtype):
    q = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, V, dtype=dtype)
    k2 = F.normalize(F.softplus(torch.randn(B, T, H, E, dtype=torch.float32)), p=2, dim=-1)
    q2 = F.normalize(F.softplus(torch.randn(B, T, H, E, dtype=torch.float32)), p=2, dim=-1)
    gk = F.logsigmoid(torch.rand(B, T, H, K, dtype=torch.float32)) * gate_scale
    g2 = F.logsigmoid(torch.rand(B, T, H, E, dtype=torch.float32)) * gate_scale
    beta = torch.rand(B, T, H, dtype=torch.float32).sigmoid()
    return [x.to(device_name) for x in (q, k, v, k2, q2, gk, g2, beta)]


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'E', 'gate_scale', 'chunk_size'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-V{}-E{}-gate_scale{}-chunk_size{}".format(*test))
        for test in [
            (2, 64, 2, 32, 32, 1, 0.5, 64),
            (2, 100, 2, 32, 48, 4, 0.5, 32),
            (1, 63, 3, 16, 16, 12, 0.5, 64),
            (1, 130, 2, 32, 32, 16, 2.0, 64),
        ]
    ],
)
def test_naive(B: int, T: int, H: int, K: int, V: int, E: int, gate_scale: float, chunk_size: int):
    torch.manual_seed(42)
    q, k, v, k2, q2, gk, g2, beta = _make_inputs(B, T, H, K, V, E, gate_scale, torch.float32)
    h0 = torch.randn(B, H, E, K, V, dtype=torch.float32, device=device_name)
    scale = K ** -0.5
    # the references return outputs in v.dtype, so keep FP64 for the direct comparison
    v_fp64 = v.double()

    ref, ref_ht = naive_recurrent_kda(
        q=(q2[..., None] * q[..., None, :]).flatten(-2),
        k=(k2[..., None] * k[..., None, :]).flatten(-2),
        v=v,
        g=(g2[..., None] + gk[..., None, :]).flatten(-2),
        beta=beta,
        scale=scale,
        initial_state=h0.flatten(2, 3),
        output_final_state=True,
    )
    tri, tri_ht = naive_recurrent_triadic_kda(
        q=q,
        k=k,
        v=v_fp64,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    assert_close('o', ref, tri, 1e-5)
    assert_close('ht', ref_ht, tri_ht.flatten(2, 3), 1e-5)

    chunk_o, chunk_ht = naive_chunk_triadic_kda(
        q=q,
        k=k,
        v=v_fp64,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    assert_close('chunk_o', ref, chunk_o, 1e-5)
    assert_close('chunk_ht', ref_ht, chunk_ht.flatten(2, 3), 1e-5)
    assert tri.dtype == chunk_o.dtype == torch.float64
    assert tri_ht.dtype == chunk_ht.dtype == torch.float32
    torch.testing.assert_close(chunk_o, tri, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(chunk_ht, tri_ht, rtol=1e-6, atol=1e-7)

    # gk = 0 recovers Triadic Gated DeltaNet
    gdn_o, gdn_ht = naive_recurrent_triadic_gdn(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        g=g2,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    zero_gk = torch.zeros_like(gk)
    tkda_gdn_o, tkda_gdn_ht = naive_chunk_triadic_kda(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        gk=zero_gk,
        g2=g2,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    assert_close('gdn_o', gdn_o, tkda_gdn_o, 1e-5)
    assert_close('gdn_ht', gdn_ht, tkda_gdn_ht, 1e-5)


@pytest.mark.parametrize(
    ('lengths', 'gate_scale', 'cu_dtype'),
    [
        pytest.param((13, 0, 50), 0.5, torch.int32, id='int32-short'),
        pytest.param((63, 0, 67), 2.0, torch.int64, id='int64-multi-chunk'),
    ],
)
def test_naive_varlen(lengths: tuple[int, ...], gate_scale: float, cu_dtype: torch.dtype):
    torch.manual_seed(42)
    H, K, V, E = 2, 32, 32, 16
    inputs = _make_inputs(1, sum(lengths), H, K, V, E, gate_scale, torch.float32)
    h0 = torch.randn(len(lengths), H, E, K, V, dtype=torch.float32, device=device_name)
    cu_seqlens = torch.tensor(list(accumulate(lengths, initial=0)), dtype=cu_dtype, device=device_name)

    ref_outputs, ref_states = [], []
    segments = zip(*(x.split(lengths, dim=1) for x in inputs))
    for i, segment_inputs in enumerate(segments):
        state = h0[i:i+1]
        if lengths[i] == 0:
            ref_states.append(state)
            continue
        ref_o, ref_ht = naive_recurrent_triadic_kda(*segment_inputs, initial_state=state, output_final_state=True)
        ref_outputs.append(ref_o)
        ref_states.append(ref_ht)
    ref_o, ref_ht = torch.cat(ref_outputs, dim=1), torch.cat(ref_states)

    for op in (naive_recurrent_triadic_kda, naive_chunk_triadic_kda):
        o, ht = op(*inputs, initial_state=h0, output_final_state=True, cu_seqlens=cu_seqlens)
        torch.testing.assert_close(o, ref_o, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(ht, ref_ht, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(ht[1], h0[1], rtol=0, atol=0)


@pytest.mark.parametrize(
    'op', [naive_recurrent_triadic_kda, naive_chunk_triadic_kda], ids=['recurrent', 'chunk'],
)
def test_naive_beta_default(op):
    torch.manual_seed(42)
    *inputs, beta = _make_inputs(1, 35, 2, 8, 8, 4, 0.5, torch.float64)
    o, ht = op(*inputs, output_final_state=True)
    ref_o, ref_ht = op(*inputs, beta=torch.ones_like(beta), output_final_state=True)
    torch.testing.assert_close(o, ref_o, rtol=0, atol=0)
    torch.testing.assert_close(ht, ref_ht, rtol=0, atol=0)


@pytest.mark.parametrize(
    'op', [naive_recurrent_triadic_kda, naive_chunk_triadic_kda], ids=['recurrent', 'chunk'],
)
def test_naive_varlen_rejects_multiple_batches(op):
    torch.manual_seed(42)
    inputs = _make_inputs(2, 3, 1, 4, 4, 2, 0.5, torch.float32)
    with pytest.raises(ValueError, match='The batch size is expected to be 1'):
        op(
            *inputs,
            cu_seqlens=torch.tensor([0, 1, 3], dtype=torch.int32, device=device_name),
        )
