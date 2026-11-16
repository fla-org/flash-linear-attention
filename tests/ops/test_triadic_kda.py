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
from fla.ops.triadic_gdn import naive_recurrent_triadic_gdn
from fla.ops.triadic_kda import naive_chunk_triadic_kda, naive_recurrent_triadic_kda
from fla.utils import assert_close, device


def _make_inputs(B: int, T: int, H: int, K: int, V: int, E: int, gate_scale: float, dtype: torch.dtype):
    q = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, V, dtype=dtype)
    k2 = F.normalize(F.softplus(torch.randn(B, T, H, E, dtype=torch.float32)), p=2, dim=-1)
    q2 = F.normalize(F.softplus(torch.randn(B, T, H, E, dtype=torch.float32)), p=2, dim=-1)
    gk = F.logsigmoid(torch.rand(B, T, H, K, dtype=torch.float32)) * gate_scale
    g2 = F.logsigmoid(torch.rand(B, T, H, E, dtype=torch.float32)) * gate_scale
    beta = torch.rand(B, T, H, dtype=torch.float32).sigmoid()
    return [x.to(device) for x in (q, k, v, k2, q2, gk, g2, beta)]


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'E', 'gate_scale'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-V{}-E{}-gate_scale{}".format(*test))
        for test in [
            (2, 64, 2, 32, 32, 1, 0.5),
            (2, 100, 2, 32, 48, 4, 0.5),
            (1, 63, 3, 16, 16, 12, 0.5),
            (1, 130, 2, 32, 32, 16, 2.0),
        ]
    ],
)
def test_naive(B: int, T: int, H: int, K: int, V: int, E: int, gate_scale: float):
    torch.manual_seed(42)
    q, k, v, k2, q2, gk, g2, beta = _make_inputs(B, T, H, K, V, E, gate_scale, torch.float32)
    h0 = torch.randn(B, H, E, K, V, dtype=torch.float32, device=device)
    scale = K ** -0.5

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
        v=v,
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
        v=v,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    assert_close('chunk_o', ref, chunk_o, 1e-5)
    assert_close('chunk_ht', ref_ht, chunk_ht.flatten(2, 3), 1e-5)

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

    if B == 1:
        cu = torch.tensor([0, 0, T // 2, T], dtype=torch.int32, device=device)
        h0_cu = torch.randn(3, H, E, K, V, dtype=torch.float32, device=device)
        rec_cu, rec_ht_cu = naive_recurrent_triadic_kda(
            q=q, k=k, v=v, k2=k2, q2=q2, gk=gk, g2=g2,
            beta=beta, scale=scale, initial_state=h0_cu, output_final_state=True, cu_seqlens=cu,
        )
        chk_cu, chk_ht_cu = naive_chunk_triadic_kda(
            q=q, k=k, v=v, k2=k2, q2=q2, gk=gk, g2=g2,
            beta=beta, scale=scale, initial_state=h0_cu, output_final_state=True, cu_seqlens=cu,
        )
        assert_close('varlen_o', rec_cu, chk_cu, 1e-5)
        assert_close('varlen_ht', rec_ht_cu, chk_ht_cu, 1e-5)
