# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For the list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.kda.naive import naive_recurrent_kda
from fla.ops.triadic_gdn import naive_chunk_triadic_gdn
from fla.ops.triadic_kda import naive_chunk_triadic_kda, naive_recurrent_triadic_kda
from fla.utils import assert_close, device


def _make_inputs(B, T, H, K, V, E, gate_logit_normalizer, dtype, gk_zero=False):
    q = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, V, dtype=dtype)
    k2 = F.normalize(F.softplus(torch.randn(B, T, H, E, dtype=torch.float32)), p=2, dim=-1)
    q2 = F.normalize(F.softplus(torch.randn(B, T, H, E, dtype=torch.float32)), p=2, dim=-1)
    g = F.logsigmoid(torch.rand(B, T, H, E, dtype=torch.float32)) / gate_logit_normalizer
    gk = torch.zeros(B, T, H, K, dtype=torch.float32) if gk_zero else (
        F.logsigmoid(torch.rand(B, T, H, K, dtype=torch.float32)) / gate_logit_normalizer
    )
    beta = torch.rand(B, T, H, dtype=torch.float32).sigmoid()
    return [x.to(device) for x in (q, k, v, k2, q2, g, gk, beta)]


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
    # the references must equal a KDA recurrence over the joint key k2 ⊗ k, with the gate
    # being the sum of the per-slice and per-channel decays
    torch.manual_seed(42)
    q, k, v, k2, q2, g, gk, beta = _make_inputs(B, T, H, K, V, E, gate_logit_normalizer, torch.float32)
    h0 = torch.randn(B, H, E, K, V, dtype=torch.float32, device=device)
    scale = K ** -0.5

    ref, ref_ht = naive_recurrent_kda(
        q=(q2[..., None] * q[..., None, :]).flatten(-2),
        k=(k2[..., None] * k[..., None, :]).flatten(-2),
        v=v,
        g=(g[..., None] + gk[..., None, :]).expand(B, T, H, E, K).flatten(-2),
        beta=beta,
        scale=scale,
        initial_state=h0.flatten(2, 3),
        output_final_state=True,
    )

    tri, tri_ht = naive_recurrent_triadic_kda(
        q=q, k=k, v=v, k2=k2, q2=q2, g=g, gk=gk, beta=beta,
        scale=scale, initial_state=h0, output_final_state=True,
    )
    assert_close('o', ref, tri, 1e-5)
    assert_close('ht', ref_ht, tri_ht.flatten(2, 3), 1e-5)

    tri, tri_ht = naive_chunk_triadic_kda(
        q=q, k=k, v=v, k2=k2, q2=q2, g=g, gk=gk, beta=beta,
        scale=scale, initial_state=h0, output_final_state=True,
    )
    assert_close('chunk_o', ref, tri, 1e-5)
    assert_close('chunk_ht', ref_ht, tri_ht.flatten(2, 3), 1e-5)


@pytest.mark.parametrize('T', [64, 100, 200])
def test_gk_zero_recovers_triadic_gdn(T: int):
    """With no per-channel decay, Triadic KDA must be exactly Triadic GDN."""
    torch.manual_seed(7)
    B, H, K, V, E = 2, 2, 32, 32, 4
    q, k, v, k2, q2, g, gk, beta = _make_inputs(B, T, H, K, V, E, 1.0, torch.float32, gk_zero=True)
    h0 = torch.randn(B, H, E, K, V, dtype=torch.float32, device=device)
    scale = K ** -0.5

    kda_o, kda_ht = naive_recurrent_triadic_kda(
        q=q, k=k, v=v, k2=k2, q2=q2, g=g, gk=gk, beta=beta,
        scale=scale, initial_state=h0, output_final_state=True,
    )
    gdn_o, gdn_ht = naive_chunk_triadic_gdn(
        q=q, k=k, v=v, k2=k2, q2=q2, g=g, beta=beta,
        scale=scale, initial_state=h0, output_final_state=True,
    )
    assert_close('o', kda_o, gdn_o, 1e-5)
    assert_close('ht', kda_ht, gdn_ht, 1e-5)
