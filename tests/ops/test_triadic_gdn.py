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
from fla.ops.triadic_gdn import naive_chunk_triadic_gdn, naive_recurrent_triadic_gdn
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
