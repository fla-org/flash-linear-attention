# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Tests of the forward-mode derivative (JVP) of the chunked gated delta rule."""

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule.chunk_jvp import chunk_gated_delta_rule_jvp_ref
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
from fla.utils import device


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative difference in the Frobenius norm: |a - b| / |b|."""
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def make_inputs(B, T, H, K, V, D, dtype, seed=0, dev=device):
    """Primal inputs as a trained model feeds them (unit-norm keys, log-sigmoid gates, beta in (0, 1)), and tangents."""
    gen = torch.Generator(device='cpu').manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float64).to(dev)
    q = randn(B, T, H, K)
    k = F.normalize(randn(B, T, H, K), dim=-1)
    v = randn(B, T, H, V)
    g = F.logsigmoid(randn(B, T, H))
    beta = torch.rand(B, T, H, generator=gen, dtype=torch.float64).to(dev)
    tangents = [randn(D, *x.shape) for x in (q, k, v, g, beta)]
    cast = [x.to(dtype) for x in (q, k, v, g, beta, *tangents)]
    return cast


@pytest.mark.parametrize('T', [64, 100, 200])
def test_chunk_jvp_reference_matches_naive_fp64(T):
    """The chunked reference equals torch.func.jvp of the token-by-token recurrence in fp64 (rel. diff < 1e-10)."""
    B, H, K, V, D = 2, 3, 32, 32, 3
    q, k, v, g, beta, *tans = make_inputs(B, T, H, K, V, D, torch.float64)
    o, o_dot = chunk_gated_delta_rule_jvp_ref(q, k, v, g, beta, *tans)

    def f(q, k, v, g, beta):
        return naive_recurrent_gated_delta_rule(q, k, v, beta, g)[0]
    for d in range(D):
        o_ref, o_dot_ref = torch.func.jvp(f, (q, k, v, g, beta), tuple(t[d] for t in tans))
        assert rel(o_dot[d], o_dot_ref) < 1e-10
    assert rel(o, o_ref) < 1e-10
