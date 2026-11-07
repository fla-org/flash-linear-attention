# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.dsa import naive_dsa, parallel_dsa
from fla.utils import assert_close, device


def make_inputs(B, T, HQ, H, K, V, topk, dtype, cu_seqlens=None):
    torch.manual_seed(42)
    q = torch.randn(B, T, HQ, K, dtype=dtype, device=device).requires_grad_()
    k = torch.randn(B, T, H, K, dtype=dtype, device=device).requires_grad_()
    v = torch.randn(B, T, H, V, dtype=dtype, device=device).requires_grad_()
    # random causal top-k without duplicates that always keeps the query itself, padded with -1
    lens = [T] if cu_seqlens is None else (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    indices = torch.full((B, T, topk), -1, dtype=torch.long, device=device)
    bos = 0
    for n in lens:
        i = torch.arange(n, device=device)
        scores = torch.rand(B, n, n, device=device).masked_fill(i[:, None] < i[None, :], -1.)
        scores[:, i, i] = 2.
        values, idx = scores.topk(min(topk, n), dim=-1)
        indices[:, bos:bos + n, :idx.shape[-1]] = idx.masked_fill(values < 0, -1)
        bos += n
    return q, k, v, indices


@pytest.mark.parametrize(
    ('B', 'T', 'HQ', 'H', 'K', 'V', 'topk', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-HQ{}-H{}-K{}-V{}-topk{}-{}".format(*test))
        for test in [
            (2, 64, 4, 4, 32, 32, 16, torch.float),
            (1, 100, 8, 2, 48, 40, 32, torch.float),
            (2, 80, 20, 1, 64, 32, 24, torch.float),
            (1, 50, 4, 1, 32, 32, 64, torch.float),
            (2, 256, 16, 1, 128, 128, 64, torch.bfloat16),
        ]
    ],
)
def test_parallel(B, T, HQ, H, K, V, topk, dtype):
    q, k, v, indices = make_inputs(B, T, HQ, H, K, V, topk, dtype)
    do = torch.randn(B, T, HQ, V, dtype=dtype, device=device)

    ref = naive_dsa(q, k, v, None, None, indices=indices)
    (ref * do).sum().backward()
    ref_dq, ref_dk, ref_dv = q.grad, k.grad, v.grad
    q.grad = k.grad = v.grad = None
    tri = parallel_dsa(q, k, v, indices)
    (tri * do).sum().backward()

    tol = 1e-3 if dtype == torch.float else 0.008
    assert_close('o', ref, tri, tol)
    assert_close('dq', ref_dq, q.grad, tol)
    assert_close('dk', ref_dk, k.grad, tol)
    assert_close('dv', ref_dv, v.grad, tol)


def test_parallel_varlen():
    cu_seqlens = [0, 5, 60, 100, 160]
    T = cu_seqlens[-1]
    cu = torch.tensor(cu_seqlens, dtype=torch.long, device=device)
    q, k, v, indices = make_inputs(1, T, 8, 2, 32, 32, 16, torch.float, cu_seqlens=cu)
    do = torch.randn(1, T, 8, 32, device=device)

    ref = naive_dsa(q, k, v, None, None, indices=indices, cu_seqlens=cu)
    (ref * do).sum().backward()
    ref_grads = [q.grad, k.grad, v.grad]
    q.grad = k.grad = v.grad = None
    tri = parallel_dsa(q, k, v, indices, cu_seqlens=cu)
    (tri * do).sum().backward()
    assert_close('o', ref, tri, 1e-3)
    for name, r, t in zip(['dq', 'dk', 'dv'], ref_grads, [q.grad, k.grad, v.grad]):
        assert_close(name, r, t, 1e-3)


def test_parallel_ignores_future_indices():
    """Selected keys after the query position are masked, as in the reference."""
    q, k, v, indices = make_inputs(1, 40, 4, 2, 32, 32, 8, torch.float)
    # replace the last slot with a key from the future where there is one
    t = torch.arange(40, device=device)
    indices[..., -1] = torch.where(t + 3 < 40, t + 3, indices[..., -1])
    with torch.no_grad():
        assert_close('o', naive_dsa(q, k, v, None, None, indices=indices), parallel_dsa(q, k, v, indices), 1e-3)
