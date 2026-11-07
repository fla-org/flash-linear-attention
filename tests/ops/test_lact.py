# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.lact import chunk_lact
from fla.ops.lact.naive import naive_lact
from fla.utils import assert_close, device


def make_inputs(B, T, H, K, D, dtype, use_momentum, N=None):
    q = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32), dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32), dim=-1).to(dtype)
    v = torch.randn(B, T, H, K, dtype=dtype)
    w0 = torch.randn(H, D, K) / K ** 0.5
    w1 = torch.randn(H, K, D) / D ** 0.5
    w2 = torch.randn(H, D, K) / K ** 0.5
    lr = torch.rand(B, T, H, 3) * 0.1
    momentum = torch.rand(B, T, H) if use_momentum else None
    inputs = [q, k, v, w0, w1, w2, lr] + ([momentum] if use_momentum else [])
    return [x.to(device).requires_grad_() for x in inputs]


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'D', 'chunk_size', 'use_muon', 'use_momentum', 'prenorm', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-D{}-C{}-muon{}-momentum{}-prenorm{}-{}".format(*test))
        for test in [
            (2, 100, 2, 16, 32, 32, False, False, False, torch.float32),
            (2, 128, 2, 16, 16, 32, False, True, False, torch.float32),
            (1, 96, 3, 32, 32, 32, True, True, False, torch.float32),
            (2, 100, 2, 16, 32, 16, True, False, True, torch.float32),
            (2, 64, 2, 16, 32, 16, False, True, True, torch.float32),
            (2, 100, 2, 32, 32, 32, False, True, False, torch.bfloat16),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    K: int,
    D: int,
    chunk_size: int,
    use_muon: bool,
    use_momentum: bool,
    prenorm: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    inputs = make_inputs(B, T, H, K, D, dtype, use_momentum)
    q, k, v, w0, w1, w2, lr = inputs[:7]
    momentum = inputs[7] if use_momentum else None
    kwargs = dict(chunk_size=chunk_size, use_muon=use_muon, prenorm=prenorm, output_final_state=True)
    tol = 0.002 if dtype == torch.float32 else 0.02

    do = torch.randn_like(v)
    ref, ref_ht = naive_lact(q, k, v, w0, w1, w2, lr, momentum, **kwargs)
    ((ref * do).sum() + sum((s * 0.1).sum() for s in ref_ht)).backward()
    ref_grads = [x.grad for x in inputs]
    for x in inputs:
        x.grad = None

    tri, tri_ht = chunk_lact(q, k, v, w0, w1, w2, lr, momentum, **kwargs)
    ((tri * do).sum() + sum((s * 0.1).sum() for s in tri_ht)).backward()
    tri_grads = [x.grad for x in inputs]

    assert_close('o', ref, tri, tol)
    assert len(ref_ht) == len(tri_ht) == (6 if use_momentum else 3)
    for i, (r, t) in enumerate(zip(ref_ht, tri_ht)):
        assert_close(f'ht{i}', r, t, tol)
    names = ['dq', 'dk', 'dv', 'dw0', 'dw1', 'dw2', 'dlr', 'dmomentum']
    for name, r, t in zip(names, ref_grads, tri_grads):
        assert_close(name, r, t, tol)


@pytest.mark.parametrize('use_momentum', [False, True])
@pytest.mark.parametrize('prenorm', [False, True])
def test_chunk_initial_state(use_momentum: bool, prenorm: bool):
    """Processing a sequence in two calls that split at a chunk boundary matches processing it in one call."""
    torch.manual_seed(42)
    B, T, H, K, D, chunk_size, split = 2, 100, 2, 16, 32, 16, 48
    inputs = make_inputs(B, T, H, K, D, torch.float32, use_momentum)
    q, k, v, w0, w1, w2, lr = inputs[:7]
    momentum = inputs[7] if use_momentum else None
    kwargs = dict(chunk_size=chunk_size, use_muon=True, prenorm=prenorm)

    ref, ref_ht = chunk_lact(q, k, v, w0, w1, w2, lr, momentum, output_final_state=True, **kwargs)
    o1, h1 = chunk_lact(
        q[:, :split], k[:, :split], v[:, :split], w0, w1, w2, lr[:, :split],
        momentum[:, :split] if use_momentum else None, output_final_state=True, **kwargs,
    )
    o2, h2 = chunk_lact(
        q[:, split:], k[:, split:], v[:, split:], w0, w1, w2, lr[:, split:],
        momentum[:, split:] if use_momentum else None, initial_state=h1, output_final_state=True, **kwargs,
    )
    assert_close('o', ref, torch.cat([o1, o2], 1), 1e-4)
    for i, (r, t) in enumerate(zip(ref_ht, h2)):
        assert_close(f'ht{i}', r, t, 1e-4)


@pytest.mark.parametrize('use_momentum', [False, True])
@pytest.mark.parametrize('cu_seqlens', [[0, 15, 100, 164, 300]])
def test_chunk_varlen(use_momentum: bool, cu_seqlens: list[int]):
    torch.manual_seed(42)
    H, K, D, chunk_size = 2, 16, 32, 32
    T, N = cu_seqlens[-1], len(cu_seqlens) - 1
    inputs = make_inputs(1, T, H, K, D, torch.float32, use_momentum)
    q, k, v, w0, w1, w2, lr = inputs[:7]
    momentum = inputs[7] if use_momentum else None
    h0 = [torch.randn(N, *w.shape) / w.shape[-1] ** 0.5 for w in (w0, w1, w2)]
    h0 += [torch.randn(N, *w.shape) * 0.01 for w in (w0, w1, w2)] if use_momentum else []
    h0 = tuple(h.to(device) for h in h0)
    kwargs = dict(chunk_size=chunk_size, use_muon=True, output_final_state=True)

    tri, tri_ht = chunk_lact(
        q, k, v, w0, w1, w2, lr, momentum, initial_state=h0,
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.long, device=device), **kwargs,
    )
    ref, ref_ht = [], []
    for i, (bos, eos) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        ref_i, ref_ht_i = naive_lact(
            q[:, bos:eos], k[:, bos:eos], v[:, bos:eos], w0, w1, w2, lr[:, bos:eos],
            momentum[:, bos:eos] if use_momentum else None, initial_state=tuple(h[i:i+1] for h in h0), **kwargs,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    assert_close('o', torch.cat(ref, 1), tri, 0.002)
    for i, (r, t) in enumerate(zip(zip(*ref_ht), tri_ht)):
        assert_close(f'ht{i}', torch.cat(r, 0), t, 0.002)
