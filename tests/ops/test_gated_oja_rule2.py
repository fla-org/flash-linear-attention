# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Correctness tests for Oja2, the Oja rule with decoupled erase and write gates.
#
# The ground truth is the pure-PyTorch ``naive_recurrent_gated_oja_rule2``. The slot code ``v`` and the log-decay ``gv``
# stay in fp32 while q, k and the two gates use the low-precision dtype, which is how the operator is fed in GSA2.

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_oja_rule2 import chunk_gated_oja_rule2, fused_recurrent_gated_oja_rule2, naive_recurrent_gated_oja_rule2
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, device


def _rand_inputs(B, T, H, K, M, dtype, N=None, seed=42):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=dtype)
    k = torch.randn(B, T, H, K, dtype=dtype)
    v = F.normalize(torch.randn(B, T, H, M, dtype=torch.float32), p=2, dim=-1)
    gv = F.logsigmoid(torch.randn(B, T, H, M, dtype=torch.float32))
    b = torch.rand(B, T, H, M, dtype=dtype)
    c = torch.rand(B, T, H, K, dtype=dtype)
    h0 = torch.randn(B if N is None else N, H, K, M, dtype=torch.float32) * 0.1
    return [x.to(device) for x in (q, k, v, gv, b, c, h0)]


def _ref(q, k, v, gv, b, c, h0, scale=None):
    return naive_recurrent_gated_oja_rule2(
        q=F.normalize(q.float(), p=2, dim=-1),
        k=F.normalize(k.float(), p=2, dim=-1),
        v=v,
        gv=gv,
        b=b.float(),
        c=c.float(),
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'M', 'scale', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-M{}-scale{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 64, 1, torch.float16),
            (2, 500, 4, 60, 64, 1, torch.float16),
            (2, 1000, 4, 128, 128, 0.1, torch.bfloat16),
            (3, 1024, 2, 128, 64, 0.1, torch.float16),
        ]
    ]
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    K: int,
    M: int,
    scale: float,
    dtype: torch.dtype,
):
    q, k, v, gv, b, c, h0 = _rand_inputs(B, T, H, K, M, dtype)
    ref, ref_ht = _ref(q, k, v, gv, b, c, h0, scale)
    tri, tri_ht = fused_recurrent_gated_oja_rule2(
        q=q,
        k=k,
        v=v,
        gv=gv,
        b=b,
        c=c,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        use_q_l2norm=True,
        use_k_l2norm=True,
    )
    assert_close('o', ref, tri, 0.002)
    assert_close('ht', ref_ht, tri_ht, 0.002)


@pytest.mark.parametrize(
    ('H', 'K', 'M', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-K{}-M{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 60, 64, [0, 1, 96, 177], torch.float16),
            (2, 128, 128, [0, 256, 500, 1000], torch.bfloat16),
        ]
    ]
)
def test_fused_recurrent_varlen(
    H: int,
    K: int,
    M: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = int(cu_seqlens[-1])
    N = len(cu_seqlens) - 1
    q, k, v, gv, b, c, h0 = _rand_inputs(1, T, H, K, M, dtype, N=N)
    tri, tri_ht = fused_recurrent_gated_oja_rule2(
        q=q,
        k=k,
        v=v,
        gv=gv,
        b=b,
        c=c,
        initial_state=h0,
        output_final_state=True,
        use_q_l2norm=True,
        use_k_l2norm=True,
        cu_seqlens=cu_seqlens,
    )
    refs, ref_hts = [], []
    for i in range(N):
        s = slice(int(cu_seqlens[i]), int(cu_seqlens[i + 1]))
        ref_i, ref_ht_i = _ref(q[:, s], k[:, s], v[:, s], gv[:, s], b[:, s], c[:, s], h0[i:i + 1])
        refs.append(ref_i)
        ref_hts.append(ref_ht_i)
    assert_close('o', torch.cat(refs, 1), tri, 0.002)
    assert_close('ht', torch.cat(ref_hts, 0), tri_ht, 0.002)


def test_fused_recurrent_backward_unsupported():
    q, k, v, gv, b, c, h0 = _rand_inputs(1, 8, 1, 32, 64, torch.float16)
    q.requires_grad_(True)
    o, _ = fused_recurrent_gated_oja_rule2(q=q, k=k, v=v, gv=gv, b=b, c=c)
    with pytest.raises(NotImplementedError):
        o.sum().backward()


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'M', 'scale', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-M{}-scale{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 64, 1, torch.float16),
            (2, 500, 3, 60, 64, 1, torch.float16),
            (2, 1000, 3, 128, 128, 0.1, torch.bfloat16),
            (1, 1024, 2, 128, 256, 0.1, torch.bfloat16),
            (4, 1024, 4, 128, 64, 1, torch.float16),
        ]
    ]
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    K: int,
    M: int,
    scale: float,
    dtype: torch.dtype,
):
    if IS_INTEL_ALCHEMIST and max(K, M) > 128:
        pytest.skip(reason='chunk_gated_oja_rule2 is not supported on alchemist for dimensions above 128')
    inputs = _rand_inputs(B, T, H, K, M, dtype)
    do = torch.randn(B, T, H, M, dtype=torch.float32, device=device)
    dht = torch.randn(B, H, K, M, dtype=torch.float32, device=device)

    def run(fn, **kwargs):
        q, k, v, gv, b, c, h0 = (x.detach().clone().requires_grad_(True) for x in inputs)
        o, ht = fn(q, k, v, gv, b, c, h0, **kwargs)
        ((o.float() * do).sum() + (ht * dht).sum()).backward()
        return o, ht, [x.grad for x in (q, k, v, gv, b, c, h0)]

    ref, ref_ht, ref_grads = run(lambda q, k, v, gv, b, c, h0: _ref(q, k, v, gv, b, c, h0, scale))
    tri, tri_ht, tri_grads = run(
        lambda q, k, v, gv, b, c, h0: chunk_gated_oja_rule2(
            q=q,
            k=k,
            v=v,
            gv=gv,
            b=b,
            c=c,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
            use_q_l2norm=True,
            use_k_l2norm=True,
        )
    )
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    for name, ratio, ref_grad, tri_grad in zip(
        ('dq', 'dk', 'dv', 'dgv', 'db', 'dc', 'dh0'),
        (0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.01),
        ref_grads,
        tri_grads,
    ):
        assert_close(name, ref_grad, tri_grad, ratio)


@pytest.mark.parametrize('T', [40, 100])
def test_chunk_strong_decay(T: int):
    # the bounded decay `-5 * sigmoid(.)` of GSA2 accumulates past exp's fp32 range within a chunk,
    # which must not leak into the gradients through the rows padded beyond T. Runs in fp32 because dgv is a difference of
    # large terms and is small under strong decay, so low-precision rounding would dominate its error
    B, H, K, M = 2, 2, 64, 64
    q, k, v, gv, b, c, h0 = _rand_inputs(B, T, H, K, M, torch.float32)
    gv = -5.0 * torch.sigmoid(torch.randn(B, T, H, M, device=device) + 2.0)
    do = torch.randn(B, T, H, M, dtype=torch.float32, device=device)

    def run(fn):
        xs = [x.detach().clone().requires_grad_(True) for x in (q, k, v, gv, b, c)]
        o, _ = fn(*xs)
        (o.float() * do).sum().backward()
        return o, [x.grad for x in xs]

    ref, ref_grads = run(lambda q, k, v, gv, b, c: _ref(q, k, v, gv, b, c, torch.zeros_like(h0)))
    tri, tri_grads = run(
        lambda q, k, v, gv, b, c: chunk_gated_oja_rule2(q=q, k=k, v=v, gv=gv, b=b, c=c, use_q_l2norm=True, use_k_l2norm=True)
    )
    assert_close('o', ref, tri, 0.005)
    for name, ref_grad, tri_grad in zip(('dq', 'dk', 'dv', 'dgv', 'db', 'dc'), ref_grads, tri_grads):
        assert torch.isfinite(tri_grad).all(), f"{name} is not finite"
        assert_close(name, ref_grad, tri_grad, 0.005)


@pytest.mark.parametrize('chunk_size', [16, 32, 64])
def test_chunk_with_chunk_size(chunk_size: int):
    B, T, H, K, M = 2, 300, 2, 64, 64
    inputs = _rand_inputs(B, T, H, K, M, torch.float16)

    def run(**kwargs):
        q, k, v, gv, b, c, h0 = (x.detach().clone().requires_grad_(True) for x in inputs)
        o, ht = chunk_gated_oja_rule2(
            q=q,
            k=k,
            v=v,
            gv=gv,
            b=b,
            c=c,
            initial_state=h0,
            output_final_state=True,
            use_q_l2norm=True,
            use_k_l2norm=True,
            **kwargs,
        )
        (o.float().square().sum() + ht.square().sum()).backward()
        return o, ht, [x.grad for x in (q, k, v, gv, b, c, h0)]

    base_o, base_ht, base_grads = run()
    o, ht, grads = run(chunk_size=chunk_size)
    assert_close('o', base_o, o, 0.005)
    assert_close('ht', base_ht, ht, 0.005)
    for name, base_grad, grad in zip(('dq', 'dk', 'dv', 'dgv', 'db', 'dc', 'dh0'), base_grads, grads):
        assert_close(name, base_grad, grad, 0.02)


@pytest.mark.parametrize(
    ('H', 'K', 'M', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-K{}-M{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 60, 64, [0, 96, 177], torch.float16),
            (4, 128, 128, [0, 256, 500, 1000], torch.bfloat16),
            (2, 64, 64, [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ]
)
@pytest.mark.smoke
def test_chunk_varlen(
    H: int,
    K: int,
    M: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    if IS_INTEL_ALCHEMIST and max(K, M) > 128:
        pytest.skip(reason='chunk_gated_oja_rule2 is not supported on alchemist for dimensions above 128')
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = int(cu_seqlens[-1])
    N = len(cu_seqlens) - 1
    inputs = _rand_inputs(1, T, H, K, M, dtype, N=N)
    do = torch.randn(1, T, H, M, dtype=torch.float32, device=device)
    dht = torch.randn(N, H, K, M, dtype=torch.float32, device=device)

    q, k, v, gv, b, c, h0 = (x.detach().clone().requires_grad_(True) for x in inputs)
    tri, tri_ht = chunk_gated_oja_rule2(
        q=q,
        k=k,
        v=v,
        gv=gv,
        b=b,
        c=c,
        initial_state=h0,
        output_final_state=True,
        use_q_l2norm=True,
        use_k_l2norm=True,
        cu_seqlens=cu_seqlens,
    )
    ((tri.float() * do).sum() + (tri_ht * dht).sum()).backward()
    tri_grads = [x.grad for x in (q, k, v, gv, b, c, h0)]

    q, k, v, gv, b, c, h0 = (x.detach().clone().requires_grad_(True) for x in inputs)
    refs, ref_hts = [], []
    for i in range(N):
        s = slice(int(cu_seqlens[i]), int(cu_seqlens[i + 1]))
        ref_i, ref_ht_i = _ref(q[:, s], k[:, s], v[:, s], gv[:, s], b[:, s], c[:, s], h0[i:i + 1])
        refs.append(ref_i)
        ref_hts.append(ref_ht_i)
    ref, ref_ht = torch.cat(refs, 1), torch.cat(ref_hts, 0)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_grads = [x.grad for x in (q, k, v, gv, b, c, h0)]

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    for name, ratio, ref_grad, tri_grad in zip(
        ('dq', 'dk', 'dv', 'dgv', 'db', 'dc', 'dh0'),
        (0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.01),
        ref_grads,
        tri_grads,
    ):
        assert_close(name, ref_grad, tri_grad, ratio)


def test_chunk_matches_fused_recurrent():
    q, k, v, gv, b, c, h0 = _rand_inputs(2, 100, 2, 64, 64, torch.float16)
    kwargs = dict(initial_state=h0, output_final_state=True, use_q_l2norm=True, use_k_l2norm=True)
    o0, ht0 = chunk_gated_oja_rule2(q=q, k=k, v=v, gv=gv, b=b, c=c, **kwargs)
    o1, ht1 = fused_recurrent_gated_oja_rule2(q=q, k=k, v=v, gv=gv, b=b, c=c, **kwargs)
    assert_close('o', o0, o1, 0.005)
    assert_close('ht', ht0, ht1, 0.005)
