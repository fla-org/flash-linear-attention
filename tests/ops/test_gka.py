# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gka import chunk_gka, fused_recurrent_gka, naive_recurrent_gka, naive_recurrent_gka_chebyshev
from fla.utils import assert_close, device


def make_inputs(B, T, H, K, V, gates, use_alpha, use_initial_state, dtype, N=None):
    N = B if N is None else N
    q = F.normalize(torch.randn(B, T, H, K), p=2, dim=-1)
    k = F.normalize(torch.randn(B, T, H, K), p=2, dim=-1) * torch.rand(B, T, H, 1).sigmoid()
    v = torch.randn(B, T, H, V)
    g = torch.empty(B, T, H).uniform_(0.8, 0.99).log() if gates in ('tied', 'g_only') else None
    gk = g if gates == 'tied' else None
    alpha = torch.rand(B, T, H).sigmoid() if use_alpha else None
    initial_state = None
    if use_initial_state:
        k0 = F.normalize(torch.randn(N, 8, H, K), p=2, dim=-1)
        initial_state = (torch.einsum('nthi,nthj->nhij', k0, k0), torch.randn(N, H, K, V) * 0.1)
        initial_state = tuple(s.to(device) for s in initial_state)
    q, k, v = (x.to(device, dtype) for x in (q, k, v))
    g, gk, alpha = (x.to(device) if x is not None else None for x in (g, gk, alpha))
    return q, k, v, g, gk, alpha, initial_state


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'gates', 'use_alpha', 'use_initial_state', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-V{}-{}-alpha{}-h0{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 64, 'tied', True, False, torch.float),
            (2, 500, 4, 60, 60, 'tied', True, True, torch.float),
            (2, 1024, 4, 128, 128, 'tied', False, True, torch.float),
            (2, 256, 4, 64, 100, 'g_only', True, True, torch.float),
            (2, 256, 4, 100, 64, 'none', True, False, torch.float),
            (2, 1024, 4, 128, 128, 'tied', True, True, torch.float16),
            (2, 500, 4, 64, 64, 'g_only', True, True, torch.float16),
        ]
    ],
)
def test_chunk_fwd(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    gates: str,
    use_alpha: bool,
    use_initial_state: bool,
    dtype: torch.dtype,
):
    """`chunk_gka`'s outputs and final states must match `naive_recurrent_gka_chebyshev` at the same `num_iter`."""
    torch.manual_seed(42)
    q, k, v, g, gk, alpha, initial_state = make_inputs(B, T, H, K, V, gates, use_alpha, use_initial_state, dtype)

    with torch.no_grad():
        tri, (tri_hkk, tri_hkv) = chunk_gka(
            q=q, k=k, v=v, g=g, gk=gk, alpha=alpha, initial_state=initial_state, output_final_state=True,
        )
        ref, (ref_hkk, ref_hkv) = naive_recurrent_gka_chebyshev(
            q=q, k=k, v=v, g=g, gk=gk, alpha=alpha, initial_state=initial_state, output_final_state=True,
        )

    # compare in float32: the RMS in `assert_close` overflows when taken in float16
    assert_close('o', ref.float(), tri.float(), 0.01)
    assert_close('h_kk_final', ref_hkk, tri_hkk, 0.005)
    assert_close('h_kv_final', ref_hkv, tri_hkv, 0.005)


@pytest.mark.parametrize(
    ('H', 'K', 'gates', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-K{}-{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (3, 50, 'tied', [0, 15], torch.float),
            (4, 64, 'tied', [0, 14, 121, 421, 500], torch.float),
            (4, 100, 'g_only', [0, 256, 500, 1000], torch.float),
            (4, 128, 'none', [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ],
)
def test_chunk_varlen_fwd(
    H: int,
    K: int,
    gates: str,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    """Packed sequences must match `naive_recurrent_gka_chebyshev` run on each sequence separately."""
    torch.manual_seed(42)
    T, N = cu_seqlens[-1], len(cu_seqlens) - 1
    q, k, v, g, gk, alpha, initial_state = make_inputs(1, T, H, K, K, gates, True, True, dtype, N=N)
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long, device=device)

    with torch.no_grad():
        tri, (tri_hkk, tri_hkv) = chunk_gka(
            q=q, k=k, v=v, g=g, gk=gk, alpha=alpha, initial_state=initial_state, output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        refs, ref_hkks, ref_hkvs = [], [], []
        for i in range(N):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            ref, (ref_hkk, ref_hkv) = naive_recurrent_gka_chebyshev(
                *(x[:, bos:eos] for x in (q, k, v)),
                *(x[:, bos:eos] if x is not None else None for x in (g, gk, alpha)),
                initial_state=tuple(s[i:i + 1] for s in initial_state),
                output_final_state=True,
            )
            refs.append(ref)
            ref_hkks.append(ref_hkk)
            ref_hkvs.append(ref_hkv)

    assert_close('o', torch.cat(refs, 1).float(), tri.float(), 0.01)
    assert_close('h_kk_final', torch.cat(ref_hkks, 0), tri_hkk, 0.005)
    assert_close('h_kv_final', torch.cat(ref_hkvs, 0), tri_hkv, 0.005)


def _leaves(q, k, v, g, gk, alpha, initial_state):
    """Fresh leaf tensors; a tied `gk` stays the same tensor as `g` so both gradients accumulate into one leaf."""
    q, k, v = (x.detach().clone().requires_grad_() for x in (q, k, v))
    tied = gk is not None and gk is g
    g = g.detach().clone().requires_grad_() if g is not None else None
    gk = g if tied else (gk.detach().clone().requires_grad_() if gk is not None else None)
    alpha = alpha.detach().clone().requires_grad_() if alpha is not None else None
    if initial_state is not None:
        initial_state = tuple(s.detach().clone().requires_grad_() for s in initial_state)
    return q, k, v, g, gk, alpha, initial_state


def _grads(fn, q, k, v, g, gk, alpha, initial_state, do, dht, **kwargs):
    q, k, v, g, gk, alpha, initial_state = _leaves(q, k, v, g, gk, alpha, initial_state)
    o, (h_kk, h_kv) = fn(
        q=q, k=k, v=v, g=g, gk=gk, alpha=alpha, initial_state=initial_state, output_final_state=True, **kwargs,
    )
    ((o.float() * do).sum() + (h_kk * dht[0]).sum() + (h_kv * dht[1]).sum()).backward()
    grads = {'dq': q.grad, 'dk': k.grad, 'dv': v.grad}
    if g is not None:
        grads['dg'] = g.grad
    if gk is not None and gk is not g:
        grads['dgk'] = gk.grad
    if alpha is not None:
        grads['dalpha'] = alpha.grad
    if initial_state is not None:
        grads['dh0_kk'], grads['dh0_kv'] = (s.grad for s in initial_state)
    return o, (h_kk, h_kv), grads


def _random_dht(N, H, K, V):
    return torch.randn(N, H, K, K, device=device) * 0.1, torch.randn(N, H, K, V, device=device) * 0.1


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'gates', 'use_alpha', 'use_initial_state', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-V{}-{}-alpha{}-h0{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 64, 'tied', True, False, torch.float),
            (2, 500, 4, 60, 60, 'tied', True, True, torch.float),
            (2, 256, 4, 128, 128, 'tied', False, True, torch.float),
            (2, 256, 4, 64, 100, 'g_only', True, True, torch.float),
            (2, 256, 4, 100, 64, 'none', True, True, torch.float),
            (2, 256, 4, 128, 128, 'tied', True, True, torch.float16),
        ]
    ],
)
def test_chunk_bwd(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    gates: str,
    use_alpha: bool,
    use_initial_state: bool,
    dtype: torch.dtype,
):
    """All gradients must match the exact-solve `naive_recurrent_gka`, for a loss on the outputs and both final states."""
    torch.manual_seed(42)
    inputs = make_inputs(B, T, H, K, V, gates, use_alpha, use_initial_state, dtype)
    do = torch.randn(B, T, H, V, device=device)
    dht = _random_dht(B, H, K, V)

    _, (tri_hkk, tri_hkv), tri = _grads(chunk_gka, *inputs, do, dht)
    _, (ref_hkk, ref_hkv), ref = _grads(naive_recurrent_gka, *inputs, do, dht)

    assert_close('h_kk_final', ref_hkk, tri_hkk, 0.005)
    assert_close('h_kv_final', ref_hkv, tri_hkv, 0.005)
    assert ref.keys() == tri.keys()
    for name in ref:
        assert_close(name, ref[name].float(), tri[name].float(), 0.02 if name in ('dg', 'dgk', 'dh0_kk') else 0.01)


@pytest.mark.parametrize(
    ('H', 'K', 'gates', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-K{}-{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (3, 50, 'tied', [0, 15], torch.float),
            (4, 64, 'tied', [0, 14, 121, 300, 400], torch.float),
            (4, 100, 'g_only', [0, 200, 256, 500], torch.float),
            (4, 128, 'tied', [0, 15, 100, 300], torch.float16),
        ]
    ],
)
def test_chunk_varlen_bwd(
    H: int,
    K: int,
    gates: str,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    """All gradients for packed sequences must match `naive_recurrent_gka` run on each sequence separately."""
    torch.manual_seed(42)
    T, N = cu_seqlens[-1], len(cu_seqlens) - 1
    inputs = make_inputs(1, T, H, K, K, gates, True, True, dtype, N=N)
    do = torch.randn(1, T, H, K, device=device)
    dht = _random_dht(N, H, K, K)
    cu = torch.tensor(cu_seqlens, dtype=torch.long, device=device)

    _, _, tri = _grads(chunk_gka, *inputs, do, dht, cu_seqlens=cu)

    # the reference runs each sequence separately on slices of the same leaves
    q, k, v, g, gk, alpha, initial_state = _leaves(*inputs)
    loss = 0
    for i in range(N):
        bos, eos = cu_seqlens[i], cu_seqlens[i + 1]
        o, (h_kk, h_kv) = naive_recurrent_gka(
            *(x[:, bos:eos] for x in (q, k, v)),
            *(x[:, bos:eos] if x is not None else None for x in (g, gk, alpha)),
            initial_state=tuple(s[i:i + 1] for s in initial_state),
            output_final_state=True,
        )
        loss = loss + (o.float() * do[:, bos:eos]).sum() + (h_kk * dht[0][i]).sum() + (h_kv * dht[1][i]).sum()
    loss.backward()
    ref = {'dq': q.grad, 'dk': k.grad, 'dv': v.grad, 'dg': g.grad, 'dalpha': alpha.grad}
    if gk is not None and gk is not g:
        ref['dgk'] = gk.grad
    ref['dh0_kk'], ref['dh0_kv'] = (s.grad for s in initial_state)

    assert ref.keys() == tri.keys()
    for name in ref:
        assert_close(name, ref[name].float(), tri[name].float(), 0.02 if name in ('dg', 'dgk', 'dh0_kk') else 0.01)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'gates', 'split'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-{}-split{}".format(*test))
        for test in [
            (2, 300, 4, 64, 'tied', 100),
            (2, 300, 4, 64, 'tied', 128),
            (2, 300, 4, 100, 'g_only', 37),
            (2, 300, 4, 64, 'none', 200),
        ]
    ],
)
def test_chunk_state_chaining(
    B: int,
    T: int,
    H: int,
    K: int,
    gates: str,
    split: int,
):
    """Two calls chained through the final states must match one call: covers dh0, dht and the tail chunk."""
    torch.manual_seed(42)
    inputs = make_inputs(B, T, H, K, K, gates, True, True, torch.float)
    do = torch.randn(B, T, H, K, device=device)
    dht = _random_dht(B, H, K, K)

    full, (full_hkk, full_hkv), ref = _grads(chunk_gka, *inputs, do, dht)

    q, k, v, g, gk, alpha, initial_state = _leaves(*inputs)

    def part(sl):
        sliced = (x[:, sl] if x is not None else None for x in (q, k, v, g, gk, alpha))
        return dict(zip(('q', 'k', 'v', 'g', 'gk', 'alpha'), sliced))
    first, state = chunk_gka(**part(slice(0, split)), initial_state=initial_state, output_final_state=True)
    second, (h_kk, h_kv) = chunk_gka(**part(slice(split, None)), initial_state=state, output_final_state=True)
    o = torch.cat([first, second], dim=1)
    ((o * do).sum() + (h_kk * dht[0]).sum() + (h_kv * dht[1]).sum()).backward()
    tri = {'dq': q.grad, 'dk': k.grad, 'dv': v.grad, 'dalpha': alpha.grad}
    if g is not None:
        tri['dg'] = g.grad
    tri['dh0_kk'], tri['dh0_kv'] = (s.grad for s in initial_state)

    assert_close('o', full, o, 0.005)
    assert_close('h_kk_final', full_hkk, h_kk, 0.005)
    assert_close('h_kv_final', full_hkv, h_kv, 0.005)
    for name in tri:
        assert_close(name, ref[name], tri[name], 0.01)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'gates'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-{}".format(*test))
        for test in [
            (2, 300, 4, 64, 'tied'),
            (2, 300, 4, 60, 'none'),
        ]
    ],
)
def test_chunk_bwd_split_matches_fused(
    B: int,
    T: int,
    H: int,
    K: int,
    gates: str,
):
    """The two-launch per-chunk backward (used for float32 with K > 64) must reproduce the single launch."""
    from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
    from fla.ops.gka.chunk_solve_bwd import chunk_gka_solve_bwd
    from fla.ops.gka.chunk_solve_fwd import chunk_gka_solve_fwd
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2

    torch.manual_seed(42)
    q, k, _, _, gk, _, initial_state = make_inputs(B, T, H, K, K, gates, False, True, torch.float)
    gk = chunk_local_cumsum(gk, chunk_size=64, scale=RCP_LN2) if gk is not None else None
    h, _ = chunk_fwd_h(k=k, v=k, g=gk, h0=initial_state[0], chunk_size=64)
    x, fro = chunk_gka_solve_fwd(q, k, h, gk=gk)
    dq, _ = chunk_gka_solve_fwd(torch.randn_like(x), k, h, gk=gk, fro=fro)
    dh, _ = chunk_bwd_dh(q=dq, k=k, v=k, do=x, h0=None, dht=torch.randn_like(initial_state[0]), scale=1., g=gk)

    fused = chunk_gka_solve_bwd(k, x, dq, h, dh, fro, gk=gk, output_dh0=True, split=False)
    split = chunk_gka_solve_bwd(k, x, dq, h, dh, fro, gk=gk, output_dh0=True, split=True)
    for name, ref, tri in zip(('dk', 'dg', 'dh0_lamb'), fused, split):
        if ref is None:
            assert tri is None
        else:
            assert_close(name, ref, tri, 1e-5)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'gates', 'use_alpha', 'use_initial_state', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-V{}-{}-alpha{}-h0{}-{}".format(*test))
        for test in [
            (1, 1, 1, 64, 64, 'tied', True, True, torch.float),
            (2, 7, 4, 60, 60, 'tied', True, True, torch.float),
            (2, 100, 4, 128, 128, 'tied', False, True, torch.float),
            (2, 50, 4, 100, 64, 'none', True, False, torch.float),
            (2, 64, 4, 64, 100, 'g_only', True, True, torch.float16),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    gates: str,
    use_alpha: bool,
    use_initial_state: bool,
    dtype: torch.dtype,
):
    """`fused_recurrent_gka`'s outputs and final states must match `naive_recurrent_gka_chebyshev` at the same `num_iter`."""
    torch.manual_seed(42)
    q, k, v, g, gk, alpha, initial_state = make_inputs(B, T, H, K, V, gates, use_alpha, use_initial_state, dtype)

    tri, (tri_hkk, tri_hkv) = fused_recurrent_gka(
        q=q, k=k, v=v, g=g, gk=gk, alpha=alpha, initial_state=initial_state, output_final_state=True,
    )
    ref, (ref_hkk, ref_hkv) = naive_recurrent_gka_chebyshev(
        q=q, k=k, v=v, g=g, gk=gk, alpha=alpha, initial_state=initial_state, output_final_state=True,
    )

    assert_close('o', ref.float(), tri.float(), 0.002)
    assert_close('h_kk_final', ref_hkk, tri_hkk, 0.001)
    assert_close('h_kv_final', ref_hkv, tri_hkv, 0.001)


@pytest.mark.parametrize(
    ('H', 'K', 'gates', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-K{}-{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 64, 'tied', [0, 1, 8, 40, 100], torch.float),
            (4, 100, 'g_only', [0, 30, 31, 90], torch.float16),
        ]
    ],
)
def test_fused_recurrent_varlen(
    H: int,
    K: int,
    gates: str,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    """Packed sequences must match `naive_recurrent_gka_chebyshev` run on each sequence separately."""
    torch.manual_seed(42)
    T, N = cu_seqlens[-1], len(cu_seqlens) - 1
    q, k, v, g, gk, alpha, initial_state = make_inputs(1, T, H, K, K, gates, True, True, dtype, N=N)

    tri, (tri_hkk, tri_hkv) = fused_recurrent_gka(
        q=q, k=k, v=v, g=g, gk=gk, alpha=alpha, initial_state=initial_state, output_final_state=True,
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.long, device=device),
    )
    refs, ref_hkks, ref_hkvs = [], [], []
    for i in range(N):
        bos, eos = cu_seqlens[i], cu_seqlens[i + 1]
        ref, (ref_hkk, ref_hkv) = naive_recurrent_gka_chebyshev(
            *(x[:, bos:eos] for x in (q, k, v)),
            *(x[:, bos:eos] if x is not None else None for x in (g, gk, alpha)),
            initial_state=tuple(s[i:i + 1] for s in initial_state),
            output_final_state=True,
        )
        refs.append(ref)
        ref_hkks.append(ref_hkk)
        ref_hkvs.append(ref_hkv)

    assert_close('o', torch.cat(refs, 1).float(), tri.float(), 0.002)
    assert_close('h_kk_final', torch.cat(ref_hkks, 0), tri_hkk, 0.001)
    assert_close('h_kv_final', torch.cat(ref_hkvs, 0), tri_hkv, 0.001)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'gates', 'prefill'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-{}-prefill{}".format(*test))
        for test in [
            (2, 140, 4, 64, 'tied', 100),
            (2, 80, 4, 128, 'g_only', 64),
            (1, 40, 2, 60, 'none', 1),
        ]
    ],
)
def test_fused_recurrent_decode_after_chunk_prefill(
    B: int,
    T: int,
    H: int,
    K: int,
    gates: str,
    prefill: int,
):
    """Prefill with `chunk_gka`, then decode one token at a time: must match `chunk_gka` on the whole sequence."""
    torch.manual_seed(42)
    q, k, v, g, gk, alpha, initial_state = make_inputs(B, T, H, K, K, gates, True, True, torch.float)

    def step(sl):
        sliced = (x[:, sl] if x is not None else None for x in (q, k, v, g, gk, alpha))
        return dict(zip(('q', 'k', 'v', 'g', 'gk', 'alpha'), sliced))

    ref, (ref_hkk, ref_hkv) = chunk_gka(**step(slice(None)), initial_state=initial_state, output_final_state=True)
    o, state = chunk_gka(**step(slice(0, prefill)), initial_state=initial_state, output_final_state=True)
    outputs = [o]
    for t in range(prefill, T):
        o, state = fused_recurrent_gka(**step(slice(t, t + 1)), initial_state=state, output_final_state=True)
        outputs.append(o)

    assert_close('o', ref, torch.cat(outputs, 1), 0.01)
    assert_close('h_kk_final', ref_hkk, state[0], 0.005)
    assert_close('h_kv_final', ref_hkv, state[1], 0.005)


def test_fused_recurrent_has_no_backward():
    q, k, v, g, gk, alpha, _ = make_inputs(1, 8, 2, 32, 32, 'tied', True, False, torch.float)
    q.requires_grad_()
    o, _ = fused_recurrent_gka(q, k, v, g=g, gk=gk, alpha=alpha)
    with pytest.raises(NotImplementedError, match='inference-only'):
        o.sum().backward()


def test_unsupported_inputs():
    q, k, v, g, gk, alpha, initial_state = make_inputs(2, 64, 2, 32, 32, 'tied', True, True, torch.float)
    with pytest.raises(ValueError, match='head dim K of at most 128'):
        big = torch.randn(1, 16, 1, 256, device=device)
        chunk_gka(big, big, big)
    with pytest.raises(ValueError, match='batch size is expected to be 1'):
        chunk_gka(q, k, v, g=g, gk=gk, cu_seqlens=torch.tensor([0, 64], dtype=torch.long, device=device))
    with pytest.raises(ValueError, match='initial_state'):
        chunk_gka(q, k, v, g=g, gk=gk, initial_state=tuple(s[:1] for s in initial_state))
    with pytest.raises(ValueError, match='`gk` must have shape'):
        chunk_gka(q, k, v, g=g, gk=gk[..., :1])
    for fn in (chunk_gka, fused_recurrent_gka):
        with pytest.raises(ValueError, match='`num_iter` must be at least 1'):
            fn(q, k, v, g=g, gk=gk, num_iter=0)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(('K', 'V'), [(32, 64), (64, 16)])
def test_chunk_small_head_dim_backward_raises(K: int, V: int, dtype: torch.dtype):
    q, k, v, g, gk, alpha, _ = make_inputs(1, 64, 2, K, V, 'tied', True, False, dtype)
    with torch.no_grad():
        chunk_gka(q, k, v, g=g, gk=gk, alpha=alpha)
    q, k, v = (x.requires_grad_() for x in (q, k, v))
    o, (_, h_kv) = chunk_gka(q, k, v, g=g, gk=gk, alpha=alpha, output_final_state=True)
    # refused before chunk_simple_gla's backward kernels launch, through the output and through the final state
    with pytest.raises(RuntimeError, match='not supported for K='):
        o.float().sum().backward(retain_graph=True)
    with pytest.raises(RuntimeError, match='not supported for K='):
        h_kv.sum().backward()
