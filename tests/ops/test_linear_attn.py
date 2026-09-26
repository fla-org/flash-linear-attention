# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.linear_attn import LinearAttention
from fla.models.utils import Cache
from fla.modules import RMSNorm
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn, fused_recurrent_linear_attn
from fla.ops.linear_attn.naive import naive_chunk_linear_attn, naive_recurrent_linear_attn
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-{}".format(*test))
        for test in [
            (1, 64, 1, 64, None, torch.float),
            (2, 512, 4, 60, None, torch.float),
            (3, 1024, 8, 128, 1., torch.float),
            (3, 1024, 8, 128, 0.1, torch.float),
            (3, 1024, 8, 128, None, torch.float),
            (2, 2048, 8, 256, None, torch.float16),
            (2, 2048, 4, 256, None, torch.float16),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float | None,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    h0 = torch.randn((B, H, D, D), dtype=torch.float, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    ref, ref_ht = naive_recurrent_linear_attn(q, k, v, scale=scale, initial_state=h0, output_final_state=True, normalize=False)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = fused_recurrent_linear_attn(q, k, v, scale=scale, initial_state=h0, output_final_state=True, normalize=False)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('o', ref, tri, 0.001)
    assert_close('ht', ref_ht, tri_ht, 0.001)
    assert_close('dq', ref_dq, tri_dq, 0.001)
    assert_close('dk', ref_dk, tri_dk, 0.001)
    assert_close('dv', ref_dv, tri_dv, 0.001)
    assert_close('dh0', ref_dh0, tri_dh0, 0.001)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'normalize', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-norm{}-{}".format(*test))
        for test in [
            (1, 128, 2, 64, False, torch.float),
            (2, 256, 4, 60, False, torch.float),
            (1, 128, 2, 64, True, torch.float),
            (2, 256, 4, 60, True, torch.float),
        ]
    ],
)
def test_naive_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    normalize: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn((B, T, H, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)

    ref, _ = naive_recurrent_linear_attn(q, k, v, normalize=normalize)
    tri = naive_chunk_linear_attn(q, k, v, normalize=normalize)

    assert_close('o', ref, tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, torch.float16),
            (2, 500, 3, 60, torch.float16),
            (2, 1000, 3, 128, torch.float16),
            (3, 1000, 4, 64, torch.float16),
            (2, 2048, 4, 256, torch.float16),
        ]
    ],
)
@pytest.mark.smoke
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    h0 = torch.randn((B, H, D, D), dtype=torch.float, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    ref, ref_ht = fused_recurrent_linear_attn(
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        initial_state=h0,
        output_final_state=True,
        normalize=False,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = chunk_linear_attn(
        q=q,
        k=k,
        v=v,
        initial_state=h0,
        output_final_state=True,
        normalize=False,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('o', ref, tri, 0.001)
    assert_close('ht', ref_ht, tri_ht, 0.001)
    assert_close('dq', ref_dq, tri_dq, 0.001)
    assert_close('dk', ref_dk, tri_dk, 0.001)
    assert_close('dv', ref_dv, tri_dv, 0.001)
    assert_close('dh0', ref_dh0, tri_dh0, 0.001)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, torch.float16),
            (2, 500, 3, 60, torch.float16),
            (2, 1000, 3, 128, torch.float16),
            (3, 1000, 4, 64, torch.float16),
            (2, 2048, 4, 256, torch.float16),
        ]
    ],
)
def test_fused_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    h0 = torch.randn((B, H, D, D), dtype=torch.float, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    ref, ref_ht = fused_recurrent_linear_attn(
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        initial_state=h0,
        output_final_state=True,
        normalize=False,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = fused_chunk_linear_attn(
        q=q,
        k=k,
        v=v,
        initial_state=h0,
        output_final_state=True,
        normalize=False,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('o', ref, tri, 0.001)
    assert_close('ht', ref_ht, tri_ht, 0.001)
    assert_close('dq', ref_dq, tri_dq, 0.001)
    assert_close('dk', ref_dk, tri_dk, 0.001)
    assert_close('dv', ref_dv, tri_dv, 0.001)
    assert_close('dh0', ref_dh0, tri_dh0, 0.001)


@pytest.mark.parametrize(
    ('fn', 'B', 'T', 'split', 'H', 'D'),
    [
        pytest.param(fn, 2, 256, 128, 4, 64, id=f"{name}-split128")
        for fn, name in [
            (fused_recurrent_linear_attn, 'fused_recurrent'),
            (fused_chunk_linear_attn, 'fused_chunk'),
            (chunk_linear_attn, 'chunk'),
        ]
    ],
)
def test_normalize_split_resume(fn, B: int, T: int, split: int, H: int, D: int):
    """Splitting at `split` and resuming via the returned (kv_state, z_state)
    must reproduce the single-call output when normalize=True."""
    torch.manual_seed(42)
    q = torch.randn((B, T, H, D), dtype=torch.float32, device=device)
    k = torch.randn((B, T, H, D), dtype=torch.float32, device=device)
    v = torch.randn((B, T, H, D), dtype=torch.float32, device=device)

    o_full, _ = fn(q=q, k=k, v=v, output_final_state=True, normalize=True)

    o_a, state_a = fn(
        q=q[:, :split], k=k[:, :split], v=v[:, :split],
        output_final_state=True, normalize=True,
    )
    o_b, _ = fn(
        q=q[:, split:], k=k[:, split:], v=v[:, split:],
        initial_state=state_a, output_final_state=True, normalize=True,
    )
    o_split = torch.cat([o_a, o_b], dim=1)

    assert_close('o', o_full, o_split, 0.002)


@pytest.mark.parametrize(
    ('fn', 'B', 'T', 'H', 'D'),
    [
        pytest.param(fn, 2, 256, 4, 64, id=f"{name}-normgrad")
        for fn, name in [
            (fused_recurrent_linear_attn, 'fused_recurrent'),
            (fused_chunk_linear_attn, 'fused_chunk'),
            (chunk_linear_attn, 'chunk'),
        ]
    ],
)
def test_normalize_grad(fn, B: int, T: int, H: int, D: int):
    """Gradient parity for `normalize=True` against the naive recurrent reference.

    Regression test for the missing `dk` denominator-path contribution: pre-fix,
    `chunk_global_cumsum` had no autograd so the chain `k -> k_cum -> denom` was
    severed and `dk` collapsed to the numerator-only contribution.

    Uses strictly-positive q, k ( normalize=True is designed for);
    """
    torch.manual_seed(42)
    eps = 0.01
    q = (torch.randn((B, T, H, D), dtype=torch.float32, device=device).abs() + eps).requires_grad_()
    k = (torch.randn((B, T, H, D), dtype=torch.float32, device=device).abs() + eps).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=torch.float32, device=device).requires_grad_()
    do = torch.randn_like(v)

    ref, _ = naive_recurrent_linear_attn(q, k, v, normalize=True)
    (ref * do).sum().backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri, _ = fn(q=q, k=k, v=v, normalize=True)
    (tri * do).sum().backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close('o', ref, tri, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)


@pytest.mark.parametrize(
    ('fn', 'B', 'T', 'split', 'H', 'D'),
    [
        pytest.param(fn, 2, 256, 128, 4, 64, id=f"{name}-zinitgrad")
        for fn, name in [
            (fused_recurrent_linear_attn, 'fused_recurrent'),
            (fused_chunk_linear_attn, 'fused_chunk'),
            (chunk_linear_attn, 'chunk'),
        ]
    ],
)
def test_normalize_zinit_grad(fn, B: int, T: int, split: int, H: int, D: int):
    """Gradient parity when chaining via `(kv_state, z_state)`: splitting at
    `split` and resuming with the prior `z_state` as `z_init` must produce the
    same q/k/v gradients as a single-call run."""
    torch.manual_seed(42)
    eps = 0.01
    q = (torch.randn((B, T, H, D), dtype=torch.float32, device=device).abs() + eps).requires_grad_()
    k = (torch.randn((B, T, H, D), dtype=torch.float32, device=device).abs() + eps).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=torch.float32, device=device).requires_grad_()
    do = torch.randn_like(v)

    o_full, _ = fn(q=q, k=k, v=v, normalize=True)
    (o_full * do).sum().backward()
    full_dq, q.grad = q.grad.clone(), None
    full_dk, k.grad = k.grad.clone(), None
    full_dv, v.grad = v.grad.clone(), None

    o_a, state_a = fn(
        q=q[:, :split], k=k[:, :split], v=v[:, :split],
        output_final_state=True, normalize=True,
    )
    o_b, _ = fn(
        q=q[:, split:], k=k[:, split:], v=v[:, split:],
        initial_state=state_a, normalize=True,
    )
    o_split = torch.cat([o_a, o_b], dim=1)
    (o_split * do).sum().backward()
    split_dq, q.grad = q.grad.clone(), None
    split_dk, k.grad = k.grad.clone(), None
    split_dv, v.grad = v.grad.clone(), None

    assert_close('o', o_full, o_split, 0.002)
    assert_close('dq', full_dq, split_dq, 0.005)
    assert_close('dk', full_dk, split_dk, 0.005)
    assert_close('dv', full_dv, split_dv, 0.005)


@pytest.mark.smoke
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize('packed', [False, True])
def test_layer_padding_grad(dtype, packed):
    torch.manual_seed(42)
    layer = LinearAttention(
        hidden_size=128,
        num_heads=2,
        feature_map='elu',
        norm_k=True,
        do_feature_map_norm=True,
        output_norm='identity',
        layer_idx=0,
    ).to(device=device, dtype=dtype)
    x = torch.randn(3, 73, 128, device=device, dtype=dtype, requires_grad=True)
    mask = torch.zeros(3, 73, device=device, dtype=torch.bool)
    mask[1, 7:72], mask[2, :67] = True, True
    cu_seqlens = torch.tensor([0, 0, 65, 132], device=device, dtype=torch.int32)
    refs = [layer(x[i:i+1, mask[i]], past_key_values=Cache(), use_cache=True) for i in [1, 2]]
    ref = torch.cat([r[0].squeeze(0) for r in refs])
    ref_z = refs[-1][2][0]['recurrent_state'][1].sum()
    cache = Cache()
    if packed:
        out = layer(x[mask].unsqueeze(0), past_key_values=cache, use_cache=True, cu_seqlens=cu_seqlens)[0].squeeze(0)
    else:
        out = layer(x, attention_mask=mask, past_key_values=cache, use_cache=True)[0]
        assert torch.count_nonzero(out[~mask]) == 0
        out = out[mask]
    assert_close('o', ref, out, 1e-3)
    do = torch.zeros_like(ref)
    do[65:] = torch.randn_like(ref[65:])
    inputs = (x, *layer.parameters())
    ref_grads = torch.autograd.grad((ref * do).sum() + ref_z, inputs)
    grads = torch.autograd.grad((out * do).sum() + cache[0]['recurrent_state'][1][-1].sum(), inputs)
    assert torch.count_nonzero(grads[0][~mask]) == 0
    assert torch.count_nonzero(grads[0][1]) == 0
    for ref_grad, grad in zip(ref_grads, grads):
        assert_close('grad', ref_grad, grad, 1e-3)
    layer.norm = RMSNorm(layer.head_v_dim).to(device=device, dtype=dtype)
    out = layer(x, attention_mask=torch.zeros_like(mask))[0]
    assert torch.count_nonzero(out) == 0
    for grad in torch.autograd.grad(out.sum(), (x, *layer.parameters())):
        assert torch.count_nonzero(grad) == 0


@pytest.mark.smoke
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize('mode', ['chunk', 'fused_chunk', 'fused_recurrent'])
@pytest.mark.parametrize('lengths', [(0, 65, 67), (0, 0, 0)])
@torch.no_grad()
def test_layer_padding_cache(dtype, mode, lengths):
    torch.manual_seed(42)
    layer = LinearAttention(
        hidden_size=128,
        num_heads=2,
        feature_map='elu',
        do_feature_map_norm=True,
        output_norm='identity',
        mode=mode,
        layer_idx=0,
    ).to(device=device, dtype=dtype)
    x = torch.randn(3, 73, 128, device=device, dtype=dtype)
    mask = torch.arange(73, device=device)[None] < torch.tensor(lengths, device=device)[:, None]
    out, _, cache = layer(x, attention_mask=mask, past_key_values=Cache(), use_cache=True)
    assert torch.count_nonzero(out[~mask]) == 0
    states = cache[0]['recurrent_state']
    next_x = torch.randn(3, 1, 128, device=device, dtype=dtype)
    layer(next_x, attention_mask=torch.zeros_like(mask[:, :1]), past_key_values=cache, use_cache=True)
    for before, after in zip(states, cache[0]['recurrent_state']):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    next_mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
    next_out = layer(next_x, attention_mask=next_mask, past_key_values=cache, use_cache=True)[0]
    assert cache.get_seq_length(0) == 75
    for i, length in enumerate(lengths):
        ref_cache = Cache()
        if length:
            ref, _, ref_cache = layer(x[i:i+1, :length], past_key_values=ref_cache, use_cache=True)
            assert_close('o', ref, out[i:i+1, :length], 1e-3)
            for ref_state, state in zip(ref_cache[0]['recurrent_state'], states):
                assert_close('state', ref_state, state[i:i+1], 1e-3)
        else:
            for state in states:
                assert torch.count_nonzero(state[i]) == 0
        ref = layer(next_x[i:i+1], past_key_values=ref_cache, use_cache=True)[0]
        assert_close('decode', ref, next_out[i:i+1], 1e-3)
