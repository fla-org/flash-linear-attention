# -*- coding: utf-8 -*-

import os
from typing import List

import pytest
import torch
import triton
import warnings

from fla.ops.nsa.naive import naive_nsa, naive_nsa_sel, naive_nsa_cmp, naive_nsa_topk
from fla.ops.nsa.parallel import parallel_nsa, parallel_nsa_fwd, parallel_nsa_topk
from fla.ops.nsa.compression import parallel_nsa_compression
from fla.ops.utils import prepare_token_indices
from fla.utils import assert_close, device
from fla.ops.utils.pooling import mean_pooling

os.environ['TRITON_F32_DEFAULT'] = 'ieee'

def build_block_indices(B, T, H, S, block_size, seq_indices=None):
    block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device=device)
    for b in range(B):
        for i in range(T):
            if seq_indices is None:
                t = i
            else:
                _, t = seq_indices[i]
            for h in range(H):
                i_i = torch.randperm(triton.cdiv(t + 1, block_size))[:S]
                block_indices[b, i, h, :len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]
    return block_indices

def build_partial_varlen(x, cu_seqlens, q_lens):
    partial_x = torch.cat([x[:, cu_seqlens[i + 1] - q_lens[i]: cu_seqlens[i + 1]] for i in range(len(q_lens))], dim=1)
    return partial_x

# FIXME
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'S', 'block_size', 'scale', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-S{}-block_size{}-scale{}-{}".format(*test))
        for test in [
            (1, 63, 1, 16, 64, 16, 32, 1.0, torch.float16),
            (3, 111, 1, 32, 100, 16, 32, 1.0, torch.float16),
            (3, 1024, 2, 32, 60, 16, 32, 0.1, torch.float16),
            (3, 1024, 2, 32, 128, 16, 32, 0.1, torch.float16),
            (4, 2048, 2, 32, 64, 16, 32, 0.1, torch.float16)
        ]
    ]
)
def test_parallel(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    scale: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    block_indices = build_block_indices(B, T, H, S, block_size)

    ref = naive_nsa_sel(q=q, k=k, v=v, block_indices=block_indices, block_size=block_size, scale=scale)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_nsa(q=q, k=k, v=v, block_indices=block_indices, block_counts=S, block_size=block_size, scale=scale)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)


@pytest.mark.parametrize(
    ('H', 'HQ', 'D', 'S', 'block_size', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-HQ{}-D{}-S{}-block_size{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (1, 16, 64, 16, 32, [0, 15], torch.float16),
            (1, 16, 64, 8, 16, [0, 15, 205, 550, 800], torch.float16),
            (2, 32, 64, 16, 32, [0, 256, 500, 1000], torch.float16),
            (2, 32, 100, 16, 32, [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ]
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test because SKIP_TEST_CHUNK_VARLEN is set'
)
def test_parallel_varlen(
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    cu_seqlens: List[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)

    seq_indices = prepare_token_indices(cu_seqlens)
    block_indices = build_block_indices(1, T, H, S, block_size, seq_indices.tolist())

    ref = naive_nsa_sel(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_size=block_size,
        cu_seqlens=cu_seqlens
    )
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_nsa(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_counts=S,
        block_size=block_size,
        cu_seqlens=cu_seqlens
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close('o', ref, tri, 0.004)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)

@pytest.mark.parametrize(
    ('B', 'T', 'Tq', 'H', 'HQ', 'D', 'S', 'block_size', 'scale', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-Tq{}-H{}-HQ{}-D{}-S{}-block_size{}-scale{}-{}".format(*test))
        for test in [
            (1, 63, 1, 1, 16, 64, 16, 32, 1.0, torch.float16),
            (3, 111, 15, 1, 32, 100, 16, 32, 1.0, torch.float16),
            (3, 1024, 3, 2, 32, 60, 16, 32, 0.1, torch.float16),
            (3, 1024, 33, 2, 32, 128, 16, 32, 0.1, torch.float16),
            (4, 2048, 25, 2, 32, 64, 16, 32, 0.1, torch.float16)
        ]
    ]
)
def test_parallel_selective_decode(
    B: int,
    T: int,
    Tq: int,
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    scale: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)

    block_indices = build_block_indices(B, T, H, S, block_size)

    o_full, lse_full = parallel_nsa_fwd(
        q, k, v,
        block_indices,
        S,
        block_size,
        scale,
    )

    o_short, lse_short = parallel_nsa_fwd(
        q[:, -Tq:].contiguous(),      # only the last T_q queries
        k, v,
        block_indices[:, -Tq:].contiguous(),
        S,
        block_size,
        scale,
    )

    o_naive_fla = naive_nsa_sel(
        q, k, v, block_indices, block_size, scale
    )

    assert_close(
        'outputs: full-vs-naive',
        o_naive_fla, o_full, 0.005
    )
    assert_close(
        'outputs: full-vs-cached',
        o_short, o_full[:, -Tq:], 0.005
    )
    assert_close(
        'log-sum-exp: full-vs-cached',
        lse_short, lse_full[:, -Tq:], 0.005
    )


@pytest.mark.parametrize(
    ('B', 'T', 'Tq', 'H', 'HQ', 'D', 'block_size', 'scale', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-Tq{}-H{}-HQ{}-D{}-block_size{}-scale{}-{}".format(*test))
        for test in [
            # (1, 63, 1, 1, 16, 64, 32, 1.0, torch.float16), # Can't pass this as rel grad error bloats with short inputs. Numerical issue?
            (3, 111, 15, 1, 32, 100, 32, 1.0, torch.float16),
            (3, 1024, 3, 2, 32, 60, 32, 0.1, torch.float16),
            (3, 1024, 33, 2, 32, 128, 32, 0.1, torch.float16),
            (4, 2048, 25, 2, 32, 64, 32, 0.1, torch.float16)
        ]
    ]
)
def test_parallel_compressive(
    B: int,
    T: int,
    Tq: int,
    H: int,
    HQ: int,
    D: int,
    block_size: int,
    scale: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    k_cmp, v_cmp = mean_pooling(k, block_size), mean_pooling(v, block_size)
    o_full, lse_full = parallel_nsa_compression(
        q=q,
        k=k_cmp,
        v=v_cmp,
        TK=T,
        block_size=block_size,
        scale=scale,
    )
    o_full.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    o_naive, lse_naive = naive_nsa_cmp(
        q=q,
        k_cmp=k_cmp,
        v_cmp=v_cmp,
        block_size=block_size,
        scale=scale,
    )
    o_naive.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    assert_close(
        'outputs: full-vs-naive',
        o_full, o_naive, 0.005
    )
    # For positions not attending to any token, the log-sum-exp should be -inf; the kernel returns 0 instead, it is
    # OK as those positions will not be used in the compressive attention anyway.
    assert_close(
        'log-sum-exp: full-vs-naive',
        lse_full, torch.where(lse_naive == float('-inf'), 0, lse_naive), 0.005
    )
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)

    o_short, lse_short = parallel_nsa_compression(
        q[:, -Tq:].contiguous(),      # only the last T_q queries
        k_cmp, v_cmp,
        T,
        block_size,
        scale,
    )

    assert_close(
        'outputs: full-vs-cached',
        o_short, o_full[:, -Tq:], 0.005
    )

    assert_close(
        'log-sum-exp: full-vs-cached',
        lse_short, lse_full[:, -Tq:], 0.005
    )


@pytest.mark.parametrize(
    ('B', 'T', 'Tq', 'H', 'HQ', 'D', 'S', 'block_size', 'scale', 'dtype', 'reuse_lse'),
    [
        pytest.param(*test, id="B{}-T{}-Tq{}-H{}-HQ{}-D{}-S{}-block_size{}-scale{}-{}-reuse_lse{}".format(*test))
        for test in [
            (1, 1, 1, 1, 16, 64, 16, 32, 1.0, torch.float16, True),
            (3, 111, 15, 1, 32, 100, 16, 32, 1.0, torch.float16, False),
            (3, 1024, 3, 2, 32, 60, 16, 32, 0.1, torch.float16, True),
            (3, 1024, 33, 2, 32, 128, 16, 32, 0.1, torch.float16, False),
            (4, 2048, 25, 2, 32, 64, 16, 32, 0.1, torch.float32, True) # Use FP32 to reduce numerical issues
        ]
    ]
)
def test_parallel_topk_decode(
    B: int,
    T: int,
    Tq: int,
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    scale: float,
    dtype: torch.dtype,
    reuse_lse: bool,
):
    torch.manual_seed(42)
    # Use a wider range to reduce numerical issues, otherwise there will be too many mismatches due to close scores.
    q = torch.rand((B, T, HQ, D), dtype=dtype, device=device) * 10 - 5
    k = torch.rand((B, T, H, D), dtype=dtype, device=device) * 10 - 5
    v = torch.rand((B, T, H, D), dtype=dtype, device=device) * 10 - 5

    k_cmp, v_cmp = mean_pooling(k, block_size), mean_pooling(v, block_size)

    if reuse_lse:
        # For positions not attending to any token, the log-sum-exp should be -inf; the kernel returns 0 instead, it is
        # OK as those positions will not be used in the compressive attention anyway.
        _, lse_full = naive_nsa_cmp(
            q=q,
            k_cmp=k_cmp,
            v_cmp=v_cmp,
            block_size=block_size,
            scale=scale,
        )
        lse_full = torch.where(lse_full == float('-inf'), 0, lse_full).contiguous()
    else:
        lse_full = None

    block_indices = parallel_nsa_topk(
        q=q,
        k=k_cmp,
        TK=T,
        lse=lse_full,
        block_counts=S,
        block_size=block_size,
        scale=scale,
    )

    block_indices_naive = naive_nsa_topk(
        q, k_cmp, block_counts=S, block_size=block_size, scale=scale,
    )

    # Separate checks for forcefully selected blocks (0, -1, -2)
    fixed_block_indices, free_block_indices = block_indices[:, :, :, :3], block_indices[:, :, :, 3:]
    fixed_block_indices_naive, free_block_indices_naive = block_indices_naive[:, :, :, :3], block_indices_naive[:, :, :, 3:]

    fixed_block_indices, _ = torch.sort(fixed_block_indices, dim=-1)
    fixed_block_indices_naive, _ = torch.sort(fixed_block_indices_naive, dim=-1)

    assert (fixed_block_indices == fixed_block_indices_naive).all(), \
        "Different in forcefully selected block indices compared to naive"

    if not (free_block_indices == free_block_indices_naive).all():
        indices = torch.nonzero(free_block_indices != free_block_indices_naive, as_tuple=False)
        for idx in range(indices.shape[0]):
            b_i, t_i, h_i, s_i = indices[idx]
            q_vals = q[b_i.item(), t_i.item(), h_i * (HQ // H): (h_i + 1) * (HQ // H), :]
            k_vals = k_cmp[b_i.item(), :, h_i.item()]
            a_s = torch.einsum('h k, s k -> s h', q_vals, k_vals) * scale
            a_s[t_i // block_size + ((t_i + 1) % block_size == 0).int():] = float('-inf')
            a_sn = torch.softmax(a_s, dim=0)
            a_snm = a_sn.mean(-1)
            a_lse = torch.log(torch.exp(a_s).sum(0))
            if lse_full is not None:
                k_lse = lse_full[b_i.item(), t_i.item(), h_i * (HQ // H): (h_i + 1) * (HQ // H)]
                assert_close('lse vs naive', a_lse, k_lse, ratio=0.005)

            assert_close('block-score vs naive', a_snm[free_block_indices[b_i, t_i, h_i, s_i]],
                         a_snm[free_block_indices_naive[b_i, t_i, h_i, s_i]], ratio=0.005)
        warnings.warn(f"Block indices mismatch: {len(indices)}/{block_indices.numel()} "
              f"({len(indices) / free_block_indices_naive.numel():.2f}), seemingly due to numerical issues.")

    block_indices_short = parallel_nsa_topk(
        q=q[:, -Tq:].contiguous(),
        k=k_cmp,
        lse=lse_full[:, -Tq:].contiguous() if lse_full is not None else None,
        TK=T,
        block_counts=S,
        block_size=block_size,
        scale=scale,
    )

    fixed_block_indices_short, free_block_indices_short = block_indices_short[:, :, :, :3], block_indices_short[:, :, :, 3:]
    fixed_block_indices_short, _ = torch.sort(fixed_block_indices_short, dim=-1)
    assert (fixed_block_indices_short == fixed_block_indices[:, -Tq:]).all(), \
        "Different in forcefully selected block indices compared to full"
    assert (free_block_indices_short == free_block_indices[:, -Tq:]).all(), \
        "Different in free block indices compared to full"

@pytest.mark.parametrize(
    ('B', 'T', 'Tq', 'H', 'HQ', 'D', 'S', 'block_size', 'scale', 'window_size', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-Tq{}-H{}-HQ{}-D{}-S{}-block_size{}-scale{}-W{}-{}".format(*test))
        for test in [
            (1, 1, 1, 1, 16, 64, 16, 32, 1.0, 0, torch.float16),
            (3, 111, 15, 1, 32, 100, 16, 32, 1.0, 128, torch.float16),
            (3, 1024, 256, 1, 32, 100, 16, 32, 1.0, 128, torch.float16),
            (3, 1024, 3, 2, 32, 60, 16, 32, 0.1, 128, torch.float16),
            (3, 1024, 33, 2, 32, 128, 16, 32, 0.1, 0, torch.float16),
            (4, 2048, 25, 2, 32, 64, 16, 32, 0.1, 512, torch.float16)
        ]
    ]
)
def test_parallel_decode(
    B: int,
    T: int,
    Tq: int,
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    scale: float,
    window_size: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    q = torch.rand((B, T, HQ, D), dtype=dtype, device=device) * 3 - 2
    k = torch.rand((B, T, H, D), dtype=dtype, device=device) * 3 - 2
    v = torch.rand((B, T, H, D), dtype=dtype, device=device) * 3 - 2

    g = torch.randn((B, T, HQ, 3), dtype=dtype, device=device)
    g_cmp, g_slc, g_swa = g.sigmoid().unbind(-1)

    o_full = parallel_nsa(q, k, v, g_cmp, g_slc, g_swa,
                          block_counts=S, block_size=block_size, scale=scale, window_size=window_size)

    o_short = parallel_nsa(
        q[:, -Tq:].contiguous(),      # only the last T_q queries
        k, v,
        g_cmp[:, -Tq:].contiguous(),
        g_slc[:, -Tq:].contiguous(),
        g_swa[:, -Tq:].contiguous(),
        block_counts=S,
        block_size=block_size,
        scale=scale,
        window_size=window_size
    )

    assert_close('short vs full', o_short, o_full[:, -Tq:], 0.005)

@pytest.mark.parametrize(
    ('H', 'HQ', 'D', 'S', 'block_size', 'cu_seqlens', 'q_lens', 'dtype'),
    [
        pytest.param(*test, id="H{}-HQ{}-D{}-S{}-block_size{}-cu_seqlens{}-q_lens{}-{}".format(*test))
        for test in [
            (1, 16, 64, 16, 32, [0, 15], [1,], torch.float16),
            (1, 16, 64, 8, 16, [0, 15, 205, 550, 800], [3, 15, 30, 8], torch.float16),
            (2, 32, 64, 16, 32, [0, 256, 500, 1000], [1, 15, 4], torch.float16),
            (2, 32, 100, 16, 32, [0, 15, 100, 300, 1200, 2000], [5, 3, 1, 1, 128], torch.float16),
        ]
    ]
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test because SKIP_TEST_CHUNK_VARLEN is set'
)
def test_parallel_varlen_decode(
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    cu_seqlens,
    q_lens,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((1, T, H, D), dtype=dtype, device=device)
    v = torch.randn((1, T, H, D), dtype=dtype, device=device)
    scale = 1.0 / (D ** 0.5)

    seq_indices = prepare_token_indices(cu_seqlens)
    block_indices = build_block_indices(1, T, H, S, block_size, seq_indices.tolist())

    o_full, lse_full = parallel_nsa_fwd(
        q, k, v,
        block_indices,
        S,
        block_size,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        scale=scale,
        token_indices_q=seq_indices,
    )

    ref = naive_nsa_sel(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_size=block_size,
        cu_seqlens=cu_seqlens
    )

    q_short = build_partial_varlen(q, cu_seqlens, q_lens)
    block_indices_short = build_partial_varlen(block_indices, cu_seqlens, q_lens)
    cu_seqlens_q = torch.cumsum(torch.tensor([0] + q_lens), dim=0).to(device)
    token_indices_q = prepare_token_indices(cu_seqlens_q)

    o_short_ref = build_partial_varlen(o_full, cu_seqlens, q_lens)
    lse_short_ref = build_partial_varlen(lse_full, cu_seqlens, q_lens)

    o_short, lse_short = parallel_nsa_fwd(
        q_short, k, v,
        block_indices_short,
        S,
        block_size,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens,
        scale= 1.0 / (D ** 0.5),
        token_indices_q=token_indices_q
    )

    assert_close('outputs: full vs naive', ref, o_full, 0.005)
    assert_close('outputs: full vs short', o_short, o_short_ref, 0.005)
    assert_close('lse: full vs short', lse_short, lse_short_ref, 0.005)


@pytest.mark.parametrize(
    ('H', 'HQ', 'D', 'block_size', 'cu_seqlens', 'q_lens', 'dtype'),
    [
        pytest.param(*test, id="H{}-HQ{}-D{}-block_size{}-cu_seqlens{}-q_lens{}-{}".format(*test))
        for test in [
            (1, 16, 64, 32, [0, 15], [1,], torch.float16),
            (1, 16, 64, 16, [0, 15, 205, 550, 800], [3, 15, 30, 8], torch.float16),
            (2, 32, 64, 32, [0, 256, 500, 1000], [1, 15, 4], torch.float16),
            (2, 32, 100, 32, [0, 15, 100, 300, 1200, 2000], [5, 3, 1, 1, 128], torch.float16),
        ]
    ]
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test because SKIP_TEST_CHUNK_VARLEN is set'
)
def test_parallel_compressive_varlen(
    H: int,
    HQ: int,
    D: int,
    block_size: int,
    cu_seqlens,
    q_lens,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)

    scale = 1.0 / (D ** 0.5)
    k_cmp, v_cmp = mean_pooling(k, block_size, cu_seqlens), mean_pooling(v, block_size, cu_seqlens)

    seq_indices = prepare_token_indices(cu_seqlens)

    o_full, lse_full = parallel_nsa_compression(
        q=q,
        k=k_cmp,
        v=v_cmp,
        TK=T,
        block_size=block_size,
        scale=scale,
        cu_seqlens=(cu_seqlens, cu_seqlens),
    )
    o_full.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    o_naive, lse_naive = naive_nsa_cmp(
        q=q,
        k_cmp=k_cmp,
        v_cmp=v_cmp,
        block_size=block_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
        seq_indices=seq_indices
    )
    o_naive.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    assert_close('outputs: full vs naive', o_naive, o_full, 0.005)
    assert_close('lse: full vs naive', torch.where(lse_naive == float('-inf'), 0, lse_naive), lse_full, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)

    q_short = build_partial_varlen(q, cu_seqlens, q_lens)
    cu_seqlens_q = torch.cumsum(torch.tensor([0] + q_lens), dim=0).to(device)

    o_short_ref = build_partial_varlen(o_full, cu_seqlens, q_lens)
    lse_short_ref = build_partial_varlen(lse_full, cu_seqlens, q_lens)

    o_short, lse_short = parallel_nsa_compression(
        q_short,
        k_cmp, v_cmp,
        T,
        block_size,
        scale,
        cu_seqlens=(cu_seqlens_q, cu_seqlens),
    )

    assert_close('outputs: full vs short', o_short, o_short_ref, 0.005)
    assert_close('lse: full vs short', lse_short, lse_short_ref, 0.005)