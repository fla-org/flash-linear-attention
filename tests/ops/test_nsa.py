
import os

import pytest
import torch
import triton

from fla.ops.nsa.naive import naive_nsa
from fla.ops.nsa.parallel import parallel_nsa
from fla.ops.utils import prepare_token_indices
from fla.utils import assert_close, device


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
            (4, 2048, 2, 32, 64, 16, 32, 0.1, torch.float16),
        ]
    ],
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
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device=device)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
                block_indices[b, t, h, :len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]

    ref = naive_nsa(q=q, k=k, v=v, block_indices=block_indices, block_size=block_size, scale=scale)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_nsa(q=q, k=k, v=v, block_indices=block_indices, block_size=block_size, scale=scale)
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
            (2, 32, 64, 16, 32, [0, 256, 500, 1000], torch.float16),
            (2, 32, 100, 16, 32, [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_parallel_varlen(
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)

    block_indices = torch.full((1, T, H, S), T, dtype=torch.long, device=device)
    seq_indices = prepare_token_indices(cu_seqlens).tolist()

    for i in range(T):
        _, t = seq_indices[i]
        for h in range(H):
            i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
            block_indices[0, i, h, :len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]

    ref = naive_nsa(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_size=block_size,
        cu_seqlens=cu_seqlens,
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
        block_size=block_size,
        cu_seqlens=cu_seqlens,
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close('o', ref, tri, 0.004)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
