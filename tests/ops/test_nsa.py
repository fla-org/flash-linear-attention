# -*- coding: utf-8 -*-

import os

import pytest
import torch
import triton

from fla.ops.common.utils import prepare_token_indices
from fla.ops.nsa.naive import naive_nsa
from fla.ops.nsa.parallel import parallel_nsa
from fla.ops.utils.testing import assert_close
from fla.utils import device

compiled_mode = os.getenv("COMPILER_MODE") == "1"
if compiled_mode:
    test_b_list = [1]
    test_t_list = [64]
    test_t_varlen_list = test_t_list
else:
    test_b_list = [2]
    test_t_list = [256, 1024, 2000]
    test_t_varlen_list = [63, 286, 300, 512]
test_h_list = [2]


# FIXME
@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("HQ", [64])
@pytest.mark.parametrize("D", [100, 64])
@pytest.mark.parametrize("S", [16])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
@pytest.mark.skipif(
    True,
    reason="TBD"
)
def test_parallel(
    B: int,
    H: int,
    HQ: int,
    T: int,
    D: int,
    S: int,
    block_size: int,
    dtype: torch.dtype,
    scale: float
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    indices = torch.full((B, T, H, S), T, dtype=torch.long, device=device)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
                indices[b, t, h, :len(i_i)] = i_i
    indices = indices.sort(-1)[0]

    ref = naive_nsa(q=q, k=k, v=v, indices=indices, block_size=block_size, scale=scale)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_nsa(q=q, k=k, v=v, indices=indices, block_size=block_size, scale=scale)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)


@pytest.mark.parametrize("N", test_b_list)
@pytest.mark.parametrize("T", test_t_varlen_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("HQ", [64])
@pytest.mark.parametrize("D", [100, 64])
@pytest.mark.parametrize("S", [16])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set"
)
@pytest.mark.skipif(
    True,
    reason="TBD"
)
def test_parallel_varlen(
    N: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    # randomly split the sequence into N segments
    offsets = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 1)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]
    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)

    indices = torch.full((1, T, H, S), T, dtype=torch.long, device=device)
    seq_indices = prepare_token_indices(offsets).tolist()

    for i in range(T):
        _, t = seq_indices[i]
        for h in range(H):
            i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
            indices[0, i, h, :len(i_i)] = i_i
    indices = indices.sort(-1)[0]

    ref = naive_nsa(
        q=q,
        k=k,
        v=v,
        indices=indices,
        block_size=block_size,
        cu_seqlens=offsets
    )
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_nsa(
        q=q,
        k=k,
        v=v,
        indices=indices,
        block_size=block_size,
        cu_seqlens=offsets
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close("  o", ref, tri, 0.004)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)
