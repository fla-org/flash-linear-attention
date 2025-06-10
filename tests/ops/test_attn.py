# -*- coding: utf-8 -*-

import os

import pytest
import torch

from fla.ops.attn.parallel import parallel_attn
from fla.ops.utils import prepare_lens
from fla.utils import COMPILER_MODE, assert_close, check_shared_mem, device

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    HAS_FLASH = True
except Exception:
    HAS_FLASH = False


if COMPILER_MODE:
    test_b_list = [2]
    test_t_list = [2048]
    test_t_varlen_list = test_t_list
    test_d_list = [64, 100, 128]
else:
    test_b_list = [2, 4]
    test_t_list = [1, 15, 63, 286, 300, 1024, 2048]
    test_t_varlen_list = [63, 286, 300, 512]
    test_d_list = [32, 64, 100]
test_hq_list = [8, 16]
test_h_list = [2]


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('HQ', test_hq_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('scale', [0.1])
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    not HAS_FLASH,
    reason="Skipping test because flash-attn is not installed"
)
def test_parallel(
    B: int,
    H: int,
    HQ: int,
    T: int,
    D: int,
    dtype: torch.dtype,
    scale: float,
):
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shard mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    ref = flash_attn_func(q=q, k=k, v=v, softmax_scale=scale, causal=True)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_attn(q=q, k=k, v=v, scale=scale)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)


@pytest.mark.parametrize('N', test_b_list)
@pytest.mark.parametrize('T', test_t_varlen_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('HQ', test_hq_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    not HAS_FLASH,
    reason="Skipping test because flash-attn is not installed"
)
def test_parallel_varlen(
    N: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    dtype: torch.dtype,
):
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shard mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    N = min(1, N) if T < 64 else N
    # randomly split the sequence into N segments
    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 16)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0].to(torch.int32)
    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)

    ref = flash_attn_varlen_func(
        q=q.squeeze(0),
        k=k.squeeze(0),
        v=v.squeeze(0),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=prepare_lens(cu_seqlens).max(),
        max_seqlen_k=prepare_lens(cu_seqlens).max(),
        causal=True
    )
    ref.backward(do.squeeze(0))
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_attn(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, 0.004)
    assert_close("dq", ref_dq.squeeze(), tri_dq.squeeze(), 0.005)
    assert_close("dk", ref_dk.squeeze(), tri_dk.squeeze(), 0.005)
    assert_close("dv", ref_dv.squeeze(), tri_dv.squeeze(), 0.005)
