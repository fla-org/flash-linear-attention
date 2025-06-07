# -*- coding: utf-8 -*-

import os
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.ops.forgetting_attn.parallel import parallel_forgetting_attn
from fla.utils import COMPILER_MODE, assert_close, check_shared_mem, device, is_intel_alchemist

if COMPILER_MODE:
    test_b_list = [1]
    test_t_list = [1024]
    test_t_varlen_list = test_t_list
    test_d_list = [64, 100]
else:
    test_b_list = [2]
    test_t_list = [3, 15, 63, 300, 1024, 2048]
    test_t_varlen_list = [63, 300, 1024, 512, 2048]
    test_d_list = [64, 100]
test_fgate_logit_range_list = [(0, 5), (5, 10)]
test_hq_list = [8, 16]
test_h_list = [2]


def naive_forgetting_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None
):
    _, T, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H
    if scale is None:
        scale = D ** -0.5
    gc = g.float().cumsum(1)
    mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=device))
    ref = torch.einsum("bqhd,bkhd->bhqk", q.float() * scale, repeat(k, "b t h d -> b t (h g) d", g=G).float())
    ref = ref + rearrange(gc, "b t h -> b h t 1") - rearrange(gc, "b t h -> b h 1 t")
    ref = ref.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -float('inf'))
    ref = torch.einsum("bhqk,bkhd->bqhd", F.softmax(ref, dim=-1), repeat(v, "b t h d -> b t (h g) d", g=G).float())
    return ref


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("HQ", test_hq_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("fgate_logit_range", test_fgate_logit_range_list)
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
@pytest.mark.skipif(
    is_intel_alchemist,
    reason="Intel Triton Failure"
)
def test_parallel(
    B: int,
    H: int,
    HQ: int,
    T: int,
    D: int,
    fgate_logit_range: Tuple[float, float],
    dtype: torch.dtype
):
    if not check_shared_mem('hopper') and D > 128:
        # maybe we can enable this test on Triton 3.3.0
        pytest.skip("Skipping test because global shared memory is not available")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    logit_min, logit_max = fgate_logit_range
    g = torch.rand((B, T, HQ), dtype=dtype, device=device) * (logit_max - logit_min) + logit_min
    g = F.logsigmoid(g).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    scale = D ** -0.5

    ref = naive_forgetting_attn(q, k, v, g, scale)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    tri = parallel_forgetting_attn(q=q, k=k, v=v, g=g, scale=scale)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)
    assert_close("dg", ref_dg, tri_dg, 0.005)


@pytest.mark.parametrize("N", test_b_list)
@pytest.mark.parametrize("T", test_t_varlen_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("HQ", test_hq_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("fgate_logit_range", test_fgate_logit_range_list)
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set"
)
@pytest.mark.skipif(
    is_intel_alchemist,
    reason="Intel Triton Failure"
)
def test_parallel_varlen(
    N: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    fgate_logit_range: Tuple[float, float],
    dtype: torch.dtype,
):
    if not check_shared_mem('hopper') and D > 128:
        # maybe we can enable this test on Triton 3.3.0
        pytest.skip("Skipping test because global shared memory is not available")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    N = min(1, N) if T < 64 else N
    # randomly split the sequence into N segments
    offsets = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 16)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0].to(torch.int32)
    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    logit_min, logit_max = fgate_logit_range
    g = torch.rand((1, T, HQ), dtype=dtype, device=device) * (logit_max - logit_min) + logit_min
    g = F.logsigmoid(g).requires_grad_(True)
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)

    ref = q.new_empty(1, T, HQ, D)
    for bos, eos in zip(offsets[:-1], offsets[1:]):
        ref[:, bos:eos] = naive_forgetting_attn(
            q=q[:, bos:eos],
            k=k[:, bos:eos],
            v=v[:, bos:eos],
            g=g[:, bos:eos]
        )
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    tri = parallel_forgetting_attn(
        q=q,
        k=k,
        v=v,
        g=g,
        cu_seqlens=offsets
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert_close("  o", ref, tri, 0.004)
    assert_close(" dq", ref_dq.squeeze(), tri_dq.squeeze(), 0.005)
    assert_close(" dk", ref_dk.squeeze(), tri_dk.squeeze(), 0.005)
    assert_close(" dv", ref_dv.squeeze(), tri_dv.squeeze(), 0.005)
    assert_close(" dg", ref_dg.squeeze(), tri_dg.squeeze(), 0.005)
