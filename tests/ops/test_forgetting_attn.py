# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.ops.forgetting_attn.parallel import parallel_forgetting_attn
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, check_shared_mem, device


def naive_forgetting_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
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


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'scale'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-scale{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 1.0),
            (3, 111, 2, 2, 100, 1.0),
            (3, 1024, 2, 8, 60, 0.1),
            (3, 1024, 2, 8, 128, 0.1),
            (4, 2048, 2, 8, 64, 0.1),
        ]
    ],
)
def test_parallel(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    scale: float,
):
    torch.manual_seed(42)
    dtype = torch.float16
    if not check_shared_mem('hopper') and D > 128:
        # maybe we can enable this test on Triton 3.3.0
        pytest.skip("Skipping test because global shared memory is not available")

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)

    g = torch.randn((B, T, HQ), dtype=dtype, device=device).uniform_(-0.1, -0.01).requires_grad_(True)

    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
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


@pytest.mark.parametrize(
    ('H', 'HQ', 'D', 'cu_seqlens'),
    [
        pytest.param(*test, id="H{}-HQ{}-D{}-cu_seqlens{}".format(*test))
        for test in [
            (2, 2, 64, [0, 15]),
            (2, 8, 64, [0, 256, 500, 1000]),
            (2, 2, 100, [0, 15, 100, 300, 1200, 2000]),
        ]
    ],
)
@pytest.mark.skipif(
    IS_INTEL_ALCHEMIST,
    reason="Intel Triton Failure",
)
def test_parallel_varlen(
    H: int,
    HQ: int,
    D: int,
    cu_seqlens: list[int],
):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    dtype = torch.float16
    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    g = torch.rand((1, T, HQ), dtype=dtype, device=device).uniform_(-0.1, -0.01).requires_grad_(True)
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)

    ref = q.new_empty(1, T, HQ, D)
    for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False):
        ref[:, bos:eos] = naive_forgetting_attn(
            q=q[:, bos:eos],
            k=k[:, bos:eos],
            v=v[:, bos:eos],
            g=g[:, bos:eos],
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
        cu_seqlens=cu_seqlens,
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
