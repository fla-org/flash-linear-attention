
import os

import pytest
import torch

from fla.ops.attn.parallel import parallel_attn
from fla.ops.utils import prepare_lens
from fla.utils import assert_close, check_shared_mem, device

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    HAS_FLASH = True
except Exception:
    HAS_FLASH = False


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
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shard mem")
    if not HAS_FLASH:
        pytest.skip(reason="Skipping test because flash-attn is not installed")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    q = torch.randn((B, T, HQ, D), dtype=torch.float16, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=torch.float16, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=torch.float16, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=torch.float16, device=device)

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
def test_parallel_varlen(
    H: int,
    HQ: int,
    D: int,
    cu_seqlens: list[int],
):
    if not HAS_FLASH:
        pytest.skip(reason="Skipping test because flash-attn is not installed")
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    dtype = torch.float16

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
        causal=True,
    )
    ref.backward(do.squeeze(0))
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_attn(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, 0.004)
    assert_close("dq", ref_dq.squeeze(), tri_dq.squeeze(), 0.005)
    assert_close("dk", ref_dk.squeeze(), tri_dk.squeeze(), 0.005)
    assert_close("dv", ref_dv.squeeze(), tri_dv.squeeze(), 0.005)
