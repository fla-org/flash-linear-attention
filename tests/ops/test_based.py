
import pytest
import torch

from fla.ops.based import fused_chunk_based, parallel_based
from fla.ops.based.naive import naive_parallel_based
from fla.utils import device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 60, torch.float16),
            (3, 111, 2, 64, torch.float16),
            (3, 1024, 4, 100, torch.float16),
            (3, 1024, 8, 128, torch.float16),
            (4, 2048, 8, 256, torch.float16),
        ]
    ],
)
def test_based(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn((B, H, T, 16), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, H, T, 16), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn_like(v)
    ref = naive_parallel_based(q, k, v, use_norm=True)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_based(q, k, v, use_norm=True)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    if dtype == torch.float32:
        assert ref.allclose(tri, 0, 1e-4)
        assert ref_dq.allclose(tri_dq, 0, 1e-4)
        assert ref_dk.allclose(tri_dk, 0, 1e-4)
        assert ref_dv.allclose(tri_dv, 0, 1e-4)

    tri = fused_chunk_based(q, k, v, use_norm=True)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    if dtype == torch.float32:
        assert ref.allclose(tri, 0, 1e-4)
        assert ref_dq.allclose(tri_dq, 0, 1e-4)
        assert ref_dk.allclose(tri_dk, 0, 1e-4)
        assert ref_dv.allclose(tri_dv, 0, 1e-4)
