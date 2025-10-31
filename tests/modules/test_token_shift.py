
import pytest
import torch

from fla.modules.token_shift import token_shift, token_shift_ref
from fla.utils import assert_close, device

test_b_list = [4]
test_t_list = [512, 4100, 8192]
test_h_list = [2560, 4096]
test_cu_seqlens_list = [
    None,
    [0, 4, 7, 40, 128],
    [0, 10, 20, 64],
    [0, 32],
    [0, 1, 3, 4],
]
test_dtype_list = [torch.float]


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('cu_seqlens_val', test_cu_seqlens_list)
@pytest.mark.parametrize('dtype', test_dtype_list)
def test_token_shift(B, T, H, cu_seqlens_val, dtype):
    if cu_seqlens_val is not None:
        B = 1
        T = cu_seqlens_val[-1]
        cu_seqlens_tensor = torch.tensor(cu_seqlens_val, dtype=torch.int32, device=device)
    else:
        cu_seqlens_tensor = None

    torch.manual_seed(42)

    x = torch.randn(B, T, H, device=device).to(dtype).requires_grad_(True)
    dy = torch.randn_like(x)

    ref = token_shift_ref(x, cu_seqlens_tensor)
    tri = token_shift(x, cu_seqlens_tensor)

    ref.backward(dy)
    ref_dx, x.grad = x.grad, None

    tri.backward(dy)
    tri_dx, x.grad = x.grad, None

    assert_close(' x', ref, tri, 1e-3)
    assert_close('dx', ref_dx, tri_dx, 1e-3)


def _split_for_passing(
    x: torch.Tensor,
    cu_seqlens,
    split_at: int = 1,
):
    assert x.size(0) == 1
    assert 0 < split_at < len(cu_seqlens) - 1

    cu0 = [t - cu_seqlens[0] for t in cu_seqlens[: split_at + 1]]
    cu1 = [t - cu_seqlens[split_at] for t in cu_seqlens[split_at:]]
    T0, T1 = cu0[-1], cu1[-1]

    x0 = x[:, :T0].contiguous()
    x1 = x[:, T0: T0 + T1].contiguous()
    cache1 = x[:, T0 - 1: T0].contiguous()
    return x0, x1, \
        torch.tensor(cu0, dtype=torch.int32, device=x.device), \
        torch.tensor(cu1, dtype=torch.int32, device=x.device), \
        cache1


def _check_passing_vs_whole(
    B: int,
    T: int,
    H: int,
    cu_seqlens: list[int] | None,
    dtype: torch.dtype,
    split_at: int = 1,
):
    torch.manual_seed(42)

    if cu_seqlens is None:
        x = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True)
        cu_seqlens_tensor = None
    else:
        B = 1
        T = cu_seqlens[-1]
        x = torch.randn(1, T, H, device=device, dtype=dtype, requires_grad=True)
        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    dy = torch.randn_like(x)
    ref_out = token_shift(x, cu_seqlens_tensor)
    ref_out.backward(dy)
    ref_dx = x.grad.clone()
    x.grad.zero_()

    if cu_seqlens is None:
        T0 = T // 2
        x0 = x[:, :T0].contiguous()
        x1 = x[:, T0:].contiguous()
        cu0, cu1 = None, None
    else:
        if split_at >= len(cu_seqlens) - 1:
            pytest.skip("invalid split_at")
        x0, x1, cu0, cu1, cache1 = _split_for_passing(x, cu_seqlens, split_at)

    out0, cache_out0 = token_shift(x0, cu0, output_cache=True)
    out1, cache_out1 = token_shift(x1, cu1, cache=cache_out0, output_cache=True)

    cat_out = torch.cat([out0, out1], dim=1)
    cat_out.backward(dy)

    cat_dx = x.grad.clone()

    assert_close("do", ref_out, cat_out, 1e-3)
    assert_close("dx", ref_dx, cat_dx, 1e-3)


@pytest.mark.parametrize(
    ("B", "T", "H", "cu_seqlens", "split_at"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-cu{}-split{}".format(*test))
        for test in [
            (2, 512, 1024, None, 1),
            (1, 8192, 1024, None, 2),
        ]
    ],
)
def test_all_with_and_without_varlen(B, T, H, cu_seqlens, split_at):
    dtype = torch.float
    assert cu_seqlens is None, "This test is for cu_seqlens=None case"
    _check_passing_vs_whole(B, T, H, cu_seqlens, dtype, split_at)
