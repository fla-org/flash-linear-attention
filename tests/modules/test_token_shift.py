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
    [0, 1, 3, 4]
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
