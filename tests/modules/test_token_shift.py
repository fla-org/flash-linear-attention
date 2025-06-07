import pytest
import torch

from fla.modules.token_shift import token_shift, token_shift_ref
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    "test_case,batch_size,seq_len,hidden_size,cu_seqlens,dtype",
    [
        ("fixed_length_standard", 8, 128, 128, None, torch.float32),
        ("fixed_length_different_dims", 4, 256, 64, None, torch.float32),
        ("var_length_standard", 1, 128, 128, [0, 4, 7, 40, 128], torch.float32),
        ("var_length_fewer_seqs", 1, 64, 64, [0, 10, 20, 64], torch.float32),
        ("var_length_single_seq", 1, 32, 32, [0, 32], torch.float32),
        ("edge_case_len_1", 1, 4, 64, [0, 1, 3, 4], torch.float32),
        ("dtype_float16", 2, 32, 64, None, torch.float16),
        ("dtype_bfloat16", 2, 32, 64, None, torch.bfloat16)
    ]
)
def test_token_shift(test_case, batch_size, seq_len, hidden_size, cu_seqlens, dtype):
    """Comprehensive test for token shift operation"""

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create test tensors
    x = torch.randn(batch_size, seq_len, hidden_size, device=device).to(dtype).requires_grad_(True)
    dy = torch.randn_like(x)

    cu_seqlens_tensor = None
    if cu_seqlens is not None:
        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    # Forward pass
    ref = token_shift_ref(x, cu_seqlens_tensor)
    tri = token_shift(x, cu_seqlens_tensor)

    ref.backward(dy)
    ref_dx, x.grad = x.grad, None

    tri.backward(dy)
    tri_dx, x.grad = x.grad, None

    assert_close(' x', ref, tri, 1e-3)
    assert_close('dx', ref_dx, tri_dx, 1e-3)
