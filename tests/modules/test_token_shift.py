import pytest
import torch

from fla.modules.token_shift import fused_token_shift, token_shift_forward_pytorch
from fla.utils import device


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
    x_pytorch = torch.randn(batch_size, seq_len, hidden_size, device=device).to(dtype).requires_grad_(True)
    x_triton = x_pytorch.clone().detach().requires_grad_(True)

    # Convert cu_seqlens to tensor if provided
    cu_seqlens_tensor = None
    if cu_seqlens is not None:
        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    # Forward pass
    out_pytorch = token_shift_forward_pytorch(x_pytorch, cu_seqlens_tensor)
    out_triton = fused_token_shift(x_triton, cu_seqlens_tensor)

    # Check forward results - use higher tolerance for lower precision dtypes
    rtol = 1e-5 if dtype == torch.float32 else 1e-3
    atol = 1e-5 if dtype == torch.float32 else 1e-3

    torch.testing.assert_close(
        out_pytorch, out_triton,
        rtol=rtol, atol=atol,
        msg=f"Forward pass failed for {test_case}"
    )

    # Backward pass - create a dummy gradient
    grad_output = torch.randn_like(out_pytorch)

    # Compute gradients
    out_pytorch.backward(grad_output)
    out_triton.backward(grad_output)

    # Check backward results
    torch.testing.assert_close(
        x_pytorch.grad, x_triton.grad,
        rtol=rtol, atol=atol,
        msg=f"Backward pass failed for {test_case}"
    )

    # Print result for debugging
    print(f"Test case '{test_case}' PASSED")
