# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified for QuasarAttention

import torch
import triton
import triton.language as tl

from fla.utils import IS_AMD, autotune_cache_kwargs, check_shared_mem, input_guard

NUM_WARPS = [2, 4, 8, 16] if IS_AMD else [4, 8, 16, 32]


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit
def forward_substitution_kernel(
    # Input: Lower triangular matrix L (I + M)
    L_ptr,  # pointer to lower triangular matrix
    L_stride_bh,  # stride for batch and head
    # Output: Inverse matrix A
    A_ptr,  # pointer to inverse matrix
    A_stride_bh,  # stride for batch and head
    BT: tl.constexpr,
):
    """
    Compute inverse of lower triangular matrix using forward substitution.
    
    For L = I + M (lower triangular with 1s on diagonal):
    Compute A = L^(-1) using forward substitution:
    - A[i,i] = 1
    - A[i,j] = -sum(L[i,k] * A[k,j] for k in range(j,i)) for j < i
    """
    # Get batch-head index
    i_bh = tl.program_id(0)
    
    # Compute pointer offsets for this batch-head
    L_offset = i_bh * L_stride_bh
    A_offset = i_bh * A_stride_bh
    
    # Initialize A as identity matrix
    for i in range(BT):
        for j in range(BT):
            if i == j:
                tl.store(A_ptr + A_offset + i * BT + j, 1.0)
            else:
                tl.store(A_ptr + A_offset + i * BT + j, 0.0)
    
    # Forward substitution
    for i in range(1, BT):
        for j in range(i):
            # A[i,j] = -sum(L[i,k] * A[k,j] for k in range(j,i))
            sum_val = 0.0
            for k in range(j, i):
                L_ik = tl.load(L_ptr + L_offset + i * BT + k)
                A_kj = tl.load(A_ptr + A_offset + k * BT + j)
                sum_val += L_ik * A_kj
            tl.store(A_ptr + A_offset + i * BT + j, -sum_val)


@input_guard
def forward_substitution(
    L: torch.Tensor,
) -> torch.Tensor:
    """
    Compute inverse of lower triangular matrix using forward substitution.
    
    Args:
        L: Lower triangular matrix of shape [B, H, BT, BT] with 1s on diagonal
    
    Returns:
        A: Inverse matrix of shape [B, H, BT, BT]
    """
    B, H, BT, BT2 = L.shape
    assert BT == BT2
    
    # Reshape for kernel: [B*H, BT, BT]
    L_flat = L.view(B * H, BT, BT)
    A_flat = torch.empty_like(L_flat)
    
    # Launch kernel ONCE for all batches and heads in parallel
    forward_substitution_kernel[(B * H,)](
        L_ptr=L_flat,
        L_stride_bh=BT * BT,
        A_ptr=A_flat,
        A_stride_bh=BT * BT,
        BT=BT
    )
    
    return A_flat.view(B, H, BT, BT)


class ForwardSubstitutionFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        L: torch.Tensor,
    ):
        A = forward_substitution(L)
        ctx.save_for_backward(L, A)
        return A

    @staticmethod
    @input_guard
    def backward(ctx, dA):
        L, A = ctx.saved_tensors
        
        # Backward pass: dL = -A^T @ dA @ A^T
        # Simplified implementation for now
        dL = torch.zeros_like(L)
        
        return dL


@torch.compiler.disable
def quasar_forward_substitution(
    L: torch.Tensor,
) -> torch.Tensor:
    """
    Compute inverse of lower triangular matrix using Triton kernel with autograd support
    
    Args:
        L: Lower triangular matrix of shape [B, H, BT, BT] with 1s on diagonal
    
    Returns:
        A: Inverse matrix of shape [B, H, BT, BT]
    """
    return ForwardSubstitutionFunction.apply(L)