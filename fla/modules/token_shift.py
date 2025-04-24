# -*- coding: utf-8 -*-


from typing import Optional

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16, 32]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['BLOCK_SIZE'],
)
@triton.jit
def token_shift_fwd_kernel(
    x_ptr,
    y_ptr,
    H,
    cu_seqlens_ptr,
    stride_b,
    stride_t,
    stride_h,
    IS_VARLEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        seq_idx = i_b

        seq_start = tl.load(cu_seqlens_ptr + seq_idx).to(tl.int32)
        seq_end = tl.load(cu_seqlens_ptr + seq_idx + 1).to(tl.int32)

        # i_t is the global position index
        # Check if this thread should process data
        if i_t < seq_start or i_t >= seq_end:
            return

        # Calculate local position within sequence
        local_pos = i_t - seq_start
        t_idx = i_t  # t_idx is the global position

        # Check if this is the first position in the sequence
        is_first_pos = (local_pos == 0)
    else:
        b_idx = i_b
        t_idx = i_t

        is_first_pos = (t_idx == 0)
    # Process the entire hidden dimension
    h_offsets = tl.arange(0, BLOCK_SIZE)
    h_mask = h_offsets < H

    # Calculate offset to current position
    if IS_VARLEN:
        base_offset = t_idx * stride_t + h_offsets * stride_h
    else:
        base_offset = b_idx * stride_b + t_idx * stride_t + h_offsets * stride_h

    # Load current values
    curr_values = tl.load(x_ptr + base_offset, mask=h_mask)

    if is_first_pos:
        # First position in sequence: delta = -hidden_states
        tl.store(y_ptr + base_offset, -curr_values, mask=h_mask)
    else:
        # Other positions: delta = prev - curr
        if IS_VARLEN:
            prev_offset = (t_idx-1) * stride_t + h_offsets * stride_h
        else:
            prev_offset = b_idx * stride_b + (t_idx-1) * stride_t + h_offsets * stride_h

        prev_values = tl.load(x_ptr + prev_offset, mask=h_mask)
        delta = prev_values - curr_values
        tl.store(y_ptr + base_offset, delta, mask=h_mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16, 32]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['BLOCK_SIZE'],
)
@triton.jit
def token_shift_bwd_kernel(
    grad_input_ptr,
    grad_output_ptr,
    H,
    T,
    cu_seqlens_ptr,
    stride_b,
    stride_t,
    stride_h,
    IS_VARLEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        seq_idx = i_b

        seq_start = tl.load(cu_seqlens_ptr + seq_idx).to(tl.int32)
        seq_end = tl.load(cu_seqlens_ptr + seq_idx + 1).to(tl.int32)
        seq_len = seq_end - seq_start

        if i_t < seq_start or i_t >= seq_end:
            return

        local_pos = i_t - seq_start
        t_idx = i_t  # t_idx is the global position

        # Check if this is the first position in the sequence
        is_last_pos = (local_pos == seq_len-1)
    else:
        b_idx = i_b
        t_idx = i_t

        is_last_pos = (t_idx == T - 1)

    # Process the entire hidden dimension
    h_offsets = tl.arange(0, BLOCK_SIZE)
    h_mask = h_offsets < H

    # Calculate offset to current position
    if IS_VARLEN:
        base_offset = t_idx * stride_t + h_offsets * stride_h
    else:
        base_offset = b_idx * stride_b + t_idx * stride_t + h_offsets * stride_h

    # Load current gradient
    curr_grad = tl.load(grad_output_ptr + base_offset, mask=h_mask)

    if is_last_pos:
        # Last position: grad = -grad_delta[t]
        grad = -curr_grad
    else:
        # Other positions: grad = -grad_delta[t] + grad_delta[t+1]
        if IS_VARLEN:
            next_offset = (t_idx+1) * stride_t + h_offsets * stride_h
        else:
            next_offset = b_idx * stride_b + (t_idx+1) * stride_t + h_offsets * stride_h

        next_grad = tl.load(grad_output_ptr + next_offset, mask=h_mask)
        grad = -curr_grad + next_grad

    # Store the result
    tl.store(grad_input_ptr + base_offset, grad, mask=h_mask)


def token_shift_forward_triton(
    x: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Implementation of token shift using Triton kernels

    Args:
        x: Input tensor of shape [batch_size, seq_len, hidden_size]
        cu_seqlens: Cumulative sequence lengths (optional)

    Returns:
        Tensor of same shape as input with token shift applied
    """
    assert x.dim() == 3, "Input must be [batch_size, seq_len, hidden_size]"
    B, T, H = x.shape

    if cu_seqlens is not None:
        # Variable length mode
        n_seqs = cu_seqlens.shape[0] - 1
        IS_VARLEN = True
    else:
        # Fixed length mode
        n_seqs = B  # Not used in fixed length mode
        IS_VARLEN = False

    # Calculate block size as power of 2 >= H
    block_size = triton.next_power_of_2(H)

    # Allocate output tensor with same shape and dtype as input
    y = torch.empty_like(x)

    # Launch kernel
    grid = (n_seqs, T)
    token_shift_fwd_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        H=H,
        cu_seqlens_ptr=cu_seqlens,
        stride_b=x.stride(0),
        stride_t=x.stride(1),
        stride_h=x.stride(2),
        IS_VARLEN=IS_VARLEN,
        BLOCK_SIZE=block_size,
    )

    return y


def token_shift_backward_triton(
    grad_output: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Backward pass for token shift using Triton kernels

    Args:
        grad_output: Gradient tensor of shape [batch_size, seq_len, hidden_size]
        cu_seqlens: Cumulative sequence lengths (optional)

    Returns:
        Gradient tensor for input of same shape
    """
    assert grad_output.dim() == 3, "Input must be [batch_size, seq_len, hidden_size]"
    B, T, H = grad_output.shape

    if cu_seqlens is not None:
        # Variable length mode
        n_seqs = cu_seqlens.shape[0] - 1
        IS_VARLEN = True
    else:
        # Fixed length mode
        n_seqs = B  # Not used in fixed length mode
        IS_VARLEN = False

    # Calculate block size as power of 2 >= H
    block_size = triton.next_power_of_2(H)

    # Allocate output tensor with same shape and dtype as input
    grad_input = torch.empty_like(grad_output)

    # Launch kernel
    grid = (n_seqs, T)
    token_shift_bwd_kernel[grid](
        grad_output_ptr=grad_output,
        grad_input_ptr=grad_input,
        H=H,
        T=T,
        cu_seqlens_ptr=cu_seqlens,
        stride_b=grad_output.stride(0),
        stride_t=grad_output.stride(1),
        stride_h=grad_output.stride(2),
        IS_VARLEN=IS_VARLEN,
        BLOCK_SIZE=block_size,
    )

    return grad_input


def token_shift_forward_pytorch(
    x: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Implementation of token shift using PyTorch

    Args:
        x: Input tensor of shape [batch_size, seq_len, hidden_size]
        cu_seqlens: Cumulative sequence lengths (optional)

    Returns:
        Tensor of same shape as input with token shift applied
    """
    if cu_seqlens is not None:
        # Variable length mode with cu_seqlens
        assert x.dim() == 3, "Input must be [batch_size, seq_len, hidden_size]"
        B, T, H = x.shape
        assert B == 1, "Batch size must be 1 when using cu_seqlens"

        result = torch.zeros_like(x)
        n_seqs = cu_seqlens.shape[0] - 1

        for i in range(n_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i+1].item()
            seq_len = end - start

            if seq_len <= 1:
                # For sequences of length 1 or 0, delta is simply -x
                result[0, start:end] = -x[0, start:end]
            else:
                # For longer sequences, handle padding manually
                shifted = torch.zeros_like(x[0, start:end])
                shifted[1:] = x[0, start:end-1]  # Shift all but first token
                delta = shifted - x[0, start:end]
                result[0, start:end] = delta

        return result
    else:
        # Fixed length mode - use ZeroPad2d
        time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))
        shifted = time_shift(x)
        delta = shifted - x
        return delta


class TokenShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cu_seqlens=None):
        # Save for backward
        ctx.save_for_backward(cu_seqlens)
        return token_shift_forward_triton(x, cu_seqlens)

    @staticmethod
    def backward(ctx, grad_output):
        cu_seqlens, = ctx.saved_tensors
        grad_input = token_shift_backward_triton(grad_output, cu_seqlens)
        return grad_input, None  # None for cu_seqlens gradient


def fused_token_shift(x, cu_seqlens=None):
    """
    Custom autograd function for token shift operation
    """
    return TokenShift.apply(x, cu_seqlens)
