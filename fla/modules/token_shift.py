# -*- coding: utf-8 -*-


from typing import Optional

import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


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
    x,
    y,
    T,
    H,
    cu_seqlens,
    IS_VARLEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = i_b
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)

        if i_t < bos or i_t >= eos:
            return

        is_first_pos = (i_t - bos == 0)
    else:
        is_first_pos = (i_t == 0)

    h_offsets = tl.arange(0, BLOCK_SIZE)
    h_mask = h_offsets < H

    if IS_VARLEN:
        base_offset = i_t * H + h_offsets
    else:
        base_offset = i_b * T*H + i_t * H + h_offsets

    curr_values = tl.load(x + base_offset, mask=h_mask)

    if is_first_pos:
        # First position in sequence: delta = -hidden_states
        tl.store(y + base_offset, -curr_values, mask=h_mask)
    else:
        # Other positions: delta = prev - curr
        if IS_VARLEN:
            prev_offset = (i_t-1) * H + h_offsets
        else:
            prev_offset = i_b * T*H + (i_t-1) * H + h_offsets

        prev_values = tl.load(x + prev_offset, mask=h_mask)
        delta = prev_values - curr_values
        tl.store(y + base_offset, delta, mask=h_mask)


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
    grad_input,
    grad_output,
    T,
    H,
    cu_seqlens,
    IS_VARLEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        i_n = i_b
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)

        if i_t < bos or i_t >= eos:
            return

        local_pos = i_t - bos
        is_last_pos = (local_pos == eos - bos - 1)
    else:
        is_last_pos = (i_t == T - 1)

    h_offsets = tl.arange(0, BLOCK_SIZE)
    h_mask = h_offsets < H

    if IS_VARLEN:
        base_offset = i_t * H + h_offsets
    else:
        base_offset = i_b * T*H + i_t * H + h_offsets

    curr_grad = tl.load(grad_output + base_offset, mask=h_mask)

    if is_last_pos:
        # Last position: grad = -grad_delta[t]
        grad = -curr_grad
    else:
        # Other positions: grad = -grad_delta[t] + grad_delta[t+1]
        if IS_VARLEN:
            next_offset = (i_t+1) * H + h_offsets
        else:
            next_offset = i_b * T*H + (i_t+1) * H + h_offsets

        next_grad = tl.load(grad_output + next_offset, mask=h_mask)
        grad = -curr_grad + next_grad

    tl.store(grad_input + base_offset, grad, mask=h_mask)


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
        n_seqs = cu_seqlens.shape[0] - 1
        IS_VARLEN = True
    else:
        n_seqs = B
        IS_VARLEN = False

    block_size = triton.next_power_of_2(H)
    y = torch.empty_like(x)

    grid = (n_seqs, T)
    token_shift_fwd_kernel[grid](
        x=x,
        y=y,
        T=T,
        H=H,
        cu_seqlens=cu_seqlens,
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
        n_seqs = cu_seqlens.shape[0] - 1
        IS_VARLEN = True
    else:
        n_seqs = B
        IS_VARLEN = False

    block_size = triton.next_power_of_2(H)
    grad_input = torch.empty_like(grad_output)

    grid = (n_seqs, T)
    token_shift_bwd_kernel[grid](
        grad_output=grad_output,
        grad_input=grad_input,
        T=T,
        H=H,
        cu_seqlens=cu_seqlens,
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
                shifted[1:] = x[0, start:end-1]
                delta = shifted - x[0, start:end]
                result[0, start:end] = delta

        return result
    else:
        time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))
        shifted = time_shift(x)
        delta = shifted - x
        return delta


class TokenShift(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, cu_seqlens=None):
        ctx.save_for_backward(cu_seqlens)
        return token_shift_forward_triton(x, cu_seqlens)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, grad_output):
        cu_seqlens, = ctx.saved_tensors
        grad_input = token_shift_backward_triton(grad_output, cu_seqlens)
        return grad_input, None


def fused_token_shift(x, cu_seqlens=None):
    """
    Custom autograd function for token shift operation
    """
    return TokenShift.apply(x, cu_seqlens)
