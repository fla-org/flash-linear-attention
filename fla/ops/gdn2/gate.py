# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.utils.op import log
from fla.utils import autotune_cache_kwargs, input_guard, is_amd

BT_LIST_AUTOTUNE = [32, 64, 128]
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [4, 8, 16, 32]


def gdn2_gate_ref(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    beta=1.0, threshold=20.0
) -> torch.Tensor:
    """
    Torch reference implementation for GDN2 gate computation.

    Computes: g = -A.exp().unsqueeze(-1) * softplus(rearrange(g, '... (h d) -> ... h d', d=head_k_dim))

    Supports both formats:
    - Standard: [batch_size, seq_len, num_heads * head_k_dim]
    - vLLM: [num_tokens, num_heads * head_k_dim]

    Args:
        g: Input tensor of shape [..., num_heads * head_k_dim]
        A: Parameter tensor of shape [num_heads]
        head_k_dim: Dimension of each head

    Returns:
        Output tensor of shape [..., num_heads, head_k_dim]
    """
    # Rearrange g to separate heads: [..., H*D] -> [..., H, D]
    g = rearrange(g, '... (h d) -> ... h d', d=head_k_dim)

    # Apply the gate computation: -A.exp().unsqueeze(-1) * softplus(g)
    # A: [H] -> [H, 1] for broadcasting
    A_exp = -A.float().exp().unsqueeze(-1)  # [H, 1]
    g_softplus = F.softplus(g.float(), beta, threshold)      # [..., H, D]

    return A_exp * g_softplus


def gdn2_gate_bwd_ref(
    grad_output: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    beta: float = 1.0,
    threshold: float = 20.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for GDN2 gate computation.

    Supports both formats:
    - Standard: grad_output [batch_size, seq_len, num_heads, head_k_dim]
    - vLLM: grad_output [num_tokens, num_heads, head_k_dim]

    Args:
        grad_output: Gradient tensor of shape [..., num_heads, head_k_dim]
        g: Input tensor from forward pass, shape [..., num_heads * head_k_dim]
        A: Parameter tensor from forward pass, shape [num_heads]
        head_k_dim: Dimension of each head
        beta: softplus beta parameter
        threshold: softplus threshold parameter

    Returns:
        tuple: (grad_g, grad_A)
            - grad_g: Gradient w.r.t. input g, shape [..., num_heads * head_k_dim]
            - grad_A: Gradient w.r.t. parameter A, shape [num_heads]
    """
    # Save original shape
    orig_shape = g.shape[:-1]

    # Reshape tensors for computation
    g = g.view(-1, g.shape[-1])  # [T, H*D]
    grad_output = grad_output.view(-1, grad_output.shape[-2], grad_output.shape[-1])  # [T, H, D]

    H = A.shape[0]
    D = head_k_dim

    # Rearrange g to separate heads: [T, H*D] -> [T, H, D]
    g_reshaped = rearrange(g, 't (h d) -> t h d', h=H, d=D)

    # Compute softplus and its derivative
    g_float = g_reshaped.float()
    g_scaled = g_float * beta

    # Use thresholding for numerical stability (same as forward)
    use_linear = g_scaled > threshold
    softplus_result = torch.where(use_linear, g_float, (1.0 / beta) * torch.log(1.0 + torch.exp(g_scaled)))

    # Compute sigmoid for softplus derivative
    sigmoid_result = torch.sigmoid(beta * g_float)

    # Compute A_exp for reuse
    A_exp = -A.float().exp().unsqueeze(-1)  # [H, 1]

    # Compute grad_g: grad_output * (-A_exp * sigmoid_result)
    grad_g_reshaped = grad_output * (A_exp.unsqueeze(0) * sigmoid_result)

    # Reshape grad_g back to original format: [T, H, D] -> [T, H*D]
    grad_g = rearrange(grad_g_reshaped, 't h d -> t (h d)')
    grad_g = grad_g.view(*orig_shape, H * D)

    # Compute grad_A: sum(grad_output * (-exp(A).unsqueeze(-1) * softplus_result))
    # Sum over all dimensions except H
    grad_A_per_token = grad_output * (torch.exp(A).unsqueeze(-1).unsqueeze(0) * softplus_result)
    grad_A = -grad_A_per_token.sum(dim=(0, 2))  # Sum over T and D dimensions

    return grad_g, grad_A


@triton.autotune(
    configs=[
        triton.Config({'BT': bt}, num_warps=nw, num_stages=ns)
        for bt in BT_LIST_AUTOTUNE
        for nw in NUM_WARPS_AUTOTUNE
        for ns in [2, 3]
    ],
    key=['H', 'D'],
    **autotune_cache_kwargs
)
@triton.jit
def gdn2_gate_fwd_kernel(
    g, A, y,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    T,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    n_t = i_t * BT

    b_a = tl.load(A + i_h).to(tl.float32)
    b_a = -tl.exp(b_a)

    stride_row = H * D
    stride_col = 1

    g_ptr = tl.make_block_ptr(
        base=g + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    y_ptr = tl.make_block_ptr(
        base=y + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    b_g = tl.load(g_ptr, boundary_check=(0, 1)).to(tl.float32)

    # softplus(x, beta) = (1/beta) * log(1 + exp(beta * x))
    # When beta * x > threshold, use linear approximation x
    # Use threshold to switch to linear when beta*x > threshold
    g_scaled = b_g * beta
    use_linear = g_scaled > threshold
    sp = tl.where(use_linear, b_g, (1.0 / beta) * log(1.0 + tl.exp(g_scaled)))
    b_y = b_a * sp

    tl.store(y_ptr, b_y.to(y.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in NUM_WARPS_AUTOTUNE
        for ns in [2, 3]
    ],
    key=['H', 'D'],
    **autotune_cache_kwargs
)
@triton.jit
def gdn2_gate_bwd_kernel(
    g,
    A,
    dy,
    dg,
    dA,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    n_t = i_t * BT

    a_h = tl.load(A + i_h).to(tl.float32)
    neg_exp_a = -tl.exp(a_h)

    stride_row = H * D
    stride_col = 1

    g_ptr = tl.make_block_ptr(
        base=g + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )
    dy_ptr = tl.make_block_ptr(
        base=dy + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )
    dg_ptr = tl.make_block_ptr(
        base=dg + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    b_g = tl.load(g_ptr, boundary_check=(0, 1)).to(tl.float32)  # [BT, BD]
    b_dy = tl.load(dy_ptr, boundary_check=(0, 1)).to(tl.float32)  # [BT, BD]

    # softplus(g)
    g_scaled = b_g * beta
    use_linear = g_scaled > threshold
    sp = tl.where(use_linear, b_g, (1.0 / beta) * log(1.0 + tl.exp(g_scaled)))

    sig = tl.sigmoid(g_scaled)

    # grad_g = dy * (-exp(A)) * sigmoid(beta*g)
    b_dg = b_dy * (neg_exp_a * sig)
    tl.store(dg_ptr, b_dg.to(dg_ptr.dtype.element_ty), boundary_check=(0, 1))

    contrib = b_dy * (neg_exp_a * sp)
    tile_sum = tl.sum(tl.sum(contrib, axis=1), axis=0)

    out_off = i_t * H + i_h
    tl.store(dA + out_off, tile_sum)


def gdn2_gate_fwd(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    beta: float = 1.0,
    threshold: float = 20.0
) -> torch.Tensor:
    """
    Forward pass for GDN2 gate:
      input g: [..., H*D]
      param A: [H]
      beta: softplus beta parameter
      threshold: softplus threshold parameter
      return  : [..., H, D]
    """
    orig_shape = g.shape[:-1]

    g = g.view(-1, g.shape[-1])
    T = g.shape[0]
    HD = g.shape[1]
    H = A.shape[0]
    assert HD == H * head_k_dim

    y = torch.empty_like(g)

    def grid(meta): return (triton.cdiv(T, meta['BT']), H)

    gdn2_gate_fwd_kernel[grid](
        g, A, y,
        beta, threshold,
        T, H, head_k_dim,
        BD=triton.next_power_of_2(head_k_dim)
    )

    y = y.view(*orig_shape, H, head_k_dim)
    return y


def gdn2_gate_bwd(
    grad_output: torch.Tensor,  # [..., H, D]
    g: torch.Tensor,            # [..., H*D]
    A: torch.Tensor,            # [H]
    head_k_dim: int,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:

    g_flat = g.view(-1, g.shape[-1])
    T = g_flat.shape[0]
    H = A.shape[0]
    D = head_k_dim

    dy = grad_output.view(T, H * D)
    dg = torch.empty_like(g_flat)

    BT = 32
    NT = triton.cdiv(T, BT)
    dA = torch.empty((NT, H), dtype=torch.float32, device=g.device)

    grid = (triton.cdiv(T, BT), H)
    gdn2_gate_bwd_kernel[grid](
        g_flat, A, dy, dg, dA,
        beta, threshold,
        T, H, D,
        BT=BT,
        BD=triton.next_power_of_2(D),
    )

    dA = dA.sum(0).to(A.dtype)
    dg = dg.view(g.shape)
    return dg, dA


class GDN2GateFunction(torch.autograd.Function):
    """
    Autograd function for GDN2 gate computation.

    Supports both formats:
    - Standard: [batch_size, seq_len, num_heads * head_k_dim]
    - vLLM: [num_tokens, num_heads * head_k_dim]
    """

    @input_guard
    @staticmethod
    def forward(ctx, g: torch.Tensor, A: torch.Tensor, head_k_dim: int, beta: float = 1.0,
                threshold: float = 20.0) -> torch.Tensor:
        ctx.save_for_backward(g, A)
        ctx.head_k_dim = head_k_dim
        ctx.beta = beta
        ctx.threshold = threshold

        return gdn2_gate_fwd(g, A, head_k_dim, beta, threshold)

    @input_guard
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        g, A = ctx.saved_tensors
        head_k_dim = ctx.head_k_dim
        beta = ctx.beta
        threshold = ctx.threshold

        grad_g, grad_A = gdn2_gate_bwd(grad_output, g, A, head_k_dim, beta, threshold)
        return grad_g, grad_A, None, None, None


def fused_gdn2_gate(g: torch.Tensor, A: torch.Tensor, head_k_dim: int,
                    beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    """
    Fused GDN2 gate computation with autograd support.

    Supports both formats:
    - Standard: [batch_size, seq_len, num_heads * head_k_dim]
    - vLLM: [num_tokens, num_heads * head_k_dim]

    Args:
        g: Input tensor of shape [..., num_heads * head_k_dim]
        A: Parameter tensor of shape [num_heads]
        head_k_dim: Dimension of each head
        beta: softplus beta parameter
        threshold: softplus threshold parameter

    Returns:
        Output tensor of shape [..., num_heads, head_k_dim]
    """
    return GDN2GateFunction.apply(g, A, head_k_dim, beta, threshold)
