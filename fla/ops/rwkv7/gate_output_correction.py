# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def gate_output_correction_ref(
    o: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    r_k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
):
    """
    o: [B, T, H*D]
    r: [B, T, H, D]
    k: [B, T, H, D]
    r_k: [H, D]
    v: [B, T, H, D]
    g: [B, T, H*D]
    """
    # Unsqueeze r_k for broadcasting across batch and time
    correction_term = ((r * k * r_k.unsqueeze(0).unsqueeze(0)).sum(-1, keepdim=True) * v).view(o.shape)
    output = (o + correction_term) * g
    return output


def gate_output_correction_backward_ref(grad_output, o, r, k, r_k, v, g):
    """
    Reference backward pass implementation in pure PyTorch.
    """
    B, T, HD = o.shape
    H, D = r.shape[-2], r.shape[-1]

    # Unsqueeze r_k for broadcasting
    r_k_b = r_k.unsqueeze(0).unsqueeze(0)
    correction_scalar = (r * k * r_k_b).sum(-1, keepdim=True)
    gated_input = o + (correction_scalar * v).view(B, T, HD)

    grad_g = grad_output * gated_input
    grad_gated_input = grad_output * g
    grad_o = grad_gated_input
    grad_correction = grad_gated_input
    grad_correction_reshaped = grad_correction.view(B, T, H, D)
    grad_v = grad_correction_reshaped * correction_scalar
    grad_correction_scalar = (grad_correction_reshaped * v).sum(-1, keepdim=True)
    grad_r_mul_k_mul_rk = grad_correction_scalar.expand_as(r)
    grad_r = grad_r_mul_k_mul_rk * k * r_k_b
    grad_k = grad_r_mul_k_mul_rk * r * r_k_b
    # Sum over batch and time, keep the head dimension
    grad_r_k = (grad_r_mul_k_mul_rk * r * k).sum(dim=(0, 1))
    return grad_o, grad_r, grad_k, grad_r_k, grad_v, grad_g


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
    ],
    key=['num_heads', 'head_dim', 'BLOCK_SIZE_D'],
)
@triton.jit
def gate_output_correction_fwd_kernel(
    o_ptr, r_ptr, k_ptr, r_k_ptr, v_ptr, g_ptr, output_ptr,
    o_b_stride, o_t_stride,
    r_b_stride, r_t_stride, r_h_stride,
    v_b_stride, v_t_stride, v_h_stride,
    r_k_h_stride,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr
):
    pid_b, pid_t, pid_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < head_dim

    # Load r_k for the specific head
    offset_rk_h = pid_h * r_k_h_stride
    vec_r_k = tl.load(r_k_ptr + offset_rk_h + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

    offset_rh = pid_b * r_b_stride + pid_t * r_t_stride + pid_h * r_h_stride
    offset_vh = pid_b * v_b_stride + pid_t * v_t_stride + pid_h * v_h_stride
    vec_r = tl.load(r_ptr + offset_rh + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    vec_k = tl.load(k_ptr + offset_rh + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    vec_v = tl.load(v_ptr + offset_vh + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

    correction = tl.sum(vec_r * vec_k * vec_r_k, axis=0) * vec_v

    offset_o = pid_b * o_b_stride + pid_t * o_t_stride + pid_h * head_dim
    vec_o = tl.load(o_ptr + offset_o + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    vec_g = tl.load(g_ptr + offset_o + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    final_output = (vec_o + correction) * vec_g

    tl.store(output_ptr + offset_o + d_offsets, final_output.to(output_ptr.dtype.element_ty), mask=d_mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 4]
    ],
    key=['num_heads', 'head_dim', 'BLOCK_SIZE_D'],
)
@triton.jit
def gate_output_correction_bwd_kernel(
    grad_output_ptr, o_ptr, r_ptr, k_ptr, r_k_ptr, v_ptr, g_ptr,
    grad_o_ptr, grad_r_ptr, grad_k_ptr, grad_r_k_intermediate_ptr, grad_v_ptr, grad_g_ptr,
    r_b_stride, r_t_stride, r_h_stride, o_b_stride, o_t_stride, r_k_h_stride,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr
):
    pid_b, pid_t, pid_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < head_dim

    rkv_offset = pid_b * r_b_stride + pid_t * r_t_stride + pid_h * r_h_stride
    vec_r = tl.load(r_ptr + rkv_offset + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    vec_k = tl.load(k_ptr + rkv_offset + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    vec_v = tl.load(v_ptr + rkv_offset + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

    og_offset = pid_b * o_b_stride + pid_t * o_t_stride + pid_h * head_dim
    vec_o = tl.load(o_ptr + og_offset + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    vec_g = tl.load(g_ptr + og_offset + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    vec_grad_output = tl.load(grad_output_ptr + og_offset + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

    # Load r_k for the specific head
    offset_rk_h = pid_h * r_k_h_stride
    vec_r_k = tl.load(r_k_ptr + offset_rk_h + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

    prod_r_k_rk = vec_r * vec_k * vec_r_k
    correction_scalar = tl.sum(prod_r_k_rk, axis=0)
    correction_vec = correction_scalar * vec_v
    gated_input = vec_o + correction_vec

    vec_grad_g = vec_grad_output * gated_input
    vec_grad_gated_input = vec_grad_output * vec_g
    vec_grad_o = vec_grad_gated_input
    vec_grad_correction = vec_grad_gated_input
    vec_grad_v = vec_grad_correction * correction_scalar
    grad_correction_scalar = tl.sum(vec_grad_correction * vec_v, axis=0)
    vec_grad_prod_r_k_rk = grad_correction_scalar
    vec_grad_r = vec_grad_prod_r_k_rk * vec_k * vec_r_k
    vec_grad_k = vec_grad_prod_r_k_rk * vec_r * vec_r_k
    local_grad_r_k = vec_grad_prod_r_k_rk * vec_r * vec_k

    tl.store(grad_o_ptr + og_offset + d_offsets, vec_grad_o.to(grad_o_ptr.dtype.element_ty), mask=d_mask)
    tl.store(grad_g_ptr + og_offset + d_offsets, vec_grad_g.to(grad_g_ptr.dtype.element_ty), mask=d_mask)
    tl.store(grad_r_ptr + rkv_offset + d_offsets, vec_grad_r.to(grad_r_ptr.dtype.element_ty), mask=d_mask)
    tl.store(grad_k_ptr + rkv_offset + d_offsets, vec_grad_k.to(grad_k_ptr.dtype.element_ty), mask=d_mask)
    tl.store(grad_v_ptr + rkv_offset + d_offsets, vec_grad_v.to(grad_v_ptr.dtype.element_ty), mask=d_mask)
    tl.store(grad_r_k_intermediate_ptr + rkv_offset + d_offsets,
             local_grad_r_k.to(grad_r_k_intermediate_ptr.dtype.element_ty), mask=d_mask)


def gate_output_correction_backward_triton(grad_output, o, r, k, r_k, v, g):
    batch_size, seq_len, _ = o.shape
    num_heads, head_dim = r.shape[-2], r.shape[-1]

    grad_o = torch.empty_like(o)
    grad_r = torch.empty_like(r)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)
    grad_g = torch.empty_like(g)
    # Keep intermediate in float32 for precision
    grad_r_k = torch.empty_like(r, dtype=torch.float32)

    grid = (batch_size, seq_len, num_heads)
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)

    gate_output_correction_bwd_kernel[grid](
        grad_output, o, r, k, r_k, v, g,
        grad_o, grad_r, grad_k, grad_r_k, grad_v, grad_g,
        r.stride(0), r.stride(1), r.stride(2),
        o.stride(0), o.stride(1),
        r_k.stride(0),
        num_heads=num_heads, head_dim=head_dim, BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    # Sum over batch and time to get the final gradient for r_k
    grad_r_k = grad_r_k.sum(dim=(0, 1)).type_as(r_k)
    return grad_o, grad_r, grad_k, grad_r_k, grad_v, grad_g


class GateOutputCorrection(torch.autograd.Function):
    @staticmethod
    @autocast_custom_fwd
    @input_guard
    def forward(ctx, o, r, k, r_k, v, g):
        assert r_k.dim() == 2 and r_k.shape[0] == r.shape[-2] and r_k.shape[1] == r.shape[-1]

        batch_size, seq_len, _ = o.shape
        num_heads, head_dim = r.shape[-2], r.shape[-1]
        output = torch.empty_like(o)
        ctx.save_for_backward(o, r, k, r_k, v, g)
        grid = (batch_size, seq_len, num_heads)

        gate_output_correction_fwd_kernel[grid](
            o, r, k, r_k, v, g, output,
            o.stride(0), o.stride(1),
            r.stride(0), r.stride(1), r.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            r_k.stride(0),
            num_heads, head_dim, BLOCK_SIZE_D=triton.next_power_of_2(head_dim)
        )
        return output

    @staticmethod
    @autocast_custom_bwd
    @input_guard
    def backward(ctx, grad_output):
        o, r, k, r_k, v, g = ctx.saved_tensors
        return gate_output_correction_backward_triton(grad_output, o, r, k, r_k, v, g)


gate_output_correction = GateOutputCorrection.apply
