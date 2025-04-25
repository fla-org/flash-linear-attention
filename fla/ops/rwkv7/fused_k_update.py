# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def k_update_ref(k: torch.Tensor, a: torch.Tensor, ka: torch.Tensor) -> torch.Tensor:
    """
    k: [batch_size, seq_len, key_dim]
    a: [batch_size, seq_len, key_dim]
    ka: [key_dim]
    k*(1 + (a-1)*self.k_a)
    """
    return k.addcmul(k * (a - 1), ka)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': block_size}, num_warps=num_warps)
        for block_size in [1024, 2048, 4096, 8192]
        for num_warps in [2, 4, 8, 16, 32]
    ],
    key=['hidden_dim'],
)
@triton.jit
def k_update_fwd_kernel(
    k,
    a,
    ka,
    out,
    xnumel,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for the k-update operation.
    """
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]

    xmask = xindex < xnumel
    x0 = xindex % hidden_dim

    b_k = tl.load(k + xindex, xmask, other=0.).to(tl.float32)
    b_a = tl.load(a + xindex, xmask, other=0.).to(tl.float32)
    b_ka = tl.load(ka + x0, eviction_policy='evict_last').to(tl.float32)

    output = b_k * (1 + (b_a - 1) * b_ka)

    tl.store(out + xindex, output.to(out.dtype.element_ty), xmask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': block_size}, num_warps=num_warps)
        for block_size in [1024, 2048, 4096, 8192]
        for num_warps in [2, 4, 8, 16, 32]
    ],
    key=['hidden_dim'],
)
@triton.jit
def k_update_bwd_kernel(
    grad_output,
    k,
    a,
    ka,
    grad_k,
    grad_a,
    grad_ka_full,
    xnumel,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing all gradients in k-update operation.
    """
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]

    xmask = xindex < xnumel
    x0 = xindex % hidden_dim

    b_grad_output = tl.load(grad_output + xindex, xmask, other=0.)
    b_k = tl.load(k + xindex, xmask, other=0.).to(tl.float32)
    b_a = tl.load(a + xindex, xmask, other=0.).to(tl.float32)
    b_ka = tl.load(ka + x0, eviction_policy='evict_last').to(tl.float32)

    b_grad_k = b_grad_output * (1 + (b_a - 1) * b_ka)
    b_grad_a = b_grad_output * b_k * b_ka
    b_grad_ka_full = b_grad_output * b_k * (b_a - 1)

    tl.store(grad_k + xindex, b_grad_k.to(grad_k.dtype.element_ty), xmask)
    tl.store(grad_a + xindex, b_grad_a.to(grad_a.dtype.element_ty), xmask)
    tl.store(grad_ka_full + xindex, b_grad_ka_full.to(grad_ka_full.dtype.element_ty), xmask)


class KUpdateFunction(torch.autograd.Function):
    @staticmethod
    @autocast_custom_fwd
    @input_guard
    def forward(ctx, k, a, ka):
        """
        Forward pass of k_update operation.
        k: [batch_size, seq_len, key_dim]
        a: [batch_size, seq_len, key_dim]
        ka: [key_dim]
        """
        ctx.save_for_backward(k, a, ka)
        out = torch.empty_like(k)

        def grid(meta): return (triton.cdiv(meta['xnumel'], meta['BLOCK_SIZE']),)

        k_update_fwd_kernel[grid](k, a, ka, out, k.numel(), k.shape[2])
        return out

    @staticmethod
    @autocast_custom_bwd
    @input_guard
    def backward(ctx, grad_output):
        k, a, ka = ctx.saved_tensors

        grad_k = torch.empty_like(k)
        grad_a = torch.empty_like(a)
        grad_ka_full = torch.empty_like(k)

        def grid(meta):
            return (triton.cdiv(meta['xnumel'], meta['BLOCK_SIZE']),)

        k_update_bwd_kernel[grid](
            grad_output, k, a, ka,
            grad_k, grad_a, grad_ka_full,
            k.numel(), k.shape[2]
        )

        grad_ka = grad_ka_full.sum(dim=(0, 1))

        return grad_k, grad_a, grad_ka


def fused_k_update(k, a, ka):
    return KUpdateFunction.apply(k, a, ka)
