# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified for QuasarAttention

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.utils import IS_AMD, autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, check_shared_mem, input_guard

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]
BT_LIST_AUTOTUNE = [32, 64, 128]
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [4, 8, 16, 32]


def naive_quasar_gate(
    beta: torch.Tensor,
    lambda_t: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Torch reference implementation for QuasarAttention gate computation.

    Computes: alpha = (1 - exp(-beta * lambda)) / (lambda + eps)

    Args:
        beta (torch.Tensor):
            Parameter tensor with `H` elements.
        lambda_t (torch.Tensor):
            Input tensor of shape `[..., H, 1]` (norm squared of keys).
        output_dtype (torch.dtype):
            Output dtype.

    Returns:
        Output tensor of shape `[..., H, 1]`.
    """
    eps = 1e-8
    alpha = (1 - torch.exp(-beta.view(-1, 1) * lambda_t)) / (lambda_t + eps)
    return alpha.to(output_dtype)


@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in BT_LIST_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3]
    ],
    key=["H", "D"],
    **autotune_cache_kwargs,
)
@triton.jit
def quasar_gate_fwd_kernel(
    lambda_t,
    beta,
    alpha,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)

    b_beta = tl.load(beta + i_h).to(tl.float32)

    p_lambda = tl.make_block_ptr(lambda_t + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_alpha = tl.make_block_ptr(alpha + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    # [BT, BD]
    b_lambda = tl.load(p_lambda, boundary_check=(0, 1)).to(tl.float32)
    
    # alpha = (1 - exp(-beta * lambda)) / (lambda + eps)
    eps = 1e-8
    b_alpha = (1 - tl.exp(-b_beta * b_lambda)) / (b_lambda + eps)
    tl.store(p_alpha, b_alpha.to(p_alpha.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def quasar_gate_fwd(
    lambda_t: torch.Tensor,
    beta: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    H, K = lambda_t.shape[-2:]
    T = lambda_t.numel() // (H * K)

    alpha = torch.empty_like(lambda_t, dtype=output_dtype)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]), H)

    quasar_gate_fwd_kernel[grid](
        lambda_t=lambda_t,
        beta=beta,
        alpha=alpha,
        T=T,
        H=H,
        D=K,
        BD=triton.next_power_of_2(K),
    )
    return alpha


class QuasarGateFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        lambda_t: torch.Tensor,
        beta: torch.Tensor,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        alpha = quasar_gate_fwd(
            lambda_t=lambda_t,
            beta=beta,
            output_dtype=output_dtype
        )
        ctx.save_for_backward(lambda_t, beta)
        return alpha

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dalpha: torch.Tensor):
        lambda_t, beta = ctx.saved_tensors
        eps = 1e-8
        
        # dalpha/dlambda and dalpha/dbeta derivatives
        # alpha = (1 - exp(-beta * lambda)) / (lambda + eps)
        # dalpha/dbeta = exp(-beta * lambda)
        beta_exp = torch.exp(-beta.view(-1, 1) * lambda_t)
        lambda_plus_eps = lambda_t + eps
        
        # dalpha/dlambda = (beta * exp(-beta * lambda) * lambda - (1 - exp(-beta * lambda))) / lambda^2
        dlambda = (beta.view(-1, 1) * beta_exp * lambda_plus_eps - (1 - beta_exp)) / (lambda_plus_eps ** 2)
        
        # dalpha/dbeta = exp(-beta * lambda)
        dbeta = beta_exp
        
        dlambda = dlambda * dalpha
        # Sum over sequence and dimensions, but preserve head dimension
        dbeta = (dbeta * dalpha).sum(dim=(0, 1))
        
        return dlambda, dbeta, None, None


@triton.jit
def fast_quasar_alpha_fwd_kernel(
    k,
    beta,
    alpha,
    T,
    stride_beta_b,
    stride_beta_t,
    stride_beta_h,
    H: tl.constexpr,
    S: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    i_bh, i_t = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    
    eps = 1e-6
    
    # Process BT tokens
    for t in range(BT):
        idx = i_t * BT + t
        if idx < T:
            # We use block ptr if we want, but simple indexing is fine here for S
            offset = (i_b * T * H + idx * H + i_h) * S
            b_k2 = 0.0
            for s in range(0, S, BK):
                mask = (s + tl.arange(0, BK)) < S
                b_k = tl.load(k + offset + s + tl.arange(0, BK), mask=mask, other=0.0).to(tl.float32)
                b_k2 += tl.sum(b_k * b_k)
            
            # Load beta for this specific token
            beta_offset = i_b * stride_beta_b + idx * stride_beta_t + i_h * stride_beta_h
            b_beta = tl.load(beta + beta_offset).to(tl.float32)
            
            # alpha = (1 - exp(-beta * |k|^2)) / (|k|^2 + eps)
            # Clamp k2 internally for stability like the torch version did
            k2_clamped = tl.where(b_k2 < 0.1, 0.1, tl.where(b_k2 > 10.0, 10.0, b_k2))
            b_alpha = (1.0 - tl.exp(-b_beta * k2_clamped)) / (k2_clamped + eps)
            
            tl.store(alpha + i_b * T * H + idx * H + i_h, b_alpha.to(alpha.dtype.element_ty))


@input_guard
def fast_quasar_alpha(
    k: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    B, T, H, S = k.shape
    alpha = torch.empty(B, T, H, device=k.device, dtype=k.dtype)
    
    if beta.ndim == 1:
        stride_beta_b, stride_beta_t, stride_beta_h = 0, 0, beta.stride(0)
    elif beta.ndim == 3:
        stride_beta_b, stride_beta_t, stride_beta_h = beta.stride(0), beta.stride(1), beta.stride(2)
    else:
        raise ValueError(f"beta must be 1D or 3D, got {beta.ndim}D")
        
    BT = 64
    grid = (B * H, triton.cdiv(T, BT))
    fast_quasar_alpha_fwd_kernel[grid](
        k=k,
        beta=beta,
        alpha=alpha,
        T=T,
        stride_beta_b=stride_beta_b,
        stride_beta_t=stride_beta_t,
        stride_beta_h=stride_beta_h,
        H=H,
        S=S,
        BK=triton.next_power_of_2(S),
        BT=BT,
    )
    return alpha


@torch.compiler.disable
def fused_quasar_gate(
    lambda_t: torch.Tensor,
    beta: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Fused QuasarAttention gate computation with autograd support.

    Computes: alpha = (1 - exp(-beta * lambda)) / (lambda + eps)

    Args:
        lambda_t (torch.Tensor):
            Input tensor of shape `[..., H, 1]` (norm squared of keys).
        beta (torch.Tensor):
            Parameter tensor with `H` elements.

    Returns:
        Output tensor of shape `[..., H, 1]`.
    """
    return QuasarGateFunction.apply(lambda_t, beta, output_dtype)