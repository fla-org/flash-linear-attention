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


def kda_gate_ref(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    beta=1.0, threshold=20.0,
) -> torch.Tensor:
    """
    Torch reference implementation for KDA gate computation.

    Computes: g = -A.exp().unsqueeze(-1) * softplus(rearrange(g, '... (h d) -> ... h d', d=head_k_dim))

    Supports both formats:
    - Standard: [batch_size, seq_len, num_heads * head_k_dim]
    - vLLM: [num_tokens, num_heads * head_k_dim]

    Args:
        g: Input tensor of shape [..., num_heads * head_k_dim]
        A: Parameter tensor of shape [num_heads] or [1, 1, num_heads, 1]
        g_bias : Optional bias tensor added to g before activation, shape [num_heads * head_k_dim]
        head_k_dim: Dimension of each head

    Returns:
        Output tensor of shape [..., num_heads, head_k_dim]
    """
    # Rearrange g to separate heads: [..., H*D] -> [..., H, D]
    A = A.view(-1)  # Flatten A to [num_heads] to handle any input shape
    if g_bias is not None:
        g = g + g_bias
    g = rearrange(g, '... (h d) -> ... h d', d=head_k_dim)

    # Apply the gate computation: -A.exp().unsqueeze(-1) * softplus(g)
    # A: [H] -> [H, 1] for broadcasting
    A_exp = -A.float().exp().unsqueeze(-1)  # [H, 1]
    g_softplus = F.softplus(g.float(), beta, threshold)      # [..., H, D]

    return A_exp * g_softplus


@triton.autotune(
    configs=[
        triton.Config({'BT': bt}, num_warps=nw, num_stages=ns)
        for bt in BT_LIST_AUTOTUNE
        for nw in NUM_WARPS_AUTOTUNE
        for ns in [2, 3]
    ],
    key=['H', 'D'],
    **autotune_cache_kwargs,
)
@triton.jit
def kda_gate_fwd_kernel(
    g, A, y,
    g_bias,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    T,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
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

    if HAS_BIAS:
        n_d = tl.arange(0, BD)
        bias_mask = n_d < D
        b_bias = tl.load(g_bias + i_h * D + n_d, mask=bias_mask, other=0.0).to(tl.float32)
        b_g = b_g + b_bias[None, :]

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
    **autotune_cache_kwargs,
)
@triton.jit
def kda_gate_bwd_kernel(
    g,
    A,
    dy,
    dg,
    dA,
    g_bias,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
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

    if HAS_BIAS:
        n_d = tl.arange(0, BD)
        bias_mask = n_d < D
        b_bias = tl.load(g_bias + i_h * D + n_d, mask=bias_mask, other=0.0).to(tl.float32)
        b_g = b_g + b_bias[None, :]

    # softplus(g + bias)
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


def kda_gate_fwd(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    """
    Forward pass for KDA gate:
      input g: [..., H*D]
      param A: [H] or [1, 1, H, 1]
      beta: softplus beta parameter
      threshold: softplus threshold parameter
      return  : [..., H, D]
    """
    orig_shape = g.shape[:-1]

    g = g.view(-1, g.shape[-1])
    T = g.shape[0]
    HD = g.shape[1]
    H = A.numel()
    assert H * head_k_dim == HD

    y = torch.empty_like(g, dtype=torch.float32)

    def grid(meta): return (triton.cdiv(T, meta['BT']), H)

    kda_gate_fwd_kernel[grid](
        g, A, y, g_bias,
        beta, threshold,
        T, H, head_k_dim,
        BD=triton.next_power_of_2(head_k_dim),
        HAS_BIAS=g_bias is not None,
    )

    y = y.view(*orig_shape, H, head_k_dim)
    return y


def kda_gate_bwd(
    grad_output: torch.Tensor,  # [..., H, D]
    g: torch.Tensor,            # [..., H*D]
    A: torch.Tensor,            # [H]
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:

    g_flat = g.view(-1, g.shape[-1])
    T = g_flat.shape[0]
    A_ori_shape = A.shape

    H = A.numel()
    D = head_k_dim

    dy = grad_output.view(T, H * D)
    dg = torch.empty_like(g_flat, dtype=torch.float32)

    BT = 32
    NT = triton.cdiv(T, BT)
    dA = torch.empty((NT, H), dtype=torch.float32, device=g.device)

    grid = (triton.cdiv(T, BT), H)
    kda_gate_bwd_kernel[grid](
        g_flat, A, dy, dg, dA, g_bias,
        beta, threshold,
        T, H, D,
        BT=BT,
        BD=triton.next_power_of_2(D),
        HAS_BIAS=g_bias is not None,
    )

    dA = dA.sum(0).view(A_ori_shape).type_as(A)
    dgbias = dg.sum(0).type_as(g_bias) if g_bias is not None else None
    dg = dg.view(g.shape).type_as(g)
    return dg, dA, dgbias


class KDAGateFunction(torch.autograd.Function):
    """
    Autograd function for KDA gate computation.

    Supports both formats:
    - Standard: [batch_size, seq_len, num_heads * head_k_dim]
    - vLLM: [num_tokens, num_heads * head_k_dim]
    """

    @input_guard
    @staticmethod
    def forward(ctx, g: torch.Tensor, A: torch.Tensor, head_k_dim: int,
                g_bias: torch.Tensor | None = None,
                beta: float = 1.0,
                threshold: float = 20.0) -> torch.Tensor:
        ctx.save_for_backward(g, A)
        ctx.g_bias = g_bias
        ctx.head_k_dim = head_k_dim
        ctx.beta = beta
        ctx.threshold = threshold

        return kda_gate_fwd(g, A, head_k_dim, g_bias, beta, threshold)

    @input_guard
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        g, A = ctx.saved_tensors
        head_k_dim = ctx.head_k_dim
        beta = ctx.beta
        threshold = ctx.threshold
        g_bias = ctx.g_bias

        grad_g, grad_A, grad_gbias = kda_gate_bwd(grad_output, g, A, head_k_dim, g_bias, beta, threshold)
        return grad_g, grad_A, None, grad_gbias, None, None


def fused_kda_gate(g: torch.Tensor, A: torch.Tensor, head_k_dim: int,
                   g_bias: torch.Tensor | None = None,
                   beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    """
    Fused KDA gate computation with autograd support.

    Supports both formats:
    - Standard: [batch_size, seq_len, num_heads * head_k_dim]
    - vLLM: [num_tokens, num_heads * head_k_dim]

    Args:
        g: Input tensor of shape [..., num_heads * head_k_dim]
        A: Parameter tensor of shape [num_heads] or [1, 1, num_heads, 1]
        head_k_dim: Dimension of each head
        beta: softplus beta parameter
        threshold: softplus threshold parameter

    Returns:
        Output tensor of shape [..., num_heads, head_k_dim]
    """
    return KDAGateFunction.apply(g, A, head_k_dim, g_bias, beta, threshold)
