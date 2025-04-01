import logging

import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, check_pytorch_version, input_guard, use_cuda_graph

logger = logging.getLogger(__name__)

if not check_pytorch_version('2.4'):
    logger.warning('PyTorch < 2.4 detected - computations may be slower due to lack of optimizations')


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': block_size})
        for block_size in [128, 256, 512, 1024, 2048, 4096, 8192]
    ],
    key=['hidden_dim'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit
def rwkv_seq_mix_kernel(
    x_ptr,
    x_prev_ptr,
    mix_k_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    token_length,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    block_idx = block_start + tl.arange(0, BLOCK_SIZE)[:]

    total_seq_dim = token_length * hidden_dim
    batch_idx = block_idx // total_seq_dim
    seq_and_feat = block_idx % total_seq_dim
    seq_idx = seq_and_feat // hidden_dim
    feat_idx = seq_and_feat % hidden_dim

    is_valid = (batch_idx < batch_size) & (seq_idx < token_length)

    x_idx = batch_idx * total_seq_dim + seq_idx * hidden_dim + feat_idx

    curr_x = tl.load(x_ptr + x_idx, mask=is_valid, other=0.0).to(tl.float32)
    k_value = tl.load(mix_k_ptr + feat_idx).to(tl.float32)

    is_first = seq_idx < 1
    prev_state_idx = batch_idx * hidden_dim + feat_idx
    prev_state = tl.load(x_prev_ptr + prev_state_idx,
                         mask=(is_first & is_valid),
                         other=0.0).to(tl.float32)

    prev_x_idx = x_idx - hidden_dim
    prev_x = tl.load(x_ptr + prev_x_idx,
                     mask=(~is_first & is_valid),
                     other=0.0).to(tl.float32)

    prev_value = tl.where(is_first, prev_state, prev_x)
    state_diff = prev_value - curr_x
    mixed = state_diff * k_value
    result = tl.cast(curr_x + mixed, dtype=output_ptr.dtype.element_ty, fp_downcast_rounding='rtne')
    tl.store(output_ptr + x_idx, result, mask=is_valid)


@triton.jit
def rwkv_channel_mixing_pow_and_relu(
    in_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr
):
    """Fused ReLU and Power operation: x = ReLU(x)^2"""
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)
    x0 = xindex
    x = tl.load(in_ptr + (x0), None)
    x = tl.maximum(x, 0.0).to(tl.float32)
    x = tl.cast(x * x, dtype=out_ptr.dtype.element_ty, fp_downcast_rounding='rtne')
    tl.store(out_ptr + (x0), x, None)


def rwkv_mix_torch(x: torch.Tensor, x_prev: torch.Tensor, x_k: torch.Tensor):
    if x_prev.dim() == 2:
        x_prev = x_prev.unsqueeze(1)  # (batch_size, 1, hidden_dim)
    xx = torch.cat((x_prev, x[:, :-1, :]), dim=1) - x
    k = x.addcmul(xx, x_k)
    return k


def rwkv_relu_and_square_torch(x: torch.Tensor):
    return torch.relu(x) ** 2


def rwkv_mix_fwd(x, x_prev, x_k):
    has_batch = x.dim() == 3

    if has_batch:
        batch_size, token_length, hidden_dim = x.shape
    else:
        token_length, hidden_dim = x.shape
        batch_size = 1
        x = x.unsqueeze(0)
        x_prev = x_prev.unsqueeze(0)

    token_length = x.shape[1]
    hidden_dim = x.shape[2]
    total_elements = batch_size * token_length * hidden_dim

    output = torch.empty_like(x)

    def grid(meta): return (
        (total_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],  # grid_0
        1,  # grid_1
        1   # grid_2
    )

    rwkv_seq_mix_kernel[grid](
        x.contiguous(),
        x_prev.contiguous(),
        x_k.squeeze(),
        output,
        batch_size=batch_size,
        token_length=token_length,
        hidden_dim=hidden_dim,
    )
    if not has_batch:
        output = output.squeeze(0)
    return output


def rwkv_relu_and_square_fwd(x: torch.Tensor, inplace: bool = True):
    """
    Triton implementation of RWKV's ReLU and square operation
    Args:
        x: Input tensor
    Returns:
        Tensor after ReLU and square operations
    """
    x = x.contiguous()
    output = x if inplace else torch.empty_like(x)

    def grid(meta): return (
        (output.numel() + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],  # grid_0
        1,  # grid_1
        1   # grid_2
    )
    rwkv_channel_mixing_pow_and_relu[grid](
        x,
        output,
        BLOCK_SIZE=4096,
    )

    return output


@triton.jit
def relu_square_bwd_kernel(
    out_ptr,
    forward_input_ptr,
    BLOCK_SIZE: tl.constexpr
):
    """ReLU(x)^2 backward kernel
    grad_input = grad_output * 2 * x if x > 0 else 0
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(forward_input_ptr + offsets).to(tl.float32)
    grad = tl.load(out_ptr + offsets).to(tl.float32)

    x = tl.maximum(x, 0.0)

    grad_input = grad * 2 * x

    tl.store(out_ptr + offsets, grad_input.to(out_ptr.dtype.element_ty))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': block_size})
        for block_size in [128, 256, 512, 1024, 2048, 4096, 8192]
    ],
    key=['hidden_dim'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit
def rwkv_mix_bwd_kenel(
    dk1_ptr0,
    xk_ptr,
    dx_ptr,
    dx_prev_ptr,
    batch_size,
    token_length,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    batch_idx = offsets // (token_length * hidden_dim)
    seq_feat = offsets % (token_length * hidden_dim)
    seq_idx = seq_feat // hidden_dim
    feat_idx = seq_feat % hidden_dim

    is_valid = offsets < (batch_size * token_length * hidden_dim)

    dk1 = tl.load(dk1_ptr0 + offsets, mask=is_valid)
    xk = tl.load(xk_ptr + feat_idx, mask=is_valid)
    prod = dk1 * xk

    mask_next = seq_idx < (token_length - 1)
    next_offset = offsets + hidden_dim
    dk1_next = tl.load(dk1_ptr0 + next_offset, mask=mask_next & is_valid, other=0.0)
    prod_next = dk1_next * xk
    dx_val = dk1 - prod + tl.where(mask_next, prod_next, 0.0)
    dx_val = tl.cast(dx_val, dtype=dx_ptr.dtype.element_ty, fp_downcast_rounding='rtne')
    tl.store(dx_ptr + offsets, dx_val, mask=is_valid)

    dx_prev_offset = batch_idx * hidden_dim + feat_idx
    is_first_step = seq_idx == 0

    tl.store(
        dx_prev_ptr + dx_prev_offset,
        tl.cast(prod, dtype=dx_prev_ptr.dtype.element_ty),
        mask=is_first_step
    )


@torch.compile(fullgraph=True)
def compute_x_k_grad(dk1, x, x_prev):
    """
    Args:
        dk1: (batch*seq_len, hidden_dim)
        x: (batch, seq_len, hidden_dim)
        x_prev: (batch, hidden_dim) or (batch, 1, hidden_dim)
    """

    if x_prev.dim() == 2:
        x_prev = x_prev.unsqueeze(1)  # (batch, 1, hidden_dim)
    xx = torch.cat((x_prev, x[:, :-1, :]), dim=1) - x  # (batch, seq_len, hidden_dim)

    # (hidden_dim,) --> (1, 1, hidden_dim)
    grad_x_k = (dk1 * xx.reshape(-1, x.shape[2])).sum(dim=0).view(1, 1, -1)
    return grad_x_k


def rwkv_channel_mixing_bwd(grad_output, x, x_prev, x_k, key_weight, value_weight, k1, k1_K, k, inplace=True):
    batch_size = x.shape[0] if x.dim() == 3 else 1
    seq_len, n_embd = x.shape[-2], x.shape[-1]

    dV = k.transpose(-2, -1) @ grad_output
    dk = grad_output @ value_weight.transpose(-2, -1)

    BLOCK_SIZE = 4096
    grid = ((dk.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    relu_square_bwd_kernel[grid](
        dk,
        k1_K,
        BLOCK_SIZE=BLOCK_SIZE
    )

    dK = k1.transpose(-2, -1) @ dk
    dk1 = dk @ key_weight.transpose(-2, -1)
    dk1 = dk1.view(-1, n_embd).contiguous()

    dk_reduced = compute_x_k_grad(dk1, x, x_prev)
    dx_prev = torch.empty_like(x_prev) if not inplace else x_prev
    dx = torch.empty_like(x) if not inplace else x

    def grid(meta): return ((batch_size * seq_len * n_embd + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], 1, 1)
    rwkv_mix_bwd_kenel[grid](
        dk1,
        x_k.squeeze(),
        dx,
        dx_prev,
        batch_size,
        seq_len,
        n_embd,
    )
    # dx_prev.shape batch_size, seq_len, n_embd
    return dx, dx_prev, dk_reduced, dK, dV


class Rwkv7ChannelMixing(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, x_prev, x_k, key_weight, value_weight, inplace: bool = True):
        k1 = rwkv_mix_fwd(x, x_prev, x_k)
        k1_K = k1 @ key_weight
        k = rwkv_relu_and_square_fwd(k1_K, inplace=True)
        ctx.save_for_backward(x, x_prev, x_k, key_weight, value_weight)
        ctx.inplace = inplace
        return k @ value_weight

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dkv):
        x, x_prev, x_k, key_weight, value_weight = ctx.saved_tensors
        k1 = rwkv_mix_fwd(x, x_prev, x_k)
        k1_K = k1 @ key_weight
        k = rwkv_relu_and_square_fwd(k1_K, inplace=False)
        dx, dx_prev, dk_reduced, dK, dV = rwkv_channel_mixing_bwd(
            dkv, x, x_prev, x_k, key_weight, value_weight, k1, k1_K, k, ctx.inplace)
        return dx, dx_prev, dk_reduced, dK, dV, None


def channel_mixing_rwkv7(x: torch.Tensor, x_prev: torch.Tensor, x_k: torch.Tensor,
                         key_weight: torch.Tensor, value_weight: torch.Tensor, inplace: bool = True):
    assert x.dim() == 3

    return Rwkv7ChannelMixing.apply(x, x_prev, x_k, key_weight, value_weight, inplace), x[-1, :]


def channel_mixing_rwkv7_torch(x, x_prev, x_k, key_weight, value_weight):
    k1 = rwkv_mix_torch(x, x_prev, x_k)
    k1_K = k1 @ key_weight
    k = rwkv_relu_and_square_torch(k1_K)
    return k @ value_weight, x[-1, :]
