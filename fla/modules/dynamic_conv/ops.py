# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.modules.activations.ops import swish_bwd
from fla.modules.causal_conv1d import causal_conv1d_update_states
from fla.ops.utils import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.jit
def _dynamic_conv1d_offsets(cu_seqlens, chunk_indices, T, BT: tl.constexpr, ND: tl.constexpr):
    i_t = tl.program_id(0).to(tl.int64) // ND
    i_n = tl.program_id(1).to(tl.int64)
    if cu_seqlens is not None:
        i_n = tl.load(chunk_indices + 2 * i_t).to(tl.int64)
        i_t = tl.load(chunk_indices + 2 * i_t + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        length = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    else:
        bos, length = i_n * T, T
    return i_n, bos, length, i_t * BT


@triton.jit
def _dynamic_conv1d_load_x(x, initial_state, i_n, bos, length, t, d, D: tl.constexpr, W: tl.constexpr):
    values = tl.load(
        pointer=x + (bos + t[:, None]) * D + d[None, :],
        mask=(t[:, None] >= 0) & (t[:, None] < length) & (d[None, :] < D),
        other=0,
    ).to(tl.float32)
    if initial_state is not None:
        values += tl.load(
            pointer=initial_state + (i_n * D + d[None, :]) * W + t[:, None] + W,
            mask=(t[:, None] >= -W) & (t[:, None] < 0) & (d[None, :] < D),
            other=0,
        ).to(tl.float32)
    return values


@triton.jit(do_not_specialize=['T'])
def dynamic_conv1d_fwd_kernel(
    x, z, weight, kernel_bias, bias, residual, initial_state, y, pre_activation, cu_seqlens, chunk_indices, T,
    D: tl.constexpr, R: tl.constexpr, W: tl.constexpr,
    BT: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr,
):
    i_n, bos, length, start = _dynamic_conv1d_offsets(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        BT=BT,
        ND=tl.cdiv(D, BD),
    )
    t = start + tl.arange(0, BT).to(tl.int64)
    d = (tl.program_id(0).to(tl.int64) % tl.cdiv(D, BD)) * BD + tl.arange(0, BD)
    r = tl.arange(0, BR).to(tl.int64)
    latent = tl.load(z + (bos + t[:, None]) * R + r[None, :], (t[:, None] < length) & (r[None, :] < R), other=0).to(tl.float32)
    out = tl.full((BT, BD), 0, tl.float32)
    for j in range(W):
        up = tl.load(
            pointer=weight + (r[:, None] * W + j) * D + d[None, :],
            mask=(r[:, None] < R) & (d[None, :] < D),
            other=0,
        ).to(tl.float32)
        kernel = tl.dot(latent, up, input_precision='ieee')
        if kernel_bias is not None:
            kernel += tl.load(kernel_bias + j * D + d, d < D, other=0)[None, :].to(tl.float32)
        values = _dynamic_conv1d_load_x(
            x=x,
            initial_state=initial_state,
            i_n=i_n,
            bos=bos,
            length=length,
            t=t - j,
            d=d,
            D=D,
            W=W,
        )
        out += kernel * values
    if bias is not None:
        out += tl.load(bias + d, d < D, other=0)[None, :].to(tl.float32)
    if pre_activation is not None:
        tl.store(pre_activation + (bos + t[:, None]) * D + d[None, :], out, (t[:, None] < length) & (d[None, :] < D))
        out *= tl.sigmoid(out)
    if residual is not None:
        out += tl.load(
            pointer=residual + (bos + t[:, None]) * D + d[None, :],
            mask=(t[:, None] < length) & (d[None, :] < D),
            other=0,
        ).to(tl.float32)
    tl.store(y + (bos + t[:, None]) * D + d[None, :], out, (t[:, None] < length) & (d[None, :] < D))


@triton.jit(do_not_specialize=['T'])
def dynamic_conv1d_bwd_kernel_dx(
    z, weight, kernel_bias, dy, dht, dx, cu_seqlens, chunk_indices, T,
    D: tl.constexpr, R: tl.constexpr, W: tl.constexpr,
    STATE: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr,
):
    if STATE:
        i_n = tl.program_id(1).to(tl.int64)
        if cu_seqlens is not None:
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            length = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
        else:
            bos, length = i_n * T, T
        t = tl.arange(0, BT).to(tl.int64) - W
        valid = t < 0
    else:
        i_n, bos, length, start = _dynamic_conv1d_offsets(
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            BT=BT,
            ND=tl.cdiv(D, BD),
        )
        t = start + tl.arange(0, BT).to(tl.int64)
        valid = t < length
    d = (tl.program_id(0).to(tl.int64) % tl.cdiv(D, BD)) * BD + tl.arange(0, BD)
    r = tl.arange(0, BR).to(tl.int64)
    grad = tl.full((BT, BD), 0, tl.float32)
    for j in range(W):
        future = t + j
        latent = tl.load(
            pointer=z + (bos + future[:, None]) * R + r[None, :],
            mask=(future[:, None] >= 0) & (future[:, None] < length) & (r[None, :] < R),
            other=0,
        ).to(tl.float32)
        up = tl.load(
            pointer=weight + (r[:, None] * W + j) * D + d[None, :],
            mask=(r[:, None] < R) & (d[None, :] < D),
            other=0,
        ).to(tl.float32)
        kernel = tl.dot(latent, up, input_precision='ieee')
        if kernel_bias is not None:
            kernel += tl.load(kernel_bias + j * D + d, d < D, other=0)[None, :].to(tl.float32)
        dout = tl.load(
            pointer=dy + (bos + future[:, None]) * D + d[None, :],
            mask=(future[:, None] >= 0) & (future[:, None] < length) & (d[None, :] < D),
            other=0,
        ).to(tl.float32)
        grad += dout * kernel
    if dht is not None:
        slot = t - length + W
        grad += tl.load(
            pointer=dht + (i_n * D + d[None, :]) * W + slot[:, None],
            mask=(slot[:, None] >= 0) & (slot[:, None] < W) & (d[None, :] < D),
            other=0,
        ).to(tl.float32)
    if STATE:
        offsets = (i_n * D + d[None, :]) * W + t[:, None] + W
    else:
        offsets = (bos + t[:, None]) * D + d[None, :]
    tl.store(dx + offsets, grad, valid[:, None] & (d[None, :] < D))


@triton.jit(do_not_specialize=['T'])
def dynamic_conv1d_bwd_kernel_dz(
    x, weight, initial_state, dy, dz, cu_seqlens, chunk_indices, T,
    D: tl.constexpr, R: tl.constexpr, W: tl.constexpr,
    BT: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr,
):
    i_n, bos, length, start = _dynamic_conv1d_offsets(cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, BT=BT, ND=1)
    t = start + tl.arange(0, BT).to(tl.int64)
    r = tl.arange(0, BR).to(tl.int64)
    grad = tl.full((BT, BR), 0, tl.float32)
    for block in range(tl.cdiv(D, BD)):
        d = block.to(tl.int64) * BD + tl.arange(0, BD)
        dout = tl.load(
            pointer=dy + (bos + t[:, None]) * D + d[None, :],
            mask=(t[:, None] < length) & (d[None, :] < D),
            other=0,
        ).to(tl.float32)
        for j in range(W):
            values = _dynamic_conv1d_load_x(
                x=x,
                initial_state=initial_state,
                i_n=i_n,
                bos=bos,
                length=length,
                t=t - j,
                d=d,
                D=D,
                W=W,
            )
            up = tl.load(
                pointer=weight + (r[None, :] * W + j) * D + d[:, None],
                mask=(r[None, :] < R) & (d[:, None] < D),
                other=0,
            ).to(tl.float32)
            grad += tl.dot(dout * values, up, input_precision='ieee')
    tl.store(dz + (bos + t[:, None]) * R + r[None, :], grad, (t[:, None] < length) & (r[None, :] < R))


@triton.jit(do_not_specialize=['T'])
def dynamic_conv1d_bwd_kernel_dw(
    x, z, initial_state, dy, dw, dkernel_bias, cu_seqlens, chunk_indices, T,
    D: tl.constexpr, R: tl.constexpr, W: tl.constexpr, NT: tl.constexpr,
    BS: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr,
):
    i_n, bos, length, start = _dynamic_conv1d_offsets(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        BT=BS,
        ND=tl.cdiv(D, BD),
    )
    part = tl.program_id(1).to(tl.int64) * NT + tl.program_id(0).to(tl.int64) // tl.cdiv(D, BD)
    d = (tl.program_id(0).to(tl.int64) % tl.cdiv(D, BD)) * BD + tl.arange(0, BD)
    r = tl.arange(0, BR).to(tl.int64)
    for j in range(W):
        grad = tl.full((BR, BD), 0, tl.float32)
        grad_bias = tl.full((BD,), 0, tl.float32)
        for offset in range(0, BS, BT):
            t = start + offset + tl.arange(0, BT).to(tl.int64)
            latent = tl.load(
                pointer=z + (bos + t[None, :]) * R + r[:, None],
                mask=(t[None, :] < length) & (r[:, None] < R),
                other=0,
            ).to(tl.float32)
            dout = tl.load(
                pointer=dy + (bos + t[:, None]) * D + d[None, :],
                mask=(t[:, None] < length) & (d[None, :] < D),
                other=0,
            ).to(tl.float32)
            values = _dynamic_conv1d_load_x(
                x=x,
                initial_state=initial_state,
                i_n=i_n,
                bos=bos,
                length=length,
                t=t - j,
                d=d,
                D=D,
                W=W,
            )
            dkernel = dout * values
            grad += tl.dot(latent, dkernel, input_precision='ieee')
            grad_bias += tl.sum(dkernel, axis=0)
        tl.store(dw + ((part * R + r[:, None]) * W + j) * D + d[None, :], grad, (r[:, None] < R) & (d[None, :] < D))
        if dkernel_bias is not None:
            tl.store(dkernel_bias + (part * W + j) * D + d, grad_bias, d < D)


class DynamicConv1dFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, z, weight, kernel_bias, bias, residual, initial_state, output_final_state, activation, cu_seqlens, indices):
        B, T, D = x.shape
        R, W, _ = weight.shape
        BT, BD, BR = 16, 32, max(16, triton.next_power_of_2(R))
        if cu_seqlens is not None and indices is None:
            indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(indices) if indices is not None else triton.cdiv(T, BT)
        y = torch.empty_like(x)
        pre_activation = torch.empty_like(x, dtype=torch.float32) if activation is not None else None
        if NT:
            dynamic_conv1d_fwd_kernel[(triton.cdiv(D, BD) * NT, B)](
                x=x,
                z=z,
                weight=weight,
                kernel_bias=kernel_bias,
                bias=bias,
                residual=residual,
                initial_state=initial_state,
                y=y,
                pre_activation=pre_activation,
                cu_seqlens=cu_seqlens,
                chunk_indices=indices,
                T=T,
                D=D,
                R=R,
                W=W,
                BT=BT,
                BD=BD,
                BR=BR,
            )
        final_state = None
        if output_final_state:
            final_state = causal_conv1d_update_states(x=x, state_len=W, initial_state=initial_state, cu_seqlens=cu_seqlens)
        ctx.save_for_backward(x, z, weight, kernel_bias, bias, residual, initial_state, pre_activation, cu_seqlens, indices)
        return y, final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dy, dht):
        x, z, weight, kernel_bias, bias, residual, initial_state, pre_activation, cu_seqlens, indices = ctx.saved_tensors
        B, T, D = x.shape
        R, W, _ = weight.shape
        BT, BD, BR = 16, 32, max(16, triton.next_power_of_2(R))
        NT = len(indices) if indices is not None else triton.cdiv(T, BT)
        dr = dy.to(residual) if residual is not None else None
        if pre_activation is not None and x.numel():
            dy = swish_bwd(x=pre_activation, dy=dy, output_contiguous=True)
        dx, dz = torch.empty_like(x), torch.empty_like(z)
        dh0 = torch.empty_like(initial_state) if initial_state is not None else None
        if NT:
            dynamic_conv1d_bwd_kernel_dx[(triton.cdiv(D, BD) * NT, B)](
                z=z,
                weight=weight,
                kernel_bias=kernel_bias,
                dy=dy,
                dht=dht,
                dx=dx,
                cu_seqlens=cu_seqlens,
                chunk_indices=indices,
                T=T,
                D=D,
                R=R,
                W=W,
                STATE=False,
                BT=BT,
                BD=BD,
                BR=BR,
            )
            dynamic_conv1d_bwd_kernel_dz[(NT, B)](
                x=x,
                weight=weight,
                initial_state=initial_state,
                dy=dy,
                dz=dz,
                cu_seqlens=cu_seqlens,
                chunk_indices=indices,
                T=T,
                D=D,
                R=R,
                W=W,
                BT=BT,
                BD=BD,
                BR=BR,
            )
        if dh0 is not None:
            dynamic_conv1d_bwd_kernel_dx[(triton.cdiv(D, BD), len(dh0))](
                z=z,
                weight=weight,
                kernel_bias=kernel_bias,
                dy=dy,
                dht=dht,
                dx=dh0,
                cu_seqlens=cu_seqlens,
                chunk_indices=None,
                T=T,
                D=D,
                R=R,
                W=W,
                STATE=True,
                BT=max(16, triton.next_power_of_2(W)),
                BD=BD,
                BR=BR,
            )
        BS = 256
        splits = prepare_chunk_indices(cu_seqlens, BS) if cu_seqlens is not None else None
        NS = len(splits) if splits is not None else triton.cdiv(T, BS)
        dw = weight.new_empty((B * NS, R, W, D), dtype=torch.float32)
        dkernel_bias = kernel_bias.new_empty((B * NS, W, D), dtype=torch.float32) if kernel_bias is not None else None
        if NS:
            dynamic_conv1d_bwd_kernel_dw[(triton.cdiv(D, BD) * NS, B)](
                x=x,
                z=z,
                initial_state=initial_state,
                dy=dy,
                dw=dw,
                dkernel_bias=dkernel_bias,
                cu_seqlens=cu_seqlens,
                chunk_indices=splits,
                T=T,
                D=D,
                R=R,
                W=W,
                NT=NS,
                BS=BS,
                BT=BT,
                BD=BD,
                BR=BR,
            )
        dw = dw.sum(0).to(weight)
        dkernel_bias = dkernel_bias.sum(0).to(kernel_bias) if dkernel_bias is not None else None
        db = dy.sum(dim=(0, 1), dtype=torch.float32).to(bias) if bias is not None else None
        return dx, dz, dw, dkernel_bias, db, dr, dh0, None, None, None, None


def dynamic_conv1d(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    kernel_bias: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply a causal depthwise convolution with low-rank, input-dependent weights.

    Args:
        x (torch.Tensor):
            Convolution inputs of shape [B, T, D].
        z (torch.Tensor):
            Per-token latent coefficients of shape [B, T, R].
        weight (torch.Tensor):
            Kernel basis of shape [R, W, D]. Tap zero multiplies the current token.
        kernel_bias (torch.Tensor, Optional):
            Static kernel of shape [W, D], added to the generated weights. Default: `None`.
        bias (torch.Tensor, Optional):
            Channel bias of shape [D]. Default: `None`.
        residual (torch.Tensor, Optional):
            Residual tensor [B, T, D], added after the output activation. Default: `None`.
        initial_state (torch.Tensor, Optional):
            Raw input history [N, D, W], ordered oldest to newest. Never mutated. Default: `None`.
        output_final_state (bool, Optional):
            Return the differentiable final input history. Default: `False`.
        activation (str, Optional):
            Apply SiLU/swish before the optional residual and output cast. Default: `None`.
        cu_seqlens (torch.Tensor, Optional):
            Packed sequence boundaries [N+1], with B=1. Default: `None`.
        chunk_indices (torch.Tensor, Optional):
            Packed chunk indices from prepare_chunk_indices(cu_seqlens, 16). Default: `None`.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]:
            Output [B, T, D] and optional final history [N, D, W].
    """
    if x.ndim != 3 or z.ndim != 3 or weight.ndim != 3:
        raise ValueError('x, z, and weight must be three-dimensional')
    if activation not in (None, 'silu', 'swish'):
        raise ValueError('activation must be None, silu, or swish')
    B, T, D = x.shape
    R, W, WD = weight.shape
    if B < 1 or D < 1 or z.shape != (B, T, R) or WD != D or not 1 <= R <= 128 or not 1 <= W <= 8:
        raise ValueError('expected matching [B,T,D], [B,T,R], [R,W,D] shapes with 1 <= R <= 128 and 1 <= W <= 8')
    if x.device.type != 'cuda' or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('dynamic_conv1d requires CUDA/HIP tensors in float16, bfloat16, or float32')
    N = B
    if cu_seqlens is not None:
        if B != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError('packed inputs require B=1 and one-dimensional cu_seqlens with at least two entries')
        if cu_seqlens.device != x.device or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError('cu_seqlens must be an int32/int64 tensor on the input device')
        N = cu_seqlens.numel() - 1
    if kernel_bias is not None and kernel_bias.shape != (W, D):
        raise ValueError('kernel_bias must have shape [W,D]')
    if bias is not None and bias.shape != (D,):
        raise ValueError('bias must have shape [D]')
    if residual is not None and residual.shape != x.shape:
        raise ValueError('residual must have the same shape as x')
    if chunk_indices is not None:
        if cu_seqlens is None or chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
            raise ValueError('chunk_indices requires packed inputs and shape [num_chunks, 2]')
        if chunk_indices.device != x.device or chunk_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError('chunk_indices must be int32/int64 on the input device')
    if initial_state is not None and initial_state.shape != (N, D, W):
        raise ValueError('initial_state must have shape [N,D,W]')
    if initial_state is not None and initial_state.dtype != x.dtype:
        raise ValueError('initial_state must have the same dtype as x')
    for tensor in (z, weight, kernel_bias, bias, residual, initial_state):
        if tensor is None:
            continue
        if tensor.device != x.device or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError('floating-point inputs must share a device and use float16, bfloat16, or float32')
    return DynamicConv1dFunction.apply(
        x,
        z,
        weight,
        kernel_bias,
        bias,
        residual,
        initial_state,
        output_final_state,
        activation,
        cu_seqlens,
        chunk_indices,
    )
