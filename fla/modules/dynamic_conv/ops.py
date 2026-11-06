# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

from fla.modules.activations.ops import swish_bwd
from fla.modules.causal_conv1d import causal_conv1d_update_states
from fla.ops.utils import prepare_chunk_indices
from fla.utils import input_guard


@triton.jit
def _dynamic_conv1d_offsets(
    cu_seqlens,
    chunk_indices,
    T,
    NB: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64) // NB
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
def _dynamic_conv1d_load_x(
    x,
    initial_state,
    i_n,
    bos,
    length,
    t,
    d,
    D: tl.constexpr,
    W: tl.constexpr,
):
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


@triton.jit(do_not_specialize=['M'])
def dynamic_conv1d_proj_kernel(
    a,
    b,
    c,
    pre_activation,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    tiles = tl.cdiv(M, BM) * tl.cdiv(N, BN)
    part, tile = pid // tiles, pid % tiles
    m = tile // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    n = tile % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    out = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(BS if BS else K, BK)):
        k = part * BS + block.to(tl.int64) * BK + tl.arange(0, BK)
        left = tl.load(
            pointer=a + m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(m[:, None] < M) & (k[None, :] < K),
            other=0,
        ).to(tl.float32)
        right = tl.load(
            pointer=b + k[:, None] * stride_bk + n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (n[None, :] < N),
            other=0,
        ).to(tl.float32)
        out += tl.dot(left, right, input_precision='ieee')
    mask = (m[:, None] < M) & (n[None, :] < N)
    offset = part * M * N + m[:, None] * N + n[None, :]
    if pre_activation is not None:
        tl.store(pre_activation + offset, out, mask=mask)
        out *= tl.sigmoid(out)
    tl.store(c + offset, out, mask=mask)


@triton.jit(do_not_specialize=['T'])
def dynamic_conv1d_fwd_kernel(
    x,
    z,
    out_proj_weight,
    out_proj_bias,
    residual,
    initial_state,
    y,
    pre_activation,
    cu_seqlens,
    chunk_indices,
    T,
    D: tl.constexpr,
    R: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
):
    i_n, bos, length, start = _dynamic_conv1d_offsets(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        NB=tl.cdiv(D, BD),
        BT=BT,
    )
    t = start + tl.arange(0, BT).to(tl.int64)
    d = (tl.program_id(0).to(tl.int64) % tl.cdiv(D, BD)) * BD + tl.arange(0, BD)
    out = tl.full((BT, BD), 0, tl.float32)
    for j in range(W):
        kernel = tl.full((BT, BD), 0, tl.float32)
        r = tl.arange(0, BR)
        latent = tl.load(
            pointer=z + (bos + t[:, None]) * R + r[None, :],
            mask=(t[:, None] < length) & (r[None, :] < R),
            other=0,
        ).to(tl.float32)
        up = tl.load(
            pointer=out_proj_weight + (d[None, :] * W + j) * R + r[:, None],
            mask=(r[:, None] < R) & (d[None, :] < D),
            other=0,
        ).to(tl.float32)
        kernel += tl.dot(latent, up, input_precision='ieee')
        kernel += tl.load(out_proj_bias + d * W + j.to(tl.int64), d < D, other=0)[None, :].to(tl.float32)
        values = _dynamic_conv1d_load_x(
            x=x,
            initial_state=initial_state,
            i_n=i_n,
            bos=bos,
            length=length,
            t=t - W + 1 + j,
            d=d,
            D=D,
            W=W,
        )
        out += kernel * values
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
    z,
    out_proj_weight,
    out_proj_bias,
    dy,
    dht,
    dx,
    cu_seqlens,
    chunk_indices,
    T,
    D: tl.constexpr,
    R: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
    STATE: tl.constexpr,
):
    if STATE:
        i_n = tl.program_id(1).to(tl.int64)
        if cu_seqlens is not None:
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            length = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
        else:
            bos, length = i_n * T, T
        start = tl.program_id(0).to(tl.int64) // tl.cdiv(D, BD) * BT
        t = start + tl.arange(0, BT).to(tl.int64) - W
        valid = t < 0
    else:
        i_n, bos, length, start = _dynamic_conv1d_offsets(
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            NB=tl.cdiv(D, BD),
            BT=BT,
        )
        t = start + tl.arange(0, BT).to(tl.int64)
        valid = t < length
    d = (tl.program_id(0).to(tl.int64) % tl.cdiv(D, BD)) * BD + tl.arange(0, BD)
    grad = tl.full((BT, BD), 0, tl.float32)
    for j in range(W):
        future = t + W - 1 - j
        kernel = tl.full((BT, BD), 0, tl.float32)
        r = tl.arange(0, BR)
        latent = tl.load(
            pointer=z + (bos + future[:, None]) * R + r[None, :],
            mask=(future[:, None] >= 0) & (future[:, None] < length) & (r[None, :] < R),
            other=0,
        ).to(tl.float32)
        up = tl.load(
            pointer=out_proj_weight + (d[None, :] * W + j) * R + r[:, None],
            mask=(r[:, None] < R) & (d[None, :] < D),
            other=0,
        ).to(tl.float32)
        kernel += tl.dot(latent, up, input_precision='ieee')
        kernel += tl.load(out_proj_bias + d * W + j.to(tl.int64), d < D, other=0)[None, :].to(tl.float32)
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
    x,
    out_proj_weight,
    initial_state,
    dy,
    dz,
    cu_seqlens,
    chunk_indices,
    T,
    D: tl.constexpr,
    R: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
):
    i_n, bos, length, start = _dynamic_conv1d_offsets(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        NB=1,
        BT=BT,
    )
    t = start + tl.arange(0, BT).to(tl.int64)
    r = tl.arange(0, BR)
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
                t=t - W + 1 + j,
                d=d,
                D=D,
                W=W,
            )
            up = tl.load(
                pointer=out_proj_weight + (d[:, None] * W + j) * R + r[None, :],
                mask=(r[None, :] < R) & (d[:, None] < D),
                other=0,
            ).to(tl.float32)
            grad += tl.dot(dout * values, up, input_precision='ieee')
    tl.store(dz + (bos + t[:, None]) * R + r[None, :], grad, (t[:, None] < length) & (r[None, :] < R))


@triton.jit(do_not_specialize=['T'])
def dynamic_conv1d_bwd_kernel_dproj(
    x,
    z,
    initial_state,
    dy,
    dout_proj_weight,
    dout_proj_bias,
    cu_seqlens,
    chunk_indices,
    T,
    D: tl.constexpr,
    R: tl.constexpr,
    W: tl.constexpr,
    NS: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
    BS: tl.constexpr,
):
    i_n, bos, length, start = _dynamic_conv1d_offsets(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        NB=tl.cdiv(D, BD),
        BT=BS,
    )
    i_d = tl.program_id(0).to(tl.int64) % tl.cdiv(D, BD)
    part = tl.program_id(1).to(tl.int64) * NS + tl.program_id(0).to(tl.int64) // tl.cdiv(D, BD)
    d = i_d * BD + tl.arange(0, BD)
    r = tl.arange(0, BR)
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
                t=t - W + 1 + j,
                d=d,
                D=D,
                W=W,
            )
            dkernel = dout * values
            grad += tl.dot(latent, dkernel, input_precision='ieee')
            grad_bias += tl.sum(dkernel, axis=0)
        tl.store(
            pointer=dout_proj_weight + ((part * D + d[None, :]) * W + j) * R + r[:, None],
            value=grad,
            mask=(r[:, None] < R) & (d[None, :] < D),
        )
        tl.store(dout_proj_bias + (part * D + d) * W + j, grad_bias, d < D)


class DynamicConv1dFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        generator_input,
        in_proj_weight,
        out_proj_weight,
        out_proj_bias,
        residual,
        initial_state,
        output_final_state,
        dynamic_activation,
        activation,
        cu_seqlens,
        chunk_indices,
    ):
        B, T, D = x.shape
        R = in_proj_weight.shape[0]
        W = out_proj_weight.shape[0] // D
        BT, BD, BR = 16, 32, max(16, triton.next_power_of_2(R))
        z = x.new_empty(B, T, R, dtype=torch.float32)
        proj_activation = torch.empty_like(z) if dynamic_activation is not None else None
        if B * T:
            dynamic_conv1d_proj_kernel[(triton.cdiv(B * T, BT),)](
                a=generator_input,
                b=in_proj_weight,
                c=z,
                pre_activation=proj_activation,
                M=B * T,
                N=R,
                K=D,
                stride_am=D,
                stride_ak=1,
                stride_bk=1,
                stride_bn=D,
                BM=BT,
                BN=BR,
                BK=32,
                BS=0,
            )
        if cu_seqlens is not None and chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT)
        NT = len(chunk_indices) if chunk_indices is not None else triton.cdiv(T, BT)
        y = torch.empty_like(x)
        pre_activation = torch.empty_like(x, dtype=torch.float32) if activation is not None else None
        if NT:
            dynamic_conv1d_fwd_kernel[(triton.cdiv(D, BD) * NT, B)](
                x=x,
                z=z,
                out_proj_weight=out_proj_weight,
                out_proj_bias=out_proj_bias,
                residual=residual,
                initial_state=initial_state,
                y=y,
                pre_activation=pre_activation,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
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
        ctx.save_for_backward(
            x, generator_input, in_proj_weight, out_proj_weight, out_proj_bias, residual, initial_state,
            z, proj_activation, pre_activation, cu_seqlens, chunk_indices,
        )
        return y, final_state

    @staticmethod
    @input_guard
    def backward(ctx, dy, dht):
        (x, generator_input, in_proj_weight, out_proj_weight, out_proj_bias, residual, initial_state,
         z, proj_activation, pre_activation, cu_seqlens, chunk_indices) = ctx.saved_tensors
        B, T, D = x.shape
        R = in_proj_weight.shape[0]
        W = out_proj_weight.shape[0] // D
        BT, BD, BR = 16, 32, max(16, triton.next_power_of_2(R))
        NT = len(chunk_indices) if chunk_indices is not None else triton.cdiv(T, BT)
        dr = dy.to(residual) if residual is not None else None
        if pre_activation is not None and x.numel():
            dy = swish_bwd(x=pre_activation, dy=dy, output_contiguous=True)
        dx, dz = torch.empty_like(x), torch.empty_like(z)
        dh0 = torch.empty_like(initial_state) if initial_state is not None else None
        if NT:
            dynamic_conv1d_bwd_kernel_dx[(triton.cdiv(D, BD) * NT, B)](
                z=z,
                out_proj_weight=out_proj_weight,
                out_proj_bias=out_proj_bias,
                dy=dy,
                dht=dht,
                dx=dx,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                T=T,
                D=D,
                R=R,
                W=W,
                BT=BT,
                BD=BD,
                BR=BR,
                STATE=False,
            )
            dynamic_conv1d_bwd_kernel_dz[(NT, B)](
                x=x,
                out_proj_weight=out_proj_weight,
                initial_state=initial_state,
                dy=dy,
                dz=dz,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                T=T,
                D=D,
                R=R,
                W=W,
                BT=BT,
                BD=BD,
                BR=BR,
            )
        if dh0 is not None:
            dynamic_conv1d_bwd_kernel_dx[(triton.cdiv(D, BD) * triton.cdiv(W, BT), len(dh0))](
                z=z,
                out_proj_weight=out_proj_weight,
                out_proj_bias=out_proj_bias,
                dy=dy,
                dht=dht,
                dx=dh0,
                cu_seqlens=cu_seqlens,
                chunk_indices=None,
                T=T,
                D=D,
                R=R,
                W=W,
                BT=BT,
                BD=BD,
                BR=BR,
                STATE=True,
            )
        BS = 256
        splits = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BS) if cu_seqlens is not None else None
        NS = len(splits) if splits is not None else triton.cdiv(T, BS)
        dw2 = out_proj_weight.new_empty(B * NS, D * W, R, dtype=torch.float32)
        db2 = out_proj_bias.new_empty(B * NS, D * W, dtype=torch.float32)
        if NS:
            dynamic_conv1d_bwd_kernel_dproj[(triton.cdiv(D, BD) * NS, B)](
                x=x,
                z=z,
                initial_state=initial_state,
                dy=dy,
                dout_proj_weight=dw2,
                dout_proj_bias=db2,
                cu_seqlens=cu_seqlens,
                chunk_indices=splits,
                T=T,
                D=D,
                R=R,
                W=W,
                NS=NS,
                BT=BT,
                BD=BD,
                BR=BR,
                BS=BS,
            )
        dw2, db2 = dw2.sum(0).to(out_proj_weight), db2.sum(0).to(out_proj_bias)
        if proj_activation is not None and z.numel():
            dz = swish_bwd(x=proj_activation, dy=dz, output_contiguous=True)
        dg = torch.empty_like(generator_input)
        NS = triton.cdiv(B * T, BS)
        dw1 = in_proj_weight.new_empty(NS, R, D, dtype=torch.float32)
        if B * T:
            dynamic_conv1d_proj_kernel[(triton.cdiv(B * T, BT) * triton.cdiv(D, BD),)](
                a=dz,
                b=in_proj_weight,
                c=dg,
                pre_activation=None,
                M=B * T,
                N=D,
                K=R,
                stride_am=R,
                stride_ak=1,
                stride_bk=D,
                stride_bn=1,
                BM=BT,
                BN=BD,
                BK=BR,
                BS=0,
            )
        if NS:
            dynamic_conv1d_proj_kernel[(triton.cdiv(R, BT) * triton.cdiv(D, BD) * NS,)](
                a=dz,
                b=generator_input,
                c=dw1,
                pre_activation=None,
                M=R,
                N=D,
                K=B * T,
                stride_am=1,
                stride_ak=R,
                stride_bk=D,
                stride_bn=1,
                BM=BT,
                BN=BD,
                BK=32,
                BS=BS,
            )
        dw1 = dw1.sum(0).to(in_proj_weight)
        return dx, dg, dw1, dw2, db2, dr, dh0, None, None, None, None, None


class DynamicShortConvolution(nn.Module):
    r"""Depthwise causal convolution with input-dependent weights.

    Let D = hidden_size, r = dynamic_rank_dim, and W = kernel_size.
    For conditioning input h_t (default: x_t), the kernel generator computes

    .. math::
        z_t = \phi(A h_t), \qquad
        K_t = \operatorname{reshape}(B z_t + b, [D, W]),

    where A has shape [r, D], B has shape [D * W, r], and b has shape [D * W].
    The convolution applies the generated weights from oldest to newest:

    .. math::
        y_t = \psi\!\left(\sum_{j=0}^{W-1} K_t[:, j] \odot x_{t-W+1+j}\right)
              + \mathrm{residual}_t.

    Phi is dynamic_activation and psi is activation; None selects the identity.
    Both projections and the convolution accumulate in FP32. The second projection is fused with convolution.
    Only the second generator projection has a bias, which is always enabled.
    The convolution has no additional bias; an absent residual is zero.

    Args:
        hidden_size (int):
            Number of conditioning, input, and output channels D.
        kernel_size (int):
            Width W of the causal convolution kernel.
        dynamic_rank_dim (int, Optional):
            Rank r of the kernel generator, at most 128. Default: 16.
        dynamic_activation (str, Optional):
            Activation between projections: `None`, `"silu"`, or `"swish"`. Default: `None`.
        activation (str, Optional):
            Convolution output activation: `None`, `"silu"`, or `"swish"`. Default: `None`.
        backend (str, Optional):
            Convolution backend; dynamic weights support `"triton"` on CUDA/HIP. Default: `"triton"`.
        device (torch.device, Optional):
            Device for the parameters. Default: `None`.
        dtype (torch.dtype, Optional):
            Parameter dtype. Default: `None`.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        dynamic_rank_dim: int = 16,
        dynamic_activation: str | None = None,
        activation: str | None = None,
        backend: str | None = 'triton',
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or kernel_size < 1:
            raise ValueError('hidden_size and kernel_size must be positive')
        if not 1 <= dynamic_rank_dim <= 128:
            raise ValueError('dynamic_rank_dim must be in [1, 128]')
        if dynamic_activation not in (None, 'silu', 'swish') or activation not in (None, 'silu', 'swish'):
            raise ValueError('dynamic_activation and activation must be None, silu, or swish')
        self.backend = os.environ.get('FLA_CONV_BACKEND', backend)
        if self.backend != 'triton':
            raise ValueError('DynamicShortConvolution requires the triton backend')
        self.hidden_size = hidden_size
        self.kernel_size = (kernel_size,)
        self.dynamic_rank_dim = dynamic_rank_dim
        self.dynamic_activation = dynamic_activation
        self.activation = activation
        self.weight = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=dynamic_rank_dim, bias=False, device=device, dtype=dtype),
            nn.Identity() if dynamic_activation is None else nn.SiLU(),
            nn.Linear(
                in_features=dynamic_rank_dim,
                out_features=hidden_size * kernel_size,
                bias=True,
                device=device,
                dtype=dtype,
            ),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight[0].weight, std=0.02)
        nn.init.zeros_(self.weight[2].weight)
        bound = self.kernel_size[0] ** -0.5
        nn.init.uniform_(self.weight[2].bias, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        *,
        generator_input: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply dynamic convolution, returning the output and optional final input history.

        Args:
            x (torch.Tensor):
                Convolution input [B, T, D]. Packed inputs require B=1.
            residual (torch.Tensor, Optional):
                Tensor [B, T, D] added after the output activation. Default: `None`.
            mask (torch.Tensor, Optional):
                Attention mask [B, T], mutually exclusive with cu_seqlens. Default: `None`.
            cache (torch.Tensor, Optional):
                Input history [N, D, W], ordered oldest to newest. Default: `None`.
                Single-token calls without autograd update it in place; other calls leave it unchanged.
            output_final_state (bool, Optional):
                Whether to return the final input history. Default: `False`.
            cu_seqlens (torch.Tensor, Optional):
                Packed sequence boundaries [N+1]. Default: `None`.
            chunk_indices (torch.Tensor, Optional):
                Packed chunk indices for 16-token chunks. Default: `None`.
            generator_input (torch.Tensor, Optional):
                Conditioning input [B, T, D]. Default: `x`.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]:
                Output [B, T, D] and final history [N, D, W] when requested, otherwise `None`.
        """
        if x.ndim != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError('x must have shape [B, T, hidden_size]')
        if x.device.type != 'cuda':
            raise ValueError('DynamicShortConvolution requires CUDA/HIP tensors')
        if kwargs.get('cp_context') is not None:
            raise ValueError('DynamicShortConvolution does not support context parallelism')
        B, T, D = x.shape
        W = self.kernel_size[0]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if cu_seqlens is not None and B != 1:
            raise ValueError('Packed inputs require batch size 1')
        if cache is not None and cache.shape != (N, D, W):
            raise ValueError('cache must have shape [N, D, W]')
        if residual is not None and residual.shape != x.shape:
            raise ValueError('residual must have the same shape as x')
        generator_input = x if generator_input is None else generator_input
        if generator_input.shape != x.shape:
            raise ValueError('generator_input must have the same shape as x')
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError('mask and cu_seqlens cannot be provided at the same time')
            x = x * mask.unsqueeze(-1).to(x.dtype)
            generator_input = generator_input * mask.unsqueeze(-1).to(generator_input.dtype)
        update_cache = not torch.is_grad_enabled() and B * T == N
        if update_cache and cu_seqlens is not None:
            update_cache = bool((cu_seqlens.diff() == 1).all())
        y, final_state = DynamicConv1dFunction.apply(
            x,
            generator_input,
            self.weight[0].weight,
            self.weight[2].weight,
            self.weight[2].bias,
            residual,
            cache,
            output_final_state or update_cache,
            self.dynamic_activation,
            self.activation,
            cu_seqlens,
            chunk_indices,
        )
        if update_cache and cache is not None:
            cache.copy_(final_state)
            final_state = cache
        return y, final_state if output_final_state else None

    def step(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cache: torch.Tensor | None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        *,
        generator_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward(
            x=x,
            residual=residual,
            cache=cache,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            generator_input=generator_input,
        )

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size[0]
