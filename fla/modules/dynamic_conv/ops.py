# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.modules.activations.ops import swish_bwd
from fla.modules.causal_conv1d import causal_conv1d_update_states
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.utils import autotune_cache_kwargs, input_guard


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [1, 2, 3]
    ],
    key=['D', 'R', 'W', 'ACTIVATION', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def dynamic_conv1d_fwd_kernel(
    x,
    z,
    w2,
    b,
    residual,
    initial_state,
    y,
    cu_seqlens,
    chunk_indices,
    T,
    D: tl.constexpr,
    R: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64)
    i_t, i_d, i_n = i // tl.cdiv(D, BD), i % tl.cdiv(D, BD), tl.program_id(1).to(tl.int64)
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = i_d * BD + tl.arange(0, BD)
    o_r = tl.arange(0, BR).to(tl.int64)
    m_t, m_d, m_r = o_t < T, o_d < D, o_r < R
    b_y = tl.full((BT, BD), 0, tl.float32)
    for i_w in range(W):
        p_z = z + (bos + o_t[:, None]) * R + o_r[None, :]
        p_w2 = w2 + (o_d[None, :] * W + i_w) * R + o_r[:, None]
        b_z = tl.load(p_z, mask=m_t[:, None] & m_r[None, :], other=0).to(tl.float32)
        b_w2 = tl.load(p_w2, mask=m_r[:, None] & m_d[None, :], other=0).to(tl.float32)
        b_w = tl.full((BT, BD), 0, tl.float32)
        b_w += tl.dot(b_z, b_w2, input_precision='ieee')
        b_w += tl.load(b + o_d * W + i_w, mask=m_d, other=0)[None, :].to(tl.float32)

        o_x = o_t - W + 1 + i_w
        p_x = x + (bos + o_x[:, None]) * D + o_d[None, :]
        m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :]
        b_x = tl.load(p_x, mask=m_x, other=0).to(tl.float32)
        if initial_state is not None:
            p_h0 = initial_state + (i_n * D + o_d[None, :]) * W + o_x[:, None] + W
            m_h0 = ((o_x >= -W) & (o_x < 0))[:, None] & m_d[None, :]
            b_x += tl.load(p_h0, mask=m_h0, other=0).to(tl.float32)
        b_y += b_w * b_x
    o_y = (bos + o_t[:, None]) * D + o_d[None, :]
    m_y = m_t[:, None] & m_d[None, :]
    if ACTIVATION == 'silu' or ACTIVATION == 'swish':
        b_y *= tl.sigmoid(b_y)
    if residual is not None:
        b_y += tl.load(residual + o_y, mask=m_y, other=0).to(tl.float32)
    tl.store(y + o_y, b_y.to(y.dtype.element_ty), mask=m_y)


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [1, 2, 3]
    ],
    key=['D', 'R', 'W', 'COMPUTE_DH0', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def dynamic_conv1d_bwd_kernel(
    x,
    z,
    w2,
    b,
    initial_state,
    dy,
    dht,
    dx,
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
    COMPUTE_DH0: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_n = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    if IS_VARLEN:
        if not COMPUTE_DH0:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    if COMPUTE_DH0:
        o_t -= W
        m_t = o_t < 0
    else:
        m_t = o_t < T
    o_r = tl.arange(0, BR).to(tl.int64)
    m_r = o_r < R
    b_dz = tl.full((BT, BR), 0, tl.float32)
    for i_d in range(tl.cdiv(D, BD)):
        o_d = i_d.to(tl.int64) * BD + tl.arange(0, BD)
        m_d = o_d < D
        b_dx = tl.full((BT, BD), 0, tl.float32)
        if not COMPUTE_DH0:
            p_dy = dy + (bos + o_t[:, None]) * D + o_d[None, :]
            b_dy = tl.load(p_dy, mask=m_t[:, None] & m_d[None, :], other=0).to(tl.float32)
        for i_w in range(W):
            p_w2 = w2 + (o_d[None, :] * W + i_w) * R + o_r[:, None]
            b_w2 = tl.load(p_w2, mask=m_r[:, None] & m_d[None, :], other=0).to(tl.float32)
            o_y = o_t + W - 1 - i_w
            m_y = (o_y >= 0) & (o_y < T)
            p_z = z + (bos + o_y[:, None]) * R + o_r[None, :]
            b_z = tl.load(p_z, mask=m_y[:, None] & m_r[None, :], other=0).to(tl.float32)
            b_w = tl.full((BT, BD), 0, tl.float32)
            b_w += tl.dot(b_z, b_w2, input_precision='ieee')
            b_w += tl.load(b + o_d * W + i_w, mask=m_d, other=0)[None, :].to(tl.float32)
            p_dy = dy + (bos + o_y[:, None]) * D + o_d[None, :]
            b_dy_shift = tl.load(p_dy, mask=m_y[:, None] & m_d[None, :], other=0).to(tl.float32)
            b_dx += b_dy_shift * b_w

            if not COMPUTE_DH0:
                o_x = o_t - W + 1 + i_w
                p_x = x + (bos + o_x[:, None]) * D + o_d[None, :]
                m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :]
                b_x = tl.load(p_x, mask=m_x, other=0).to(tl.float32)
                if initial_state is not None:
                    p_h0 = initial_state + (i_n * D + o_d[None, :]) * W + o_x[:, None] + W
                    m_h0 = ((o_x >= -W) & (o_x < 0))[:, None] & m_d[None, :]
                    b_x += tl.load(p_h0, mask=m_h0, other=0).to(tl.float32)
                b_dz += tl.dot(b_dy * b_x, tl.trans(b_w2), input_precision='ieee')
        if dht is not None:
            o_h = o_t - T + W
            p_dht = dht + (i_n * D + o_d[None, :]) * W + o_h[:, None]
            m_h = ((o_h >= 0) & (o_h < W))[:, None] & m_d[None, :]
            b_dx += tl.load(p_dht, mask=m_h, other=0).to(tl.float32)
        if COMPUTE_DH0:
            p_dx = dx + (i_n * D + o_d[None, :]) * W + o_t[:, None] + W
        else:
            p_dx = dx + (bos + o_t[:, None]) * D + o_d[None, :]
        tl.store(p_dx, b_dx.to(dx.dtype.element_ty), mask=m_t[:, None] & m_d[None, :])
    if not COMPUTE_DH0:
        p_dz = dz + (bos + o_t[:, None]) * R + o_r[None, :]
        tl.store(p_dz, b_dz.to(dz.dtype.element_ty), mask=m_t[:, None] & m_r[None, :])


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [1, 2, 3]
    ],
    key=['D', 'R', 'W', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def dynamic_conv1d_bwd_kernel_dw2(
    x,
    z,
    initial_state,
    dy,
    dw2,
    db,
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
    IS_VARLEN: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64)
    i_t, i_d, i_n = i // tl.cdiv(D, BD), i % tl.cdiv(D, BD), tl.program_id(1).to(tl.int64)
    i_s = i_n * NS + i_t
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_d = i_d * BD + tl.arange(0, BD)
    o_r = tl.arange(0, BR).to(tl.int64)
    m_d, m_r = o_d < D, o_r < R
    for i_w in range(W):
        b_dw2 = tl.full((BR, BD), 0, tl.float32)
        b_db = tl.full((BD,), 0, tl.float32)
        for i_b in range(0, BS, BT):
            o_t = i_t * BS + i_b + tl.arange(0, BT)
            m_t = o_t < T
            p_z = z + (bos + o_t[None, :]) * R + o_r[:, None]
            p_dy = dy + (bos + o_t[:, None]) * D + o_d[None, :]
            b_z = tl.load(p_z, mask=m_r[:, None] & m_t[None, :], other=0).to(tl.float32)
            b_dy = tl.load(p_dy, mask=m_t[:, None] & m_d[None, :], other=0).to(tl.float32)

            o_x = o_t - W + 1 + i_w
            p_x = x + (bos + o_x[:, None]) * D + o_d[None, :]
            m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :]
            b_x = tl.load(p_x, mask=m_x, other=0).to(tl.float32)
            if initial_state is not None:
                p_h0 = initial_state + (i_n * D + o_d[None, :]) * W + o_x[:, None] + W
                m_h0 = ((o_x >= -W) & (o_x < 0))[:, None] & m_d[None, :]
                b_x += tl.load(p_h0, mask=m_h0, other=0).to(tl.float32)
            b_dw = b_dy * b_x
            b_dw2 += tl.dot(b_z, b_dw, input_precision='ieee')
            b_db += tl.sum(b_dw, axis=0)
        p_dw2 = dw2 + ((i_s * D + o_d[None, :]) * W + i_w) * R + o_r[:, None]
        p_db = db + (i_s * D + o_d) * W + i_w
        tl.store(p_dw2, b_dw2.to(dw2.dtype.element_ty), mask=m_r[:, None] & m_d[None, :])
        tl.store(p_db, b_db.to(db.dtype.element_ty), mask=m_d)


class DynamicConv1dFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        z,
        w2,
        b,
        residual,
        initial_state,
        output_final_state,
        activation,
        cu_seqlens,
    ):
        B, T, D = x.shape
        R = z.shape[-1]
        W = w2.shape[0] // D
        BT, BD, BR = 16, 32, triton.next_power_of_2(R)
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT) if cu_seqlens is not None else None
        NT = len(chunk_indices) if chunk_indices is not None else triton.cdiv(T, BT)
        y = torch.empty_like(x)
        dynamic_conv1d_fwd_kernel[(triton.cdiv(D, BD) * NT, B)](
            x=x,
            z=z,
            w2=w2,
            b=b,
            residual=residual,
            initial_state=initial_state,
            y=y,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            D=D,
            R=R,
            W=W,
            BT=BT,
            BD=BD,
            BR=BR,
            ACTIVATION=activation,
        )
        ht = None
        if output_final_state:
            ht = causal_conv1d_update_states(x=x, state_len=W, initial_state=initial_state, cu_seqlens=cu_seqlens)
        ctx.save_for_backward(x, z, w2, b, residual, initial_state, cu_seqlens, chunk_indices)
        ctx.activation = activation
        return y, ht

    @staticmethod
    @input_guard
    def backward(ctx, dy, dht):
        x, z, w2, b, residual, initial_state, cu_seqlens, chunk_indices = ctx.saved_tensors
        B, T, D = x.shape
        R = z.shape[-1]
        W = w2.shape[0] // D
        BT, BD, BR = 16, 32, triton.next_power_of_2(R)
        NT = len(chunk_indices) if chunk_indices is not None else triton.cdiv(T, BT)
        dr = dy if residual is not None else None
        if ctx.activation is not None and x.numel():
            y = torch.empty_like(x, dtype=torch.float32)
            dynamic_conv1d_fwd_kernel[(triton.cdiv(D, BD) * NT, B)](
                x=x,
                z=z,
                w2=w2,
                b=b,
                residual=None,
                initial_state=initial_state,
                y=y,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                T=T,
                D=D,
                R=R,
                W=W,
                BT=BT,
                BD=BD,
                BR=BR,
                ACTIVATION=None,
            )
            dy = swish_bwd(x=y, dy=dy, output_contiguous=True)
            del y
        dx, dz = torch.empty_like(x), torch.empty_like(z)
        dh0 = torch.empty_like(initial_state) if initial_state is not None else None
        dynamic_conv1d_bwd_kernel[(NT, B)](
            x=x,
            z=z,
            w2=w2,
            b=b,
            initial_state=initial_state,
            dy=dy,
            dht=dht,
            dx=dx,
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
            COMPUTE_DH0=False,
        )
        if dh0 is not None:
            dynamic_conv1d_bwd_kernel[(triton.cdiv(W, BT), len(dh0))](
                x=x,
                z=z,
                w2=w2,
                b=b,
                initial_state=initial_state,
                dy=dy,
                dht=dht,
                dx=dh0,
                dz=None,
                cu_seqlens=cu_seqlens,
                chunk_indices=None,
                T=T,
                D=D,
                R=R,
                W=W,
                BT=BT,
                BD=BD,
                BR=BR,
                COMPUTE_DH0=True,
            )
        BS = 256
        chunk_indices_dw = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BS) if cu_seqlens is not None else None
        NS = len(chunk_indices_dw) if chunk_indices_dw is not None else triton.cdiv(T, BS)
        dw2 = w2.new_empty(B * NS, D * W, R, dtype=torch.float32)
        db = b.new_empty(B * NS, D * W, dtype=torch.float32)
        dynamic_conv1d_bwd_kernel_dw2[(triton.cdiv(D, BD) * NS, B)](
            x=x,
            z=z,
            initial_state=initial_state,
            dy=dy,
            dw2=dw2,
            db=db,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices_dw,
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
        dw2, db = dw2.sum(0).to(w2), db.sum(0).to(b)
        return dx, dz, dw2, db, dr, dh0, None, None, None


def dynamic_conv1d(
    x: torch.Tensor,
    h: torch.Tensor | None,
    dynamic_proj_w1: torch.Tensor,
    dynamic_proj_w2: torch.Tensor,
    dynamic_proj_bias: torch.Tensor,
    dynamic_activation: str | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Apply dynamic depthwise causal convolution with a low-rank kernel generator.

    A two-layer projection of h_t generates a separate [D, W] convolution kernel at each position.

    .. math::
        z_t = \phi(W_1 h_t), \qquad w_t = \operatorname{reshape}(W_2 z_t + b, [D, W]),

        y_t = \psi\!\left(\sum_{j=0}^{W-1} w_t[:, j] \odot x_{t-W+1+j}\right)
              + \mathrm{residual}_t.

    Args:
        x (torch.Tensor):
            Convolution input [B, T, D]. Packed inputs require B=1.
        h (torch.Tensor | None):
            Conditioning input [B, T, D]. `None` uses x.
        dynamic_proj_w1 (torch.Tensor):
            First projection W_1 with shape [R, D], where 16 <= R <= 128.
        dynamic_proj_w2 (torch.Tensor):
            Second projection W_2 with shape [D * W, R].
        dynamic_proj_bias (torch.Tensor):
            Second projection bias b with shape [D * W].
        dynamic_activation (str, Optional):
            Activation phi between projections: `None`, `"silu"`, or `"swish"`. Default: `None`.
        residual (torch.Tensor, Optional):
            Tensor [B, T, D] added after the output activation. Default: `None`.
        initial_state (torch.Tensor, Optional):
            Input history [N, D, W], ordered oldest to newest. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the final input history. Default: `False`.
        activation (str, Optional):
            Output activation psi: `None`, `"silu"`, or `"swish"`. Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Packed sequence boundaries [N+1]. Default: `None`.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]:
            Output [B, T, D] and final history [N, D, W] when requested, otherwise `None`.
    """
    if not 16 <= dynamic_proj_w1.shape[0] <= 128:
        raise ValueError('dynamic_rank_dim must be in [16, 128]')
    if dynamic_activation not in (None, 'silu', 'swish') or activation not in (None, 'silu', 'swish'):
        raise ValueError('dynamic_activation and activation must be None, silu, or swish')
    h = x if h is None else h
    z = F.linear(input=h, weight=dynamic_proj_w1)
    if dynamic_activation in ('silu', 'swish'):
        z = F.silu(z)
    return DynamicConv1dFunction.apply(
        x,
        z,
        dynamic_proj_w2,
        dynamic_proj_bias,
        residual,
        initial_state,
        output_final_state,
        activation,
        cu_seqlens,
    )


class DynamicShortConvolution(nn.Module):
    r"""Depthwise causal convolution with input-dependent weights.

    A two-layer projection of h_t generates a separate [D, W] convolution kernel at each position.

    .. math::
        z_t = \phi(W_1 h_t), \qquad w_t = \operatorname{reshape}(W_2 z_t + b, [D, W]),

        y_t = \psi\!\left(\sum_{j=0}^{W-1} w_t[:, j] \odot x_{t-W+1+j}\right)
              + \mathrm{residual}_t.

    Args:
        hidden_size (int):
            Number of conditioning, input, and output channels D.
        kernel_size (int):
            Width W of the causal convolution kernel.
        dynamic_rank_dim (int, Optional):
            Rank R of the kernel generator (16–128). Default: 16.
        dynamic_activation (str, Optional):
            Activation phi between projections: `None`, `"silu"`, or `"swish"`. Default: `None`.
        activation (str, Optional):
            Output activation psi: `None`, `"silu"`, or `"swish"`. Default: `None`.
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
        if not 16 <= dynamic_rank_dim <= 128:
            raise ValueError('dynamic_rank_dim must be in [16, 128]')
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
        self.proj = nn.Sequential(
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
        nn.init.normal_(tensor=self.proj[0].weight, std=0.02)
        nn.init.normal_(tensor=self.proj[2].weight, std=0.02)
        nn.init.zeros_(tensor=self.proj[2].bias)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        *,
        h: torch.Tensor | None = None,
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
            cu_seqlens (torch.LongTensor, Optional):
                Packed sequence boundaries [N+1]. Default: `None`.
            h (torch.Tensor, Optional):
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
        h = x if h is None else h
        if h.shape != x.shape:
            raise ValueError('h must have the same shape as x')
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError('mask and cu_seqlens cannot be provided at the same time')
            x = x * mask.unsqueeze(-1).to(x.dtype)
            h = h * mask.unsqueeze(-1).to(h.dtype)
        update_cache = not torch.is_grad_enabled() and B * T == N
        if update_cache and cu_seqlens is not None:
            update_cache = bool((cu_seqlens.diff() == 1).all())
        y, ht = dynamic_conv1d(
            x=x,
            h=h,
            dynamic_proj_w1=self.proj[0].weight,
            dynamic_proj_w2=self.proj[2].weight,
            dynamic_proj_bias=self.proj[2].bias,
            dynamic_activation=self.dynamic_activation,
            residual=residual,
            initial_state=cache,
            output_final_state=output_final_state or update_cache,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
        )
        if update_cache and cache is not None:
            cache.copy_(ht)
            ht = cache
        return y, ht if output_final_state else None

    def step(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cache: torch.Tensor | None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        *,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward(
            x=x,
            residual=residual,
            cache=cache,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            h=h,
        )

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size[0]
