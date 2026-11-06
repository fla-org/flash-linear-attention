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


@triton.jit(do_not_specialize=['M'])
def dynamic_conv1d_fwd_kernel_proj(
    a,
    b,
    c,
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
    ACTIVATION: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64)
    NM, NN = tl.cdiv(M, BM), tl.cdiv(N, BN)
    i_s, i_m, i_n = i // (NM * NN), (i // NN) % NM, i % NN
    o_m = i_m * BM + tl.arange(0, BM)
    o_n = i_n * BN + tl.arange(0, BN)
    m_m, m_n = o_m < M, o_n < N

    b_c = tl.full((BM, BN), 0, tl.float32)
    for i_k in range(tl.cdiv(BS if BS else K, BK)):
        o_k = i_s * BS + i_k.to(tl.int64) * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_a = a + o_m[:, None] * stride_am + o_k[None, :] * stride_ak
        p_b = b + o_k[:, None] * stride_bk + o_n[None, :] * stride_bn
        b_a = tl.load(p_a, mask=m_m[:, None] & m_k[None, :], other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=m_k[:, None] & m_n[None, :], other=0).to(tl.float32)
        b_c += tl.dot(b_a, b_b, input_precision='ieee')
    o_c = i_s * M * N + o_m[:, None] * N + o_n[None, :]
    m_c = m_m[:, None] & m_n[None, :]
    if ACTIVATION == 'silu' or ACTIVATION == 'swish':
        b_c *= tl.sigmoid(b_c)
    tl.store(c + o_c, b_c.to(c.dtype.element_ty), mask=m_c)


@triton.jit(do_not_specialize=['T'])
def dynamic_conv1d_fwd_kernel(
    x,
    z,
    weight,
    bias,
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
):
    i = tl.program_id(0).to(tl.int64)
    i_t, i_d, i_n = i // tl.cdiv(D, BD), i % tl.cdiv(D, BD), tl.program_id(1).to(tl.int64)
    if cu_seqlens is not None:
        i_n = tl.load(chunk_indices + 2 * i_t).to(tl.int64)
        i_t = tl.load(chunk_indices + 2 * i_t + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_n * T, (i_n + 1) * T
    T = eos - bos

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = i_d * BD + tl.arange(0, BD)
    o_r = tl.arange(0, BR).to(tl.int64)
    m_t, m_d, m_r = o_t < T, o_d < D, o_r < R
    b_y = tl.full((BT, BD), 0, tl.float32)
    for i_w in range(W):
        p_z = z + (bos + o_t[:, None]) * R + o_r[None, :]
        p_w = weight + (o_d[None, :] * W + i_w) * R + o_r[:, None]
        b_z = tl.load(p_z, mask=m_t[:, None] & m_r[None, :], other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=m_r[:, None] & m_d[None, :], other=0).to(tl.float32)
        b_k = tl.full((BT, BD), 0, tl.float32)
        b_k += tl.dot(b_z, b_w, input_precision='ieee')
        b_k += tl.load(bias + o_d * W + i_w, mask=m_d, other=0)[None, :].to(tl.float32)

        o_x = o_t - W + 1 + i_w
        p_x = x + (bos + o_x[:, None]) * D + o_d[None, :]
        m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :]
        b_x = tl.load(p_x, mask=m_x, other=0).to(tl.float32)
        if initial_state is not None:
            p_h0 = initial_state + (i_n * D + o_d[None, :]) * W + o_x[:, None] + W
            m_h0 = ((o_x >= -W) & (o_x < 0))[:, None] & m_d[None, :]
            b_x += tl.load(p_h0, mask=m_h0, other=0).to(tl.float32)
        b_y += b_k * b_x
    o_y = (bos + o_t[:, None]) * D + o_d[None, :]
    m_y = m_t[:, None] & m_d[None, :]
    if ACTIVATION == 'silu' or ACTIVATION == 'swish':
        b_y *= tl.sigmoid(b_y)
    if residual is not None:
        b_y += tl.load(residual + o_y, mask=m_y, other=0).to(tl.float32)
    tl.store(y + o_y, b_y.to(y.dtype.element_ty), mask=m_y)


@triton.jit(do_not_specialize=['T'])
def dynamic_conv1d_bwd_kernel_dx(
    z,
    weight,
    bias,
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
    COMPUTE_DH0: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64)
    i_t, i_d, i_n = i // tl.cdiv(D, BD), i % tl.cdiv(D, BD), tl.program_id(1).to(tl.int64)
    if cu_seqlens is not None:
        if not COMPUTE_DH0:
            i_n = tl.load(chunk_indices + 2 * i_t).to(tl.int64)
            i_t = tl.load(chunk_indices + 2 * i_t + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_n * T, (i_n + 1) * T
    T = eos - bos

    o_t = i_t * BT + tl.arange(0, BT)
    if COMPUTE_DH0:
        o_t -= W
        m_t = o_t < 0
    else:
        m_t = o_t < T
    o_d = i_d * BD + tl.arange(0, BD)
    o_r = tl.arange(0, BR).to(tl.int64)
    m_d, m_r = o_d < D, o_r < R
    b_dx = tl.full((BT, BD), 0, tl.float32)
    for i_w in range(W):
        o_y = o_t + W - 1 - i_w
        m_y = (o_y >= 0) & (o_y < T)
        p_z = z + (bos + o_y[:, None]) * R + o_r[None, :]
        p_w = weight + (o_d[None, :] * W + i_w) * R + o_r[:, None]
        b_z = tl.load(p_z, mask=m_y[:, None] & m_r[None, :], other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=m_r[:, None] & m_d[None, :], other=0).to(tl.float32)
        b_k = tl.full((BT, BD), 0, tl.float32)
        b_k += tl.dot(b_z, b_w, input_precision='ieee')
        b_k += tl.load(bias + o_d * W + i_w, mask=m_d, other=0)[None, :].to(tl.float32)
        p_dy = dy + (bos + o_y[:, None]) * D + o_d[None, :]
        b_dy = tl.load(p_dy, mask=m_y[:, None] & m_d[None, :], other=0).to(tl.float32)
        b_dx += b_dy * b_k
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


@triton.jit(do_not_specialize=['T'])
def dynamic_conv1d_bwd_kernel_dz(
    x,
    weight,
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
    i_t, i_n = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    if cu_seqlens is not None:
        i_n = tl.load(chunk_indices + 2 * i_t).to(tl.int64)
        i_t = tl.load(chunk_indices + 2 * i_t + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_n * T, (i_n + 1) * T
    T = eos - bos

    o_t = i_t * BT + tl.arange(0, BT)
    o_r = tl.arange(0, BR).to(tl.int64)
    m_t, m_r = o_t < T, o_r < R
    b_dz = tl.full((BT, BR), 0, tl.float32)
    for i_d in range(tl.cdiv(D, BD)):
        o_d = i_d.to(tl.int64) * BD + tl.arange(0, BD)
        m_d = o_d < D
        p_dy = dy + (bos + o_t[:, None]) * D + o_d[None, :]
        b_dy = tl.load(p_dy, mask=m_t[:, None] & m_d[None, :], other=0).to(tl.float32)
        for i_w in range(W):
            o_x = o_t - W + 1 + i_w
            p_x = x + (bos + o_x[:, None]) * D + o_d[None, :]
            m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :]
            b_x = tl.load(p_x, mask=m_x, other=0).to(tl.float32)
            if initial_state is not None:
                p_h0 = initial_state + (i_n * D + o_d[None, :]) * W + o_x[:, None] + W
                m_h0 = ((o_x >= -W) & (o_x < 0))[:, None] & m_d[None, :]
                b_x += tl.load(p_h0, mask=m_h0, other=0).to(tl.float32)
            p_w = weight + (o_d[:, None] * W + i_w) * R + o_r[None, :]
            b_w = tl.load(p_w, mask=m_d[:, None] & m_r[None, :], other=0).to(tl.float32)
            b_dz += tl.dot(b_dy * b_x, b_w, input_precision='ieee')
    p_dz = dz + (bos + o_t[:, None]) * R + o_r[None, :]
    tl.store(p_dz, b_dz.to(dz.dtype.element_ty), mask=m_t[:, None] & m_r[None, :])


@triton.jit(do_not_specialize=['T'])
def dynamic_conv1d_bwd_kernel_dw(
    x,
    z,
    initial_state,
    dy,
    dw,
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
):
    i = tl.program_id(0).to(tl.int64)
    i_t, i_d, i_n = i // tl.cdiv(D, BD), i % tl.cdiv(D, BD), tl.program_id(1).to(tl.int64)
    i_s = i_n * NS + i_t
    if cu_seqlens is not None:
        i_n = tl.load(chunk_indices + 2 * i_t).to(tl.int64)
        i_t = tl.load(chunk_indices + 2 * i_t + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_n * T, (i_n + 1) * T
    T = eos - bos

    o_d = i_d * BD + tl.arange(0, BD)
    o_r = tl.arange(0, BR).to(tl.int64)
    m_d, m_r = o_d < D, o_r < R
    for i_w in range(W):
        b_dw = tl.full((BR, BD), 0, tl.float32)
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
            b_dk = b_dy * b_x
            b_dw += tl.dot(b_z, b_dk, input_precision='ieee')
            b_db += tl.sum(b_dk, axis=0)
        p_dw = dw + ((i_s * D + o_d[None, :]) * W + i_w) * R + o_r[:, None]
        p_db = db + (i_s * D + o_d) * W + i_w
        tl.store(p_dw, b_dw.to(dw.dtype.element_ty), mask=m_r[:, None] & m_d[None, :])
        tl.store(p_db, b_db.to(db.dtype.element_ty), mask=m_d)


class DynamicConv1dFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        h,
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
        if B * T:
            dynamic_conv1d_fwd_kernel_proj[(triton.cdiv(B * T, BT),)](
                a=h,
                b=in_proj_weight,
                c=z,
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
                ACTIVATION=dynamic_activation,
            )
        if cu_seqlens is not None and chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT)
        NT = len(chunk_indices) if chunk_indices is not None else triton.cdiv(T, BT)
        y = torch.empty_like(x)
        if NT:
            dynamic_conv1d_fwd_kernel[(triton.cdiv(D, BD) * NT, B)](
                x=x,
                z=z,
                weight=out_proj_weight,
                bias=out_proj_bias,
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
        final_state = None
        if output_final_state:
            final_state = causal_conv1d_update_states(x=x, state_len=W, initial_state=initial_state, cu_seqlens=cu_seqlens)
        ctx.save_for_backward(
            x,
            h,
            in_proj_weight,
            out_proj_weight,
            out_proj_bias,
            residual,
            initial_state,
            z,
            cu_seqlens,
            chunk_indices,
        )
        ctx.dynamic_activation = dynamic_activation
        ctx.activation = activation
        return y, final_state

    @staticmethod
    @input_guard
    def backward(ctx, dy, dht):
        (
            x,
            h,
            in_proj_weight,
            out_proj_weight,
            out_proj_bias,
            residual,
            initial_state,
            z,
            cu_seqlens,
            chunk_indices,
        ) = ctx.saved_tensors
        B, T, D = x.shape
        R = in_proj_weight.shape[0]
        W = out_proj_weight.shape[0] // D
        BT, BD, BR = 16, 32, max(16, triton.next_power_of_2(R))
        NT = len(chunk_indices) if chunk_indices is not None else triton.cdiv(T, BT)
        dresidual = dy.to(residual) if residual is not None else None
        if ctx.activation is not None and x.numel():
            y = torch.empty_like(x, dtype=torch.float32)
            dynamic_conv1d_fwd_kernel[(triton.cdiv(D, BD) * NT, B)](
                x=x,
                z=z,
                weight=out_proj_weight,
                bias=out_proj_bias,
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
        if NT:
            dynamic_conv1d_bwd_kernel_dx[(triton.cdiv(D, BD) * NT, B)](
                z=z,
                weight=out_proj_weight,
                bias=out_proj_bias,
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
                COMPUTE_DH0=False,
            )
            dynamic_conv1d_bwd_kernel_dz[(NT, B)](
                x=x,
                weight=out_proj_weight,
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
                weight=out_proj_weight,
                bias=out_proj_bias,
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
                COMPUTE_DH0=True,
            )
        BS = 256
        grad_chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BS) if cu_seqlens is not None else None
        NS = len(grad_chunk_indices) if grad_chunk_indices is not None else triton.cdiv(T, BS)
        dout_proj_weight = out_proj_weight.new_empty(B * NS, D * W, R, dtype=torch.float32)
        dout_proj_bias = out_proj_bias.new_empty(B * NS, D * W, dtype=torch.float32)
        if NS:
            dynamic_conv1d_bwd_kernel_dw[(triton.cdiv(D, BD) * NS, B)](
                x=x,
                z=z,
                initial_state=initial_state,
                dy=dy,
                dw=dout_proj_weight,
                db=dout_proj_bias,
                cu_seqlens=cu_seqlens,
                chunk_indices=grad_chunk_indices,
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
        dout_proj_weight, dout_proj_bias = dout_proj_weight.sum(0).to(out_proj_weight), dout_proj_bias.sum(0).to(out_proj_bias)
        if ctx.dynamic_activation is not None and z.numel():
            z = torch.empty_like(z)
            dynamic_conv1d_fwd_kernel_proj[(triton.cdiv(B * T, BT),)](
                a=h,
                b=in_proj_weight,
                c=z,
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
                ACTIVATION=None,
            )
            dz = swish_bwd(x=z, dy=dz, output_contiguous=True)
            del z
        dh = torch.empty_like(h)
        NS = triton.cdiv(B * T, BS)
        din_proj_weight = in_proj_weight.new_empty(NS, R, D, dtype=torch.float32)
        if B * T:
            dynamic_conv1d_fwd_kernel_proj[(triton.cdiv(B * T, BT) * triton.cdiv(D, BD),)](
                a=dz,
                b=in_proj_weight,
                c=dh,
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
                ACTIVATION=None,
            )
        if NS:
            dynamic_conv1d_fwd_kernel_proj[(triton.cdiv(R, BT) * triton.cdiv(D, BD) * NS,)](
                a=dz,
                b=h,
                c=din_proj_weight,
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
                ACTIVATION=None,
            )
        din_proj_weight = din_proj_weight.sum(0).to(in_proj_weight)
        return dx, dh, din_proj_weight, dout_proj_weight, dout_proj_bias, dresidual, dh0, None, None, None, None, None


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

    Phi is dynamic_activation and psi is activation; `None` selects the identity.
    Both projections and the convolution accumulate in FP32. The second projection is fused with convolution.
    Activation inputs are recomputed during backward.
    Only the second generator projection has a bias, which is always enabled.
    The convolution has no additional bias; an absent residual is zero.
    The second projection weight starts at zero, so its bias initializes a static filter shared across tokens.

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
        nn.init.normal_(self.proj[0].weight, std=0.02)
        nn.init.zeros_(self.proj[2].weight)
        bound = self.kernel_size[0] ** -0.5
        nn.init.uniform_(self.proj[2].bias, -bound, bound)

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
            cu_seqlens (torch.Tensor, Optional):
                Packed sequence boundaries [N+1]. Default: `None`.
            chunk_indices (torch.Tensor, Optional):
                Packed chunk indices for 16-token chunks. Default: `None`.
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
        y, final_state = DynamicConv1dFunction.apply(
            x,
            h,
            self.proj[0].weight,
            self.proj[2].weight,
            self.proj[2].bias,
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
