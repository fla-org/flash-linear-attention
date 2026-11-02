# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard


@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
        for BT in [2, 4, 8]
    ],
    key=['H', 'K', 'V', 'BD'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def gate_output_correction_fwd_kernel(
    o,
    r,
    k,
    r_k,
    v,
    g,
    y,
    T,
    t_offset,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_b = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_h = tl.program_id(2).to(tl.int64)
    o_t = t_offset + i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD).to(tl.int64)
    m_k = (o_t[:, None] < T) & (o_d[None, :] < K)
    m_v = (o_t[:, None] < T) & (o_d[None, :] < V)
    o_h = (i_b * T + o_t[:, None]) * H + i_h
    o_k = o_h * K + o_d[None, :]
    o_v = o_h * V + o_d[None, :]

    b_r_k = tl.load(r_k + i_h * K + o_d, mask=o_d < K, other=0).to(tl.float32)
    b_r = tl.load(r + o_k, mask=m_k, other=0).to(tl.float32)
    b_k = tl.load(k + o_k, mask=m_k, other=0).to(tl.float32)
    b_v = tl.load(v + o_v, mask=m_v, other=0).to(tl.float32)
    b_c = tl.sum(b_r * b_k * b_r_k[None, :], axis=1)[:, None]
    b_o = tl.load(o + o_v, mask=m_v, other=0).to(tl.float32)
    b_g = tl.load(g + o_v, mask=m_v, other=0).to(tl.float32)
    b_y = (b_o + b_c * b_v) * b_g
    tl.store(y + o_v, b_y.to(y.dtype.element_ty), mask=m_v)


@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 4]
        for BT in [2, 4, 8]
    ],
    key=['H', 'K', 'V', 'BD'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def gate_output_correction_bwd_kernel(
    o,
    r,
    k,
    r_k,
    v,
    g,
    dy,
    do,
    dr,
    dk,
    dr_k,
    dv,
    dg,
    T,
    t_offset,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_b = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_h = tl.program_id(2).to(tl.int64)
    o_t = t_offset + i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD).to(tl.int64)
    m_k = (o_t[:, None] < T) & (o_d[None, :] < K)
    m_v = (o_t[:, None] < T) & (o_d[None, :] < V)
    o_h = (i_b * T + o_t[:, None]) * H + i_h
    o_k = o_h * K + o_d[None, :]
    o_v = o_h * V + o_d[None, :]

    b_r = tl.load(r + o_k, mask=m_k, other=0).to(tl.float32)
    b_k = tl.load(k + o_k, mask=m_k, other=0).to(tl.float32)
    b_v = tl.load(v + o_v, mask=m_v, other=0).to(tl.float32)
    b_o = tl.load(o + o_v, mask=m_v, other=0).to(tl.float32)
    b_g = tl.load(g + o_v, mask=m_v, other=0).to(tl.float32)
    b_dy = tl.load(dy + o_v, mask=m_v, other=0).to(tl.float32)
    b_r_k = tl.load(r_k + i_h * K + o_d, mask=o_d < K, other=0).to(tl.float32)

    b_c = tl.sum(b_r * b_k * b_r_k[None, :], axis=1)[:, None]
    b_dg = b_dy * (b_o + b_c * b_v)
    b_do = b_dy * b_g
    b_dv = b_do * b_c
    b_dc = tl.sum(b_do * b_v, axis=1)[:, None]
    b_dr = b_dc * b_k * b_r_k[None, :]
    b_dk = b_dc * b_r * b_r_k[None, :]
    b_dr_k = b_dc * b_r * b_k

    tl.store(do + o_v, b_do.to(do.dtype.element_ty), mask=m_v)
    tl.store(dg + o_v, b_dg.to(dg.dtype.element_ty), mask=m_v)
    tl.store(dr + o_k, b_dr.to(dr.dtype.element_ty), mask=m_k)
    tl.store(dk + o_k, b_dk.to(dk.dtype.element_ty), mask=m_k)
    tl.store(dv + o_v, b_dv.to(dv.dtype.element_ty), mask=m_v)
    tl.store(dr_k + o_k, b_dr_k, mask=m_k)


def gate_output_correction_fwd(
    o: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    r_k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    B, T, H, K = r.shape
    V = v.shape[-1]
    BD = triton.next_power_of_2(max(K, V))
    y = torch.empty_like(o)
    for t_offset in range(0, T, 65536):
        # bound the second grid dimension for long sequences
        chunk_size = min(65536, T - t_offset)

        def grid(meta):
            return (B, triton.cdiv(chunk_size, meta['BT']), H)

        gate_output_correction_fwd_kernel[grid](
            o=o,
            r=r,
            k=k,
            r_k=r_k,
            v=v,
            g=g,
            y=y,
            T=T,
            t_offset=t_offset,
            H=H,
            K=K,
            V=V,
            BD=BD,
        )
    return y


def gate_output_correction_bwd(
    o: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    r_k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    dy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = r.shape
    V = v.shape[-1]
    BD = triton.next_power_of_2(max(K, V))
    do = torch.empty_like(o)
    dr = torch.empty_like(r)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g)
    # accumulate the shared weight gradient in fp32 before reducing batch and time
    dr_k = torch.empty_like(r, dtype=torch.float32)
    for t_offset in range(0, T, 65536):
        chunk_size = min(65536, T - t_offset)

        def grid(meta):
            return (B, triton.cdiv(chunk_size, meta['BT']), H)

        gate_output_correction_bwd_kernel[grid](
            o=o,
            r=r,
            k=k,
            r_k=r_k,
            v=v,
            g=g,
            dy=dy,
            do=do,
            dr=dr,
            dk=dk,
            dr_k=dr_k,
            dv=dv,
            dg=dg,
            T=T,
            t_offset=t_offset,
            H=H,
            K=K,
            V=V,
            BD=BD,
        )
    dr_k = dr_k.sum(dim=(0, 1)).to(r_k)
    return do, dr, dk, dr_k, dv, dg


class GateOutputCorrectionFunction(torch.autograd.Function):
    @staticmethod
    @autocast_custom_fwd
    @input_guard
    def forward(ctx, o, r, k, r_k, v, g):
        assert r_k.dim() == 2 and r_k.shape[0] == r.shape[-2] and r_k.shape[1] == r.shape[-1]
        y = gate_output_correction_fwd(o=o, r=r, k=k, r_k=r_k, v=v, g=g)
        ctx.save_for_backward(o, r, k, r_k, v, g)
        return y

    @staticmethod
    @autocast_custom_bwd
    @input_guard
    def backward(ctx, dy):
        o, r, k, r_k, v, g = ctx.saved_tensors
        return gate_output_correction_bwd(o=o, r=r, k=k, r_k=r_k, v=v, g=g, dy=dy)


@torch.compiler.disable
def gate_output_correction(
    o: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    r_k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the RWKV7 output correction and output gate.

    Computes ``(o + correction) * g``, where
    ``correction = ((r * k * r_k).sum(-1, keepdim=True) * v).view_as(o)``.

    Args:
        o (torch.Tensor):
            Output tensor before correction, shape `[B, T, H * V]`.
        r (torch.Tensor):
            Receptance tensor, shape `[B, T, H, K]`.
        k (torch.Tensor):
            Key tensor, shape `[B, T, H, K]`.
        r_k (torch.Tensor):
            Per-head correction weight, shape `[H, K]`.
        v (torch.Tensor):
            Value tensor, shape `[B, T, H, V]`.
        g (torch.Tensor):
            Output gate tensor, shape `[B, T, H * V]`.

    Returns:
        y (torch.Tensor):
            Corrected and gated output tensor with the same shape as `o`.
    """
    return GateOutputCorrectionFunction.apply(o, r, k, r_k, v, g)
