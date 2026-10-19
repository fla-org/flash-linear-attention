# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from fla.backends import dispatch
from fla.utils import IS_NVIDIA, autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.jit(do_not_specialize=['T'])
def hc_fwd_kernel(
    x,
    w,
    y,
    m,
    o,
    T,
    N: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    HAS_Y: tl.constexpr,
    HAS_M: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    o_n = tl.arange(0, BN).to(tl.int64)
    o_d = i_c * BD + tl.arange(0, BD).to(tl.int64)
    m_nd = (o_n[:, None] < N) & (o_d[None, :] < D)
    b_w = tl.load(w + i_t * N + o_n, mask=o_n < N, other=0).to(tl.float32)
    if HAS_Y:
        b_y = tl.load(y + i_t * D + o_d, mask=o_d < D, other=0).to(tl.float32)
        if HAS_M:
            b_o = tl.full((BN, BD), 0, dtype=tl.float32)
            for i_n in tl.static_range(N):
                b_x = tl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(tl.float32)
                b_m = tl.load(m + (i_t * N + o_n) * N + i_n, mask=o_n < N, other=0).to(tl.float32)
                b_o += b_m[:, None] * b_x[None, :]
        else:
            b_o = tl.load(x + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(tl.float32)
        b_o += b_w[:, None] * b_y[None, :]
        tl.store(o + (i_t * N + o_n[:, None]) * D + o_d[None, :], b_o, mask=m_nd)
    else:
        # keep the stream reduction in registers to avoid a shared-memory layout conversion
        b_o = tl.full((BD,), 0, dtype=tl.float32)
        for i_n in tl.static_range(N):
            b_x = tl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(tl.float32)
            b_wj = tl.load(w + i_t * N + i_n).to(tl.float32)
            b_o += b_wj * b_x
        tl.store(o + i_t * D + o_d, b_o, mask=o_d < D)


@triton.jit(do_not_specialize=['T'])
def hc_bwd_kernel(
    x,
    w,
    y,
    m,
    do,
    dx,
    dy,
    dw_partial,
    dm_partial,
    T,
    N: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    ND: tl.constexpr,
    HAS_Y: tl.constexpr,
    HAS_M: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    o_n = tl.arange(0, BN).to(tl.int64)
    o_d = i_c * BD + tl.arange(0, BD).to(tl.int64)
    m_nd = (o_n[:, None] < N) & (o_d[None, :] < D)
    b_w = tl.load(w + i_t * N + o_n, mask=o_n < N, other=0).to(tl.float32)
    if HAS_Y:
        b_do = tl.load(do + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(tl.float32)
        b_y = tl.load(y + i_t * D + o_d, mask=o_d < D, other=0).to(tl.float32)
        b_dw = tl.sum(b_do * b_y[None, :], axis=1)
        b_dy = tl.sum(b_do * b_w[:, None], axis=0)
        tl.store(dy + i_t * D + o_d, b_dy, mask=o_d < D)
        if HAS_M:
            b_dx = tl.full((BN, BD), 0, dtype=tl.float32)
            for i_n in tl.static_range(N):
                b_doj = tl.load(do + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(tl.float32)
                b_m = tl.load(m + (i_t * N + i_n) * N + o_n, mask=o_n < N, other=0).to(tl.float32)
                b_dx += b_m[:, None] * b_doj[None, :]
                b_xj = tl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(tl.float32)
                b_dm = tl.sum(b_do * b_xj[None, :], axis=1)
                tl.store(dm_partial + ((i_t * ND + i_c) * N + o_n) * N + i_n, b_dm, mask=o_n < N)
        else:
            b_dx = b_do
    else:
        b_do = tl.load(do + i_t * D + o_d, mask=o_d < D, other=0).to(tl.float32)
        b_x = tl.load(x + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(tl.float32)
        b_dx = b_w[:, None] * b_do[None, :]
        b_dw = tl.sum(b_x * b_do[None, :], axis=1)
    tl.store(dx + (i_t * N + o_n[:, None]) * D + o_d[None, :], b_dx, mask=m_nd)
    tl.store(dw_partial + (i_t * ND + i_c) * N + o_n, b_dw, mask=o_n < N)


@triton.jit(do_not_specialize=['T'])
def hc_bwd_kernel_reduce(partial, grad, T, C: tl.constexpr, S: tl.constexpr, BC: tl.constexpr, BS: tl.constexpr):
    i_t = tl.program_id(0).to(tl.int64)
    o_c = tl.arange(0, BC).to(tl.int64)
    o_s = tl.arange(0, BS).to(tl.int64)
    b_partial = tl.load(
        partial + (i_t * C + o_c[:, None]) * S + o_s[None, :],
        mask=(o_c[:, None] < C) & (o_s[None, :] < S),
        other=0,
    )
    tl.store(grad + i_t * S + o_s, tl.sum(b_partial, axis=0), mask=o_s < S)


def _validate_inputs(x: torch.Tensor, *others: torch.Tensor | None) -> None:
    if not IS_NVIDIA or not x.is_cuda:
        raise ValueError('HC kernels require NVIDIA CUDA tensors')
    for value in (x, *others):
        if value is not None and (
            value.device != x.device or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise ValueError('Inputs must share a CUDA device and use FP16, BF16, or FP32')
    if x.ndim < 2 or not 1 <= x.shape[-2] <= 8 or x.shape[-1] < 1:
        raise ValueError('Expected [..., N, D] with 1 <= N <= 8 and D >= 1')


@dispatch
def hc_fwd(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor | None = None, m: torch.Tensor | None = None):
    N, D = x.shape[-2:]
    T = x.numel() // (N * D)
    BD = min(256, triton.next_power_of_2(D))
    ND = triton.cdiv(D, BD)
    o = torch.empty_like(x) if y is not None else x.new_empty((*x.shape[:-2], D))
    if T:
        hc_fwd_kernel[(T, ND)](
            x=x,
            w=w,
            y=y,
            m=m,
            o=o,
            T=T,
            N=N,
            D=D,
            BN=triton.next_power_of_2(N),
            BD=BD,
            HAS_Y=y is not None,
            HAS_M=m is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return o, hc_bwd


def hc_bwd(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor | None,
    m: torch.Tensor | None,
    do: torch.Tensor,
):
    N, D = x.shape[-2:]
    T = x.numel() // (N * D)
    BD = min(256, triton.next_power_of_2(D))
    ND = triton.cdiv(D, BD)
    dx, dw = torch.empty_like(x), torch.empty_like(w)
    dy = torch.empty_like(y) if y is not None else None
    dm = torch.empty_like(m) if m is not None else None
    dw_partial = torch.empty((T, ND, N), device=x.device, dtype=torch.float32)
    dm_partial = torch.empty((T, ND, N * N), device=x.device, dtype=torch.float32) if m is not None else None
    if T:
        hc_bwd_kernel[(T, ND)](
            x=x,
            w=w,
            y=y,
            m=m,
            do=do,
            dx=dx,
            dy=dy,
            dw_partial=dw_partial,
            dm_partial=dm_partial,
            T=T,
            N=N,
            D=D,
            BN=triton.next_power_of_2(N),
            BD=BD,
            ND=ND,
            HAS_Y=y is not None,
            HAS_M=m is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
        for partial, grad, size in ((dw_partial, dw, N), (dm_partial, dm, N * N)):
            if grad is not None:
                hc_bwd_kernel_reduce[(T,)](
                    partial=partial,
                    grad=grad,
                    T=T,
                    C=ND,
                    S=size,
                    BC=triton.next_power_of_2(ND),
                    BS=triton.next_power_of_2(size),
                    num_warps=4,
                    enable_fp_fusion=False,
                )
    return dx, dw, dy, dm


class FusedHCFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, w, y, m):
        o, ctx.bwd = hc_fwd(x=x, w=w, y=y, m=m)
        ctx.save_for_backward(x, w, y, m)
        return o

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    @once_differentiable
    def backward(ctx, do):
        x, w, y, m = ctx.saved_tensors
        return ctx.bwd(x=x, w=w, y=y, m=m, do=do)


def fused_hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Read out residual streams with activated routing weights.

    Args:
        x (torch.Tensor):
            Residual streams of shape `[..., N, D]`.
        pre (torch.Tensor):
            Routing weights of shape `[..., N]` with the same leading dimensions as `x`.

    Returns:
        torch.Tensor:
            Weighted sum of shape `[..., D]`, accumulated in FP32 and returned in `x.dtype`.
    """
    _validate_inputs(x, pre)
    if pre.shape != x.shape[:-1]:
        raise ValueError('pre must have shape [..., N] matching x')
    return FusedHCFunction.apply(x, pre, None, None)


def fused_hc_post(x: torch.Tensor, branch: torch.Tensor, post: torch.Tensor, mixing: torch.Tensor | None = None) -> torch.Tensor:
    """Write a branch into residual streams using iHC or mHC routing.

    Args:
        x (torch.Tensor):
            Residual streams of shape `[..., N, D]`.
        branch (torch.Tensor):
            Branch output of shape `[..., D]` with the same leading dimensions as `x`.
        post (torch.Tensor):
            Activated routing weights of shape `[..., N]`.
        mixing (torch.Tensor, Optional):
            Residual mixing matrices of shape `[..., N, N]` for mHC; `None` uses identity mixing for iHC. Default: `None`.

    Returns:
        torch.Tensor:
            Updated streams of shape `[..., N, D]`, accumulated in FP32 and returned in `x.dtype`.
    """
    _validate_inputs(x, branch, post, mixing)
    if branch.shape != (*x.shape[:-2], x.shape[-1]) or post.shape != x.shape[:-1]:
        raise ValueError('branch and post must have shapes [..., D] and [..., N] matching x')
    if mixing is not None and mixing.shape != (*x.shape[:-1], x.shape[-2]):
        raise ValueError('mixing must have shape [..., N, N] matching x')
    return FusedHCFunction.apply(x, post, branch, mixing)
