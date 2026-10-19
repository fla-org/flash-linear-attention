# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
from torch.autograd.function import once_differentiable
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from fla.backends import GluonBackend, register
from fla.ops.hc.fused import (
    _validate_hc_post,
    _validate_hc_pre,
    fused_hc_post,
    fused_hc_pre,
)
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@gluon.jit(do_not_specialize=['T'])
def hc_fwd_kernel(
    x,
    weight,
    branch,
    mixing,
    o,
    T,
    N: gl.constexpr,
    D: gl.constexpr,
    BN: gl.constexpr,
    BD: gl.constexpr,
    HAS_BRANCH: gl.constexpr,
    HAS_MIXING: gl.constexpr,
):
    i_t = gl.program_id(0).to(gl.int64)
    i_c = gl.program_id(1).to(gl.int64)
    if HAS_BRANCH:
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 32], [1, 4], [1, 0])
    o_n = gl.arange(0, BN, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    o_d = i_c * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    m_nd = (o_n[:, None] < N) & (o_d[None, :] < D)
    b_weight = gl.load(weight + i_t * N + o_n, mask=o_n < N, other=0).to(gl.float32)
    if HAS_BRANCH:
        b_branch = gl.load(branch + i_t * D + o_d, mask=o_d < D, other=0).to(gl.float32)
        if HAS_MIXING:
            b_o = gl.full((BN, BD), 0, dtype=gl.float32, layout=layout)
            for i_n in gl.static_range(N):
                b_x = gl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(gl.float32)
                b_mixing = gl.load(mixing + (i_t * N + o_n) * N + i_n, mask=o_n < N, other=0).to(gl.float32)
                b_o += b_mixing[:, None] * b_x[None, :]
        else:
            b_o = gl.load(x + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(gl.float32)
        b_o += b_weight[:, None] * b_branch[None, :]
        gl.store(o + (i_t * N + o_n[:, None]) * D + o_d[None, :], b_o, mask=m_nd)
    else:
        # keep the stream reduction in registers to avoid a shared-memory layout conversion
        b_o = gl.full((BD,), 0, dtype=gl.float32, layout=gl.SliceLayout(0, layout))
        for i_n in gl.static_range(N):
            b_x = gl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(gl.float32)
            b_weight_j = gl.load(weight + i_t * N + i_n).to(gl.float32)
            b_o += b_weight_j * b_x
        gl.store(o + i_t * D + o_d, b_o, mask=o_d < D)


@gluon.jit(do_not_specialize=['T'])
def hc_bwd_kernel(
    x,
    weight,
    branch,
    mixing,
    do,
    dx,
    dbranch,
    dweight_partial,
    dmixing_partial,
    T,
    N: gl.constexpr,
    D: gl.constexpr,
    BN: gl.constexpr,
    BD: gl.constexpr,
    CD: gl.constexpr,
    HAS_BRANCH: gl.constexpr,
    HAS_MIXING: gl.constexpr,
):
    i_t = gl.program_id(0).to(gl.int64)
    i_c = gl.program_id(1).to(gl.int64)
    if (not HAS_BRANCH and N <= 4) or (HAS_MIXING and N > 4):
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 32], [1, 4], [1, 0])
    o_n = gl.arange(0, BN, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    o_d = i_c * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    m_nd = (o_n[:, None] < N) & (o_d[None, :] < D)
    b_weight = gl.load(weight + i_t * N + o_n, mask=o_n < N, other=0).to(gl.float32)
    if HAS_BRANCH:
        b_do = gl.load(do + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(gl.float32)
        b_branch = gl.load(branch + i_t * D + o_d, mask=o_d < D, other=0).to(gl.float32)
        b_dweight = gl.sum(b_do * b_branch[None, :], axis=1)
        b_dbranch = gl.sum(b_do * b_weight[:, None], axis=0)
        gl.store(dbranch + i_t * D + o_d, b_dbranch, mask=o_d < D)
        if HAS_MIXING:
            b_dx = gl.full((BN, BD), 0, dtype=gl.float32, layout=layout)
            for i_n in gl.static_range(N):
                b_do_j = gl.load(do + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(gl.float32)
                b_mixing = gl.load(mixing + (i_t * N + i_n) * N + o_n, mask=o_n < N, other=0).to(gl.float32)
                b_dx += b_mixing[:, None] * b_do_j[None, :]
                b_x_j = gl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(gl.float32)
                b_dmixing = gl.sum(b_do * b_x_j[None, :], axis=1)
                gl.store(dmixing_partial + ((i_t * CD + i_c) * N + o_n) * N + i_n, b_dmixing, mask=o_n < N)
        else:
            b_dx = b_do
    else:
        b_do = gl.load(do + i_t * D + o_d, mask=o_d < D, other=0).to(gl.float32)
        b_x = gl.load(x + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(gl.float32)
        b_dx = b_weight[:, None] * b_do[None, :]
        b_dweight = gl.sum(b_x * b_do[None, :], axis=1)
    gl.store(dx + (i_t * N + o_n[:, None]) * D + o_d[None, :], b_dx, mask=m_nd)
    gl.store(dweight_partial + (i_t * CD + i_c) * N + o_n, b_dweight, mask=o_n < N)


@gluon.jit(do_not_specialize=['T'])
def hc_bwd_kernel_reduce(partial, grad, T, C: gl.constexpr, S: gl.constexpr, BC: gl.constexpr, BS: gl.constexpr):
    i_t = gl.program_id(0).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    o_c = gl.arange(0, BC, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    o_s = gl.arange(0, BS, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    b_partial = gl.load(
        partial + (i_t * C + o_c[:, None]) * S + o_s[None, :],
        mask=(o_c[:, None] < C) & (o_s[None, :] < S),
        other=0,
    )
    gl.store(grad + i_t * S + o_s, gl.sum(b_partial, axis=0), mask=o_s < S)


def hc_fwd(x, weight, branch=None, mixing=None):
    N, D = x.shape[-2:]
    T = x.numel() // (N * D)
    BD = min(256 if branch is not None else 1024, triton.next_power_of_2(D))
    CD = triton.cdiv(D, BD)
    out = torch.empty_like(x) if branch is not None else x.new_empty((*x.shape[:-2], D))
    if T:
        hc_fwd_kernel[(T, CD)](
            x=x,
            weight=weight,
            branch=branch,
            mixing=mixing,
            o=out,
            T=T,
            N=N,
            D=D,
            BN=triton.next_power_of_2(N),
            BD=BD,
            HAS_BRANCH=branch is not None,
            HAS_MIXING=mixing is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


def hc_bwd(x, weight, branch, mixing, dout):
    N, D = x.shape[-2:]
    T = x.numel() // (N * D)
    BD = 256 if (branch is None and N <= 4) or (mixing is not None and N > 4) else 1024
    BD = min(BD, triton.next_power_of_2(D))
    CD = triton.cdiv(D, BD)
    dx, dweight = torch.empty_like(x), torch.empty_like(weight)
    dbranch = torch.empty_like(branch) if branch is not None else None
    dmixing = torch.empty_like(mixing) if mixing is not None else None
    dweight_partial = torch.empty((T, CD, N), device=x.device, dtype=torch.float32)
    dmixing_partial = torch.empty((T, CD, N * N), device=x.device, dtype=torch.float32) if mixing is not None else None
    if T:
        hc_bwd_kernel[(T, CD)](
            x=x,
            weight=weight,
            branch=branch,
            mixing=mixing,
            do=dout,
            dx=dx,
            dbranch=dbranch,
            dweight_partial=dweight_partial,
            dmixing_partial=dmixing_partial,
            T=T,
            N=N,
            D=D,
            BN=triton.next_power_of_2(N),
            BD=BD,
            CD=CD,
            HAS_BRANCH=branch is not None,
            HAS_MIXING=mixing is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
        for partial, grad, size in ((dweight_partial, dweight, N), (dmixing_partial, dmixing, N * N)):
            if grad is not None:
                hc_bwd_kernel_reduce[(T,)](
                    partial=partial,
                    grad=grad,
                    T=T,
                    C=CD,
                    S=size,
                    BC=triton.next_power_of_2(CD),
                    BS=triton.next_power_of_2(size),
                    num_warps=4,
                    enable_fp_fusion=False,
                )
    return dx, dweight, dbranch, dmixing


class FusedHCGluonFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, weight, branch, mixing):
        out = hc_fwd(x=x, weight=weight, branch=branch, mixing=mixing)
        ctx.save_for_backward(x, weight, branch, mixing)
        return out

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    @once_differentiable
    def backward(ctx, dout):
        x, weight, branch, mixing = ctx.saved_tensors
        return hc_bwd(x=x, weight=weight, branch=branch, mixing=mixing, dout=dout)


@register(fused_hc_pre, env_var='FLA_HC_GLUON')
@register(fused_hc_post, env_var='FLA_HC_GLUON')
class HCGluonBackend(GluonBackend):
    default_enable = False

    def fused_hc_pre(self, x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        _validate_hc_pre(x=x, pre=pre)
        return FusedHCGluonFunction.apply(x, pre, None, None)

    def fused_hc_post(
        self,
        x: torch.Tensor,
        branch: torch.Tensor,
        post: torch.Tensor,
        mixing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_hc_post(x=x, branch=branch, post=post, mixing=mixing)
        return FusedHCGluonFunction.apply(x, post, branch, mixing)
