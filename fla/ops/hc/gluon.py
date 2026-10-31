# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from fla.backends import GluonBackend, register
from fla.ops.hc.fused import hc_fwd


@gluon.jit(do_not_specialize=['T'])
def hc_fwd_kernel(
    x,
    w,
    y,
    m,
    o,
    T,
    N: gl.constexpr,
    D: gl.constexpr,
    BN: gl.constexpr,
    BD: gl.constexpr,
    HAS_Y: gl.constexpr,
    HAS_M: gl.constexpr,
):
    i_t = gl.program_id(0).to(gl.int64)
    i_c = gl.program_id(1).to(gl.int64)
    if HAS_Y:
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 32], [1, 4], [1, 0])
    o_n = gl.arange(0, BN, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    o_d = i_c * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    m_nd = (o_n[:, None] < N) & (o_d[None, :] < D)
    # [BN]
    b_w = gl.load(w + i_t * N + o_n, mask=o_n < N, other=0).to(gl.float32)
    if HAS_Y:
        # [BD]
        b_y = gl.load(y + i_t * D + o_d, mask=o_d < D, other=0).to(gl.float32)
        if HAS_M:
            b_o = gl.full((BN, BD), 0, dtype=gl.float32, layout=layout)
            for i_n in gl.static_range(N):
                # [BD]
                b_x = gl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(gl.float32)
                # [BN], one input stream's weights across output streams
                b_m = gl.load(m + (i_t * N + o_n) * N + i_n, mask=o_n < N, other=0).to(gl.float32)
                # [BN, 1] * [1, BD] -> [BN, BD]
                b_o += b_m[:, None] * b_x[None, :]
        else:
            # [BN, BD]
            b_o = gl.load(x + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(gl.float32)
        b_o += b_w[:, None] * b_y[None, :]
        gl.store(o + (i_t * N + o_n[:, None]) * D + o_d[None, :], b_o, mask=m_nd)
    else:
        # keep the stream reduction in registers to avoid a shared-memory layout conversion
        b_o = gl.full((BD,), 0, dtype=gl.float32, layout=gl.SliceLayout(0, layout))
        for i_n in gl.static_range(N):
            # [BD], one residual stream
            b_x = gl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(gl.float32)
            b_wj = gl.load(w + i_t * N + i_n).to(gl.float32)
            b_o += b_wj * b_x
        gl.store(o + i_t * D + o_d, b_o, mask=o_d < D)


@gluon.jit(do_not_specialize=['T'])
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
    N: gl.constexpr,
    D: gl.constexpr,
    BN: gl.constexpr,
    BD: gl.constexpr,
    ND: gl.constexpr,
    HAS_Y: gl.constexpr,
    HAS_M: gl.constexpr,
):
    i_t = gl.program_id(0).to(gl.int64)
    i_c = gl.program_id(1).to(gl.int64)
    if (not HAS_Y and N <= 4) or (HAS_M and N > 4):
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 32], [1, 4], [1, 0])
    o_n = gl.arange(0, BN, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    o_d = i_c * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    m_nd = (o_n[:, None] < N) & (o_d[None, :] < D)
    # [BN]
    b_w = gl.load(w + i_t * N + o_n, mask=o_n < N, other=0).to(gl.float32)
    if HAS_Y:
        # [BN, BD]
        b_do = gl.load(do + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(gl.float32)
        # [BD]
        b_y = gl.load(y + i_t * D + o_d, mask=o_d < D, other=0).to(gl.float32)
        # [BN, BD] -> [BN]
        b_dw = gl.sum(b_do * b_y[None, :], axis=1)
        # [BN, BD] -> [BD]
        b_dy = gl.sum(b_do * b_w[:, None], axis=0)
        gl.store(dy + i_t * D + o_d, b_dy, mask=o_d < D)
        if HAS_M:
            b_dx = gl.full((BN, BD), 0, dtype=gl.float32, layout=layout)
            for i_n in gl.static_range(N):
                # [BD]
                b_doj = gl.load(do + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(gl.float32)
                # [BN], one output stream's weights across input streams
                b_m = gl.load(m + (i_t * N + i_n) * N + o_n, mask=o_n < N, other=0).to(gl.float32)
                b_dx += b_m[:, None] * b_doj[None, :]
                # [BD]
                b_xj = gl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(gl.float32)
                # [BN], one column of the mixing gradient for this feature tile
                b_dm = gl.sum(b_do * b_xj[None, :], axis=1)
                gl.store(dm_partial + ((i_t * ND + i_c) * N + o_n) * N + i_n, b_dm, mask=o_n < N)
        else:
            b_dx = b_do
    else:
        # [BD]
        b_do = gl.load(do + i_t * D + o_d, mask=o_d < D, other=0).to(gl.float32)
        # [BN, BD]
        b_x = gl.load(x + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(gl.float32)
        b_dx = b_w[:, None] * b_do[None, :]
        # [BN, BD] -> [BN]
        b_dw = gl.sum(b_x * b_do[None, :], axis=1)
    gl.store(dx + (i_t * N + o_n[:, None]) * D + o_d[None, :], b_dx, mask=m_nd)
    gl.store(dw_partial + (i_t * ND + i_c) * N + o_n, b_dw, mask=o_n < N)


@gluon.jit(do_not_specialize=['T'])
def hc_bwd_kernel_reduce(partial, grad, T, C: gl.constexpr, S: gl.constexpr, BC: gl.constexpr, BS: gl.constexpr):
    i_t = gl.program_id(0).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    o_c = gl.arange(0, BC, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    o_s = gl.arange(0, BS, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    # [BC, BS], feature tiles by routing-weight entries
    b_partial = gl.load(
        partial + (i_t * C + o_c[:, None]) * S + o_s[None, :],
        mask=(o_c[:, None] < C) & (o_s[None, :] < S),
        other=0,
    )
    # [BC, BS] -> [BS]
    gl.store(grad + i_t * S + o_s, gl.sum(b_partial, axis=0), mask=o_s < S)


@register(hc_fwd, backend=GluonBackend)
def hc_fwd_gluon(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor | None = None,
    m: torch.Tensor | None = None,
):
    N, D = x.shape[-2:]
    # flatten all leading batch/sequence dimensions into T tokens
    T = x.numel() // (N * D)
    BD = min(256 if y is not None else 1024, triton.next_power_of_2(D))
    ND = triton.cdiv(D, BD)
    # [..., N, D] for post; [..., D] for pre
    o = torch.empty_like(x) if y is not None else x.new_empty((*x.shape[:-2], D))
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
    return o, hc_bwd_gluon


def hc_bwd_gluon(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor | None,
    m: torch.Tensor | None,
    do: torch.Tensor,
):
    N, D = x.shape[-2:]
    T = x.numel() // (N * D)
    BD = 256 if (y is None and N <= 4) or (m is not None and N > 4) else 1024
    BD = min(BD, triton.next_power_of_2(D))
    ND = triton.cdiv(D, BD)
    dx, dw = torch.empty_like(x), torch.empty_like(w)
    dy = torch.empty_like(y) if y is not None else None
    dm = torch.empty_like(m) if m is not None else None
    # [T, ND, N] and [T, ND, N * N], partial gradients along feature tiles
    dw_partial = torch.empty((T, ND, N), device=x.device, dtype=torch.float32)
    dm_partial = torch.empty((T, ND, N * N), device=x.device, dtype=torch.float32) if m is not None else None
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
