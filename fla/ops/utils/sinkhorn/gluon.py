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
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.sinkhorn.ops import sinkhorn_fwd


@gluon.jit(do_not_specialize=['T'])
def sinkhorn_fwd_kernel(
    x,
    o,
    T,
    scale: gl.constexpr,
    N: gl.constexpr,
    BN: gl.constexpr,
    BT: gl.constexpr,
    NUM_ITERS: gl.constexpr,
):
    # independent matrices share a warp; row and column reductions never cross warps
    RI: gl.constexpr = 4 if BN >= 4 else BN
    RJ: gl.constexpr = 8 if BN >= 8 else BN
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [32 // (RI * RJ), RI, RJ], [4, 1, 1], [2, 1, 0])
    o_t = gl.program_id(0).to(gl.int64) * BT + gl.arange(0, BT, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    o_i = gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout))).to(gl.int64)
    o_j = gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout))).to(gl.int64)
    o_ij = (o_t[:, None, None] * N + o_i[None, :, None]) * N + o_j[None, None, :]
    m_ij = (o_t[:, None, None] < T) & (o_i[None, :, None] < N) & (o_j[None, None, :] < N)
    b_x = gl.load(x + o_ij, mask=m_ij, other=float('-inf')).to(gl.float32)
    b_x = (b_x - gl.max(b_x, axis=1)[:, None, :]) * scale
    for _ in range(NUM_ITERS):
        b_max = gl.max(b_x, axis=1)
        b_x = b_x - b_max[:, None, :]
        b_x = gl.where(m_ij, b_x - gl.log2(gl.sum(gl.exp2(b_x), axis=1))[:, None, :], float('-inf'))
        b_max = gl.max(b_x, axis=2)
        b_x = b_x - b_max[:, :, None]
        b_x = gl.where(m_ij, b_x - gl.log2(gl.sum(gl.exp2(b_x), axis=2))[:, :, None], float('-inf'))
    gl.store(o + o_ij, gl.exp2(b_x), mask=m_ij)


@gluon.jit(do_not_specialize=['T'])
def sinkhorn_bwd_kernel(
    x,
    do,
    dx,
    T,
    scale: gl.constexpr,
    N: gl.constexpr,
    BN: gl.constexpr,
    BT: gl.constexpr,
    NUM_ITERS: gl.constexpr,
):
    # independent matrices share a warp; row and column reductions never cross warps
    RI: gl.constexpr = 4 if BN >= 4 else BN
    RJ: gl.constexpr = 8 if BN >= 8 else BN
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [32 // (RI * RJ), RI, RJ], [4, 1, 1], [2, 1, 0])
    o_t = gl.program_id(0).to(gl.int64) * BT + gl.arange(0, BT, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    o_i = gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout))).to(gl.int64)
    o_j = gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout))).to(gl.int64)
    o_ij = (o_t[:, None, None] * N + o_i[None, :, None]) * N + o_j[None, None, :]
    m_ij = (o_t[:, None, None] < T) & (o_i[None, :, None] < N) & (o_j[None, None, :] < N)
    b_input = gl.load(x + o_ij, mask=m_ij, other=float('-inf')).to(gl.float32)
    b_input = (b_input - gl.max(b_input, axis=1)[:, None, :]) * scale
    b_dx = gl.load(do + o_ij, mask=m_ij, other=0).to(gl.float32)
    # recompute earlier blocks to keep the shared probability cache within 40 KiB.
    BI: gl.constexpr = NUM_ITERS if NUM_ITERS < 20 else 20
    b_ps = gl.allocate_shared_memory(gl.float32, [2 * BI, BT, BN, BN], gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0]))
    for i_b in range(triton.cdiv(NUM_ITERS, BI) - 1, -1, -1):
        b_x = b_input
        for step in range(gl.minimum((i_b + 1) * BI, NUM_ITERS)):
            b_x = b_x - gl.max(b_x, axis=1)[:, None, :]
            b_x = gl.where(m_ij, b_x - gl.log2(gl.sum(gl.exp2(b_x), axis=1))[:, None, :], float('-inf'))
            if step >= i_b * BI:
                b_ps.index(2 * (step - i_b * BI)).store(gl.exp2(b_x))
            b_x = b_x - gl.max(b_x, axis=2)[:, :, None]
            b_x = gl.where(m_ij, b_x - gl.log2(gl.sum(gl.exp2(b_x), axis=2))[:, :, None], float('-inf'))
            if step >= i_b * BI:
                b_ps.index(2 * (step - i_b * BI) + 1).store(gl.exp2(b_x))
        if i_b == triton.cdiv(NUM_ITERS, BI) - 1:
            b_dx = b_dx * gl.exp2(b_x)
        for step in range(gl.minimum(BI, NUM_ITERS - i_b * BI) - 1, -1, -1):
            b_p = b_ps.index(2 * step + 1).load(layout)
            b_dx = b_dx - b_p * gl.sum(b_dx, axis=2)[:, :, None]
            b_p = b_ps.index(2 * step).load(layout)
            b_dx = b_dx - b_p * gl.sum(b_dx, axis=1)[:, None, :]
    gl.store(dx + o_ij, b_dx, mask=m_ij)


@register(sinkhorn_fwd, backend=GluonBackend)
def sinkhorn_fwd_gluon(logits: torch.Tensor, num_iters: int):
    N = logits.shape[-1]
    T = logits.numel() // (N * N)
    out = torch.empty_like(logits, dtype=torch.float32)
    BT = 1 if T < 128 else max(4, 128 // min(32, triton.next_power_of_2(N) ** 2))
    if T:
        sinkhorn_fwd_kernel[(triton.cdiv(T, BT),)](
            x=logits,
            o=out,
            T=T,
            scale=RCP_LN2,
            N=N,
            BN=triton.next_power_of_2(N),
            BT=BT,
            NUM_ITERS=num_iters,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out, sinkhorn_bwd_gluon


def sinkhorn_bwd_gluon(logits: torch.Tensor, dout: torch.Tensor, num_iters: int) -> torch.Tensor:
    N = logits.shape[-1]
    T = logits.numel() // (N * N)
    dx = torch.empty_like(logits)
    BT = 1 if T < 128 else max(4, 128 // min(32, triton.next_power_of_2(N) ** 2))
    if T:
        sinkhorn_bwd_kernel[(triton.cdiv(T, BT),)](
            x=logits,
            do=dout,
            dx=dx,
            T=T,
            scale=RCP_LN2,
            N=N,
            BN=triton.next_power_of_2(N),
            BT=BT,
            NUM_ITERS=num_iters,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return dx
