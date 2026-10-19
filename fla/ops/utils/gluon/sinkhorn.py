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
from fla.ops.utils.sinkhorn import sinkhorn_fwd as sinkhorn_fwd_dispatch


@gluon.jit(do_not_specialize=['T'])
def sinkhorn_fwd_kernel(
    x,
    o,
    history,
    T,
    N: gl.constexpr,
    BN: gl.constexpr,
    BT: gl.constexpr,
    NUM_ITERS: gl.constexpr,
    SAVE_HISTORY: gl.constexpr,
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
    for step in range(NUM_ITERS):
        b_max = gl.max(b_x, axis=1)
        b_x = b_x - b_max[:, None, :]
        b_x = gl.where(m_ij, b_x - gl.log(gl.sum(gl.exp(b_x), axis=1))[:, None, :], float('-inf'))
        if SAVE_HISTORY:
            gl.store(history + (step.to(gl.int64) * 2 * T * N * N) + o_ij, gl.exp(b_x), mask=m_ij)
        b_max = gl.max(b_x, axis=2)
        b_x = b_x - b_max[:, :, None]
        b_x = gl.where(m_ij, b_x - gl.log(gl.sum(gl.exp(b_x), axis=2))[:, :, None], float('-inf'))
        if SAVE_HISTORY:
            gl.store(history + ((step.to(gl.int64) * 2 + 1) * T * N * N) + o_ij, gl.exp(b_x), mask=m_ij)
    gl.store(o + o_ij, gl.exp(b_x), mask=m_ij)


@gluon.jit(do_not_specialize=['T'])
def sinkhorn_bwd_kernel(
    o,
    history,
    do,
    dx,
    T,
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
    b_dx = gl.load(do + o_ij, mask=m_ij, other=0).to(gl.float32) * gl.load(o + o_ij, mask=m_ij, other=0)
    for step in range(NUM_ITERS - 1, -1, -1):
        b_probability = gl.load(history + ((step.to(gl.int64) * 2 + 1) * T * N * N) + o_ij, mask=m_ij, other=0)
        b_dx = b_dx - b_probability * gl.sum(b_dx, axis=2)[:, :, None]
        b_probability = gl.load(history + (step.to(gl.int64) * 2 * T * N * N) + o_ij, mask=m_ij, other=0)
        b_dx = b_dx - b_probability * gl.sum(b_dx, axis=1)[:, None, :]
    gl.store(dx + o_ij, b_dx, mask=m_ij)


def sinkhorn_fwd(logits, num_iters, save_history):
    N = logits.shape[-1]
    T = logits.numel() // (N * N)
    out = torch.empty_like(logits, dtype=torch.float32)
    history = torch.empty((2 * num_iters, T, N, N), device=logits.device, dtype=torch.float32) if save_history else None
    BT = 1 if T < 128 else max(4, 128 // min(32, triton.next_power_of_2(N) ** 2))
    if T:
        sinkhorn_fwd_kernel[(triton.cdiv(T, BT),)](
            x=logits,
            o=out,
            history=history,
            T=T,
            N=N,
            BN=triton.next_power_of_2(N),
            BT=BT,
            NUM_ITERS=num_iters,
            SAVE_HISTORY=save_history,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out, history, sinkhorn_bwd


def sinkhorn_bwd(out, history, dout, num_iters, dtype):
    N, T = out.shape[-1], out.numel() // out.shape[-1] ** 2
    dx = torch.empty_like(out, dtype=dtype)
    BT = 1 if T < 128 else max(4, 128 // min(32, triton.next_power_of_2(N) ** 2))
    if T:
        sinkhorn_bwd_kernel[(triton.cdiv(T, BT),)](
            o=out,
            history=history,
            do=dout,
            dx=dx,
            T=T,
            N=N,
            BN=triton.next_power_of_2(N),
            BT=BT,
            NUM_ITERS=num_iters,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return dx


@register(sinkhorn_fwd_dispatch)
class SinkhornGluonBackend(GluonBackend):
    default_enable = False

    def sinkhorn_fwd(self, logits, num_iters, save_history):
        return sinkhorn_fwd(logits=logits, num_iters=num_iters, save_history=save_history)
