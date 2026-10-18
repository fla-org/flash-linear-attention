# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


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
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1] if HAS_BRANCH else [1, 4], [1, 0])
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
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
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


@gluon.jit(do_not_specialize=['T'])
def sinkhorn_fwd_kernel(
    x,
    o,
    history,
    T,
    N: gl.constexpr,
    BN: gl.constexpr,
    NUM_ITERS: gl.constexpr,
    SAVE_HISTORY: gl.constexpr,
):
    i_t = gl.program_id(0).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0])
    o_i = gl.arange(0, BN, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    o_j = gl.arange(0, BN, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    o_ij = (i_t * N + o_i[:, None]) * N + o_j[None, :]
    m_ij = (o_i[:, None] < N) & (o_j[None, :] < N)
    b_x = gl.load(x + o_ij, mask=m_ij, other=float('-inf')).to(gl.float32)
    for step in range(NUM_ITERS):
        b_max = gl.max(b_x, axis=0)
        b_x = b_x - b_max[None, :]
        b_x = gl.where(m_ij, b_x - gl.log(gl.sum(gl.exp(b_x), axis=0))[None, :], float('-inf'))
        if SAVE_HISTORY:
            gl.store(history + (step.to(gl.int64) * 2 * T * N * N) + o_ij, gl.exp(b_x), mask=m_ij)
        b_max = gl.max(b_x, axis=1)
        b_x = b_x - b_max[:, None]
        b_x = gl.where(m_ij, b_x - gl.log(gl.sum(gl.exp(b_x), axis=1))[:, None], float('-inf'))
        if SAVE_HISTORY:
            gl.store(history + ((step.to(gl.int64) * 2 + 1) * T * N * N) + o_ij, gl.exp(b_x), mask=m_ij)
    gl.store(o + o_ij, gl.exp(b_x), mask=m_ij)


@gluon.jit(do_not_specialize=['T'])
def sinkhorn_bwd_kernel(o, history, do, dx, T, N: gl.constexpr, BN: gl.constexpr, NUM_ITERS: gl.constexpr):
    i_t = gl.program_id(0).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0])
    o_i = gl.arange(0, BN, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    o_j = gl.arange(0, BN, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    o_ij = (i_t * N + o_i[:, None]) * N + o_j[None, :]
    m_ij = (o_i[:, None] < N) & (o_j[None, :] < N)
    b_dx = gl.load(do + o_ij, mask=m_ij, other=0).to(gl.float32) * gl.load(o + o_ij, mask=m_ij, other=0)
    for step in range(NUM_ITERS - 1, -1, -1):
        b_probability = gl.load(history + ((step.to(gl.int64) * 2 + 1) * T * N * N) + o_ij, mask=m_ij, other=0)
        b_dx = b_dx - b_probability * gl.sum(b_dx, axis=1)[:, None]
        b_probability = gl.load(history + (step.to(gl.int64) * 2 * T * N * N) + o_ij, mask=m_ij, other=0)
        b_dx = b_dx - b_probability * gl.sum(b_dx, axis=0)[None, :]
    gl.store(dx + o_ij, b_dx, mask=m_ij)
