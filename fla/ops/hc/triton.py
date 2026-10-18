# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def hc_fwd_kernel(
    x,
    weight,
    branch,
    mixing,
    o,
    T,
    N: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    HAS_BRANCH: tl.constexpr,
    HAS_MIXING: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    o_n = tl.arange(0, BN).to(tl.int64)
    o_d = i_c * BD + tl.arange(0, BD).to(tl.int64)
    m_nd = (o_n[:, None] < N) & (o_d[None, :] < D)
    b_weight = tl.load(weight + i_t * N + o_n, mask=o_n < N, other=0).to(tl.float32)
    if HAS_BRANCH:
        b_branch = tl.load(branch + i_t * D + o_d, mask=o_d < D, other=0).to(tl.float32)
        if HAS_MIXING:
            b_o = tl.full((BN, BD), 0, dtype=tl.float32)
            for i_n in tl.static_range(N):
                b_x = tl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(tl.float32)
                b_mixing = tl.load(mixing + (i_t * N + o_n) * N + i_n, mask=o_n < N, other=0).to(tl.float32)
                b_o += b_mixing[:, None] * b_x[None, :]
        else:
            b_o = tl.load(x + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(tl.float32)
        b_o += b_weight[:, None] * b_branch[None, :]
        tl.store(o + (i_t * N + o_n[:, None]) * D + o_d[None, :], b_o, mask=m_nd)
    else:
        # keep the stream reduction in registers to avoid a shared-memory layout conversion
        b_o = tl.full((BD,), 0, dtype=tl.float32)
        for i_n in tl.static_range(N):
            b_x = tl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(tl.float32)
            b_weight_j = tl.load(weight + i_t * N + i_n).to(tl.float32)
            b_o += b_weight_j * b_x
        tl.store(o + i_t * D + o_d, b_o, mask=o_d < D)


@triton.jit(do_not_specialize=['T'])
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
    N: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    CD: tl.constexpr,
    HAS_BRANCH: tl.constexpr,
    HAS_MIXING: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    o_n = tl.arange(0, BN).to(tl.int64)
    o_d = i_c * BD + tl.arange(0, BD).to(tl.int64)
    m_nd = (o_n[:, None] < N) & (o_d[None, :] < D)
    b_weight = tl.load(weight + i_t * N + o_n, mask=o_n < N, other=0).to(tl.float32)
    if HAS_BRANCH:
        b_do = tl.load(do + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(tl.float32)
        b_branch = tl.load(branch + i_t * D + o_d, mask=o_d < D, other=0).to(tl.float32)
        b_dweight = tl.sum(b_do * b_branch[None, :], axis=1)
        b_dbranch = tl.sum(b_do * b_weight[:, None], axis=0)
        tl.store(dbranch + i_t * D + o_d, b_dbranch, mask=o_d < D)
        if HAS_MIXING:
            b_dx = tl.full((BN, BD), 0, dtype=tl.float32)
            for i_n in tl.static_range(N):
                b_do_j = tl.load(do + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(tl.float32)
                b_mixing = tl.load(mixing + (i_t * N + i_n) * N + o_n, mask=o_n < N, other=0).to(tl.float32)
                b_dx += b_mixing[:, None] * b_do_j[None, :]
                b_x_j = tl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(tl.float32)
                b_dmixing = tl.sum(b_do * b_x_j[None, :], axis=1)
                tl.store(dmixing_partial + ((i_t * CD + i_c) * N + o_n) * N + i_n, b_dmixing, mask=o_n < N)
        else:
            b_dx = b_do
    else:
        b_do = tl.load(do + i_t * D + o_d, mask=o_d < D, other=0).to(tl.float32)
        b_x = tl.load(x + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(tl.float32)
        b_dx = b_weight[:, None] * b_do[None, :]
        b_dweight = tl.sum(b_x * b_do[None, :], axis=1)
    tl.store(dx + (i_t * N + o_n[:, None]) * D + o_d[None, :], b_dx, mask=m_nd)
    tl.store(dweight_partial + (i_t * CD + i_c) * N + o_n, b_dweight, mask=o_n < N)


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


@triton.jit(do_not_specialize=['T'])
def sinkhorn_fwd_kernel(
    x,
    o,
    history,
    T,
    N: tl.constexpr,
    BN: tl.constexpr,
    NUM_ITERS: tl.constexpr,
    SAVE_HISTORY: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    o_i = tl.arange(0, BN).to(tl.int64)
    o_j = tl.arange(0, BN).to(tl.int64)
    o_ij = (i_t * N + o_i[:, None]) * N + o_j[None, :]
    m_ij = (o_i[:, None] < N) & (o_j[None, :] < N)
    b_x = tl.load(x + o_ij, mask=m_ij, other=float('-inf')).to(tl.float32)
    for step in range(NUM_ITERS):
        b_max = tl.max(b_x, axis=0)
        b_x = b_x - b_max[None, :]
        b_x = tl.where(m_ij, b_x - tl.log(tl.sum(tl.exp(b_x), axis=0))[None, :], float('-inf'))
        if SAVE_HISTORY:
            tl.store(history + (step.to(tl.int64) * 2 * T * N * N) + o_ij, tl.exp(b_x), mask=m_ij)
        b_max = tl.max(b_x, axis=1)
        b_x = b_x - b_max[:, None]
        b_x = tl.where(m_ij, b_x - tl.log(tl.sum(tl.exp(b_x), axis=1))[:, None], float('-inf'))
        if SAVE_HISTORY:
            tl.store(history + ((step.to(tl.int64) * 2 + 1) * T * N * N) + o_ij, tl.exp(b_x), mask=m_ij)
    tl.store(o + o_ij, tl.exp(b_x), mask=m_ij)


@triton.jit(do_not_specialize=['T'])
def sinkhorn_bwd_kernel(o, history, do, dx, T, N: tl.constexpr, BN: tl.constexpr, NUM_ITERS: tl.constexpr):
    i_t = tl.program_id(0).to(tl.int64)
    o_i = tl.arange(0, BN).to(tl.int64)
    o_j = tl.arange(0, BN).to(tl.int64)
    o_ij = (i_t * N + o_i[:, None]) * N + o_j[None, :]
    m_ij = (o_i[:, None] < N) & (o_j[None, :] < N)
    b_dx = tl.load(do + o_ij, mask=m_ij, other=0).to(tl.float32) * tl.load(o + o_ij, mask=m_ij, other=0)
    for step in range(NUM_ITERS - 1, -1, -1):
        b_probability = tl.load(history + ((step.to(tl.int64) * 2 + 1) * T * N * N) + o_ij, mask=m_ij, other=0)
        b_dx = b_dx - b_probability * tl.sum(b_dx, axis=1)[:, None]
        b_probability = tl.load(history + (step.to(tl.int64) * 2 * T * N * N) + o_ij, mask=m_ij, other=0)
        b_dx = b_dx - b_probability * tl.sum(b_dx, axis=0)[None, :]
    tl.store(dx + o_ij, b_dx, mask=m_ij)
