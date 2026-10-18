# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit(do_not_specialize=['T'])
def hyper_connection_fwd_kernel(X, W, B, M, O, T, N: gl.constexpr, D: gl.constexpr,
                                BN: gl.constexpr, BD: gl.constexpr, POST: gl.constexpr, MIX: gl.constexpr):
    t = gl.program_id(0).to(gl.int64)
    c = gl.program_id(1).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1] if POST else [1, 4], [1, 0])
    n = gl.arange(0, BN, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    d = c * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    mask = (n[:, None] < N) & (d[None, :] < D)
    w = gl.load(W + t * N + n, n < N, other=0).to(gl.float32)
    if POST:
        b = gl.load(B + t * D + d, d < D, other=0).to(gl.float32)
        if MIX:
            value = gl.full((BN, BD), 0, gl.float32, layout=layout)
            for j in gl.static_range(N):
                x = gl.load(X + (t * N + j) * D + d, d < D, other=0).to(gl.float32)
                m = gl.load(M + (t * N + n) * N + j, n < N, other=0).to(gl.float32)
                value += m[:, None] * x[None, :]
        else:
            value = gl.load(X + (t * N + n[:, None]) * D + d[None, :], mask, other=0).to(gl.float32)
        value += w[:, None] * b[None, :]
        gl.store(O + (t * N + n[:, None]) * D + d[None, :], value, mask)
    else:
        # keep the stream reduction in registers to avoid a shared-memory layout conversion
        value = gl.full((BD,), 0, gl.float32, layout=gl.SliceLayout(0, layout))
        for j in gl.static_range(N):
            x = gl.load(X + (t * N + j) * D + d, d < D, other=0).to(gl.float32)
            weight = gl.load(W + t * N + j).to(gl.float32)
            value += weight * x
        gl.store(O + t * D + d, value, d < D)


@gluon.jit(do_not_specialize=['T'])
def hyper_connection_bwd_kernel(X, W, B, M, DO, DX, DB, PW, PM, T, N: gl.constexpr, D: gl.constexpr,
                                BN: gl.constexpr, BD: gl.constexpr, CD: gl.constexpr,
                                POST: gl.constexpr, MIX: gl.constexpr):
    t = gl.program_id(0).to(gl.int64)
    c = gl.program_id(1).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    n = gl.arange(0, BN, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    d = c * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    mask = (n[:, None] < N) & (d[None, :] < D)
    w = gl.load(W + t * N + n, n < N, other=0).to(gl.float32)
    if POST:
        do = gl.load(DO + (t * N + n[:, None]) * D + d[None, :], mask, other=0).to(gl.float32)
        b = gl.load(B + t * D + d, d < D, other=0).to(gl.float32)
        dw = gl.sum(do * b[None, :], axis=1)
        db = gl.sum(do * w[:, None], axis=0)
        gl.store(DB + t * D + d, db, d < D)
        if MIX:
            dx = gl.full((BN, BD), 0, gl.float32, layout=layout)
            for j in gl.static_range(N):
                do_j = gl.load(DO + (t * N + j) * D + d, d < D, other=0).to(gl.float32)
                m = gl.load(M + (t * N + j) * N + n, n < N, other=0).to(gl.float32)
                dx += m[:, None] * do_j[None, :]
                x_j = gl.load(X + (t * N + j) * D + d, d < D, other=0).to(gl.float32)
                dm = gl.sum(do * x_j[None, :], axis=1)
                gl.store(PM + ((t * CD + c) * N + n) * N + j, dm, n < N)
        else:
            dx = do
    else:
        do = gl.load(DO + t * D + d, d < D, other=0).to(gl.float32)
        x = gl.load(X + (t * N + n[:, None]) * D + d[None, :], mask, other=0).to(gl.float32)
        dx = w[:, None] * do[None, :]
        dw = gl.sum(x * do[None, :], axis=1)
    gl.store(DX + (t * N + n[:, None]) * D + d[None, :], dx, mask)
    gl.store(PW + (t * CD + c) * N + n, dw, n < N)


@gluon.jit(do_not_specialize=['T'])
def hyper_connection_reduce_kernel(P, G, T, C: gl.constexpr, S: gl.constexpr, BC: gl.constexpr, BS: gl.constexpr):
    t = gl.program_id(0).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    c = gl.arange(0, BC, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    s = gl.arange(0, BS, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    p = gl.load(P + (t * C + c[:, None]) * S + s[None, :], (c[:, None] < C) & (s[None, :] < S), other=0)
    gl.store(G + t * S + s, gl.sum(p, axis=0), s < S)


@gluon.jit(do_not_specialize=['T'])
def sinkhorn_fwd_kernel(X, O, H, T, N: gl.constexpr, BN: gl.constexpr, ITERS: gl.constexpr, SAVE: gl.constexpr):
    t = gl.program_id(0).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0])
    i = gl.arange(0, BN, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    j = gl.arange(0, BN, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    offsets = (t * N + i[:, None]) * N + j[None, :]
    mask = (i[:, None] < N) & (j[None, :] < N)
    value = gl.load(X + offsets, mask, other=float('-inf')).to(gl.float32)
    for step in range(ITERS):
        maximum = gl.max(value, axis=0)
        value = value - maximum[None, :]
        value = gl.where(mask, value - gl.log(gl.sum(gl.exp(value), axis=0))[None, :], float('-inf'))
        if SAVE:
            gl.store(H + (step.to(gl.int64) * 2 * T * N * N) + offsets, gl.exp(value), mask)
        maximum = gl.max(value, axis=1)
        value = value - maximum[:, None]
        value = gl.where(mask, value - gl.log(gl.sum(gl.exp(value), axis=1))[:, None], float('-inf'))
        if SAVE:
            gl.store(H + ((step.to(gl.int64) * 2 + 1) * T * N * N) + offsets, gl.exp(value), mask)
    gl.store(O + offsets, gl.exp(value), mask)


@gluon.jit(do_not_specialize=['T'])
def sinkhorn_bwd_kernel(O, H, DO, DX, T, N: gl.constexpr, BN: gl.constexpr, ITERS: gl.constexpr):
    t = gl.program_id(0).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0])
    i = gl.arange(0, BN, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    j = gl.arange(0, BN, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    offsets = (t * N + i[:, None]) * N + j[None, :]
    mask = (i[:, None] < N) & (j[None, :] < N)
    value = gl.load(DO + offsets, mask, other=0).to(gl.float32) * gl.load(O + offsets, mask, other=0)
    for step in range(ITERS - 1, -1, -1):
        probability = gl.load(H + ((step.to(gl.int64) * 2 + 1) * T * N * N) + offsets, mask, other=0)
        value = value - probability * gl.sum(value, axis=1)[:, None]
        probability = gl.load(H + (step.to(gl.int64) * 2 * T * N * N) + offsets, mask, other=0)
        value = value - probability * gl.sum(value, axis=0)[None, :]
    gl.store(DX + offsets, value, mask)
