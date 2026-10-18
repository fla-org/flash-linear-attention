# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def hyper_connection_fwd_kernel(X, W, B, M, O, T, N: tl.constexpr, D: tl.constexpr,
                                BN: tl.constexpr, BD: tl.constexpr, POST: tl.constexpr, MIX: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    n = tl.arange(0, BN).to(tl.int64)
    d = c * BD + tl.arange(0, BD).to(tl.int64)
    mask = (n[:, None] < N) & (d[None, :] < D)
    w = tl.load(W + t * N + n, n < N, other=0).to(tl.float32)
    if POST:
        b = tl.load(B + t * D + d, d < D, other=0).to(tl.float32)
        if MIX:
            value = tl.full((BN, BD), 0, tl.float32)
            for j in tl.static_range(N):
                x = tl.load(X + (t * N + j) * D + d, d < D, other=0).to(tl.float32)
                m = tl.load(M + (t * N + n) * N + j, n < N, other=0).to(tl.float32)
                value += m[:, None] * x[None, :]
        else:
            value = tl.load(X + (t * N + n[:, None]) * D + d[None, :], mask, other=0).to(tl.float32)
        value += w[:, None] * b[None, :]
        tl.store(O + (t * N + n[:, None]) * D + d[None, :], value, mask)
    else:
        x = tl.load(X + (t * N + n[:, None]) * D + d[None, :], mask, other=0).to(tl.float32)
        value = tl.sum(x * w[:, None], axis=0)
        tl.store(O + t * D + d, value, d < D)


@triton.jit(do_not_specialize=['T'])
def hyper_connection_bwd_kernel(X, W, B, M, DO, DX, DB, PW, PM, T, N: tl.constexpr, D: tl.constexpr,
                                BN: tl.constexpr, BD: tl.constexpr, CD: tl.constexpr,
                                POST: tl.constexpr, MIX: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    n = tl.arange(0, BN).to(tl.int64)
    d = c * BD + tl.arange(0, BD).to(tl.int64)
    mask = (n[:, None] < N) & (d[None, :] < D)
    w = tl.load(W + t * N + n, n < N, other=0).to(tl.float32)
    if POST:
        do = tl.load(DO + (t * N + n[:, None]) * D + d[None, :], mask, other=0).to(tl.float32)
        b = tl.load(B + t * D + d, d < D, other=0).to(tl.float32)
        dw = tl.sum(do * b[None, :], axis=1)
        db = tl.sum(do * w[:, None], axis=0)
        tl.store(DB + t * D + d, db, d < D)
        if MIX:
            dx = tl.full((BN, BD), 0, tl.float32)
            for j in tl.static_range(N):
                do_j = tl.load(DO + (t * N + j) * D + d, d < D, other=0).to(tl.float32)
                m = tl.load(M + (t * N + j) * N + n, n < N, other=0).to(tl.float32)
                dx += m[:, None] * do_j[None, :]
                x_j = tl.load(X + (t * N + j) * D + d, d < D, other=0).to(tl.float32)
                dm = tl.sum(do * x_j[None, :], axis=1)
                tl.store(PM + ((t * CD + c) * N + n) * N + j, dm, n < N)
        else:
            dx = do
    else:
        do = tl.load(DO + t * D + d, d < D, other=0).to(tl.float32)
        x = tl.load(X + (t * N + n[:, None]) * D + d[None, :], mask, other=0).to(tl.float32)
        dx = w[:, None] * do[None, :]
        dw = tl.sum(x * do[None, :], axis=1)
    tl.store(DX + (t * N + n[:, None]) * D + d[None, :], dx, mask)
    tl.store(PW + (t * CD + c) * N + n, dw, n < N)


@triton.jit(do_not_specialize=['T'])
def hyper_connection_reduce_kernel(P, G, T, C: tl.constexpr, S: tl.constexpr, BC: tl.constexpr, BS: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BC).to(tl.int64)
    s = tl.arange(0, BS).to(tl.int64)
    p = tl.load(P + (t * C + c[:, None]) * S + s[None, :], (c[:, None] < C) & (s[None, :] < S), other=0)
    tl.store(G + t * S + s, tl.sum(p, axis=0), s < S)


@triton.jit(do_not_specialize=['T'])
def sinkhorn_fwd_kernel(X, O, H, T, N: tl.constexpr, BN: tl.constexpr, ITERS: tl.constexpr, SAVE: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, BN).to(tl.int64)
    j = tl.arange(0, BN).to(tl.int64)
    offsets = (t * N + i[:, None]) * N + j[None, :]
    mask = (i[:, None] < N) & (j[None, :] < N)
    value = tl.load(X + offsets, mask, other=float('-inf')).to(tl.float32)
    for step in range(ITERS):
        maximum = tl.max(value, axis=0)
        value = value - maximum[None, :]
        value = tl.where(mask, value - tl.log(tl.sum(tl.exp(value), axis=0))[None, :], float('-inf'))
        if SAVE:
            tl.store(H + (step.to(tl.int64) * 2 * T * N * N) + offsets, tl.exp(value), mask)
        maximum = tl.max(value, axis=1)
        value = value - maximum[:, None]
        value = tl.where(mask, value - tl.log(tl.sum(tl.exp(value), axis=1))[:, None], float('-inf'))
        if SAVE:
            tl.store(H + ((step.to(tl.int64) * 2 + 1) * T * N * N) + offsets, tl.exp(value), mask)
    tl.store(O + offsets, tl.exp(value), mask)


@triton.jit(do_not_specialize=['T'])
def sinkhorn_bwd_kernel(O, H, DO, DX, T, N: tl.constexpr, BN: tl.constexpr, ITERS: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, BN).to(tl.int64)
    j = tl.arange(0, BN).to(tl.int64)
    offsets = (t * N + i[:, None]) * N + j[None, :]
    mask = (i[:, None] < N) & (j[None, :] < N)
    value = tl.load(DO + offsets, mask, other=0).to(tl.float32) * tl.load(O + offsets, mask, other=0)
    for step in range(ITERS - 1, -1, -1):
        probability = tl.load(H + ((step.to(tl.int64) * 2 + 1) * T * N * N) + offsets, mask, other=0)
        value = value - probability * tl.sum(value, axis=1)[:, None]
        probability = tl.load(H + (step.to(tl.int64) * 2 * T * N * N) + offsets, mask, other=0)
        value = value - probability * tl.sum(value, axis=0)[None, :]
    tl.store(DX + offsets, value, mask)
