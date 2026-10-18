# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--fmad=false']})
def _connection_fwd(tokens, N, D, BN, BD, POST, MIX, xd, wd, bd, md):
    CD = tilelang.cdiv(D, BD)

    @T.prim_func
    def kernel(
        X: T.Tensor((tokens * N * D,), xd), W: T.Tensor((tokens * N,), wd),
        B: T.Tensor((tokens * D if POST else 1,), bd), M: T.Tensor((tokens * N * N if MIX else 1,), md),
        O: T.Tensor((tokens * N * D if POST else tokens * D,), xd),
    ):
        with T.Kernel(tokens, CD, threads=128) as (token, block):
            t = T.cast(token, 'int64')
            c = T.cast(block, 'int64')
            value = T.alloc_fragment((BN, BD), 'float32')
            reduced = T.alloc_fragment((BD,), 'float32')
            if POST:
                if MIX:
                    T.clear(value)
                    for k in T.serial(N):
                        for i, d in T.Parallel(BN, BD):
                            value[i, d] += T.if_then_else(
                                i < N and c * BD + d < D,
                                T.cast(M[(t * N + i) * N + k], 'float32') * T.cast(X[(t * N + k) * D + c * BD + d], 'float32'),
                                0.0,
                            )
                else:
                    for i, d in T.Parallel(BN, BD):
                        value[i, d] = T.if_then_else(i < N and c * BD + d < D, X[(t * N + i) * D + c * BD + d], 0)
                for i, d in T.Parallel(BN, BD):
                    if i < N and c * BD + d < D:
                        O[(t * N + i) * D + c * BD + d] = (
                            value[i, d] + T.cast(W[t * N + i], 'float32') * T.cast(B[t * D + c * BD + d], 'float32')
                        )
            else:
                for i, d in T.Parallel(BN, BD):
                    value[i, d] = T.if_then_else(
                        i < N and c * BD + d < D,
                        T.cast(X[(t * N + i) * D + c * BD + d], 'float32') * T.cast(W[t * N + i], 'float32'), 0.0,
                    )
                T.reduce_sum(value, reduced, dim=0)
                for d in T.Parallel(BD):
                    if c * BD + d < D:
                        O[t * D + c * BD + d] = reduced[d]
    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--fmad=false']})
def _connection_bwd(tokens, N, D, BN, BD, POST, MIX, xd, wd, bd, md):
    CD = tilelang.cdiv(D, BD)

    @T.prim_func
    def kernel(
        X: T.Tensor((tokens * N * D,), xd), W: T.Tensor((tokens * N,), wd),
        B: T.Tensor((tokens * D if POST else 1,), bd), M: T.Tensor((tokens * N * N if MIX else 1,), md),
        DO: T.Tensor((tokens * N * D if POST else tokens * D,), xd), DX: T.Tensor((tokens * N * D,), xd),
        DB: T.Tensor((tokens * D if POST else 1,), bd), PW: T.Tensor((tokens * CD * N,), 'float32'),
        PM: T.Tensor((tokens * CD * N * N if MIX else 1,), 'float32'),
    ):
        with T.Kernel(tokens, CD, threads=128) as (token, block):
            t = T.cast(token, 'int64')
            c = T.cast(block, 'int64')
            value = T.alloc_fragment((BN, BD), 'float32')
            product = T.alloc_fragment((BN, BD), 'float32')
            dx = T.alloc_fragment((BN, BD), 'float32')
            dw = T.alloc_fragment((BN,), 'float32')
            dm = T.alloc_fragment((BN,), 'float32')
            db = T.alloc_fragment((BD,), 'float32')
            if POST:
                for i, d in T.Parallel(BN, BD):
                    value[i, d] = T.if_then_else(i < N and c * BD + d < D, DO[(t * N + i) * D + c * BD + d], 0)
                    product[i, d] = value[i, d] * T.if_then_else(c * BD + d < D, T.cast(B[t * D + c * BD + d], 'float32'), 0.0)
                T.reduce_sum(product, dw, dim=1)
                for i, d in T.Parallel(BN, BD):
                    product[i, d] = value[i, d] * T.if_then_else(i < N, T.cast(W[t * N + i], 'float32'), 0.0)
                T.reduce_sum(product, db, dim=0)
                for d in T.Parallel(BD):
                    if c * BD + d < D:
                        DB[t * D + c * BD + d] = db[d]
                if MIX:
                    T.clear(dx)
                    for k in T.serial(N):
                        for i, d in T.Parallel(BN, BD):
                            dx[i, d] += T.if_then_else(
                                i < N and c * BD + d < D,
                                T.cast(M[(t * N + k) * N + i], 'float32') *
                                T.cast(DO[(t * N + k) * D + c * BD + d], 'float32'),
                                0.0,
                            )
                            product[i, d] = value[i, d] * T.if_then_else(
                                c * BD + d < D, T.cast(X[(t * N + k) * D + c * BD + d], 'float32'), 0.0,
                            )
                        T.reduce_sum(product, dm, dim=1)
                        for i in T.Parallel(BN):
                            if i < N:
                                PM[((t * CD + c) * N + i) * N + k] = dm[i]
                else:
                    T.copy(value, dx)
            else:
                for i, d in T.Parallel(BN, BD):
                    value[i, d] = T.if_then_else(c * BD + d < D, DO[t * D + c * BD + d], 0)
                    dx[i, d] = value[i, d] * T.if_then_else(i < N, T.cast(W[t * N + i], 'float32'), 0.0)
                    product[i, d] = value[i, d] * T.if_then_else(
                        i < N and c * BD + d < D, T.cast(X[(t * N + i) * D + c * BD + d], 'float32'), 0.0,
                    )
                T.reduce_sum(product, dw, dim=1)
            for i, d in T.Parallel(BN, BD):
                if i < N and c * BD + d < D:
                    DX[(t * N + i) * D + c * BD + d] = dx[i, d]
            for i in T.Parallel(BN):
                if i < N:
                    PW[(t * CD + c) * N + i] = dw[i]
    return kernel


@tilelang.jit
def _reduce(tokens, C, S, BC, BS, dtype):
    @T.prim_func
    def kernel(P: T.Tensor((tokens * C * S,), 'float32'), G: T.Tensor((tokens * S,), dtype)):
        with T.Kernel(tokens, threads=128) as token:
            t = T.cast(token, 'int64')
            value = T.alloc_fragment((BC, BS), 'float32')
            reduced = T.alloc_fragment((BS,), 'float32')
            for c, s in T.Parallel(BC, BS):
                value[c, s] = T.if_then_else(c < C and s < S, P[(t * C + c) * S + s], 0)
            T.reduce_sum(value, reduced, dim=0)
            for s in T.Parallel(BS):
                if s < S:
                    G[t * S + s] = reduced[s]
    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--fmad=false']})
def _sinkhorn_fwd(tokens, N, BN, ITERS, SAVE, dtype):
    @T.prim_func
    def kernel(
        X: T.Tensor((tokens * N * N,), dtype), O: T.Tensor((tokens * N * N,), 'float32'),
        H: T.Tensor((2 * ITERS * tokens * N * N if SAVE else 1,), 'float32'),
    ):
        with T.Kernel(tokens, threads=128) as token:
            t = T.cast(token, 'int64')
            value = T.alloc_fragment((BN, BN), 'float32')
            prob = T.alloc_fragment((BN, BN), 'float32')
            col_max = T.alloc_fragment((BN,), 'float32')
            row_max = T.alloc_fragment((BN,), 'float32')
            col_sum = T.alloc_fragment((BN,), 'float32')
            row_sum = T.alloc_fragment((BN,), 'float32')
            for i, j in T.Parallel(BN, BN):
                value[i, j] = T.if_then_else(i < N and j < N, X[(t * N + i) * N + j], -T.infinity('float32'))
            for step in T.serial(ITERS):
                T.reduce_max(value, col_max, dim=0)
                for i, j in T.Parallel(BN, BN):
                    value[i, j] = value[i, j] - col_max[j]
                    prob[i, j] = T.exp(value[i, j])
                T.reduce_sum(prob, col_sum, dim=0)
                for i, j in T.Parallel(BN, BN):
                    value[i, j] = T.if_then_else(i < N and j < N, value[i, j] - T.log(col_sum[j]), -T.infinity('float32'))
                    if SAVE and i < N and j < N:
                        H[((T.cast(step, 'int64') * 2 * tokens + t) * N + i) * N + j] = T.exp(value[i, j])
                T.reduce_max(value, row_max, dim=1)
                for i, j in T.Parallel(BN, BN):
                    value[i, j] = value[i, j] - row_max[i]
                    prob[i, j] = T.exp(value[i, j])
                T.reduce_sum(prob, row_sum, dim=1)
                for i, j in T.Parallel(BN, BN):
                    value[i, j] = T.if_then_else(i < N and j < N, value[i, j] - T.log(row_sum[i]), -T.infinity('float32'))
                    if SAVE and i < N and j < N:
                        H[(((T.cast(step, 'int64') * 2 + 1) * tokens + t) * N + i) * N + j] = T.exp(value[i, j])
            for i, j in T.Parallel(BN, BN):
                if i < N and j < N:
                    O[(t * N + i) * N + j] = T.exp(value[i, j])
    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--fmad=false']})
def _sinkhorn_bwd(tokens, N, BN, ITERS, dtype):
    @T.prim_func
    def kernel(
        O: T.Tensor((tokens * N * N,), 'float32'), H: T.Tensor((2 * ITERS * tokens * N * N,), 'float32'),
        DO: T.Tensor((tokens * N * N,), 'float32'), DX: T.Tensor((tokens * N * N,), dtype),
    ):
        with T.Kernel(tokens, threads=128) as token:
            t = T.cast(token, 'int64')
            value = T.alloc_fragment((BN, BN), 'float32')
            col_sum = T.alloc_fragment((BN,), 'float32')
            row_sum = T.alloc_fragment((BN,), 'float32')
            for i, j in T.Parallel(BN, BN):
                value[i, j] = T.if_then_else(i < N and j < N, DO[(t * N + i) * N + j] * O[(t * N + i) * N + j], 0.0)
            for reverse in T.serial(ITERS):
                step = T.cast(ITERS - 1 - reverse, 'int64')
                T.reduce_sum(value, row_sum, dim=1)
                for i, j in T.Parallel(BN, BN):
                    if i < N and j < N:
                        value[i, j] -= H[(((step * 2 + 1) * tokens + t) * N + i) * N + j] * row_sum[i]
                T.reduce_sum(value, col_sum, dim=0)
                for i, j in T.Parallel(BN, BN):
                    if i < N and j < N:
                        value[i, j] -= H[((step * 2 * tokens + t) * N + i) * N + j] * col_sum[j]
            for i, j in T.Parallel(BN, BN):
                if i < N and j < N:
                    DX[(t * N + i) * N + j] = value[i, j]
    return kernel


def launch(name, **kwargs):
    tokens = kwargs['T']
    if name.startswith('hyper_connection_') and 'reduce' not in name:
        tensors = ['X', 'W', 'B', 'M', 'O'] if 'fwd' in name else ['X', 'W', 'B', 'M', 'DO', 'DX', 'DB', 'PW', 'PM']
        factory = _connection_fwd if 'fwd' in name else _connection_bwd

        def dtype(key):
            return str(kwargs[key].dtype if kwargs[key] is not None else kwargs['X'].dtype).split('.')[-1]

        kernel = factory(tokens, *(kwargs[key] for key in ('N', 'D', 'BN', 'BD', 'POST', 'MIX')),
                         *(dtype(key) for key in ('X', 'W', 'B', 'M')))
        dummy = kwargs['X'].flatten()[:1]
        fp32_dummy = kwargs.get('PW', dummy).flatten()[:1]
    elif name == 'hyper_connection_reduce_kernel':
        tensors = ['P', 'G']
        kernel = _reduce(tokens, *(kwargs[key] for key in ('C', 'S', 'BC', 'BS')), str(kwargs['G'].dtype).split('.')[-1])
    elif name == 'sinkhorn_fwd_kernel':
        tensors = ['X', 'O', 'H']
        kernel = _sinkhorn_fwd(tokens, *(kwargs[key]
                               for key in ('N', 'BN', 'ITERS', 'SAVE')), str(kwargs['X'].dtype).split('.')[-1])
        dummy = fp32_dummy = kwargs['O'].flatten()[:1]
    else:
        tensors = ['O', 'H', 'DO', 'DX']
        kernel = _sinkhorn_bwd(tokens, *(kwargs[key] for key in ('N', 'BN', 'ITERS')), str(kwargs['DX'].dtype).split('.')[-1])
    args = [kwargs[key].flatten() if kwargs[key] is not None else fp32_dummy if key in ('PM', 'H') else dummy for key in tensors]
    kernel(*args)
