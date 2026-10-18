# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--fmad=false']})
def _hc_fwd_kernel(tokens, N, D, BN, BD, HAS_BRANCH, HAS_MIXING, x_dtype, weight_dtype, branch_dtype, mixing_dtype):
    CD = tilelang.cdiv(D, BD)

    @T.prim_func
    def kernel(
        x: T.Tensor((tokens * N * D,), x_dtype),
        weight: T.Tensor((tokens * N,), weight_dtype),
        branch: T.Tensor((tokens * D if HAS_BRANCH else 1,), branch_dtype),
        mixing: T.Tensor((tokens * N * N if HAS_MIXING else 1,), mixing_dtype),
        o: T.Tensor((tokens * N * D if HAS_BRANCH else tokens * D,), x_dtype),
    ):
        with T.Kernel(tokens, CD, threads=128) as (token, block):
            i_t = T.cast(token, 'int64')
            i_c = T.cast(block, 'int64')
            b_value = T.alloc_fragment((BN, BD), 'float32')
            b_reduced = T.alloc_fragment((BD,), 'float32')
            if HAS_BRANCH:
                if HAS_MIXING:
                    T.clear(b_value)
                    for k in T.serial(N):
                        for i, d in T.Parallel(BN, BD):
                            b_value[i, d] += T.if_then_else(
                                i < N and i_c * BD + d < D,
                                T.cast(mixing[(i_t * N + i) * N + k], 'float32') *
                                T.cast(x[(i_t * N + k) * D + i_c * BD + d], 'float32'),
                                0.0,
                            )
                else:
                    for i, d in T.Parallel(BN, BD):
                        b_value[i, d] = T.if_then_else(i < N and i_c * BD + d < D, x[(i_t * N + i) * D + i_c * BD + d], 0)
                for i, d in T.Parallel(BN, BD):
                    if i < N and i_c * BD + d < D:
                        o[(i_t * N + i) * D + i_c * BD + d] = (
                            b_value[i, d] + T.cast(weight[i_t * N + i], 'float32') *
                            T.cast(branch[i_t * D + i_c * BD + d], 'float32')
                        )
            else:
                for i, d in T.Parallel(BN, BD):
                    b_value[i, d] = T.if_then_else(
                        i < N and i_c * BD + d < D,
                        T.cast(x[(i_t * N + i) * D + i_c * BD + d], 'float32') * T.cast(weight[i_t * N + i], 'float32'),
                        0.0,
                    )
                T.reduce_sum(b_value, b_reduced, dim=0)
                for d in T.Parallel(BD):
                    if i_c * BD + d < D:
                        o[i_t * D + i_c * BD + d] = b_reduced[d]
    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--fmad=false']})
def _hc_bwd_kernel(tokens, N, D, BN, BD, HAS_BRANCH, HAS_MIXING, x_dtype, weight_dtype, branch_dtype, mixing_dtype):
    CD = tilelang.cdiv(D, BD)

    @T.prim_func
    def kernel(
        x: T.Tensor((tokens * N * D,), x_dtype),
        weight: T.Tensor((tokens * N,), weight_dtype),
        branch: T.Tensor((tokens * D if HAS_BRANCH else 1,), branch_dtype),
        mixing: T.Tensor((tokens * N * N if HAS_MIXING else 1,), mixing_dtype),
        do: T.Tensor((tokens * N * D if HAS_BRANCH else tokens * D,), x_dtype),
        dx: T.Tensor((tokens * N * D,), x_dtype),
        dbranch: T.Tensor((tokens * D if HAS_BRANCH else 1,), branch_dtype),
        dweight_partial: T.Tensor((tokens * CD * N,), 'float32'),
        dmixing_partial: T.Tensor((tokens * CD * N * N if HAS_MIXING else 1,), 'float32'),
    ):
        with T.Kernel(tokens, CD, threads=128) as (token, block):
            i_t = T.cast(token, 'int64')
            i_c = T.cast(block, 'int64')
            b_value = T.alloc_fragment((BN, BD), 'float32')
            b_product = T.alloc_fragment((BN, BD), 'float32')
            b_dx = T.alloc_fragment((BN, BD), 'float32')
            b_dweight = T.alloc_fragment((BN,), 'float32')
            b_dmixing = T.alloc_fragment((BN,), 'float32')
            b_dbranch = T.alloc_fragment((BD,), 'float32')
            if HAS_BRANCH:
                for i, d in T.Parallel(BN, BD):
                    b_value[i, d] = T.if_then_else(i < N and i_c * BD + d < D, do[(i_t * N + i) * D + i_c * BD + d], 0)
                    b_product[i, d] = b_value[i, d] * T.if_then_else(
                        i_c * BD + d < D,
                        T.cast(branch[i_t * D + i_c * BD + d], 'float32'),
                        0.0,
                    )
                T.reduce_sum(b_product, b_dweight, dim=1)
                for i, d in T.Parallel(BN, BD):
                    b_product[i, d] = b_value[i, d] * T.if_then_else(i < N, T.cast(weight[i_t * N + i], 'float32'), 0.0)
                T.reduce_sum(b_product, b_dbranch, dim=0)
                for d in T.Parallel(BD):
                    if i_c * BD + d < D:
                        dbranch[i_t * D + i_c * BD + d] = b_dbranch[d]
                if HAS_MIXING:
                    T.clear(b_dx)
                    for k in T.serial(N):
                        for i, d in T.Parallel(BN, BD):
                            b_dx[i, d] += T.if_then_else(
                                i < N and i_c * BD + d < D,
                                T.cast(mixing[(i_t * N + k) * N + i], 'float32') *
                                T.cast(do[(i_t * N + k) * D + i_c * BD + d], 'float32'),
                                0.0,
                            )
                            b_product[i, d] = b_value[i, d] * T.if_then_else(
                                i_c * BD + d < D,
                                T.cast(x[(i_t * N + k) * D + i_c * BD + d], 'float32'),
                                0.0,
                            )
                        T.reduce_sum(b_product, b_dmixing, dim=1)
                        for i in T.Parallel(BN):
                            if i < N:
                                dmixing_partial[((i_t * CD + i_c) * N + i) * N + k] = b_dmixing[i]
                else:
                    T.copy(b_value, b_dx)
            else:
                for i, d in T.Parallel(BN, BD):
                    b_value[i, d] = T.if_then_else(i_c * BD + d < D, do[i_t * D + i_c * BD + d], 0)
                    b_dx[i, d] = b_value[i, d] * T.if_then_else(i < N, T.cast(weight[i_t * N + i], 'float32'), 0.0)
                    b_product[i, d] = b_value[i, d] * T.if_then_else(
                        i < N and i_c * BD + d < D,
                        T.cast(x[(i_t * N + i) * D + i_c * BD + d], 'float32'),
                        0.0,
                    )
                T.reduce_sum(b_product, b_dweight, dim=1)
            for i, d in T.Parallel(BN, BD):
                if i < N and i_c * BD + d < D:
                    dx[(i_t * N + i) * D + i_c * BD + d] = b_dx[i, d]
            for i in T.Parallel(BN):
                if i < N:
                    dweight_partial[(i_t * CD + i_c) * N + i] = b_dweight[i]
    return kernel


@tilelang.jit
def _hc_bwd_kernel_reduce(tokens, C, S, BC, BS, dtype):
    @T.prim_func
    def kernel(partial: T.Tensor((tokens * C * S,), 'float32'), grad: T.Tensor((tokens * S,), dtype)):
        with T.Kernel(tokens, threads=128) as token:
            i_t = T.cast(token, 'int64')
            b_value = T.alloc_fragment((BC, BS), 'float32')
            b_reduced = T.alloc_fragment((BS,), 'float32')
            for i_c, s in T.Parallel(BC, BS):
                b_value[i_c, s] = T.if_then_else(i_c < C and s < S, partial[(i_t * C + i_c) * S + s], 0)
            T.reduce_sum(b_value, b_reduced, dim=0)
            for s in T.Parallel(BS):
                if s < S:
                    grad[i_t * S + s] = b_reduced[s]
    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--fmad=false']})
def _sinkhorn_fwd_kernel(tokens, N, BN, NUM_ITERS, SAVE_HISTORY, dtype):
    @T.prim_func
    def kernel(
        x: T.Tensor((tokens * N * N,), dtype),
        o: T.Tensor((tokens * N * N,), 'float32'),
        history: T.Tensor((2 * NUM_ITERS * tokens * N * N if SAVE_HISTORY else 1,), 'float32'),
    ):
        with T.Kernel(tokens, threads=128) as token:
            i_t = T.cast(token, 'int64')
            b_value = T.alloc_fragment((BN, BN), 'float32')
            b_probability = T.alloc_fragment((BN, BN), 'float32')
            b_col_max = T.alloc_fragment((BN,), 'float32')
            b_row_max = T.alloc_fragment((BN,), 'float32')
            b_col_sum = T.alloc_fragment((BN,), 'float32')
            b_row_sum = T.alloc_fragment((BN,), 'float32')
            for i, j in T.Parallel(BN, BN):
                b_value[i, j] = T.if_then_else(i < N and j < N, x[(i_t * N + i) * N + j], -T.infinity('float32'))
            for step in T.serial(NUM_ITERS):
                T.reduce_max(b_value, b_col_max, dim=0)
                for i, j in T.Parallel(BN, BN):
                    b_value[i, j] = b_value[i, j] - b_col_max[j]
                    b_probability[i, j] = T.exp(b_value[i, j])
                T.reduce_sum(b_probability, b_col_sum, dim=0)
                for i, j in T.Parallel(BN, BN):
                    b_value[i, j] = T.if_then_else(
                        i < N and j < N,
                        b_value[i, j] - T.log(b_col_sum[j]),
                        -T.infinity('float32'),
                    )
                    if SAVE_HISTORY and i < N and j < N:
                        history[((T.cast(step, 'int64') * 2 * tokens + i_t) * N + i) * N + j] = T.exp(b_value[i, j])
                T.reduce_max(b_value, b_row_max, dim=1)
                for i, j in T.Parallel(BN, BN):
                    b_value[i, j] = b_value[i, j] - b_row_max[i]
                    b_probability[i, j] = T.exp(b_value[i, j])
                T.reduce_sum(b_probability, b_row_sum, dim=1)
                for i, j in T.Parallel(BN, BN):
                    b_value[i, j] = T.if_then_else(
                        i < N and j < N,
                        b_value[i, j] - T.log(b_row_sum[i]),
                        -T.infinity('float32'),
                    )
                    if SAVE_HISTORY and i < N and j < N:
                        history[(((T.cast(step, 'int64') * 2 + 1) * tokens + i_t) * N + i) * N + j] = T.exp(b_value[i, j])
            for i, j in T.Parallel(BN, BN):
                if i < N and j < N:
                    o[(i_t * N + i) * N + j] = T.exp(b_value[i, j])
    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--fmad=false']})
def _sinkhorn_bwd_kernel(tokens, N, BN, NUM_ITERS, dtype):
    @T.prim_func
    def kernel(
        o: T.Tensor((tokens * N * N,), 'float32'),
        history: T.Tensor((2 * NUM_ITERS * tokens * N * N,), 'float32'),
        do: T.Tensor((tokens * N * N,), 'float32'),
        dx: T.Tensor((tokens * N * N,), dtype),
    ):
        with T.Kernel(tokens, threads=128) as token:
            i_t = T.cast(token, 'int64')
            b_value = T.alloc_fragment((BN, BN), 'float32')
            b_col_sum = T.alloc_fragment((BN,), 'float32')
            b_row_sum = T.alloc_fragment((BN,), 'float32')
            for i, j in T.Parallel(BN, BN):
                b_value[i, j] = T.if_then_else(i < N and j < N, do[(i_t * N + i) * N + j] * o[(i_t * N + i) * N + j], 0.0)
            for reverse in T.serial(NUM_ITERS):
                step = T.cast(NUM_ITERS - 1 - reverse, 'int64')
                T.reduce_sum(b_value, b_row_sum, dim=1)
                for i, j in T.Parallel(BN, BN):
                    if i < N and j < N:
                        b_value[i, j] -= history[(((step * 2 + 1) * tokens + i_t) * N + i) * N + j] * b_row_sum[i]
                T.reduce_sum(b_value, b_col_sum, dim=0)
                for i, j in T.Parallel(BN, BN):
                    if i < N and j < N:
                        b_value[i, j] -= history[((step * 2 * tokens + i_t) * N + i) * N + j] * b_col_sum[j]
            for i, j in T.Parallel(BN, BN):
                if i < N and j < N:
                    dx[(i_t * N + i) * N + j] = b_value[i, j]
    return kernel


def _launch_kernel(name: str, **kwargs):
    tokens = kwargs['T']
    if name in ('hc_fwd_kernel', 'hc_bwd_kernel'):
        if name == 'hc_fwd_kernel':
            tensors = ('x', 'weight', 'branch', 'mixing', 'o')
            factory = _hc_fwd_kernel
        else:
            tensors = ('x', 'weight', 'branch', 'mixing', 'do', 'dx', 'dbranch', 'dweight_partial', 'dmixing_partial')
            factory = _hc_bwd_kernel

        def dtype(key: str) -> str:
            tensor = kwargs[key] if kwargs[key] is not None else kwargs['x']
            return str(tensor.dtype).split('.')[-1]

        kernel = factory(
            tokens=tokens,
            N=kwargs['N'],
            D=kwargs['D'],
            BN=kwargs['BN'],
            BD=kwargs['BD'],
            HAS_BRANCH=kwargs['HAS_BRANCH'],
            HAS_MIXING=kwargs['HAS_MIXING'],
            x_dtype=dtype('x'),
            weight_dtype=dtype('weight'),
            branch_dtype=dtype('branch'),
            mixing_dtype=dtype('mixing'),
        )
        dummy = kwargs['x'].flatten()[:1]
        fp32_dummy = kwargs.get('dweight_partial', dummy).flatten()[:1]
    elif name == 'hc_bwd_kernel_reduce':
        tensors = ('partial', 'grad')
        kernel = _hc_bwd_kernel_reduce(
            tokens=tokens,
            C=kwargs['C'],
            S=kwargs['S'],
            BC=kwargs['BC'],
            BS=kwargs['BS'],
            dtype=str(kwargs['grad'].dtype).split('.')[-1],
        )
    elif name == 'sinkhorn_fwd_kernel':
        tensors = ('x', 'o', 'history')
        kernel = _sinkhorn_fwd_kernel(
            tokens=tokens,
            N=kwargs['N'],
            BN=kwargs['BN'],
            NUM_ITERS=kwargs['NUM_ITERS'],
            SAVE_HISTORY=kwargs['SAVE_HISTORY'],
            dtype=str(kwargs['x'].dtype).split('.')[-1],
        )
        dummy = fp32_dummy = kwargs['o'].flatten()[:1]
    elif name == 'sinkhorn_bwd_kernel':
        tensors = ('o', 'history', 'do', 'dx')
        kernel = _sinkhorn_bwd_kernel(
            tokens=tokens,
            N=kwargs['N'],
            BN=kwargs['BN'],
            NUM_ITERS=kwargs['NUM_ITERS'],
            dtype=str(kwargs['dx'].dtype).split('.')[-1],
        )
    else:
        raise ValueError(f'Unknown HC kernel: {name}')

    args = []
    for key in tensors:
        tensor = kwargs[key]
        if tensor is None:
            tensor = fp32_dummy if key in ('dmixing_partial', 'history') else dummy
        args.append(tensor.flatten())
    kernel(*args)
