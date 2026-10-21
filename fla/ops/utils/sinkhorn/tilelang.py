# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import tilelang
import tilelang.language as T
import torch
import triton

from fla.backends import TileLangBackend, register
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.sinkhorn.ops import sinkhorn_fwd


class SinkhornTileLangBackend(TileLangBackend):
    default_enable = False


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--fmad=false']})
def sinkhorn_fwd_kernel(NT, scale, N, BN, BT, NUM_ITERS, dtype):
    @T.prim_func
    def kernel(
        x: T.Tensor((NT * N * N,), dtype),
        o: T.Tensor((NT * N * N,), 'float32'),
    ):
        with T.Kernel(T.ceildiv(NT, BT), threads=128) as token:
            i_t = T.cast(token, 'int64') * BT
            b_x = T.alloc_fragment((BT, BN, BN), 'float32')
            b_p = T.alloc_fragment((BT, BN, BN), 'float32')
            b_col_max = T.alloc_fragment((BT, BN), 'float32')
            b_row_max = T.alloc_fragment((BT, BN), 'float32')
            b_col_sum = T.alloc_fragment((BT, BN), 'float32')
            b_row_sum = T.alloc_fragment((BT, BN), 'float32')
            if BT > 1:
                layout = T.Fragment(
                    shape=(BT, BN, BN),
                    forward_thread_fn=lambda t, i, j: t * min(32, BN * BN) + (i * BN + j) % 32,
                    forward_index_fn=lambda t, i, j: (i * BN + j) // 32,
                )
                T.annotate_layout({b_x: layout})
            for t, i, j in T.Parallel(BT, BN, BN):
                b_x[t, i, j] = T.if_then_else(
                    i_t + t < NT and i < N and j < N,
                    x[((i_t + t) * N + i) * N + j],
                    -T.infinity('float32'),
                )
            T.reduce_max(b_x, b_col_max, dim=1)
            for t, i, j in T.Parallel(BT, BN, BN):
                b_x[t, i, j] = (b_x[t, i, j] - b_col_max[t, j]) * scale
            for step in T.serial(NUM_ITERS):
                T.reduce_max(b_x, b_col_max, dim=1)
                for t, i, j in T.Parallel(BT, BN, BN):
                    b_x[t, i, j] = b_x[t, i, j] - b_col_max[t, j]
                    b_p[t, i, j] = T.exp2(b_x[t, i, j])
                T.reduce_sum(b_p, b_col_sum, dim=1)
                for t, i, j in T.Parallel(BT, BN, BN):
                    b_x[t, i, j] = T.if_then_else(
                        i_t + t < NT and i < N and j < N,
                        b_x[t, i, j] - T.log2(b_col_sum[t, j]),
                        -T.infinity('float32'),
                    )
                T.reduce_max(b_x, b_row_max, dim=2)
                for t, i, j in T.Parallel(BT, BN, BN):
                    b_x[t, i, j] = b_x[t, i, j] - b_row_max[t, i]
                    b_p[t, i, j] = T.exp2(b_x[t, i, j])
                T.reduce_sum(b_p, b_row_sum, dim=2)
                for t, i, j in T.Parallel(BT, BN, BN):
                    b_x[t, i, j] = T.if_then_else(
                        i_t + t < NT and i < N and j < N,
                        b_x[t, i, j] - T.log2(b_row_sum[t, i]),
                        -T.infinity('float32'),
                    )
            for t, i, j in T.Parallel(BT, BN, BN):
                if i_t + t < NT and i < N and j < N:
                    o[((i_t + t) * N + i) * N + j] = T.exp2(b_x[t, i, j])
    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--fmad=false']})
def sinkhorn_bwd_kernel(NT, scale, N, BN, BT, NUM_ITERS, dtype):
    # recompute earlier blocks to keep the shared probability cache within 40 KiB.
    BI = min(NUM_ITERS, 20)

    @T.prim_func
    def kernel(
        x: T.Tensor((NT * N * N,), dtype),
        do: T.Tensor((NT * N * N,), 'float32'),
        dx: T.Tensor((NT * N * N,), dtype),
    ):
        with T.Kernel(T.ceildiv(NT, BT), threads=128) as token:
            i_t = T.cast(token, 'int64') * BT
            b_dx = T.alloc_fragment((BT, BN, BN), 'float32')
            b_ps = T.alloc_shared((2 * BI, BT, BN, BN), 'float32')
            b_x = T.alloc_fragment((BT, BN, BN), 'float32')
            b_p = T.alloc_fragment((BT, BN, BN), 'float32')
            b_col_max = T.alloc_fragment((BT, BN), 'float32')
            b_row_max = T.alloc_fragment((BT, BN), 'float32')
            b_col_sum = T.alloc_fragment((BT, BN), 'float32')
            b_row_sum = T.alloc_fragment((BT, BN), 'float32')
            if BT > 1:
                layout = T.Fragment(
                    shape=(BT, BN, BN),
                    forward_thread_fn=lambda t, i, j: t * min(32, BN * BN) + (i * BN + j) % 32,
                    forward_index_fn=lambda t, i, j: (i * BN + j) // 32,
                )
                T.annotate_layout({b_x: layout, b_dx: layout})
            for t, i, j in T.Parallel(BT, BN, BN):
                b_dx[t, i, j] = T.if_then_else(
                    i_t + t < NT and i < N and j < N,
                    do[((i_t + t) * N + i) * N + j],
                    0.0,
                )
            for block in T.serial(T.ceildiv(NUM_ITERS, BI)):
                i_b = T.ceildiv(NUM_ITERS, BI) - 1 - block
                for t, i, j in T.Parallel(BT, BN, BN):
                    b_x[t, i, j] = T.if_then_else(
                        i_t + t < NT and i < N and j < N,
                        x[((i_t + t) * N + i) * N + j],
                        -T.infinity('float32'),
                    )
                T.reduce_max(b_x, b_col_max, dim=1)
                for t, i, j in T.Parallel(BT, BN, BN):
                    b_x[t, i, j] = (b_x[t, i, j] - b_col_max[t, j]) * scale
                for step in T.serial(T.min((i_b + 1) * BI, NUM_ITERS)):
                    T.reduce_max(b_x, b_col_max, dim=1)
                    for t, i, j in T.Parallel(BT, BN, BN):
                        b_x[t, i, j] = b_x[t, i, j] - b_col_max[t, j]
                        b_p[t, i, j] = T.exp2(b_x[t, i, j])
                    T.reduce_sum(b_p, b_col_sum, dim=1)
                    for t, i, j in T.Parallel(BT, BN, BN):
                        b_x[t, i, j] = T.if_then_else(
                            i_t + t < NT and i < N and j < N,
                            b_x[t, i, j] - T.log2(b_col_sum[t, j]),
                            -T.infinity('float32'),
                        )
                        if step >= i_b * BI:
                            b_ps[2 * (step - i_b * BI), t, i, j] = T.exp2(b_x[t, i, j])
                    T.reduce_max(b_x, b_row_max, dim=2)
                    for t, i, j in T.Parallel(BT, BN, BN):
                        b_x[t, i, j] = b_x[t, i, j] - b_row_max[t, i]
                        b_p[t, i, j] = T.exp2(b_x[t, i, j])
                    T.reduce_sum(b_p, b_row_sum, dim=2)
                    for t, i, j in T.Parallel(BT, BN, BN):
                        b_x[t, i, j] = T.if_then_else(
                            i_t + t < NT and i < N and j < N,
                            b_x[t, i, j] - T.log2(b_row_sum[t, i]),
                            -T.infinity('float32'),
                        )
                        if step >= i_b * BI:
                            b_ps[2 * (step - i_b * BI) + 1, t, i, j] = T.exp2(b_x[t, i, j])
                if i_b == T.ceildiv(NUM_ITERS, BI) - 1:
                    for t, i, j in T.Parallel(BT, BN, BN):
                        b_dx[t, i, j] *= T.exp2(b_x[t, i, j])
                for reverse in T.serial(T.min(BI, NUM_ITERS - i_b * BI)):
                    step = T.min(BI, NUM_ITERS - i_b * BI) - 1 - reverse
                    T.reduce_sum(b_dx, b_row_sum, dim=2)
                    for t, i, j in T.Parallel(BT, BN, BN):
                        b_dx[t, i, j] -= b_ps[2 * step + 1, t, i, j] * b_row_sum[t, i]
                    T.reduce_sum(b_dx, b_col_sum, dim=1)
                    for t, i, j in T.Parallel(BT, BN, BN):
                        b_dx[t, i, j] -= b_ps[2 * step, t, i, j] * b_col_sum[t, j]
            for t, i, j in T.Parallel(BT, BN, BN):
                if i_t + t < NT and i < N and j < N:
                    dx[((i_t + t) * N + i) * N + j] = b_dx[t, i, j]
    return kernel


@register(sinkhorn_fwd, backend=SinkhornTileLangBackend)
def sinkhorn_fwd_tilelang(logits: torch.Tensor, num_iters: int):
    N = logits.shape[-1]
    NT = logits.numel() // (N * N)
    out = torch.empty_like(logits, dtype=torch.float32)
    if NT:
        kernel = sinkhorn_fwd_kernel(
            NT=NT,
            scale=RCP_LN2,
            N=N,
            BN=triton.next_power_of_2(N),
            BT=1 if NT < 128 else max(4, 128 // min(32, triton.next_power_of_2(N) ** 2)),
            NUM_ITERS=num_iters,
            dtype=str(logits.dtype).removeprefix('torch.'),
        )
        kernel(logits.flatten(), out.flatten())
    return out, sinkhorn_bwd_tilelang


def sinkhorn_bwd_tilelang(logits: torch.Tensor, dout: torch.Tensor, num_iters: int) -> torch.Tensor:
    N = logits.shape[-1]
    NT = logits.numel() // (N * N)
    dx = torch.empty_like(logits)
    if NT:
        kernel = sinkhorn_bwd_kernel(
            NT=NT,
            scale=RCP_LN2,
            N=N,
            BN=triton.next_power_of_2(N),
            BT=1 if NT < 128 else max(4, 128 // min(32, triton.next_power_of_2(N) ** 2)),
            NUM_ITERS=num_iters,
            dtype=str(logits.dtype).removeprefix('torch.'),
        )
        kernel(logits.flatten(), dout.flatten(), dx.flatten())
    return dx
