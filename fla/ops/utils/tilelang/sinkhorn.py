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
from fla.ops.utils.sinkhorn import sinkhorn_fwd as sinkhorn_fwd_dispatch


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--fmad=false']})
def sinkhorn_fwd_kernel(NT, N, BN, BT, NUM_ITERS, SAVE_HISTORY, dtype):
    @T.prim_func
    def kernel(
        x: T.Tensor((NT * N * N,), dtype),
        o: T.Tensor((NT * N * N,), 'float32'),
        history: T.Tensor((2 * NUM_ITERS * NT * N * N if SAVE_HISTORY else 1,), 'float32'),
    ):
        with T.Kernel(T.ceildiv(NT, BT), threads=128) as token:
            i_t = T.cast(token, 'int64') * BT
            b_x = T.alloc_fragment((BT, BN, BN), 'float32')
            b_probability = T.alloc_fragment((BT, BN, BN), 'float32')
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
            for step in T.serial(NUM_ITERS):
                T.reduce_max(b_x, b_col_max, dim=1)
                for t, i, j in T.Parallel(BT, BN, BN):
                    b_x[t, i, j] = b_x[t, i, j] - b_col_max[t, j]
                    b_probability[t, i, j] = T.exp(b_x[t, i, j])
                T.reduce_sum(b_probability, b_col_sum, dim=1)
                for t, i, j in T.Parallel(BT, BN, BN):
                    b_x[t, i, j] = T.if_then_else(
                        i_t + t < NT and i < N and j < N,
                        b_x[t, i, j] - T.log(b_col_sum[t, j]),
                        -T.infinity('float32'),
                    )
                    if SAVE_HISTORY and i_t + t < NT and i < N and j < N:
                        history[((T.cast(step, 'int64') * 2 * NT + i_t + t) * N + i) * N + j] = T.exp(b_x[t, i, j])
                T.reduce_max(b_x, b_row_max, dim=2)
                for t, i, j in T.Parallel(BT, BN, BN):
                    b_x[t, i, j] = b_x[t, i, j] - b_row_max[t, i]
                    b_probability[t, i, j] = T.exp(b_x[t, i, j])
                T.reduce_sum(b_probability, b_row_sum, dim=2)
                for t, i, j in T.Parallel(BT, BN, BN):
                    b_x[t, i, j] = T.if_then_else(
                        i_t + t < NT and i < N and j < N,
                        b_x[t, i, j] - T.log(b_row_sum[t, i]),
                        -T.infinity('float32'),
                    )
                    if SAVE_HISTORY and i_t + t < NT and i < N and j < N:
                        history[(((T.cast(step, 'int64') * 2 + 1) * NT + i_t + t) * N + i) * N + j] = T.exp(b_x[t, i, j])
            for t, i, j in T.Parallel(BT, BN, BN):
                if i_t + t < NT and i < N and j < N:
                    o[((i_t + t) * N + i) * N + j] = T.exp(b_x[t, i, j])
    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--fmad=false']})
def sinkhorn_bwd_kernel(NT, N, BN, BT, NUM_ITERS, dtype):
    @T.prim_func
    def kernel(
        o: T.Tensor((NT * N * N,), 'float32'),
        history: T.Tensor((2 * NUM_ITERS * NT * N * N,), 'float32'),
        do: T.Tensor((NT * N * N,), 'float32'),
        dx: T.Tensor((NT * N * N,), dtype),
    ):
        with T.Kernel(T.ceildiv(NT, BT), threads=128) as token:
            i_t = T.cast(token, 'int64') * BT
            b_dx = T.alloc_fragment((BT, BN, BN), 'float32')
            b_col_sum = T.alloc_fragment((BT, BN), 'float32')
            b_row_sum = T.alloc_fragment((BT, BN), 'float32')
            if BT > 1:
                layout = T.Fragment(
                    shape=(BT, BN, BN),
                    forward_thread_fn=lambda t, i, j: t * min(32, BN * BN) + (i * BN + j) % 32,
                    forward_index_fn=lambda t, i, j: (i * BN + j) // 32,
                )
                T.annotate_layout({b_dx: layout})
            for t, i, j in T.Parallel(BT, BN, BN):
                b_dx[t, i, j] = T.if_then_else(
                    i_t + t < NT and i < N and j < N,
                    do[((i_t + t) * N + i) * N + j] * o[((i_t + t) * N + i) * N + j],
                    0.0,
                )
            for reverse in T.serial(NUM_ITERS):
                step = T.cast(NUM_ITERS - 1 - reverse, 'int64')
                T.reduce_sum(b_dx, b_row_sum, dim=2)
                for t, i, j in T.Parallel(BT, BN, BN):
                    if i_t + t < NT and i < N and j < N:
                        b_dx[t, i, j] -= history[(((step * 2 + 1) * NT + i_t + t) * N + i) * N + j] * b_row_sum[t, i]
                T.reduce_sum(b_dx, b_col_sum, dim=1)
                for t, i, j in T.Parallel(BT, BN, BN):
                    if i_t + t < NT and i < N and j < N:
                        b_dx[t, i, j] -= history[((step * 2 * NT + i_t + t) * N + i) * N + j] * b_col_sum[t, j]
            for t, i, j in T.Parallel(BT, BN, BN):
                if i_t + t < NT and i < N and j < N:
                    dx[((i_t + t) * N + i) * N + j] = b_dx[t, i, j]
    return kernel


def sinkhorn_fwd(logits, num_iters, save_history):
    N = logits.shape[-1]
    NT = logits.numel() // (N * N)
    out = torch.empty_like(logits, dtype=torch.float32)
    history = torch.empty((2 * num_iters, NT, N, N), device=logits.device, dtype=torch.float32) if save_history else None
    if NT:
        kernel = sinkhorn_fwd_kernel(
            NT=NT,
            N=N,
            BN=triton.next_power_of_2(N),
            BT=1 if NT < 128 else max(4, 128 // min(32, triton.next_power_of_2(N) ** 2)),
            NUM_ITERS=num_iters,
            SAVE_HISTORY=save_history,
            dtype=str(logits.dtype).removeprefix('torch.'),
        )
        kernel(logits.flatten(), out.flatten(), history.flatten() if history is not None else out.flatten()[:1])
    return out, history, sinkhorn_bwd


def sinkhorn_bwd(out, history, dout, num_iters, dtype):
    N, NT = out.shape[-1], out.numel() // out.shape[-1] ** 2
    dx = torch.empty_like(out, dtype=dtype)
    if NT:
        kernel = sinkhorn_bwd_kernel(
            NT=NT,
            N=N,
            BN=triton.next_power_of_2(N),
            BT=1 if NT < 128 else max(4, 128 // min(32, triton.next_power_of_2(N) ** 2)),
            NUM_ITERS=num_iters,
            dtype=str(dtype).removeprefix('torch.'),
        )
        kernel(out.flatten(), history.flatten(), dout.flatten(), dx.flatten())
    return dx


@register(sinkhorn_fwd_dispatch)
class SinkhornTileLangBackend(TileLangBackend):
    default_enable = False

    def sinkhorn_fwd(self, logits, num_iters, save_history):
        return sinkhorn_fwd(logits=logits, num_iters=num_iters, save_history=save_history)
