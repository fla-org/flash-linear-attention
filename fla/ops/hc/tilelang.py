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
from fla.ops.hc.fused import hc_fwd


class HCTileLangBackend(TileLangBackend):
    default_enable = False


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--fmad=false']})
def hc_fwd_kernel(NT, N, D, BN, BD, HAS_BRANCH, HAS_MIXING, x_dtype, weight_dtype, branch_dtype, mixing_dtype):
    CD = tilelang.cdiv(D, BD)

    @T.prim_func
    def kernel(
        x: T.Tensor((NT * N * D,), x_dtype),
        weight: T.Tensor((NT * N,), weight_dtype),
        branch: T.Tensor((NT * D if HAS_BRANCH else 1,), branch_dtype),
        mixing: T.Tensor((NT * N * N if HAS_MIXING else 1,), mixing_dtype),
        o: T.Tensor((NT * N * D if HAS_BRANCH else NT * D,), x_dtype),
    ):
        with T.Kernel(NT, CD, threads=128) as (token, block):
            i_t = T.cast(token, 'int64')
            i_c = T.cast(block, 'int64')
            b_o = T.alloc_fragment((BN, BD), 'float32')
            b_reduced = T.alloc_fragment((BD,), 'float32')
            if not HAS_BRANCH and N <= 4:
                T.annotate_layout({
                    b_o: T.Fragment(
                        shape=(BN, BD),
                        forward_thread_fn=lambda i, d: d % 128,
                        forward_index_fn=lambda i, d: i * (BD // 128) + d // 128,
                    ),
                })
            if HAS_BRANCH:
                if HAS_MIXING:
                    T.clear(b_o)
                    for k in T.serial(N):
                        for i, d in T.Parallel(BN, BD):
                            b_o[i, d] += T.if_then_else(
                                i < N and i_c * BD + d < D,
                                T.cast(mixing[(i_t * N + i) * N + k], 'float32') *
                                T.cast(x[(i_t * N + k) * D + i_c * BD + d], 'float32'),
                                0.0,
                            )
                else:
                    for i, d in T.Parallel(BN, BD):
                        b_o[i, d] = T.if_then_else(i < N and i_c * BD + d < D, x[(i_t * N + i) * D + i_c * BD + d], 0)
                for i, d in T.Parallel(BN, BD):
                    if i < N and i_c * BD + d < D:
                        o[(i_t * N + i) * D + i_c * BD + d] = (
                            b_o[i, d] + T.cast(weight[i_t * N + i], 'float32') *
                            T.cast(branch[i_t * D + i_c * BD + d], 'float32')
                        )
            else:
                for i, d in T.Parallel(BN, BD):
                    b_o[i, d] = T.if_then_else(
                        i < N and i_c * BD + d < D,
                        T.cast(x[(i_t * N + i) * D + i_c * BD + d], 'float32') * T.cast(weight[i_t * N + i], 'float32'),
                        0.0,
                    )
                T.reduce_sum(b_o, b_reduced, dim=0)
                for d in T.Parallel(BD):
                    if i_c * BD + d < D:
                        o[i_t * D + i_c * BD + d] = b_reduced[d]
    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--fmad=false']})
def hc_bwd_kernel(NT, N, D, BN, BD, HAS_BRANCH, HAS_MIXING, x_dtype, weight_dtype, branch_dtype, mixing_dtype):
    CD = tilelang.cdiv(D, BD)

    @T.prim_func
    def kernel(
        x: T.Tensor((NT * N * D,), x_dtype),
        weight: T.Tensor((NT * N,), weight_dtype),
        branch: T.Tensor((NT * D if HAS_BRANCH else 1,), branch_dtype),
        mixing: T.Tensor((NT * N * N if HAS_MIXING else 1,), mixing_dtype),
        do: T.Tensor((NT * N * D if HAS_BRANCH else NT * D,), x_dtype),
        dx: T.Tensor((NT * N * D,), x_dtype),
        dbranch: T.Tensor((NT * D if HAS_BRANCH else 1,), branch_dtype),
        dweight_partial: T.Tensor((NT * CD * N,), 'float32'),
        dmixing_partial: T.Tensor((NT * CD * N * N if HAS_MIXING else 1,), 'float32'),
    ):
        with T.Kernel(NT, CD, threads=128) as (token, block):
            i_t = T.cast(token, 'int64')
            i_c = T.cast(block, 'int64')
            b_do = T.alloc_fragment((BN, BD), 'float32')
            b_product = T.alloc_fragment((BN, BD), 'float32')
            b_dx = T.alloc_fragment((BN, BD), 'float32')
            b_dw = T.alloc_fragment((BN,), 'float32')
            b_dm = T.alloc_fragment((BN,), 'float32')
            b_dy = T.alloc_fragment((BD,), 'float32')
            if HAS_BRANCH:
                for i, d in T.Parallel(BN, BD):
                    b_do[i, d] = T.if_then_else(i < N and i_c * BD + d < D, do[(i_t * N + i) * D + i_c * BD + d], 0)
                    b_product[i, d] = b_do[i, d] * T.if_then_else(
                        i_c * BD + d < D,
                        T.cast(branch[i_t * D + i_c * BD + d], 'float32'),
                        0.0,
                    )
                T.reduce_sum(b_product, b_dw, dim=1)
                for i, d in T.Parallel(BN, BD):
                    b_product[i, d] = b_do[i, d] * T.if_then_else(i < N, T.cast(weight[i_t * N + i], 'float32'), 0.0)
                T.reduce_sum(b_product, b_dy, dim=0)
                for d in T.Parallel(BD):
                    if i_c * BD + d < D:
                        dbranch[i_t * D + i_c * BD + d] = b_dy[d]
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
                            b_product[i, d] = b_do[i, d] * T.if_then_else(
                                i_c * BD + d < D,
                                T.cast(x[(i_t * N + k) * D + i_c * BD + d], 'float32'),
                                0.0,
                            )
                        T.reduce_sum(b_product, b_dm, dim=1)
                        for i in T.Parallel(BN):
                            if i < N:
                                dmixing_partial[((i_t * CD + i_c) * N + i) * N + k] = b_dm[i]
                else:
                    T.copy(b_do, b_dx)
            else:
                for i, d in T.Parallel(BN, BD):
                    b_do[i, d] = T.if_then_else(i_c * BD + d < D, do[i_t * D + i_c * BD + d], 0)
                    b_dx[i, d] = b_do[i, d] * T.if_then_else(i < N, T.cast(weight[i_t * N + i], 'float32'), 0.0)
                    b_product[i, d] = b_do[i, d] * T.if_then_else(
                        i < N and i_c * BD + d < D,
                        T.cast(x[(i_t * N + i) * D + i_c * BD + d], 'float32'),
                        0.0,
                    )
                T.reduce_sum(b_product, b_dw, dim=1)
            for i, d in T.Parallel(BN, BD):
                if i < N and i_c * BD + d < D:
                    dx[(i_t * N + i) * D + i_c * BD + d] = b_dx[i, d]
            for i in T.Parallel(BN):
                if i < N:
                    dweight_partial[(i_t * CD + i_c) * N + i] = b_dw[i]
    return kernel


@tilelang.jit
def hc_bwd_kernel_reduce(NT, C, S, BC, BS, dtype):
    @T.prim_func
    def kernel(partial: T.Tensor((NT * C * S,), 'float32'), grad: T.Tensor((NT * S,), dtype)):
        with T.Kernel(NT, threads=128) as token:
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


@register(hc_fwd, backend=HCTileLangBackend)
def hc_fwd_tilelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    branch: torch.Tensor | None = None,
    mixing: torch.Tensor | None = None,
):
    N, D = x.shape[-2:]
    NT = x.numel() // (N * D)
    BD = max(128, min(256 if NT < 128 or branch is not None or N > 4 else 1024, triton.next_power_of_2(D)))
    out = torch.empty_like(x) if branch is not None else x.new_empty((*x.shape[:-2], D))
    if NT:
        kernel = hc_fwd_kernel(
            NT=NT,
            N=N,
            D=D,
            BN=triton.next_power_of_2(N),
            BD=BD,
            HAS_BRANCH=branch is not None,
            HAS_MIXING=mixing is not None,
            x_dtype=str(x.dtype).removeprefix('torch.'),
            weight_dtype=str(weight.dtype).removeprefix('torch.'),
            branch_dtype=str(branch.dtype if branch is not None else x.dtype).removeprefix('torch.'),
            mixing_dtype=str(mixing.dtype if mixing is not None else x.dtype).removeprefix('torch.'),
        )
        dummy = x.flatten()[:1]
        kernel(
            x.flatten(),
            weight.flatten(),
            branch.flatten() if branch is not None else dummy,
            mixing.flatten() if mixing is not None else dummy,
            out.flatten(),
        )
    return out, hc_bwd_tilelang


def hc_bwd_tilelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    branch: torch.Tensor | None,
    mixing: torch.Tensor | None,
    dout: torch.Tensor,
):
    N, D = x.shape[-2:]
    NT = x.numel() // (N * D)
    BD = 256 if NT < 128 or mixing is not None or (branch is None and N <= 4) else 512
    BD = min(BD, triton.next_power_of_2(D))
    CD = triton.cdiv(D, BD)
    dx, dweight = torch.empty_like(x), torch.empty_like(weight)
    dbranch = torch.empty_like(branch) if branch is not None else None
    dmixing = torch.empty_like(mixing) if mixing is not None else None
    dweight_partial = torch.empty((NT, CD, N), device=x.device, dtype=torch.float32)
    dmixing_partial = torch.empty((NT, CD, N * N), device=x.device, dtype=torch.float32) if mixing is not None else None
    if NT:
        kernel = hc_bwd_kernel(
            NT=NT,
            N=N,
            D=D,
            BN=triton.next_power_of_2(N),
            BD=BD,
            HAS_BRANCH=branch is not None,
            HAS_MIXING=mixing is not None,
            x_dtype=str(x.dtype).removeprefix('torch.'),
            weight_dtype=str(weight.dtype).removeprefix('torch.'),
            branch_dtype=str(branch.dtype if branch is not None else x.dtype).removeprefix('torch.'),
            mixing_dtype=str(mixing.dtype if mixing is not None else x.dtype).removeprefix('torch.'),
        )
        dummy = x.flatten()[:1]
        fp32_dummy = dweight_partial.flatten()[:1]
        kernel(
            x.flatten(),
            weight.flatten(),
            branch.flatten() if branch is not None else dummy,
            mixing.flatten() if mixing is not None else dummy,
            dout.flatten(),
            dx.flatten(),
            dbranch.flatten() if dbranch is not None else dummy,
            dweight_partial.flatten(),
            dmixing_partial.flatten() if dmixing_partial is not None else fp32_dummy,
        )
        for partial, grad, size in ((dweight_partial, dweight, N), (dmixing_partial, dmixing, N * N)):
            if grad is not None:
                kernel = hc_bwd_kernel_reduce(
                    NT=NT,
                    C=CD,
                    S=size,
                    BC=triton.next_power_of_2(CD),
                    BS=triton.next_power_of_2(size),
                    dtype=str(grad.dtype).removeprefix('torch.'),
                )
                kernel(partial.flatten(), grad.flatten())
    return dx, dweight, dbranch, dmixing
