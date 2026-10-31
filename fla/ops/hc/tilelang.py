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
from torch.autograd.function import once_differentiable

from fla.backends import TileLangBackend, register
from fla.ops.hc.fused import hc_post, hc_pre
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


class HCTileLangBackend(TileLangBackend):
    default_enable = False


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--fmad=false']})
def hc_fwd_kernel(NT, N, D, BN, BD, IS_POST, HAS_MIXING, x_dtype, w_dtype, branch_dtype, mixing_dtype):
    ND = tilelang.cdiv(D, BD)

    # flattened buffers: x [NT, N, D], w [NT, N], branch [NT, D], mixing [NT, N, N]
    @T.prim_func
    def kernel(
        x: T.Tensor((NT * N * D,), x_dtype),
        w: T.Tensor((NT * N,), w_dtype),
        branch: T.Tensor((NT * D if IS_POST else 1,), branch_dtype),
        mixing: T.Tensor((NT * N * N if HAS_MIXING else 1,), mixing_dtype),
        o: T.Tensor((NT * N * D if IS_POST else NT * D,), x_dtype),
    ):
        with T.Kernel(NT, ND, threads=128) as (token, block):
            i_t = T.cast(token, 'int64')
            i_c = T.cast(block, 'int64')
            b_o = T.alloc_fragment((BN, BD), 'float32')
            b_reduced = T.alloc_fragment((BD,), 'float32')
            if not IS_POST and N <= 4:
                T.annotate_layout({
                    b_o: T.Fragment(
                        shape=(BN, BD),
                        forward_thread_fn=lambda i, d: d % 128,
                        forward_index_fn=lambda i, d: i * (BD // 128) + d // 128,
                    ),
                })
            if IS_POST:
                if HAS_MIXING:
                    T.clear(b_o)
                    # mixing[token, output stream, input stream] mixes x into a [BN, BD] tile
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
                            b_o[i, d] + T.cast(w[i_t * N + i], 'float32') *
                            T.cast(branch[i_t * D + i_c * BD + d], 'float32')
                        )
            else:
                for i, d in T.Parallel(BN, BD):
                    b_o[i, d] = T.if_then_else(
                        i < N and i_c * BD + d < D,
                        T.cast(x[(i_t * N + i) * D + i_c * BD + d], 'float32') * T.cast(w[i_t * N + i], 'float32'),
                        0.0,
                    )
                # [BN, BD] -> [BD], reducing over residual streams
                T.reduce_sum(b_o, b_reduced, dim=0)
                for d in T.Parallel(BD):
                    if i_c * BD + d < D:
                        o[i_t * D + i_c * BD + d] = b_reduced[d]
    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ['--fmad=false']})
def hc_bwd_kernel(NT, N, D, BN, BD, IS_POST, HAS_MIXING, x_dtype, w_dtype, branch_dtype, mixing_dtype):
    ND = tilelang.cdiv(D, BD)

    # partial gradients: dw [NT, ND, N], dmixing [NT, ND, N, N]
    @T.prim_func
    def kernel(
        x: T.Tensor((NT * N * D,), x_dtype),
        w: T.Tensor((NT * N,), w_dtype),
        branch: T.Tensor((NT * D if IS_POST else 1,), branch_dtype),
        mixing: T.Tensor((NT * N * N if HAS_MIXING else 1,), mixing_dtype),
        do: T.Tensor((NT * N * D if IS_POST else NT * D,), x_dtype),
        dx: T.Tensor((NT * N * D,), x_dtype),
        dbranch: T.Tensor((NT * D if IS_POST else 1,), branch_dtype),
        dw_partial: T.Tensor((NT * ND * N,), 'float32'),
        dmixing_partial: T.Tensor((NT * ND * N * N if HAS_MIXING else 1,), 'float32'),
    ):
        with T.Kernel(NT, ND, threads=128) as (token, block):
            i_t = T.cast(token, 'int64')
            i_c = T.cast(block, 'int64')
            b_do = T.alloc_fragment((BN, BD), 'float32')
            b_tmp = T.alloc_fragment((BN, BD), 'float32')
            b_dx = T.alloc_fragment((BN, BD), 'float32')
            b_dw = T.alloc_fragment((BN,), 'float32')
            b_dmixing = T.alloc_fragment((BN,), 'float32')
            b_dbranch = T.alloc_fragment((BD,), 'float32')
            if IS_POST:
                for i, d in T.Parallel(BN, BD):
                    b_do[i, d] = T.if_then_else(i < N and i_c * BD + d < D, do[(i_t * N + i) * D + i_c * BD + d], 0)
                    b_tmp[i, d] = b_do[i, d] * T.if_then_else(
                        i_c * BD + d < D,
                        T.cast(branch[i_t * D + i_c * BD + d], 'float32'),
                        0.0,
                    )
                # [BN, BD] -> [BN], reducing over this feature tile
                T.reduce_sum(b_tmp, b_dw, dim=1)
                for i, d in T.Parallel(BN, BD):
                    b_tmp[i, d] = b_do[i, d] * T.if_then_else(i < N, T.cast(w[i_t * N + i], 'float32'), 0.0)
                # [BN, BD] -> [BD], reducing over output streams
                T.reduce_sum(b_tmp, b_dbranch, dim=0)
                for d in T.Parallel(BD):
                    if i_c * BD + d < D:
                        dbranch[i_t * D + i_c * BD + d] = b_dbranch[d]
                if HAS_MIXING:
                    T.clear(b_dx)
                    # transpose the mixing axes to propagate gradients to input streams
                    for k in T.serial(N):
                        for i, d in T.Parallel(BN, BD):
                            b_dx[i, d] += T.if_then_else(
                                i < N and i_c * BD + d < D,
                                T.cast(mixing[(i_t * N + k) * N + i], 'float32') *
                                T.cast(do[(i_t * N + k) * D + i_c * BD + d], 'float32'),
                                0.0,
                            )
                            b_tmp[i, d] = b_do[i, d] * T.if_then_else(
                                i_c * BD + d < D,
                                T.cast(x[(i_t * N + k) * D + i_c * BD + d], 'float32'),
                                0.0,
                            )
                        # [BN, BD] -> [BN], one column of the mixing gradient
                        T.reduce_sum(b_tmp, b_dmixing, dim=1)
                        for i in T.Parallel(BN):
                            if i < N:
                                dmixing_partial[((i_t * ND + i_c) * N + i) * N + k] = b_dmixing[i]
                else:
                    T.copy(b_do, b_dx)
            else:
                # broadcast the pre-output gradient [BD] across residual streams [BN, BD]
                for i, d in T.Parallel(BN, BD):
                    b_do[i, d] = T.if_then_else(i_c * BD + d < D, do[i_t * D + i_c * BD + d], 0)
                    b_dx[i, d] = b_do[i, d] * T.if_then_else(i < N, T.cast(w[i_t * N + i], 'float32'), 0.0)
                    b_tmp[i, d] = b_do[i, d] * T.if_then_else(
                        i < N and i_c * BD + d < D,
                        T.cast(x[(i_t * N + i) * D + i_c * BD + d], 'float32'),
                        0.0,
                    )
                # [BN, BD] -> [BN]
                T.reduce_sum(b_tmp, b_dw, dim=1)
            for i, d in T.Parallel(BN, BD):
                if i < N and i_c * BD + d < D:
                    dx[(i_t * N + i) * D + i_c * BD + d] = b_dx[i, d]
            for i in T.Parallel(BN):
                if i < N:
                    dw_partial[(i_t * ND + i_c) * N + i] = b_dw[i]
    return kernel


@tilelang.jit
def hc_bwd_kernel_reduce(NT, C, S, BC, BS, dtype):
    # feature tiles along C; S is N for routing weights or N * N for mixing weights
    @T.prim_func
    def kernel(partial: T.Tensor((NT * C * S,), 'float32'), grad: T.Tensor((NT * S,), dtype)):
        with T.Kernel(NT, threads=128) as token:
            i_t = T.cast(token, 'int64')
            b_partial = T.alloc_fragment((BC, BS), 'float32')
            b_reduced = T.alloc_fragment((BS,), 'float32')
            for i_c, s in T.Parallel(BC, BS):
                b_partial[i_c, s] = T.if_then_else(i_c < C and s < S, partial[(i_t * C + i_c) * S + s], 0)
            # [BC, BS] -> [BS]
            T.reduce_sum(b_partial, b_reduced, dim=0)
            for s in T.Parallel(BS):
                if s < S:
                    grad[i_t * S + s] = b_reduced[s]
    return kernel


def hc_fwd_tilelang(
    x: torch.Tensor,
    w: torch.Tensor,
    branch: torch.Tensor | None,
    mixing: torch.Tensor | None,
    is_post: bool,
):
    N, D = x.shape[-2:]
    # flatten all leading batch/sequence dimensions into NT tokens
    NT = x.numel() // (N * D)
    BD = max(128, min(256 if NT < 128 or is_post or N > 4 else 1024, triton.next_power_of_2(D)))
    # [..., N, D] for post; [..., D] for pre
    o = torch.empty_like(x) if is_post else x.new_empty((*x.shape[:-2], D))
    kernel = hc_fwd_kernel(
        NT=NT,
        N=N,
        D=D,
        BN=triton.next_power_of_2(N),
        BD=BD,
        IS_POST=is_post,
        HAS_MIXING=mixing is not None,
        x_dtype=str(x.dtype).removeprefix('torch.'),
        w_dtype=str(w.dtype).removeprefix('torch.'),
        branch_dtype=str(branch.dtype if is_post else x.dtype).removeprefix('torch.'),
        mixing_dtype=str(mixing.dtype if mixing is not None else x.dtype).removeprefix('torch.'),
    )
    dummy = x.flatten()[:1]
    kernel(
        x.flatten(),
        w.flatten(),
        branch.flatten() if is_post else dummy,
        mixing.flatten() if mixing is not None else dummy,
        o.flatten(),
    )
    return o


def hc_bwd_tilelang(
    x: torch.Tensor,
    w: torch.Tensor,
    branch: torch.Tensor | None,
    mixing: torch.Tensor | None,
    do: torch.Tensor,
    is_post: bool,
):
    N, D = x.shape[-2:]
    NT = x.numel() // (N * D)
    BD = 256 if NT < 128 or mixing is not None or (not is_post and N <= 4) else 512
    BD = min(BD, triton.next_power_of_2(D))
    ND = triton.cdiv(D, BD)
    dx, dw = torch.empty_like(x), torch.empty_like(w)
    dbranch = torch.empty_like(branch) if is_post else None
    dmixing = torch.empty_like(mixing) if mixing is not None else None
    dw_partial = torch.empty((NT, ND, N), device=x.device, dtype=torch.float32)
    dmixing_partial = torch.empty((NT, ND, N * N), device=x.device, dtype=torch.float32) if mixing is not None else None
    kernel = hc_bwd_kernel(
        NT=NT,
        N=N,
        D=D,
        BN=triton.next_power_of_2(N),
        BD=BD,
        IS_POST=is_post,
        HAS_MIXING=mixing is not None,
        x_dtype=str(x.dtype).removeprefix('torch.'),
        w_dtype=str(w.dtype).removeprefix('torch.'),
        branch_dtype=str(branch.dtype if is_post else x.dtype).removeprefix('torch.'),
        mixing_dtype=str(mixing.dtype if mixing is not None else x.dtype).removeprefix('torch.'),
    )
    dummy = x.flatten()[:1]
    fp32_dummy = dw_partial.flatten()[:1]
    kernel(
        x.flatten(),
        w.flatten(),
        branch.flatten() if is_post else dummy,
        mixing.flatten() if mixing is not None else dummy,
        do.flatten(),
        dx.flatten(),
        dbranch.flatten() if dbranch is not None else dummy,
        dw_partial.flatten(),
        dmixing_partial.flatten() if dmixing_partial is not None else fp32_dummy,
    )
    for partial, grad, size in ((dw_partial, dw, N), (dmixing_partial, dmixing, N * N)):
        if grad is not None:
            kernel = hc_bwd_kernel_reduce(
                NT=NT,
                C=ND,
                S=size,
                BC=triton.next_power_of_2(ND),
                BS=triton.next_power_of_2(size),
                dtype=str(grad.dtype).removeprefix('torch.'),
            )
            kernel(partial.flatten(), grad.flatten())
    return dx, dw, dbranch, dmixing


class FusedHCPreTileLangFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, pre):
        o = hc_fwd_tilelang(x=x, w=pre, branch=None, mixing=None, is_post=False)
        ctx.save_for_backward(x, pre)
        return o

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    @once_differentiable
    def backward(ctx, do):
        x, pre = ctx.saved_tensors
        dx, dpre, _, _ = hc_bwd_tilelang(x=x, w=pre, branch=None, mixing=None, do=do, is_post=False)
        return dx, dpre


class FusedHCPostTileLangFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, branch, post, mixing):
        o = hc_fwd_tilelang(x=x, w=post, branch=branch, mixing=mixing, is_post=True)
        ctx.save_for_backward(x, branch, post, mixing)
        return o

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    @once_differentiable
    def backward(ctx, do):
        x, branch, post, mixing = ctx.saved_tensors
        dx, dpost, dbranch, dmixing = hc_bwd_tilelang(x=x, w=post, branch=branch, mixing=mixing, do=do, is_post=True)
        return dx, dbranch, dpost, dmixing


@register(hc_pre, backend=HCTileLangBackend)
def hc_pre_tilelang(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    return FusedHCPreTileLangFunction.apply(x, pre)


@register(hc_post, backend=HCTileLangBackend)
def hc_post_tilelang(x: torch.Tensor, branch: torch.Tensor, post: torch.Tensor, mixing: torch.Tensor | None = None) -> torch.Tensor:
    return FusedHCPostTileLangFunction.apply(x, branch, post, mixing)
