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


@tilelang.jit
def hc_pre_fwd_kernel(NT, N, D, x_dtype, pre_dtype, BD):
    BN = triton.next_power_of_2(N)

    @T.prim_func
    def kernel(
        x: T.Tensor((NT, N, D), x_dtype),
        pre: T.Tensor((NT, N), pre_dtype),
        o: T.Tensor((NT, D), x_dtype),
    ):
        with T.Kernel(NT, T.ceildiv(D, BD), threads=128) as (token, block):
            i_t = T.cast(token, 'int64')
            i_d = T.cast(block, 'int64') * BD
            b_o = T.alloc_fragment((BN, BD), 'float32')
            b_pre = T.alloc_fragment((BN,), 'float32')
            b_reduced = T.alloc_fragment((BD,), 'float32')
            if N <= 4:
                T.annotate_layout({
                    b_o: T.Fragment(
                        shape=(BN, BD),
                        forward_thread_fn=lambda i, d: d % 128,
                        forward_index_fn=lambda i, d: i * (BD // 128) + d // 128,
                    ),
                })
            T.copy(x[i_t, 0, i_d], b_o)
            T.copy(pre[i_t, 0], b_pre)
            for i, d in T.Parallel(BN, BD):
                b_o[i, d] *= b_pre[i]
            T.reduce_sum(b_o, b_reduced, dim=0)
            T.copy(b_reduced, o[i_t, i_d])
    return kernel


@tilelang.jit
def hc_post_fwd_kernel(NT, N, D, x_dtype, branch_dtype, post_dtype, mixing_dtype, BD, HAS_MIXING):
    BN = triton.next_power_of_2(N)

    @T.prim_func
    def kernel(
        x: T.Tensor((NT, N, D), x_dtype),
        branch: T.Tensor((NT, D), branch_dtype),
        post: T.Tensor((NT, N), post_dtype),
        mixing: T.Tensor((NT, N, N), mixing_dtype),
        o: T.Tensor((NT, N, D), x_dtype),
    ):
        with T.Kernel(NT, T.ceildiv(D, BD), threads=128) as (token, block):
            i_t = T.cast(token, 'int64')
            i_d = T.cast(block, 'int64') * BD
            b_o = T.alloc_fragment((BN, BD), 'float32')
            # keep broadcast loads in the 2D loop so layout inference can distribute both axes
            if HAS_MIXING:
                T.clear(b_o)
                for k in T.unroll(N):
                    for i, d in T.Parallel(BN, BD):
                        b_o[i, d] += T.if_then_else(
                            i < N and i_d + d < D,
                            T.cast(mixing[i_t, i, k], 'float32') * T.cast(x[i_t, k, i_d + d], 'float32'),
                            0.0,
                        )
            else:
                T.copy(x[i_t, 0, i_d], b_o)
            for i, d in T.Parallel(BN, BD):
                b_o[i, d] += T.if_then_else(
                    i < N and i_d + d < D,
                    T.cast(post[i_t, i], 'float32') * T.cast(branch[i_t, i_d + d], 'float32'),
                    0.0,
                )
            T.copy(b_o, o[i_t, 0, i_d])
    return kernel


@tilelang.jit
def hc_pre_bwd_kernel(NT, N, D, x_dtype, pre_dtype, BD):
    BN = triton.next_power_of_2(N)
    ND = tilelang.cdiv(D, BD)

    @T.prim_func
    def kernel(
        x: T.Tensor((NT, N, D), x_dtype),
        pre: T.Tensor((NT, N), pre_dtype),
        do: T.Tensor((NT, D), x_dtype),
        dx: T.Tensor((NT, N, D), x_dtype),
        dpre: T.Tensor((NT, ND, N), 'float32'),
    ):
        with T.Kernel(NT, ND, threads=128) as (token, block):
            i_t = T.cast(token, 'int64')
            i_c = T.cast(block, 'int64')
            i_d = i_c * BD
            b_do = T.alloc_fragment((BD,), 'float32')
            b_pre = T.alloc_fragment((BN,), 'float32')
            b_tmp = T.alloc_fragment((BN, BD), 'float32')
            b_dx = T.alloc_fragment((BN, BD), 'float32')
            b_dpre = T.alloc_fragment((BN,), 'float32')
            T.copy(x[i_t, 0, i_d], b_tmp)
            T.copy(pre[i_t, 0], b_pre)
            T.copy(do[i_t, i_d], b_do)
            for i, d in T.Parallel(BN, BD):
                b_tmp[i, d] *= b_do[d]
                b_dx[i, d] = b_pre[i] * b_do[d]
            T.reduce_sum(b_tmp, b_dpre, dim=1)
            T.copy(b_dx, dx[i_t, 0, i_d])
            T.copy(b_dpre, dpre[i_t, i_c, 0])
    return kernel


@tilelang.jit
def hc_post_bwd_kernel(NT, N, D, x_dtype, branch_dtype, post_dtype, mixing_dtype, BD, HAS_MIXING):
    BN = triton.next_power_of_2(N)
    # keep feature reductions within a warp and limit live do/dx values per thread
    WN = triton.cdiv(BN, 4)
    R = min(4, BD) if D % 4 == 0 else 1
    TD = min(BD // R, 32)
    layout = T.Fragment(
        shape=(BN, BD),
        replicate=32 // TD,
        forward_thread_fn=lambda i, d, rep=0: (i % WN) * TD + (d // R) % TD + rep * WN * TD,
        forward_index_fn=lambda i, d: (i // WN) * triton.cdiv(BD, TD) + (d // (R * TD)) * R + d % R,
    )
    ND = tilelang.cdiv(D, BD)

    @T.prim_func
    def kernel(
        x: T.Tensor((NT, N, D), x_dtype),
        branch: T.Tensor((NT, D), branch_dtype),
        post: T.Tensor((NT, N), post_dtype),
        mixing: T.Tensor((NT, N, N), mixing_dtype),
        do: T.Tensor((NT, N, D), x_dtype),
        dx: T.Tensor((NT, N, D), x_dtype),
        dbranch: T.Tensor((NT, D), branch_dtype),
        dpost: T.Tensor((NT, ND, N), 'float32'),
        dmixing: T.Tensor((NT, ND, N, N), 'float32'),
    ):
        with T.Kernel(NT, ND, threads=32 * WN if HAS_MIXING else 128) as (token, block):
            i_t = T.cast(token, 'int64')
            i_c = T.cast(block, 'int64')
            i_d = i_c * BD
            b_do = T.alloc_fragment((BN, BD), 'float32')
            b_tmp = T.alloc_fragment((BN, BD), 'float32')
            b_dx = T.alloc_fragment((BN, BD), 'float32')
            b_dpost = T.alloc_fragment((BN,), 'float32')
            b_dbranch = T.alloc_fragment((BD,), 'float32')
            if HAS_MIXING and N > 4:
                T.annotate_layout({b_do: layout, b_tmp: layout, b_dx: layout})
            for i, d in T.Parallel(BN, BD):
                b_do[i, d] = T.if_then_else(i < N and i_d + d < D, do[i_t, i, i_d + d], 0)
                b_tmp[i, d] = b_do[i, d] * T.if_then_else(
                    i_d + d < D, T.cast(branch[i_t, i_d + d], 'float32'), 0.0,
                )
            T.reduce_sum(b_tmp, b_dpost, dim=1)
            for i, d in T.Parallel(BN, BD):
                b_tmp[i, d] = b_do[i, d] * T.if_then_else(i < N, T.cast(post[i_t, i], 'float32'), 0.0)
            T.reduce_sum(b_tmp, b_dbranch, dim=0)
            T.copy(b_dbranch, dbranch[i_t, i_d])
            if HAS_MIXING:
                b_dmixing = T.alloc_fragment((BN,), 'float32')
                T.clear(b_dx)
                # transpose the mixing axes to propagate gradients to input streams
                for k in T.unroll(N):
                    for i, d in T.Parallel(BN, BD):
                        b_dx[i, d] += T.if_then_else(
                            i < N and i_d + d < D,
                            T.cast(mixing[i_t, k, i], 'float32') * T.cast(do[i_t, k, i_d + d], 'float32'),
                            0.0,
                        )
                        b_tmp[i, d] = b_do[i, d] * T.if_then_else(
                            i_d + d < D, T.cast(x[i_t, k, i_d + d], 'float32'), 0.0,
                        )
                    T.reduce_sum(b_tmp, b_dmixing, dim=1)
                    for i in T.Parallel(BN):
                        if i < N:
                            dmixing[i_t, i_c, i, k] = b_dmixing[i]
            else:
                T.copy(b_do, b_dx)
            T.copy(b_dx, dx[i_t, 0, i_d])
            T.copy(b_dpost, dpost[i_t, i_c, 0])
    return kernel


@tilelang.jit
def hc_bwd_kernel_reduce(NT, C, S, dtype):
    BC, BS = triton.next_power_of_2(C), triton.next_power_of_2(S)

    @T.prim_func
    def kernel(partial: T.Tensor((NT, C, S), 'float32'), grad: T.Tensor((NT, S), dtype)):
        with T.Kernel(NT, threads=128) as token:
            i_t = T.cast(token, 'int64')
            b_partial = T.alloc_fragment((BC, BS), 'float32')
            b_reduced = T.alloc_fragment((BS,), 'float32')
            T.copy(partial[i_t, 0, 0], b_partial)
            T.reduce_sum(b_partial, b_reduced, dim=0)
            T.copy(b_reduced, grad[i_t, 0])
    return kernel


def hc_fwd_tilelang(
    x: torch.Tensor,
    w: torch.Tensor,
    branch: torch.Tensor | None,
    mixing: torch.Tensor | None,
    is_post: bool,
):
    N, D = x.shape[-2:]
    NT = x.numel() // (N * D)
    BD = 256 if NT < 128 or (not is_post and N > 4) else 1024
    if mixing is not None and NT < 128:
        # expose more feature tiles when there are too few tokens to fill the GPU
        BD = 128
    BD = max(128, min(BD, triton.next_power_of_2(D)))
    if is_post:
        o = torch.empty_like(x)
        kernel = hc_post_fwd_kernel(
            NT=NT,
            N=N,
            D=D,
            x_dtype=x.dtype,
            branch_dtype=branch.dtype,
            post_dtype=w.dtype,
            mixing_dtype=mixing.dtype if mixing is not None else x.dtype,
            BD=BD,
            HAS_MIXING=mixing is not None,
        )
        kernel(
            x.view(NT, N, D),
            branch.view(NT, D),
            w.view(NT, N),
            mixing.view(NT, N, N) if mixing is not None else None,
            o.view(NT, N, D),
        )
    else:
        o = x.new_empty((*x.shape[:-2], D))
        kernel = hc_pre_fwd_kernel(NT=NT, N=N, D=D, x_dtype=x.dtype, pre_dtype=w.dtype, BD=BD)
        kernel(x.view(NT, N, D), w.view(NT, N), o.view(NT, D))
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
    dw_partial = torch.empty((NT, ND, N), device=x.device, dtype=torch.float32)
    dbranch, dmixing, dmixing_partial = None, None, None
    if is_post:
        dbranch = torch.empty_like(branch)
        if mixing is not None:
            dmixing = torch.empty_like(mixing)
            dmixing_partial = torch.empty((NT, ND, N, N), device=x.device, dtype=torch.float32)
        kernel = hc_post_bwd_kernel(
            NT=NT,
            N=N,
            D=D,
            x_dtype=x.dtype,
            branch_dtype=branch.dtype,
            post_dtype=w.dtype,
            mixing_dtype=mixing.dtype if mixing is not None else x.dtype,
            BD=BD,
            HAS_MIXING=mixing is not None,
        )
        kernel(
            x.view(NT, N, D),
            branch.view(NT, D),
            w.view(NT, N),
            mixing.view(NT, N, N) if mixing is not None else None,
            do.view(NT, N, D),
            dx.view(NT, N, D),
            dbranch.view(NT, D),
            dw_partial,
            dmixing_partial,
        )
    else:
        kernel = hc_pre_bwd_kernel(NT=NT, N=N, D=D, x_dtype=x.dtype, pre_dtype=w.dtype, BD=BD)
        kernel(x.view(NT, N, D), w.view(NT, N), do.view(NT, D), dx.view(NT, N, D), dw_partial)
    for partial, grad, size in ((dw_partial, dw, N), (dmixing_partial, dmixing, N * N)):
        if grad is not None:
            kernel = hc_bwd_kernel_reduce(NT=NT, C=ND, S=size, dtype=grad.dtype)
            kernel(partial.view(NT, ND, size), grad.view(NT, size))
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


@register(hc_pre, backend=TileLangBackend, default_enable=False)
def hc_pre_tilelang(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    return FusedHCPreTileLangFunction.apply(x, pre)


@register(hc_post, backend=TileLangBackend, default_enable=False)
def hc_post_tilelang(
    x: torch.Tensor,
    branch: torch.Tensor,
    post: torch.Tensor,
    mixing: torch.Tensor | None = None,
) -> torch.Tensor:
    return FusedHCPostTileLangFunction.apply(x, branch, post, mixing)
