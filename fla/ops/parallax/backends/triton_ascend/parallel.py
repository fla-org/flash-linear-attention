# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend launch wrappers for the Parallax training kernels.

At ``D > 128`` (``BK >= 256``) the mainline ``BT = 64`` tiles exceed the
192KiB UB budget (hivm-plan-memory reports 2097152 bits for the backward
preprocess kernel at ``BT = 64, BK = 256``), so the wrappers relaunch the
unchanged kernels with a UB-budgeted ``BT`` shared by forward, backward,
and the varlen chunk indices: 64 for ``BK <= 128`` (the mainline value)
and 32 for ``BK = 256``.
"""

from __future__ import annotations

import torch
import triton
from einops import reduce

from fla.ops.parallax.parallel import (
    parallel_parallax_bwd_kernel_dkv,
    parallel_parallax_bwd_kernel_dqr,
    parallel_parallax_bwd_kernel_preprocess,
    parallel_parallax_fwd_kernel,
)
from fla.ops.utils import prepare_chunk_indices
from fla.utils import ascend_compile_kwargs, autocast_custom_bwd, autocast_custom_fwd, contiguous
from fla.utils.ascend_ub_manager import compute_row_tile_block_size

# Live-set slabs per (BT, BK) element, counted in dtype_size (4 B) units.
# Calibrate per kernel by dividing its compile-failure required-bits report
# by (BT * BK * 32), then take the max across the four kernels: they and
# the varlen chunk indices share one BT schedule.
_TRAIN_MEM_MULT = 4.0
_SAFETY_MARGIN = 0.80
_FALLBACK_BT = 32
_MIN_BT = 16
_MAX_BT = 64


def _train_bt(BK: int) -> int:
    return compute_row_tile_block_size(
        _MAX_BT,
        BK,
        _TRAIN_MEM_MULT,
        tiling_row=True,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=_FALLBACK_BT,
        min_block=_MIN_BT,
        max_block=_MAX_BT,
    )


def parallel_parallax_fwd_npu(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    window_size_left: int = -1,
):
    # Mirror of the mainline forward orchestration; only BT differs — it
    # comes from the UB budget instead of the fixed _block_size.
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    BK = triton.next_power_of_2(K)
    BT = _train_bt(BK)
    o = torch.empty_like(q)
    barv = torch.empty_like(q)
    d1 = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)
    bart = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)
    m = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)

    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    grid = (NT, B * HQ)
    parallel_parallax_fwd_kernel[grid](
        q, r, k, v, o, barv, d1, bart, m,
        scale, cu_seqlens, chunk_indices, T,
        HQ=HQ, H=H, G=G, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left, BT=BT, BS=BT,
        **ascend_compile_kwargs(),
    )
    return o, barv, d1, bart, m


def parallel_parallax_bwd_npu(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    barv: torch.Tensor,
    d1: torch.Tensor,
    bart: torch.Tensor,
    m: torch.Tensor,
    grad_o: torch.Tensor,
    scale,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    window_size_left: int = -1,
):
    # Mirror of the mainline backward orchestration; BT matches the forward.
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    BK = triton.next_power_of_2(K)
    BT = _train_bt(BK)

    grad_q = torch.empty_like(q)
    grad_r = torch.empty_like(r)
    # dK/dV are written per q-head, then folded back to the kv-head axis.
    grad_k_buf = torch.empty((B, T, HQ, K), device=q.device, dtype=q.dtype)
    grad_v_buf = torch.empty((B, T, HQ, K), device=q.device, dtype=q.dtype)
    delta_t = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)
    delta_b = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)

    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    grid = (NT, B * HQ)

    parallel_parallax_bwd_kernel_preprocess[grid](
        grad_o, o, barv, delta_t, delta_b,
        cu_seqlens, chunk_indices, T,
        HQ=HQ, K=K, BK=BK, BT=BT,
        **ascend_compile_kwargs(),
    )
    parallel_parallax_bwd_kernel_dqr[grid](
        q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_q, grad_r,
        scale, cu_seqlens, chunk_indices, T,
        HQ=HQ, H=H, G=G, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left, BT=BT, BS=BT,
        **ascend_compile_kwargs(),
    )
    parallel_parallax_bwd_kernel_dkv[grid](
        q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_k_buf, grad_v_buf,
        scale, cu_seqlens, chunk_indices, T,
        HQ=HQ, H=H, G=G, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left, BT=BT, BS=BT,
        **ascend_compile_kwargs(),
    )

    if G == 1:
        grad_k = grad_k_buf
        grad_v = grad_v_buf
    else:
        grad_k = reduce(grad_k_buf, 'b t (h g) k -> b t h k', g=G, reduction='sum')
        grad_v = reduce(grad_v_buf, 'b t (h g) k -> b t h k', g=G, reduction='sum')
    return grad_q, grad_r, grad_k, grad_v


class ParallaxNpuFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, r, k, v, scale, window_size_left, cu_seqlens):
        chunk_indices = prepare_chunk_indices(cu_seqlens, _train_bt(triton.next_power_of_2(q.shape[-1]))) \
            if cu_seqlens is not None else None
        o, barv, d1, bart, m = parallel_parallax_fwd_npu(q, r, k, v, scale, cu_seqlens, chunk_indices, window_size_left)
        ctx.save_for_backward(q, r, k, v, o, barv, d1, bart, m)
        ctx.scale = scale
        ctx.window_size_left = window_size_left
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        return o

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, r, k, v, o, barv, d1, bart, m = ctx.saved_tensors
        gq, gr, gk, gv = parallel_parallax_bwd_npu(
            q, r, k, v, o, barv, d1, bart, m, do,
            ctx.scale, ctx.cu_seqlens, ctx.chunk_indices, ctx.window_size_left,
        )
        return gq.to(q), gr.to(r), gk.to(k), gv.to(v), None, None, None


def parallel_parallax_npu(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    # Mirror of the mainline host orchestration (fla/ops/parallax/parallel.py).
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"parallel_parallax requires bf16 or fp16 inputs, got q.dtype={q.dtype}")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    HQ, H = q.shape[2], k.shape[2]
    if H == 0 or HQ % H != 0:
        raise ValueError(f"The number of query heads ({HQ}) must be divisible by the number of key/value heads ({H}).")
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
            f"Please flatten variable-length inputs before processing.",
        )
    window_size_left = -1 if window_size is None else window_size
    return ParallaxNpuFunction.apply(q, r, k, v, float(scale), window_size_left, cu_seqlens)


__all__ = ['parallel_parallax_npu']
