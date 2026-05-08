# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp
from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    input_guard,
)

ATTNRES_FWD_BWD_CONFIGS = [
    triton.Config({'BD': BD}, num_warps=num_warps, num_stages=num_stages)
    for BD, num_warps in [(64, 1), (128, 2), (256, 4), (512, 4), (1024, 8)]
    for num_stages in [3, 4]
]

ATTNRES_DW_CONFIGS = [
    triton.Config({'BN': BN, 'BD': BD}, num_warps=num_warps, num_stages=num_stages)
    for BN, BD, num_warps in [(1024, 16, 4), (2048, 32, 4), (2048, 32, 8), (4096, 32, 8), (4096, 64, 8)]
    for num_stages in [3, 4]
]


@fla_cache_autotune(
    configs=ATTNRES_FWD_BWD_CONFIGS,
    key=['L', 'D'],
    **autotune_cache_kwargs,
)
@triton.jit
def attnres_fwd_kernel(
    q,
    v,
    w,
    o,
    rstd,
    p,
    N,
    L: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BL: tl.constexpr,
    BD: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)

    # [BL]
    m_l = tl.arange(0, BL) < L

    # [BL]
    b_var = tl.zeros([BL], dtype=tl.float32)
    b_logits = tl.zeros([BL], dtype=tl.float32)
    for i_d in range(0, D, BD):
        # [BD]
        o_d = i_d + tl.arange(0, BD)
        m_d = o_d < D
        p_v = tl.make_block_ptr(v + i_n * D, (L, D), (N * D, 1), (0, i_d), (BL, BD), (1, 0))
        # [BL, BD]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        # [BD]
        b_w = tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)
        b_q = tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32)

        b_var += tl.sum(b_v * b_v, axis=1)
        b_logits += tl.sum(b_v * (b_w * b_q)[None, :], axis=1)

    # [BL]
    b_rstd = tl.rsqrt(b_var / D + eps)
    b_logits *= b_rstd * scale
    b_logits = tl.where(m_l, b_logits, -float("inf"))
    b_logits = exp(b_logits - tl.max(b_logits, axis=0))
    # [BL]
    b_p = b_logits / tl.sum(b_logits, axis=0)

    p_rstd = tl.make_block_ptr(rstd + i_n, (L,), (N,), (0,), (BL,), (0,))
    p_p = tl.make_block_ptr(p + i_n, (L,), (N,), (0,), (BL,), (0,))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))
    tl.store(p_p, b_p.to(p_p.dtype.element_ty), boundary_check=(0,))

    for i_d in range(0, D, BD):
        p_v = tl.make_block_ptr(v + i_n * D, (L, D), (N * D, 1), (0, i_d), (BL, BD), (1, 0))
        p_o = tl.make_block_ptr(o + i_n * D, (D,), (1,), (i_d,), (BD,), (0,))
        # [BL, BD]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        # [BD]
        b_o = tl.sum(b_v * b_p[:, None], axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@fla_cache_autotune(
    configs=ATTNRES_FWD_BWD_CONFIGS,
    key=['L', 'D'],
    **autotune_cache_kwargs,
)
@triton.jit
def attnres_bwd_kernel_dv(
    q,
    v,
    w,
    rstd,
    p,
    do,
    dv,
    dqw,
    N,
    scale: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BL: tl.constexpr,
    BD: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)

    p_p = tl.make_block_ptr(p + i_n, (L,), (N,), (0,), (BL,), (0,))
    p_rstd = tl.make_block_ptr(rstd + i_n, (L,), (N,), (0,), (BL,), (0,))
    # [BL]
    b_p = tl.load(p_p, boundary_check=(0,), padding_option="zero").to(tl.float32)
    b_rstd = tl.load(p_rstd, boundary_check=(0,), padding_option="zero").to(tl.float32)

    # [BL]
    b_dp = tl.zeros([BL], dtype=tl.float32)
    b_z = tl.zeros([BL], dtype=tl.float32)
    for i_d in range(0, D, BD):
        # [BD]
        o_d = i_d + tl.arange(0, BD)
        m_d = o_d < D
        p_v = tl.make_block_ptr(v + i_n * D, (L, D), (N * D, 1), (0, i_d), (BL, BD), (1, 0))
        p_do = tl.make_block_ptr(do + i_n * D, (D,), (1,), (i_d,), (BD,), (0,))
        # [BL, BD]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        # [BD]
        b_do = tl.load(p_do, boundary_check=(0,)).to(tl.float32)
        b_w = tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)
        b_q = tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32)

        # [BL, BD]
        b_xhat = b_v * b_rstd[:, None]
        b_dp += tl.sum(b_v * b_do[None, :], axis=1)
        b_z += tl.sum(b_xhat * (b_w * b_q)[None, :], axis=1)

    # softmax bwd via the standard delta trick: delta = sum_l p*dp = sum_d do*o
    # [1]
    b_delta = tl.sum(b_p * b_dp, axis=0)
    # [BL]
    b_ds = b_p * (b_dp - b_delta) * scale
    # rstd-coupling correction (same role as `c1` in layernorm bwd)
    # [BL]
    b_c1 = b_z / D

    for i_d in range(0, D, BD):
        # [BD]
        o_d = i_d + tl.arange(0, BD)
        m_d = o_d < D
        p_v = tl.make_block_ptr(v + i_n * D, (L, D), (N * D, 1), (0, i_d), (BL, BD), (1, 0))
        p_do = tl.make_block_ptr(do + i_n * D, (D,), (1,), (i_d,), (BD,), (0,))
        # [BL, BD]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        # [BD]
        b_do = tl.load(p_do, boundary_check=(0,)).to(tl.float32)
        b_w = tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)
        b_q = tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32)

        # [BL, BD]
        b_xhat = b_v * b_rstd[:, None]
        b_dv = b_p[:, None] * b_do[None, :] + b_ds[:, None] * b_rstd[:, None] * (
            (b_w * b_q)[None, :] - b_xhat * b_c1[:, None]
        )

        p_dv = tl.make_block_ptr(dv + i_n * D, (L, D), (N * D, 1), (0, i_d), (BL, BD), (1, 0))
        p_base = tl.make_block_ptr(dqw + i_n * D, (D,), (1,), (i_d,), (BD,), (0,))
        tl.store(p_dv, b_dv.to(dv.dtype.element_ty), boundary_check=(0, 1))
        # [BD]
        tl.store(p_base, tl.sum(b_ds[:, None] * b_xhat, axis=0), boundary_check=(0,))


@fla_cache_autotune(
    configs=ATTNRES_DW_CONFIGS,
    key=['N', 'D'],
    **autotune_cache_kwargs,
)
@triton.jit
def attnres_bwd_kernel_dqdw(
    q,
    w,
    dqw,
    dq,
    dw,
    N,
    D: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    i_d = tl.program_id(0).to(tl.int32)

    # [BD]
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    # [BD]
    b_acc = tl.zeros([BD], dtype=tl.float32)
    for i_n in range(0, N, BN):
        p_base = tl.make_block_ptr(dqw, (N, D), (D, 1), (i_n, i_d * BD), (BN, BD), (1, 0))
        # [BN, BD]
        b_base = tl.load(p_base, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_acc += tl.sum(b_base, axis=0)

    # [BD]
    b_q = tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32)
    b_w = tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)

    tl.store(dq + o_d, b_acc * b_w, mask=m_d)
    tl.store(dw + o_d, b_acc * b_q, mask=m_d)


def fused_attnres_fwd(
    query: torch.Tensor,
    residuals: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not residuals.is_cuda:
        raise ValueError("Triton attnres requires CUDA tensors")
    if residuals.shape[0] < 1:
        raise ValueError("Triton attnres requires at least one residual source")

    output_shape = residuals.shape[1:]
    L, N, D = residuals.shape[0], residuals[0].numel() // residuals.shape[-1], residuals.shape[-1]
    q, v, w = query.view(-1), residuals.view(L, N, D), rms_weight.view(-1)

    o = torch.empty((N, D), device=residuals.device, dtype=residuals.dtype)
    stats_shape = (L, *output_shape[:-1])
    rstd = torch.empty((L, N), device=residuals.device, dtype=torch.float32)
    p = torch.empty_like(rstd)

    BL = max(8, triton.next_power_of_2(L))
    grid = (N,)

    attnres_fwd_kernel[grid](
        q=q,
        v=v,
        w=w,
        o=o,
        rstd=rstd,
        p=p,
        N=N,
        L=L,
        D=D,
        eps=rms_eps,
        scale=scale,
        BL=BL,
    )
    return o.view(output_shape), rstd.view(stats_shape), p.view(stats_shape)


def fused_attnres_bwd(
    do: torch.Tensor,
    rstd: torch.Tensor,
    p: torch.Tensor,
    query: torch.Tensor,
    residuals: torch.Tensor,
    rms_weight: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L, N, D = residuals.shape[0], do.numel() // do.shape[-1], do.shape[-1]
    residuals_shape = residuals.shape
    query_shape = query.shape
    rms_weight_shape = rms_weight.shape

    q = query.view(-1)
    v = residuals.view(L, N, D)
    p = p.view(L, N)
    w = rms_weight.view(-1)
    rstd = rstd.view(L, N)

    do = do.view(N, D)
    dv = torch.empty((L, N, D), dtype=do.dtype, device=do.device)
    dqw = torch.empty((N, D), dtype=torch.float32, device=do.device)
    dq = torch.empty_like(q)
    dw = torch.empty_like(w)

    BL = max(8, triton.next_power_of_2(L))
    attnres_bwd_kernel_dv[(N,)](
        q=q,
        v=v,
        w=w,
        rstd=rstd,
        p=p,
        do=do,
        dv=dv,
        dqw=dqw,
        N=N,
        scale=scale,
        L=L,
        D=D,
        BL=BL,
    )

    def grid(meta): return (triton.cdiv(D, meta['BD']),)
    attnres_bwd_kernel_dqdw[grid](
        q=q,
        w=w,
        dqw=dqw,
        dq=dq,
        dw=dw,
        N=N,
        D=D,
    )

    return dv.view(residuals_shape), dq.view(query_shape), dw.view(rms_weight_shape)


class FusedAttnresFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        query: torch.Tensor,
        residuals: torch.Tensor,
        rms_weight: torch.Tensor,
        rms_eps: float,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        o, rstd, p = fused_attnres_fwd(query, residuals, rms_weight, rms_eps, scale)
        ctx.save_for_backward(rstd, p, query, residuals, rms_weight)
        ctx.scale = scale
        ctx.mark_non_differentiable(p)
        return o, p

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        del dp
        rstd, p, query, residuals, rms_weight = ctx.saved_tensors
        dv, dq, dw = fused_attnres_bwd(
            do,
            rstd,
            p,
            query,
            residuals,
            rms_weight,
            ctx.scale,
        )
        return dq, dv, dw, None, None


def fused_attnres(
    query: torch.Tensor,
    residuals: torch.Tensor | Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
    return_residuals: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""
    Apply AttnRes residual aggregation.

    AttnRes normalizes each residual source with RMSNorm, scores it against
    `query`, applies softmax over the residual-source dimension, and returns
    the weighted sum of residual sources.
    See `Attention Residuals <https://arxiv.org/abs/2603.15031>`_.

    Args:
        query (torch.Tensor):
            Per-layer pseudo-query of shape `[D]` or `[D, 1]`, where `D` is
            the hidden size.
        residuals (torch.Tensor or Sequence[torch.Tensor]):
            Residual sources of shape `[L, ..., D]`, or a sequence of tensors
            each with shape `[..., D]`, where `L` is the number of residual
            sources.
        rms_weight (torch.Tensor):
            RMSNorm scale for key normalization of shape `[D]`.
        rms_eps (float):
            RMSNorm epsilon. Default: `1e-6`.
        scale (float):
            Scale factor applied to AttnRes logits before softmax. Default: `1.0`.
        return_weights (bool):
            Whether to return depth softmax probabilities. Default: `False`.
        return_residuals (bool):
            Whether to return the stacked residual tensor. Useful when the
            input was a list and the caller wants the materialized
            `[L, ..., D]` tensor without re-stacking. Default: `False`.

    Returns:
        o (torch.Tensor):
            Mixed residual of shape `[..., D]`.
        p (torch.Tensor):
            Depth softmax probabilities of shape `[L, ...]` if
            `return_weights=True`, otherwise not returned.
        residuals (torch.Tensor):
            Stacked residuals of shape `[L, ..., D]` if `return_residuals=True`,
            otherwise not returned.
    """
    output_shape = None
    if isinstance(residuals, Sequence) and not isinstance(residuals, torch.Tensor):
        if len(residuals) == 0:
            raise ValueError("residuals must contain at least one source")
        output_shape = residuals[0].shape
        D = output_shape[-1]
        residuals = torch.stack(tuple(residual.view(-1, D) for residual in residuals), dim=0)

    o, p = FusedAttnresFunction.apply(query, residuals, rms_weight, rms_eps, scale)
    if output_shape is not None:
        o = o.view(output_shape)

    outputs = [o]
    if return_weights:
        if output_shape is not None:
            p = p.view(residuals.shape[0], *output_shape[:-1])
        outputs.append(p)
    if return_residuals:
        if output_shape is not None:
            residuals = residuals.view(residuals.shape[0], *output_shape)
        outputs.append(residuals)
    return tuple(outputs) if len(outputs) > 1 else o


__all__ = ["fused_attnres"]
