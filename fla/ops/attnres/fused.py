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
    key=['L', 'D', 'HAS_ONORM'],
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
    ow,         # output rms weight, [D]; None when HAS_ONORM=False
    o_rstd,     # output rstd, [N]; None when HAS_ONORM=False
    N,
    L: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BL: tl.constexpr,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
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

    # second pass: compute mix `b_o = sum_l p_l * v_l` and write to `o`.
    # under HAS_ONORM we also accumulate `b_var_o` for the output rstd. `o`
    # is allocated as fp32 by the python wrapper when HAS_ONORM=True so the
    # un-normed mix round-trips losslessly through gmem before pass 3
    # normalizes and overwrites it in place; the final cast to `v.dtype`
    # happens once at the wrapper boundary.
    b_var_o = tl.zeros([], dtype=tl.float32)
    for i_d in range(0, D, BD):
        o_d = i_d + tl.arange(0, BD)
        m_d = o_d < D
        p_v = tl.make_block_ptr(v + i_n * D, (L, D), (N * D, 1), (0, i_d), (BL, BD), (1, 0))
        p_o = tl.make_block_ptr(o + i_n * D, (D,), (1,), (i_d,), (BD,), (0,))
        # [BL, BD]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        # [BD]
        b_o = tl.sum(b_v * b_p[:, None], axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))
        if HAS_ONORM:
            b_var_o += tl.sum(tl.where(m_d, b_o * b_o, 0.0), axis=0)

    if HAS_ONORM:
        b_rstd_o = tl.rsqrt(b_var_o / D + eps)
        tl.store(o_rstd + i_n, b_rstd_o)
        # ensure pass-2 stores to `o` are visible before pass 3 reads them back.
        tl.debug_barrier()
        # third pass: load staged `o`, apply RMSNorm with `ow`, write back to `o`.
        for i_d in range(0, D, BD):
            o_d = i_d + tl.arange(0, BD)
            m_d = o_d < D
            p_o = tl.make_block_ptr(o + i_n * D, (D,), (1,), (i_d,), (BD,), (0,))
            # [BD]
            b_o = tl.load(p_o, boundary_check=(0,)).to(tl.float32)
            b_ow = tl.load(ow + o_d, mask=m_d, other=0.).to(tl.float32)
            tl.store(p_o, (b_o * b_rstd_o * b_ow).to(p_o.dtype.element_ty), boundary_check=(0,))


@fla_cache_autotune(
    configs=ATTNRES_FWD_BWD_CONFIGS,
    key=['L', 'D', 'HAS_ONORM'],
    **autotune_cache_kwargs,
)
@triton.jit
def attnres_bwd_kernel_dv(
    q,
    v,
    w,
    ow,
    p,
    rstd,
    o_rstd,
    do,
    dv,
    dqw,
    dow_partial,
    N,
    scale: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BL: tl.constexpr,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)

    p_p = tl.make_block_ptr(p + i_n, (L,), (N,), (0,), (BL,), (0,))
    p_rstd = tl.make_block_ptr(rstd + i_n, (L,), (N,), (0,), (BL,), (0,))
    # [BL]
    b_p = tl.load(p_p, boundary_check=(0,), padding_option="zero").to(tl.float32)
    b_rstd = tl.load(p_rstd, boundary_check=(0,), padding_option="zero").to(tl.float32)

    # the optional output RMSNorm bwd is folded into pass 1. Following fla's
    # layernorm bwd factoring (`b_wdy = w * dy`; see `fla/modules/layernorm.py`):
    #
    #   do_pre = (wdy_o - xhat_o * c1_o) * o_rstd     where wdy_o = ow * do
    #
    # Plugging into the softmax-bwd's `b_dp[l] = sum_d v[l,d] * do_pre[d]`:
    #
    #   b_dp[l] = o_rstd * (sum_d v[l,d] * wdy_o[d] - c1_o * sum_d v[l,d] * xhat_o[d])
    #           = o_rstd * (b_dp_w[l]               - c1_o * b_dp_x[l])
    #
    # `b_dp_w` and `b_dp_x` (and `c1_o` + `dow_partial`) accumulate cleanly
    # over the same D-loop that already loads `v` for `b_dp` / `b_z`, so we
    # don't need a separate pre-pass and only load `v` once across pass 1.
    b_o_rstd = tl.zeros([], dtype=tl.float32)
    b_c1_o = tl.zeros([], dtype=tl.float32)
    b_dp_w = tl.zeros([BL], dtype=tl.float32)
    b_dp_x = tl.zeros([BL], dtype=tl.float32)
    if HAS_ONORM:
        b_o_rstd = tl.load(o_rstd + i_n).to(tl.float32)

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

        if HAS_ONORM:
            b_ow = tl.load(ow + o_d, mask=m_d, other=0.).to(tl.float32)
            # rebuild `o_pre` and `xhat_o` from the just-loaded `b_v` and
            # the saved softmax probs `b_p` — no extra v load.
            b_o_pre = tl.sum(b_v * b_p[:, None], axis=0)
            b_xhat_o = b_o_pre * b_o_rstd
            b_wdy_o = b_ow * b_do
            b_c1_o += tl.sum(tl.where(m_d, b_xhat_o * b_wdy_o, 0.0), axis=0)
            b_dp_w += tl.sum(b_v * b_wdy_o[None, :], axis=1)
            b_dp_x += tl.sum(b_v * b_xhat_o[None, :], axis=1)
            p_dow = tl.make_block_ptr(dow_partial + i_n * D, (D,), (1,), (i_d,), (BD,), (0,))
            tl.store(p_dow, (b_xhat_o * b_do).to(p_dow.dtype.element_ty), boundary_check=(0,))
        else:
            b_dp += tl.sum(b_v * b_do[None, :], axis=1)

        # [BL, BD]
        b_xhat = b_v * b_rstd[:, None]
        b_z += tl.sum(b_xhat * (b_w * b_q)[None, :], axis=1)

    if HAS_ONORM:
        b_c1_o = b_c1_o / D
        b_dp = b_o_rstd * (b_dp_w - b_c1_o * b_dp_x)

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

        if HAS_ONORM:
            b_o_pre = tl.sum(b_v * b_p[:, None], axis=0)
            b_ow = tl.load(ow + o_d, mask=m_d, other=0.).to(tl.float32)
            b_xhat_o = b_o_pre * b_o_rstd
            b_do = (b_ow * b_do - b_xhat_o * b_c1_o) * b_o_rstd

        # [BL, BD]
        b_xhat = b_v * b_rstd[:, None]
        b_dv = b_p[:, None] * b_do[None, :] + b_ds[:, None] * b_rstd[:, None] * (
            (b_w * b_q)[None, :] - b_xhat * b_c1[:, None]
        )

        p_dv = tl.make_block_ptr(dv + i_n * D, (L, D), (N * D, 1), (0, i_d), (BL, BD), (1, 0))
        p_dqw = tl.make_block_ptr(dqw + i_n * D, (D,), (1,), (i_d,), (BD,), (0,))
        tl.store(p_dv, b_dv.to(dv.dtype.element_ty), boundary_check=(0, 1))
        # [BD]
        tl.store(p_dqw, tl.sum(b_ds[:, None] * b_xhat, axis=0), boundary_check=(0,))


@fla_cache_autotune(
    configs=ATTNRES_DW_CONFIGS,
    key=['N', 'D', 'HAS_ONORM'],
    **autotune_cache_kwargs,
)
@triton.jit
def attnres_bwd_kernel_dqdw(
    q,
    w,
    dqw,
    dq,
    dw,
    dow_partial,
    dow,
    N,
    D: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
):
    i_d = tl.program_id(0).to(tl.int32)

    # [BD]
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    # [BD]
    b_acc = tl.zeros([BD], dtype=tl.float32)
    b_acc_o = tl.zeros([BD], dtype=tl.float32)
    for i_n in range(0, N, BN):
        p_dqw = tl.make_block_ptr(dqw, (N, D), (D, 1), (i_n, i_d * BD), (BN, BD), (1, 0))
        # [BN, BD]
        b_dqw = tl.load(p_dqw, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_acc += tl.sum(b_dqw, axis=0)
        if HAS_ONORM:
            p_dow = tl.make_block_ptr(dow_partial, (N, D), (D, 1), (i_n, i_d * BD), (BN, BD), (1, 0))
            b_dow = tl.load(p_dow, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            b_acc_o += tl.sum(b_dow, axis=0)

    # [BD]
    b_q = tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32)
    b_w = tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)

    tl.store(dq + o_d, b_acc * b_w, mask=m_d)
    tl.store(dw + o_d, b_acc * b_q, mask=m_d)
    if HAS_ONORM:
        tl.store(dow + o_d, b_acc_o, mask=m_d)


def fused_attnres_fwd(
    q: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    ow: torch.Tensor | None,
    eps: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not v.is_cuda:
        raise ValueError("Triton attnres requires CUDA tensors")
    if v.shape[0] < 1:
        raise ValueError("Triton attnres requires at least one residual source")

    L, N, D = v.shape[0], v[0].numel() // v.shape[-1], v.shape[-1]
    output_shape = v.shape[1:]
    stats_shape = (L, *output_shape[:-1])

    # under HAS_ONORM the kernel needs `o` to round-trip across pass 2 → pass 3
    # without precision loss, so we keep it in fp32 inside the kernel and cast
    # to `v.dtype` once at the end. bwd recomputes `o_pre = sum_l p_l * v_l`
    # inline from saved `p` and `v`, so we don't keep an [N, D] activation
    # across fwd→bwd.
    has_onorm = ow is not None
    o = torch.empty(output_shape, device=v.device, dtype=torch.float32 if has_onorm else v.dtype)
    p = torch.empty(stats_shape, device=v.device, dtype=torch.float32)
    rstd = torch.empty_like(p)
    if has_onorm:
        o_rstd = torch.empty(output_shape[:-1], device=v.device, dtype=torch.float32)
    else:
        o_rstd = None

    BL = max(8, triton.next_power_of_2(L))
    grid = (N,)

    attnres_fwd_kernel[grid](
        q=q,
        v=v,
        w=w,
        o=o,
        rstd=rstd,
        p=p,
        ow=ow,
        o_rstd=o_rstd,
        N=N,
        L=L,
        D=D,
        eps=eps,
        scale=scale,
        BL=BL,
        HAS_ONORM=has_onorm,
    )

    if has_onorm:
        o = o.to(v.dtype)

    return o, p, rstd, o_rstd


def fused_attnres_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    ow: torch.Tensor | None,
    p: torch.Tensor,
    rstd: torch.Tensor,
    o_rstd: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    L, N, D = v.shape[0], do.numel() // do.shape[-1], do.shape[-1]

    # optional output RMSNorm bwd is folded into the attnres kernels:
    # `attnres_bwd_kernel_dv` derives `c1_o` and `dow_partial` in a pre-pass,
    # rebuilds `o_pre = sum_l p_l * v_l` inline (so we don't have to save it
    # across fwd→bwd), and recomputes `do_pre = (ow * do - xhat_o * c1_o)
    # * o_rstd` per BD chunk so we never materialize `do_pre` in gmem either.
    # `attnres_bwd_kernel_dqdw` finishes by reducing `dow_partial` over N
    # alongside the existing dq / dw.
    has_onorm = ow is not None
    if has_onorm:
        dow_partial = torch.empty_like(do, dtype=torch.float32)
        dow = torch.empty_like(ow)
    else:
        dow_partial = dow = None

    dv = torch.empty_like(v)
    dqw = torch.empty_like(do, dtype=torch.float32)
    dq = torch.empty_like(q)
    dw = torch.empty_like(w)

    BL = max(8, triton.next_power_of_2(L))
    attnres_bwd_kernel_dv[(N,)](
        q=q,
        v=v,
        w=w,
        ow=ow,
        p=p,
        rstd=rstd,
        o_rstd=o_rstd,
        do=do,
        dv=dv,
        dqw=dqw,
        dow_partial=dow_partial,
        N=N,
        scale=scale,
        L=L,
        D=D,
        BL=BL,
        HAS_ONORM=has_onorm,
    )

    def grid(meta): return (triton.cdiv(D, meta['BD']),)
    attnres_bwd_kernel_dqdw[grid](
        q=q,
        w=w,
        dqw=dqw,
        dq=dq,
        dw=dw,
        dow_partial=dow_partial,
        dow=dow,
        N=N,
        D=D,
        HAS_ONORM=has_onorm,
    )

    return dv, dq, dw, dow


class FusedAttnresFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        query: torch.Tensor,
        residuals: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor | None,
        rms_eps: float,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        o, p, rstd, o_rstd = fused_attnres_fwd(
            q=query,
            v=residuals,
            w=rms_weight,
            ow=output_rms_weight,
            eps=rms_eps,
            scale=scale,
        )
        ctx.save_for_backward(query, residuals, rms_weight, output_rms_weight, p, rstd, o_rstd)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, None, None]:
        del dp
        query, residuals, rms_weight, output_rms_weight, p, rstd, o_rstd = ctx.saved_tensors
        dv, dq, dw, dow = fused_attnres_bwd(
            do=do,
            q=query,
            v=residuals,
            w=rms_weight,
            ow=output_rms_weight,
            p=p,
            rstd=rstd,
            o_rstd=o_rstd,
            scale=ctx.scale,
        )
        return dq, dv, dw, dow, None, None


def fused_attnres(
    query: torch.Tensor,
    residuals: torch.Tensor | Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
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
        output_rms_weight (torch.Tensor, optional):
            If set, an extra RMSNorm with this weight is applied to the mixed
            residual before returning, fusing the prenorm that would otherwise
            follow the AttnRes call (e.g. `attn_norm` / `mlp_norm`). Default:
            `None`.
        rms_eps (float):
            RMSNorm epsilon (also used for `output_rms_weight` when set).
            Default: `1e-6`.
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

    o, p = FusedAttnresFunction.apply(query, residuals, rms_weight, output_rms_weight, rms_eps, scale)
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
