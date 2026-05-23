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


@fla_cache_autotune(
    configs=[
        triton.Config({'BL': BL}, num_warps=num_warps, num_stages=num_stages)
        for BL in [1, 2, 4, 8]
        for num_warps in [4, 8, 16]
        for num_stages in [2, 3]
    ],
    key=['L2', 'D', 'HAS_ONORM', 'SAVE_OPRE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['L'])
def attnres_fwd_kernel(
    q,
    res,
    w,
    ow,
    o,
    o_pre,
    rstd,
    logit,
    lse,
    N,
    L,
    L2: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BL: tl.constexpr,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
    SAVE_OPRE: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)

    # [BD]
    o_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, BD), BD), BD)
    m_d = o_d < D
    # [BD] q * w, reused across all residual-source tiles
    b_qw = tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32) * tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)

    # online softmax over L; b_o accumulates in registers so each v tile is read once (logit + weighted sum)
    b_m = tl.full([], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([], dtype=tl.float32)
    b_o = tl.zeros([BD], dtype=tl.float32)
    for i_l in range(tl.cdiv(L, BL)):
        # [BL]
        o_l = i_l * BL + tl.arange(0, BL)
        m_l = o_l < L
        # per-tile base pointers from the length-L2 padded tuple; OOB rows keep res[0] and are masked by m_l
        p_v = res[0] + o_l * 0
        for i in tl.static_range(1, L2):
            p_v = tl.where(o_l == i, res[i], p_v)
        p_v = tl.multiple_of(p_v, 16)

        # [BL, BD] gather: row l from source l at offset i_n*D + o_d
        b_v = tl.load(
            tl.multiple_of(p_v[:, None] + (i_n * D + o_d[None, :]), (1, 16)),
            mask=m_l[:, None] & m_d[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        # [BL]
        b_rstd = tl.rsqrt(tl.sum(b_v * b_v, axis=1) / D + eps)
        b_logit = tl.sum(b_v * b_qw[None, :], axis=1) * b_rstd
        b_s = tl.where(m_l, b_logit * scale, float('-inf'))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, axis=0)), b_m
        b_r = exp(b_mp - b_m)
        # [BL]
        b_p = exp(b_s - b_m)
        b_acc = b_acc * b_r + tl.sum(b_p, axis=0)
        # [BD]
        b_o = b_o * b_r + tl.sum(b_p[:, None] * b_v, axis=0)

        # rstd and logit saved for bwd_dv
        p_rstd = tl.make_block_ptr(rstd + i_n, (L,), (N,), (i_l * BL,), (BL,), (0,))
        p_logit = tl.make_block_ptr(logit + i_n, (L,), (N,), (i_l * BL,), (BL,), (0,))
        tl.store(p_rstd, b_rstd.to(rstd.dtype.element_ty), boundary_check=(0,))
        tl.store(p_logit, b_logit.to(logit.dtype.element_ty), boundary_check=(0,))

    tl.store(lse + i_n, b_m + tl.log(b_acc))

    # [BD] pre-norm mixed residual sum_l p_l * v_l
    b_o = b_o / b_acc
    if SAVE_OPRE:
        p_o_pre = tl.make_block_ptr(o_pre + i_n * D, (D,), (1,), (0,), (BD,), (0,))
        tl.store(p_o_pre, b_o.to(p_o_pre.dtype.element_ty), boundary_check=(0,))
    # fold the optional output RMSNorm into the returned output o (o_rstd is recomputed from o_pre in bwd, not stored)
    if HAS_ONORM:
        b_o_rstd = tl.rsqrt(tl.sum(tl.where(m_d, b_o * b_o, 0.0), axis=0) / D + eps)
        b_ow = tl.load(ow + o_d, mask=m_d, other=0.).to(tl.float32)
        b_o = b_o * b_o_rstd * b_ow
    p_o = tl.make_block_ptr(o + i_n * D, (D,), (1,), (0,), (BD,), (0,))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@fla_cache_autotune(
    configs=[
        triton.Config({'BL': BL}, num_warps=num_warps, num_stages=num_stages)
        for BL in [1, 2, 4, 8]
        for num_warps in [4, 8, 16]
        for num_stages in [2, 3]
    ],
    key=['L2', 'D', 'HAS_ONORM', 'SAVE_OPRE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['L'])
def attnres_bwd_kernel_dv(
    q,
    res,
    w,
    ow,
    o_pre,
    rstd,
    logit,
    lse,
    do,
    dres,
    dqw,
    dow_partial,
    N,
    L,
    L2: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BL: tl.constexpr,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
    SAVE_OPRE: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)

    # [BD]
    o_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, BD), BD), BD)
    m_d = o_d < D
    b_qw = tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32) * tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)
    b_lse = tl.load(lse + i_n).to(tl.float32)
    p_do = tl.make_block_ptr(do + i_n * D, (D,), (1,), (0,), (BD,), (0,))
    b_do = tl.load(p_do, boundary_check=(0,), padding_option="zero").to(tl.float32)
    if SAVE_OPRE:
        p_o_pre = tl.make_block_ptr(o_pre + i_n * D, (D,), (1,), (0,), (BD,), (0,))
        b_o_pre = tl.load(p_o_pre, boundary_check=(0,), padding_option="zero").to(tl.float32)
    else:
        # level 1: recompute the mix sum_l p_l * v_l from V
        b_o_pre = tl.zeros([BD], dtype=tl.float32)
        for i_l in range(tl.cdiv(L, BL)):
            o_l = i_l * BL + tl.arange(0, BL)
            m_l = o_l < L
            p_v = res[0] + o_l * 0
            for i in tl.static_range(1, L2):
                p_v = tl.where(o_l == i, res[i], p_v)
            p_v = tl.multiple_of(p_v, 16)
            b_v = tl.load(
                tl.multiple_of(p_v[:, None] + (i_n * D + o_d[None, :]), (1, 16)),
                mask=m_l[:, None] & m_d[None, :],
                other=0.0,
            ).to(tl.float32)
            p_logit = tl.make_block_ptr(logit + i_n, (L,), (N,), (i_l * BL,), (BL,), (0,))
            b_logit = tl.load(p_logit, boundary_check=(0,), padding_option="zero").to(tl.float32)
            b_p = tl.where(m_l, exp(b_logit * scale - b_lse), 0.0)
            b_o_pre += tl.sum(b_p[:, None] * b_v, axis=0)

    # output RMSNorm bwd: turn b_do into the gradient w.r.t. the pre-norm output and stage dow_partial
    if HAS_ONORM:
        b_o_rstd = tl.rsqrt(tl.sum(tl.where(m_d, b_o_pre * b_o_pre, 0.0), axis=0) / D + eps)
        b_ow = tl.load(ow + o_d, mask=m_d, other=0.).to(tl.float32)
        b_xhat = b_o_pre * b_o_rstd
        b_c1 = tl.sum(tl.where(m_d, b_xhat * b_ow * b_do, 0.0), axis=0) / D
        p_dow = tl.make_block_ptr(dow_partial + i_n * D, (D,), (1,), (0,), (BD,), (0,))
        tl.store(p_dow, (b_xhat * b_do).to(p_dow.dtype.element_ty), boundary_check=(0,))
        b_do = (b_ow * b_do - b_xhat * b_c1) * b_o_rstd
    # delta = sum_l p*dp = <do_pre, o_pre>
    b_delta = tl.sum(tl.where(m_d, b_do * b_o_pre, 0.0), axis=0)

    # [BD] dqw accumulates over the L tiles
    b_dqw = tl.zeros([BD], dtype=tl.float32)
    for i_l in range(tl.cdiv(L, BL)):
        # [BL]
        o_l = i_l * BL + tl.arange(0, BL)
        m_l = o_l < L
        m_v = m_l[:, None] & m_d[None, :]
        # per-tile source / dv base pointers from the length-L2 padded tuple
        p_v = res[0] + o_l * 0
        p_dv = dres[0] + o_l * 0
        for i in tl.static_range(1, L2):
            p_v = tl.where(o_l == i, res[i], p_v)
            p_dv = tl.where(o_l == i, dres[i], p_dv)
        p_v = tl.multiple_of(p_v, 16)
        p_dv = tl.multiple_of(p_dv, 16)
        # [BL, BD] v tile, read once and reused for dp / dv / dqw
        b_v = tl.load(
            tl.multiple_of(p_v[:, None] + (i_n * D + o_d[None, :]), (1, 16)),
            mask=m_v,
            other=0.0,
        ).to(tl.float32)

        p_rstd = tl.make_block_ptr(rstd + i_n, (L,), (N,), (i_l * BL,), (BL,), (0,))
        p_logit = tl.make_block_ptr(logit + i_n, (L,), (N,), (i_l * BL,), (BL,), (0,))
        # [BL]; recompute probs from logit + lse, OOB rows masked to 0
        b_rstd = tl.load(p_rstd, boundary_check=(0,), padding_option="zero").to(tl.float32)
        b_logit = tl.load(p_logit, boundary_check=(0,), padding_option="zero").to(tl.float32)
        b_p = tl.where(m_l, exp(b_logit * scale - b_lse), 0.0)

        # softmax bwd with delta already known
        b_dp = tl.sum(b_v * b_do[None, :], axis=1)
        b_ds = b_p * (b_dp - b_delta) * scale
        # [BL, BD]
        b_k = b_v * b_rstd[:, None]
        b_dv = b_p[:, None] * b_do[None, :] + (b_ds * b_rstd)[:, None] * (b_qw[None, :] - b_k * (b_logit / D)[:, None])
        tl.store(
            tl.multiple_of(p_dv[:, None] + (i_n * D + o_d[None, :]), (1, 16)),
            b_dv.to(dres[0].dtype.element_ty),
            mask=m_v,
        )
        # [BD]
        b_dqw += tl.sum(b_ds[:, None] * b_k, axis=0)

    p_dqw = tl.make_block_ptr(dqw + i_n * D, (D,), (1,), (0,), (BD,), (0,))
    tl.store(p_dqw, b_dqw, boundary_check=(0,))


@fla_cache_autotune(
    configs=[
        triton.Config({'BN': BN, 'BD': BD}, num_warps=num_warps, num_stages=num_stages)
        for BN, BD, num_warps in [(1024, 16, 4), (2048, 32, 4), (2048, 32, 8), (4096, 32, 8), (4096, 64, 8)]
        for num_stages in [3, 4]
    ],
    key=['N', 'D', 'HAS_ONORM'],
    **autotune_cache_kwargs,
)
@triton.jit
def attnres_bwd_kernel_dqdw(
    q,
    w,
    dqw,
    dow_partial,
    dq,
    dw,
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

    # column-sum dqw (and dow_partial) over the N axis
    # [BD]
    b_dqw = tl.zeros([BD], dtype=tl.float32)
    b_dow = tl.zeros([BD], dtype=tl.float32)
    for i_n in range(0, N, BN):
        p_dqw = tl.make_block_ptr(dqw, (N, D), (D, 1), (i_n, i_d * BD), (BN, BD), (1, 0))
        b_dqw += tl.sum(tl.load(p_dqw, boundary_check=(0, 1), padding_option="zero").to(tl.float32), axis=0)
        if HAS_ONORM:
            p_dow = tl.make_block_ptr(dow_partial, (N, D), (D, 1), (i_n, i_d * BD), (BN, BD), (1, 0))
            b_dow += tl.sum(tl.load(p_dow, boundary_check=(0, 1), padding_option="zero").to(tl.float32), axis=0)

    # the logit uses the q * w product, so dq = (sum_n dqw) * w and dw = (sum_n dqw) * q
    # [BD]
    b_q = tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32)
    b_w = tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)
    tl.store(dq + o_d, b_dqw * b_w, mask=m_d)
    tl.store(dw + o_d, b_dqw * b_q, mask=m_d)
    if HAS_ONORM:
        tl.store(dow + o_d, b_dow, mask=m_d)


def _build_ptr_table(tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    # pad the per-source tensor tuple to a fixed length so Triton can compile a single kernel per L2 bucket.
    # the tuple length is part of the kernel's compile signature; padded slots are address-only (never read/written).
    L2 = max(8, triton.next_power_of_2(len(tensors)))
    assert 1 <= len(tensors) <= L2
    for t in tensors:
        assert t.data_ptr() % 16 == 0, "attnres residual sources must be 16-byte aligned"
    return tuple(tensors) + (tensors[0],) * (L2 - len(tensors))


def fused_attnres_fwd(
    q: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    res: tuple[torch.Tensor, ...],
    w: torch.Tensor,
    ow: torch.Tensor | None,
    eps: float,
    scale: float,
    checkpoint_level: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not residuals[0].is_cuda:
        raise ValueError("Triton attnres requires CUDA tensors")

    output_shape = residuals[0].shape
    L, N, D = len(residuals), residuals[0].numel() // output_shape[-1], output_shape[-1]

    dtype = residuals[0].dtype
    stats_shape = (L, *output_shape[:-1])

    has_onorm = ow is not None
    save_opre = checkpoint_level == 0
    o = torch.empty(output_shape, device=residuals[0].device, dtype=dtype)
    o_pre = torch.empty(output_shape, device=residuals[0].device, dtype=dtype) if save_opre else None
    lse = torch.empty(output_shape[:-1], device=residuals[0].device, dtype=torch.float32)
    rstd = torch.empty(stats_shape, device=residuals[0].device, dtype=torch.float32)
    logit = torch.empty_like(rstd)

    L2 = max(8, triton.next_power_of_2(L))
    attnres_fwd_kernel[(N,)](
        q=q,
        res=res,
        w=w,
        ow=ow,
        o=o,
        o_pre=o_pre,
        rstd=rstd,
        logit=logit,
        lse=lse,
        N=N,
        L=L,
        L2=L2,
        D=D,
        eps=eps,
        scale=scale,
        BD=triton.next_power_of_2(D),
        HAS_ONORM=has_onorm,
        SAVE_OPRE=save_opre,
    )

    return o, o_pre, rstd, logit, lse


def fused_attnres_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    res: tuple[torch.Tensor, ...],
    w: torch.Tensor,
    ow: torch.Tensor | None,
    o_pre: torch.Tensor | None,
    rstd: torch.Tensor,
    logit: torch.Tensor,
    lse: torch.Tensor,
    eps: float,
    scale: float,
    checkpoint_level: int,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor | None]:
    L, N, D = len(residuals), do.numel() // do.shape[-1], do.shape[-1]

    # bwd_dv produces dvs + dqw (and dow_partial when fusing output RMSNorm); bwd_dqdw reduces dqw / dow_partial over N
    # into dq / dw / dow.
    has_onorm = ow is not None
    if has_onorm:
        dow_partial = torch.empty_like(do, dtype=torch.float32)
        dow = torch.empty_like(ow)
    else:
        dow_partial = dow = None

    dvs = [torch.empty_like(r) for r in residuals]
    dres = _build_ptr_table(dvs)
    dqw = torch.empty_like(do, dtype=torch.float32)
    dq = torch.empty_like(q)
    dw = torch.empty_like(w)

    L2 = max(8, triton.next_power_of_2(L))
    attnres_bwd_kernel_dv[(N,)](
        q=q,
        res=res,
        w=w,
        ow=ow,
        o_pre=o_pre,
        rstd=rstd,
        logit=logit,
        lse=lse,
        do=do,
        dres=dres,
        dqw=dqw,
        dow_partial=dow_partial,
        N=N,
        L=L,
        L2=L2,
        D=D,
        eps=eps,
        scale=scale,
        BD=triton.next_power_of_2(D),
        HAS_ONORM=has_onorm,
        SAVE_OPRE=checkpoint_level == 0,
    )

    def grid(meta): return (triton.cdiv(D, meta['BD']),)
    attnres_bwd_kernel_dqdw[grid](
        q=q,
        w=w,
        dqw=dqw,
        dow_partial=dow_partial,
        dq=dq,
        dw=dw,
        dow=dow,
        N=N,
        D=D,
        HAS_ONORM=has_onorm,
    )

    return dvs, dq, dw, dow


class FusedAttnresFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        query: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor | None,
        rms_eps: float,
        scale: float,
        return_weights: bool,
        checkpoint_level: int,
        *residuals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # `res` is built once here and threaded through fwd/bwd so neither internal wrapper rebuilds it.
        # the tuple is pure Python, no H2D copy.
        res = _build_ptr_table(residuals)
        o, o_pre, rstd, logit, lse = fused_attnres_fwd(
            q=query,
            residuals=residuals,
            res=res,
            w=rms_weight,
            ow=output_rms_weight,
            eps=rms_eps,
            scale=scale,
            checkpoint_level=checkpoint_level,
        )
        ctx.save_for_backward(query, rms_weight, output_rms_weight, o_pre, rstd, logit, lse, *residuals)
        ctx.eps = rms_eps
        ctx.scale = scale
        ctx.checkpoint_level = checkpoint_level
        ctx.res = res
        # probs are materialized only when requested; bwd recomputes them from logit + lse
        p = (logit * scale - lse).exp() if return_weights else o.new_empty(0)
        ctx.mark_non_differentiable(p)
        return o, p

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dp: torch.Tensor | None = None,
    ):
        del dp
        query, rms_weight, output_rms_weight, o_pre, rstd, logit, lse, *residuals = ctx.saved_tensors
        dvs, dq, dw, dow = fused_attnres_bwd(
            do=do,
            q=query,
            residuals=residuals,
            res=ctx.res,
            w=rms_weight,
            ow=output_rms_weight,
            o_pre=o_pre,
            rstd=rstd,
            logit=logit,
            lse=lse,
            eps=ctx.eps,
            scale=ctx.scale,
            checkpoint_level=ctx.checkpoint_level,
        )
        return (dq, dw, dow, None, None, None, None, *dvs)


def fused_attnres(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
    checkpoint_level: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""
    Apply AttnRes residual aggregation.

    AttnRes normalizes each residual source with RMSNorm, scores it against `query`, applies softmax over the
    residual-source dimension, and returns the weighted sum of residual sources.
    See `Attention Residuals <https://arxiv.org/abs/2603.15031>`_.

    Args:
        query (torch.Tensor):
            Per-layer pseudo-query of shape `[D]` or `[D, 1]`, where `D` is the hidden size.
        residuals (Sequence[torch.Tensor]):
            Non-empty sequence of same-dtype, same-`D` residual sources, each of shape `[..., D]`.
        rms_weight (torch.Tensor):
            RMSNorm scale for key normalization of shape `[D]`.
        output_rms_weight (torch.Tensor, optional):
            If set, an extra RMSNorm with this weight is applied to the mixed residual before returning, fusing the
            prenorm that would otherwise follow the AttnRes call (e.g. `attn_norm` / `mlp_norm`). Default: `None`.
        rms_eps (float):
            RMSNorm epsilon (also used for `output_rms_weight` when set). Default: `1e-6`.
        scale (float):
            Scale factor applied to AttnRes logits before softmax. Default: `1.0`.
        return_weights (bool):
            Whether to return depth softmax probabilities. Default: `False`.
        checkpoint_level (int):
            Backward memory/recompute trade-off.
            `0` keeps the mixed residual;
            `1` drops it and recomputes it from the sources in backward (less memory, one extra read).
            Default: `1`.

    Returns:
        o (torch.Tensor):
            Mixed residual of shape `[..., D]`.
        p (torch.Tensor):
            Depth softmax probabilities of shape `[L, ...]` if `return_weights=True`, otherwise not returned.
    """
    if len(residuals) == 0:
        raise ValueError("residuals must contain at least one source")
    if checkpoint_level not in (0, 1):
        raise ValueError(f"checkpoint_level must be 0 or 1, got {checkpoint_level}")

    output_shape = residuals[0].shape
    D = output_shape[-1]
    flat_residuals = tuple(r.reshape(-1, D).contiguous() for r in residuals)

    o, p = FusedAttnresFunction.apply(
        query, rms_weight, output_rms_weight, rms_eps, scale, return_weights, checkpoint_level, *flat_residuals,
    )
    o = o.view(output_shape)

    if return_weights:
        p = p.view(len(residuals), *output_shape[:-1])
        return o, p
    return o


__all__ = ["fused_attnres"]
