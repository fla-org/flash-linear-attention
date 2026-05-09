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

# residual sources are passed as a list of separately-allocated tensors and
# accessed inside the kernels via int64 pointer tables — the upstream caller
# never has to `torch.stack` / `torch.cat` them. each source is read with a
# 2D pointer-tile gather: BL int64 base addresses are loaded once, cast to
# `tl.pointer_type(DTYPE)`, then broadcast against the inner D-tile to form
# `[BL, BD]` of independent global addresses. OOB rows (l >= L) load the
# dummy address 0 but are masked off by `m_l`, so the zero pointer is never
# dereferenced.
_TORCH_TO_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}


@fla_cache_autotune(
    configs=ATTNRES_FWD_BWD_CONFIGS,
    key=['L', 'D', 'HAS_ONORM'],
    **autotune_cache_kwargs,
)
@triton.jit
def attnres_fwd_kernel(
    q,
    res,
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
    DTYPE: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)

    # [BL]
    o_l = tl.arange(0, BL)
    m_l = o_l < L

    # one-time gather of source base pointers; reused across all D-loops below.
    p_v = tl.load(res + o_l, mask=m_l, other=0).to(tl.pointer_type(DTYPE))
    # each residual storage is 16-byte aligned (torch CUDA allocator is
    # 256-byte aligned), so tell Triton it can use wide vector loads.
    p_v = tl.multiple_of(p_v, 16)

    # [BL]
    b_var = tl.zeros([BL], dtype=tl.float32)
    b_logits = tl.zeros([BL], dtype=tl.float32)
    for i_d in range(0, D, BD):
        # [BD]
        o_d = i_d + tl.arange(0, BD)
        # tell Triton that o_d has BD contiguous elements with stride 1 — this
        # is the hint the compiler needs to coalesce the inner-D load across
        # the [BL, BD] pointer tile (without it, Triton sees `p_v[:, None] +
        # offs[None, :]` as opaque and emits per-element scattered loads).
        o_d = tl.max_contiguous(tl.multiple_of(o_d, BD), BD)
        m_d = o_d < D
        # [BL, BD] gather: row l from source l's storage at offset i_n*D + o_d
        b_v = tl.load(
            p_v[:, None] + (i_n * D + o_d[None, :]),
            mask=m_l[:, None] & m_d[None, :],
            other=0.0,
        ).to(tl.float32)
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

    # second pass: compute mix `b_o = sum_l p_l * v_l` and write to `o`. under
    # HAS_ONORM we also accumulate `b_o_var` for the output rstd. `o` is
    # allocated as fp32 by the python wrapper when HAS_ONORM=True so the
    # un-normed mix round-trips losslessly through gmem before pass 3
    # normalizes and overwrites it in place; the final cast to the residual
    # dtype happens once at the wrapper boundary.
    b_o_var = tl.zeros([], dtype=tl.float32)
    for i_d in range(0, D, BD):
        o_d = i_d + tl.arange(0, BD)
        # tell Triton that o_d has BD contiguous elements with stride 1 — this
        # is the hint the compiler needs to coalesce the inner-D load across
        # the [BL, BD] pointer tile (without it, Triton sees `p_v[:, None] +
        # offs[None, :]` as opaque and emits per-element scattered loads).
        o_d = tl.max_contiguous(tl.multiple_of(o_d, BD), BD)
        m_d = o_d < D
        p_o = tl.make_block_ptr(o + i_n * D, (D,), (1,), (i_d,), (BD,), (0,))
        # [BL, BD]
        b_v = tl.load(
            p_v[:, None] + (i_n * D + o_d[None, :]),
            mask=m_l[:, None] & m_d[None, :],
            other=0.0,
        ).to(tl.float32)
        # [BD]
        b_o = tl.sum(b_v * b_p[:, None], axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))
        if HAS_ONORM:
            b_o_var += tl.sum(tl.where(m_d, b_o * b_o, 0.0), axis=0)

    if HAS_ONORM:
        b_o_rstd = tl.rsqrt(b_o_var / D + eps)
        tl.store(o_rstd + i_n, b_o_rstd)
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
            tl.store(p_o, (b_o * b_o_rstd * b_ow).to(p_o.dtype.element_ty), boundary_check=(0,))


@fla_cache_autotune(
    configs=ATTNRES_FWD_BWD_CONFIGS,
    key=['L', 'D', 'HAS_ONORM'],
    **autotune_cache_kwargs,
)
@triton.jit
def attnres_bwd_kernel_dv(
    q,
    res,     # int64 [L]
    w,
    ow,
    p,
    rstd,
    o_rstd,
    do,
    dres,    # int64 [L]; data_ptr() of each per-source dv allocation
    dqw,
    dow_partial,
    N,
    scale: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BL: tl.constexpr,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
    DTYPE: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)

    o_l = tl.arange(0, BL)
    m_l = o_l < L
    p_v = tl.load(res + o_l, mask=m_l, other=0).to(tl.pointer_type(DTYPE))
    # each residual storage is 16-byte aligned (torch CUDA allocator is
    # 256-byte aligned), so tell Triton it can use wide vector loads.
    p_v = tl.multiple_of(p_v, 16)
    p_dv = tl.load(dres + o_l, mask=m_l, other=0).to(tl.pointer_type(DTYPE))
    p_dv = tl.multiple_of(p_dv, 16)

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
    b_o_c1 = tl.zeros([], dtype=tl.float32)
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
        # tell Triton that o_d has BD contiguous elements with stride 1 — this
        # is the hint the compiler needs to coalesce the inner-D load across
        # the [BL, BD] pointer tile (without it, Triton sees `p_v[:, None] +
        # offs[None, :]` as opaque and emits per-element scattered loads).
        o_d = tl.max_contiguous(tl.multiple_of(o_d, BD), BD)
        m_d = o_d < D
        p_do = tl.make_block_ptr(do + i_n * D, (D,), (1,), (i_d,), (BD,), (0,))
        # [BL, BD]
        b_v = tl.load(
            p_v[:, None] + (i_n * D + o_d[None, :]),
            mask=m_l[:, None] & m_d[None, :],
            other=0.0,
        ).to(tl.float32)
        # [BD]
        b_do = tl.load(p_do, boundary_check=(0,)).to(tl.float32)
        b_w = tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)
        b_q = tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32)

        if HAS_ONORM:
            b_ow = tl.load(ow + o_d, mask=m_d, other=0.).to(tl.float32)
            # rebuild `o_pre` and `xhat_o` from the just-loaded `b_v` and
            # the saved softmax probs `b_p` — no extra v load.
            b_o_pre = tl.sum(b_v * b_p[:, None], axis=0)
            b_o_xhat = b_o_pre * b_o_rstd
            b_o_wdy = b_ow * b_do
            b_o_c1 += tl.sum(tl.where(m_d, b_o_xhat * b_o_wdy, 0.0), axis=0)
            b_dp_w += tl.sum(b_v * b_o_wdy[None, :], axis=1)
            b_dp_x += tl.sum(b_v * b_o_xhat[None, :], axis=1)
            p_dow = tl.make_block_ptr(dow_partial + i_n * D, (D,), (1,), (i_d,), (BD,), (0,))
            tl.store(p_dow, (b_o_xhat * b_do).to(p_dow.dtype.element_ty), boundary_check=(0,))
        else:
            b_dp += tl.sum(b_v * b_do[None, :], axis=1)

        # [BL, BD]
        b_xhat = b_v * b_rstd[:, None]
        b_z += tl.sum(b_xhat * (b_w * b_q)[None, :], axis=1)

    if HAS_ONORM:
        b_o_c1 = b_o_c1 / D
        b_dp = b_o_rstd * (b_dp_w - b_o_c1 * b_dp_x)

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
        # tell Triton that o_d has BD contiguous elements with stride 1 — this
        # is the hint the compiler needs to coalesce the inner-D load across
        # the [BL, BD] pointer tile (without it, Triton sees `p_v[:, None] +
        # offs[None, :]` as opaque and emits per-element scattered loads).
        o_d = tl.max_contiguous(tl.multiple_of(o_d, BD), BD)
        m_d = o_d < D
        m_v = m_l[:, None] & m_d[None, :]
        p_do = tl.make_block_ptr(do + i_n * D, (D,), (1,), (i_d,), (BD,), (0,))
        # [BL, BD]
        b_v = tl.load(p_v[:, None] + (i_n * D + o_d[None, :]), mask=m_v, other=0.0).to(tl.float32)
        # [BD]
        b_do = tl.load(p_do, boundary_check=(0,)).to(tl.float32)
        b_w = tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)
        b_q = tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32)

        if HAS_ONORM:
            b_o_pre = tl.sum(b_v * b_p[:, None], axis=0)
            b_ow = tl.load(ow + o_d, mask=m_d, other=0.).to(tl.float32)
            b_o_xhat = b_o_pre * b_o_rstd
            b_do = (b_ow * b_do - b_o_xhat * b_o_c1) * b_o_rstd

        # [BL, BD]
        b_xhat = b_v * b_rstd[:, None]
        b_dv = b_p[:, None] * b_do[None, :] + b_ds[:, None] * b_rstd[:, None] * (
            (b_w * b_q)[None, :] - b_xhat * b_c1[:, None]
        )

        p_dqw = tl.make_block_ptr(dqw + i_n * D, (D,), (1,), (i_d,), (BD,), (0,))
        tl.store(p_dv[:, None] + (i_n * D + o_d[None, :]), b_dv.to(DTYPE), mask=m_v)
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
    b_o_acc = tl.zeros([BD], dtype=tl.float32)
    for i_n in range(0, N, BN):
        p_dqw = tl.make_block_ptr(dqw, (N, D), (D, 1), (i_n, i_d * BD), (BN, BD), (1, 0))
        # [BN, BD]
        b_dqw = tl.load(p_dqw, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_acc += tl.sum(b_dqw, axis=0)
        if HAS_ONORM:
            p_dow = tl.make_block_ptr(dow_partial, (N, D), (D, 1), (i_n, i_d * BD), (BN, BD), (1, 0))
            b_dow = tl.load(p_dow, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            b_o_acc += tl.sum(b_dow, axis=0)

    # [BD]
    b_q = tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32)
    b_w = tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)

    tl.store(dq + o_d, b_acc * b_w, mask=m_d)
    tl.store(dw + o_d, b_acc * b_q, mask=m_d)
    if HAS_ONORM:
        tl.store(dow + o_d, b_o_acc, mask=m_d)


def _build_ptr_table(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    # build the int64 ptr table on **pinned** CPU memory, then issue a
    # `non_blocking=True` H2D. with this combo PyTorch knows the source
    # storage outlives the async copy and skips the implicit
    # `cudaStreamSynchronize` that `torch.tensor([...], device='cuda')`
    # otherwise inserts to keep the pageable staging buffer alive
    # (~600µs of CPU stall per call, observed via `torch.profiler`).
    cpu_t = torch.tensor(
        [t.data_ptr() for t in tensors],
        dtype=torch.int64,
        pin_memory=True,
    )
    return cpu_t.to(tensors[0].device, non_blocking=True)


def fused_attnres_fwd(
    q: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    res: torch.Tensor,
    w: torch.Tensor,
    ow: torch.Tensor | None,
    eps: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not residuals[0].is_cuda:
        raise ValueError("Triton attnres requires CUDA tensors")

    output_shape = residuals[0].shape
    L, N, D = len(residuals), residuals[0].numel() // output_shape[-1], output_shape[-1]

    dtype = residuals[0].dtype
    if dtype not in _TORCH_TO_TL_DTYPE:
        raise ValueError(f"Unsupported residual dtype for fused_attnres: {dtype}")
    DTYPE = _TORCH_TO_TL_DTYPE[dtype]

    stats_shape = (L, *output_shape[:-1])

    # under HAS_ONORM the kernel needs `o` to round-trip across pass 2 → pass 3
    # without precision loss, so we keep it in fp32 inside the kernel and cast
    # to the residual dtype once at the end. bwd recomputes
    # `o_pre = sum_l p_l * v_l` inline from saved `p` and `res`, so we
    # don't keep an [N, D] activation across fwd→bwd.
    has_onorm = ow is not None
    o = torch.empty(output_shape, device=residuals[0].device, dtype=torch.float32 if has_onorm else dtype)
    p = torch.empty(stats_shape, device=residuals[0].device, dtype=torch.float32)
    rstd = torch.empty_like(p)
    if has_onorm:
        o_rstd = torch.empty(output_shape[:-1], device=residuals[0].device, dtype=torch.float32)
    else:
        o_rstd = None

    BL = max(8, triton.next_power_of_2(L))
    attnres_fwd_kernel[(N,)](
        q=q,
        res=res,
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
        DTYPE=DTYPE,
    )

    if has_onorm:
        o = o.to(dtype)

    return o, p, rstd, o_rstd


def fused_attnres_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    res: torch.Tensor,
    w: torch.Tensor,
    ow: torch.Tensor | None,
    p: torch.Tensor,
    rstd: torch.Tensor,
    o_rstd: torch.Tensor | None,
    scale: float,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor | None]:
    L, N, D = len(residuals), do.numel() // do.shape[-1], do.shape[-1]
    dtype = residuals[0].dtype
    DTYPE = _TORCH_TO_TL_DTYPE[dtype]

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

    dvs = [torch.empty_like(r) for r in residuals]
    dres = _build_ptr_table(dvs)
    dqw = torch.empty_like(do, dtype=torch.float32)
    dq = torch.empty_like(q)
    dw = torch.empty_like(w)

    BL = max(8, triton.next_power_of_2(L))
    attnres_bwd_kernel_dv[(N,)](
        q=q,
        res=res,
        w=w,
        ow=ow,
        p=p,
        rstd=rstd,
        o_rstd=o_rstd,
        do=do,
        dres=dres,
        dqw=dqw,
        dow_partial=dow_partial,
        N=N,
        scale=scale,
        L=L,
        D=D,
        BL=BL,
        HAS_ONORM=has_onorm,
        DTYPE=DTYPE,
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
        *residuals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # `res` is built once here and threaded through fwd/bwd so neither
        # internal wrapper has to rebuild it (the H2D copy + cudaMemcpy
        # launch is ~tens of µs, comparable to the kernel itself for small N).
        res = _build_ptr_table(residuals)
        o, p, rstd, o_rstd = fused_attnres_fwd(
            q=query,
            residuals=residuals,
            res=res,
            w=rms_weight,
            ow=output_rms_weight,
            eps=rms_eps,
            scale=scale,
        )
        ctx.save_for_backward(query, rms_weight, output_rms_weight, p, rstd, o_rstd, *residuals)
        ctx.scale = scale
        ctx.res = res
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
        query, rms_weight, output_rms_weight, p, rstd, o_rstd, *residuals = ctx.saved_tensors
        dvs, dq, dw, dow = fused_attnres_bwd(
            do=do,
            q=query,
            residuals=residuals,
            res=ctx.res,
            w=rms_weight,
            ow=output_rms_weight,
            p=p,
            rstd=rstd,
            o_rstd=o_rstd,
            scale=ctx.scale,
        )
        # gradient order matches forward signature:
        # query, rms_weight, output_rms_weight, rms_eps (None), scale (None), *residuals.
        return (dq, dw, dow, None, None, *dvs)


def fused_attnres(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""
    Apply AttnRes residual aggregation.

    AttnRes normalizes each residual source with RMSNorm, scores it against
    `query`, applies softmax over the residual-source dimension, and returns
    the weighted sum of residual sources.
    See `Attention Residuals <https://arxiv.org/abs/2603.15031>`_.

    Residual sources are passed as a sequence of independently allocated
    tensors and accessed inside the kernel via a pointer table — there is no
    upstream `torch.stack` / `torch.cat`, and per-source `dv` is written back
    into separately allocated tensors so autograd routes each gradient to its
    own leaf.

    Args:
        query (torch.Tensor):
            Per-layer pseudo-query of shape `[D]` or `[D, 1]`, where `D` is
            the hidden size.
        residuals (Sequence[torch.Tensor]):
            Non-empty sequence of same-dtype, same-`D` residual sources, each
            of shape `[..., D]`.
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

    Returns:
        o (torch.Tensor):
            Mixed residual of shape `[..., D]`.
        p (torch.Tensor):
            Depth softmax probabilities of shape `[L, ...]` if
            `return_weights=True`, otherwise not returned.
    """
    if len(residuals) == 0:
        raise ValueError("residuals must contain at least one source")

    output_shape = residuals[0].shape
    D = output_shape[-1]
    flat_residuals = tuple(r.reshape(-1, D).contiguous() for r in residuals)

    o, p = FusedAttnresFunction.apply(
        query, rms_weight, output_rms_weight, rms_eps, scale, *flat_residuals,
    )
    o = o.view(output_shape)

    if return_weights:
        p = p.view(len(residuals), *output_shape[:-1])
        return o, p
    return o


__all__ = ["fused_attnres"]
