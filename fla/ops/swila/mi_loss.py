# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Dense probability-input MI loss for SwiLA responsibilities.

The public entry point is behaviorally complete: it uses a fixed-config Triton
implementation for contiguous NVIDIA CUDA ``E=4`` inputs and otherwise evaluates the
same formula eagerly.  There are intentionally no environment switches in this
module.  Packed sequences must keep using the sequence-aware layer reduction.
"""

from __future__ import annotations

import math

import torch

from fla.ops.swila.autograd import autocast_custom_bwd, autocast_custom_fwd

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only installations.
    triton = None
    tl = None


if triton is not None:
    from fla.utils import input_guard

    _input_guard = input_guard
else:
    def _input_guard(fn):
        return fn


_EPS = 1.0e-8
_NUM_EXPERTS = 4
_LOG_NUM_EXPERTS = math.log(float(_NUM_EXPERTS))
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Fixed metadata selected by the upstream E4 probability-path profiling.  The
# forward partial/finalize split bounds temporary storage by T / _FWD_BT; the
# backward tile covers two flattened (H, D_v) rows per program.
_FWD_BT = 64
_FWD_BR = 1
_FWD_PARTIAL_WARPS = 1
_FWD_PARTIAL_STAGES = 1
_FWD_FINALIZE_BR = 1
_FWD_FINALIZE_WARPS = 8
_FWD_FINALIZE_STAGES = 3
_FWD_BNT_MULTIPLIER = 2
_MI_REDUCE_BLOCK_ROWS = 256
_BWD_BT = 64
_BWD_BR = 2
_BWD_WARPS = 1
_BWD_STAGES = 1


def _validate_dense_probability_inputs(r: torch.Tensor, rho: torch.Tensor) -> None:
    if not isinstance(r, torch.Tensor) or not isinstance(rho, torch.Tensor):
        raise TypeError("r and rho must both be torch.Tensor instances")
    if r.ndim != 5 or rho.ndim != 5:
        raise ValueError(
            "dense SwiLA responsibilities must have shape [B, T, H, D_v, E]; "
            f"got r={tuple(r.shape)} and rho={tuple(rho.shape)}"
        )
    if r.shape != rho.shape:
        raise ValueError(
            "r and rho must have identical shapes; "
            f"got r={tuple(r.shape)} and rho={tuple(rho.shape)}"
        )
    if r.device != rho.device:
        raise ValueError(f"r and rho must share a device; got {r.device} and {rho.device}")
    if not r.is_floating_point() or not rho.is_floating_point():
        raise TypeError(f"r and rho must be floating point; got {r.dtype} and {rho.dtype}")
    if any(size <= 0 for size in r.shape):
        raise ValueError(f"responsibility dimensions must be positive; got {tuple(r.shape)}")


def _eager_dense_mi_loss_from_probs(
    r: torch.Tensor,
    rho: torch.Tensor,
    *,
    mi_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reference dense recurrence diagnostic formula used by ``swila.py``."""
    p_bar_r = r.mean(dim=1)
    p_bar_rho = rho.mean(dim=1)
    h_marginal_r = -(p_bar_r * torch.log(p_bar_r + _EPS)).sum(dim=-1)
    h_marginal_rho = -(p_bar_rho * torch.log(p_bar_rho + _EPS)).sum(dim=-1)

    token_entropy_r = -(r * torch.log(r + _EPS)).sum(dim=-1)
    token_entropy_rho = -(rho * torch.log(rho + _EPS)).sum(dim=-1)
    h_cond_r = token_entropy_r.mean(dim=1)
    h_cond_rho = token_entropy_rho.mean(dim=1)

    mi_r = (h_marginal_r - h_cond_r).mean()
    mi_rho = (h_marginal_rho - h_cond_rho).mean()
    if r.shape[-1] == 1:
        # Match the layer's connected-zero convention for the degenerate case.
        graph_zero = r.sum() * 0.0 + rho.sum() * 0.0
        mi_r = graph_zero
        mi_rho = graph_zero
        mi_loss_r = graph_zero
        mi_loss_rho = graph_zero
        normalized_mi_r = graph_zero
        normalized_mi_rho = graph_zero
    else:
        log_k = r.new_tensor(float(r.shape[-1])).log()
        mi_loss_r = (-h_marginal_r + h_cond_r).mean() / log_k
        mi_loss_rho = (-h_marginal_rho + h_cond_rho).mean() / log_k
        normalized_mi_r = mi_r / log_k
        normalized_mi_rho = mi_rho / log_k

    aux_loss = float(mi_loss_weight) * 0.5 * (mi_loss_r + mi_loss_rho)
    stats = {
        "H_marginal_r": h_marginal_r.mean().detach(),
        "H_marginal_rho": h_marginal_rho.mean().detach(),
        "H_cond_r": h_cond_r.mean().detach(),
        "H_cond_rho": h_cond_rho.mean().detach(),
        "MI_r": mi_r.detach(),
        "MI_rho": mi_rho.detach(),
        "expert_util_r": p_bar_r.detach(),
        "expert_util_rho": p_bar_rho.detach(),
        "MI_r_normalized": normalized_mi_r.detach(),
        "MI_rho_normalized": normalized_mi_rho.detach(),
    }
    return aux_loss, stats


if triton is not None:

    @triton.jit(do_not_specialize=['T', 'NT'])
    def _mi_fwd_partials_e4_kernel(
        r,
        rho,
        sum_r,
        cond_r,
        sum_rho,
        cond_rho,
        T,
        N: tl.constexpr,
        NT,
        BT: tl.constexpr,
        BR: tl.constexpr,
    ):
        T = T.to(tl.int64)
        NT = NT.to(tl.int64)
        i_b = tl.program_id(0).to(tl.int64)
        i_n_block = tl.program_id(1).to(tl.int64)
        i_t_block = tl.program_id(2).to(tl.int64)
        offs_t = i_t_block * BT + tl.arange(0, BT).to(tl.int64)
        offs_e = tl.arange(0, 4).to(tl.int64)

        for i_r in tl.static_range(0, BR):
            n = i_n_block * BR + i_r
            mask = (offs_t[:, None] < T) & (n < N)
            offsets = ((i_b * T + offs_t[:, None]) * N + n) * 4 + offs_e[None, :]

            r_values = tl.load(r + offsets, mask=mask, other=0.0).to(tl.float32)
            r_values = tl.where(mask, r_values, 0.0)
            r_sum = tl.sum(r_values, axis=0)
            r_entropy = -r_values * tl.log(r_values + 1.0e-8)
            r_entropy_sum = tl.sum(tl.sum(tl.trans(r_entropy), axis=0), axis=0)

            rho_values = tl.load(rho + offsets, mask=mask, other=0.0).to(tl.float32)
            rho_values = tl.where(mask, rho_values, 0.0)
            rho_sum = tl.sum(rho_values, axis=0)
            rho_entropy = -rho_values * tl.log(rho_values + 1.0e-8)
            rho_entropy_sum = tl.sum(tl.sum(tl.trans(rho_entropy), axis=0), axis=0)

            partial_base = ((i_b * NT + i_t_block) * N + n) * 4
            row_partial = (i_b * NT + i_t_block) * N + n
            tl.store(sum_r + partial_base + offs_e, r_sum, mask=n < N)
            tl.store(cond_r + row_partial, r_entropy_sum, mask=n < N)
            tl.store(sum_rho + partial_base + offs_e, rho_sum, mask=n < N)
            tl.store(cond_rho + row_partial, rho_entropy_sum, mask=n < N)

    @triton.jit(do_not_specialize=['T', 'NT'])
    def _mi_fwd_finalize_e4_kernel(
        sum_r,
        cond_r,
        sum_rho,
        cond_rho,
        p_bar_r,
        p_bar_rho,
        stats_rows,
        T,
        N: tl.constexpr,
        NT,
        BNT: tl.constexpr,
        BR: tl.constexpr,
    ):
        T = T.to(tl.int64)
        NT = NT.to(tl.int64)
        i_b = tl.program_id(0).to(tl.int64)
        i_n_block = tl.program_id(1).to(tl.int64)
        offs_block = tl.arange(0, BNT).to(tl.int64)
        offs_e = tl.arange(0, 4).to(tl.int64)

        for i_r in tl.static_range(0, BR):
            n = i_n_block * BR + i_r
            block_mask = (offs_block < NT) & (n < N)
            partial_offsets = (
                ((i_b * NT + offs_block[:, None]) * N + n) * 4 + offs_e[None, :]
            )
            cond_offsets = (i_b * NT + offs_block) * N + n

            r_partials = tl.load(
                sum_r + partial_offsets, mask=block_mask[:, None], other=0.0
            ).to(tl.float32)
            p_r = tl.sum(r_partials, axis=0) / T
            h_marginal_r = -tl.sum(p_r * tl.log(p_r + 1.0e-8), axis=0)
            h_cond_r = tl.sum(
                tl.load(cond_r + cond_offsets, mask=block_mask, other=0.0).to(tl.float32),
                axis=0,
            ) / T

            rho_partials = tl.load(
                sum_rho + partial_offsets, mask=block_mask[:, None], other=0.0
            ).to(tl.float32)
            p_rho = tl.sum(rho_partials, axis=0) / T
            h_marginal_rho = -tl.sum(p_rho * tl.log(p_rho + 1.0e-8), axis=0)
            h_cond_rho = tl.sum(
                tl.load(cond_rho + cond_offsets, mask=block_mask, other=0.0).to(tl.float32),
                axis=0,
            ) / T

            row = i_b * N + n
            valid_row = n < N
            tl.store(p_bar_r + row * 4 + offs_e, p_r, mask=valid_row)
            tl.store(p_bar_rho + row * 4 + offs_e, p_rho, mask=valid_row)
            tl.store(stats_rows + row * 4 + 0, h_marginal_r, mask=valid_row)
            tl.store(stats_rows + row * 4 + 1, h_cond_r, mask=valid_row)
            tl.store(stats_rows + row * 4 + 2, h_marginal_rho, mask=valid_row)
            tl.store(stats_rows + row * 4 + 3, h_cond_rho, mask=valid_row)

    @triton.jit
    def _mi_reduce_stats_kernel(
        stats_rows,
        stats_blocks,
        ROWS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        block = tl.program_id(0).to(tl.int64)
        channel = tl.program_id(1).to(tl.int64)
        rows = block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS).to(tl.int64)
        values = tl.load(
            stats_rows + rows * 4 + channel, mask=rows < ROWS, other=0.0,
        ).to(tl.float32)
        total = tl.sum(values, axis=0)
        tl.store(stats_blocks + block * 4 + channel, total)

    @triton.jit
    def _mi_fwd_loss_e4_kernel(
        stats_sums,
        aux_loss,
        stats_out,
        ROWS: tl.constexpr,
        MI_LOSS_WEIGHT: tl.constexpr,
    ):
        log_k = tl.log(4.0)
        h_marginal_r = tl.load(stats_sums + 0).to(tl.float32) / ROWS
        h_cond_r = tl.load(stats_sums + 1).to(tl.float32) / ROWS
        h_marginal_rho = tl.load(stats_sums + 2).to(tl.float32) / ROWS
        h_cond_rho = tl.load(stats_sums + 3).to(tl.float32) / ROWS
        mi_r = h_marginal_r - h_cond_r
        mi_rho = h_marginal_rho - h_cond_rho
        aux = MI_LOSS_WEIGHT * 0.5 * (
            (-h_marginal_r + h_cond_r) / log_k
            + (-h_marginal_rho + h_cond_rho) / log_k
        )

        tl.store(aux_loss, aux)
        tl.store(stats_out + 0, h_marginal_r)
        tl.store(stats_out + 1, h_marginal_rho)
        tl.store(stats_out + 2, h_cond_r)
        tl.store(stats_out + 3, h_cond_rho)
        tl.store(stats_out + 4, mi_r)
        tl.store(stats_out + 5, mi_rho)
        tl.store(stats_out + 6, mi_r / log_k)
        tl.store(stats_out + 7, mi_rho / log_k)

    @triton.jit(do_not_specialize=['T', 'SCALE'])
    def _mi_prob_bwd_e4_kernel(
        x,
        p_bar,
        dx,
        d_aux,
        T,
        N: tl.constexpr,
        BT: tl.constexpr,
        BR: tl.constexpr,
        BRE: tl.constexpr,
        SCALE,
    ):
        T = T.to(tl.int64)
        i_b = tl.program_id(0).to(tl.int64)
        i_n_block = tl.program_id(1).to(tl.int64)
        i_t_block = tl.program_id(2).to(tl.int64)
        offs_t = i_t_block * BT + tl.arange(0, BT).to(tl.int64)
        offs_re = tl.arange(0, BRE).to(tl.int64)
        offs_n = i_n_block * BR + offs_re // 4
        offs_e = offs_re % 4
        mask = (offs_t[:, None] < T) & (offs_n[None, :] < N)
        offsets = (
            ((i_b * T + offs_t[:, None]) * N + offs_n[None, :]) * 4
            + offs_e[None, :]
        )
        values = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)

        row = i_b * N + offs_n
        p = tl.load(p_bar + row * 4 + offs_e, mask=offs_n < N, other=0.0).to(
            tl.float32
        )
        marginal_derivative = tl.log(p + 1.0e-8) + p / (p + 1.0e-8)
        conditional_derivative = (
            tl.log(values + 1.0e-8) + values / (values + 1.0e-8)
        )
        gradient = (
            tl.load(d_aux).to(tl.float32)
            * SCALE
            * (marginal_derivative[None, :] - conditional_derivative)
        )
        tl.store(dx + offsets, gradient.to(dx.dtype.element_ty), mask=mask)


def _reduce_mi_stats(stats_rows: torch.Tensor) -> torch.Tensor:
    """Reduce four FP32 row statistics without serialized atomic summation.

    Each program owns one channel of a block. Subsequent launches reduce those
    block sums in the same stream, providing the inter-level synchronization.
    The final buffer has shape [1, 4]; no probability or backward math changes.
    """
    rows = stats_rows.shape[0]
    while rows > 1:
        blocks = triton.cdiv(rows, _MI_REDUCE_BLOCK_ROWS)
        stats_blocks = torch.empty((blocks, 4), device=stats_rows.device, dtype=torch.float32)
        _mi_reduce_stats_kernel[(blocks, 4)](
            stats_rows, stats_blocks, rows, _MI_REDUCE_BLOCK_ROWS, num_warps=4,
        )
        stats_rows = stats_blocks
        rows = blocks
    return stats_rows


def _triton_dense_e4_forward(
    r: torch.Tensor,
    rho: torch.Tensor,
    *,
    mi_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:  # pragma: no cover - protected by the eligibility check.
        raise RuntimeError("Triton is required for the fused E4 MI path")

    b, t, n, e = r.shape
    if e != _NUM_EXPERTS:
        raise RuntimeError(f"the fused MI path requires E=4, got E={e}")
    nt = triton.cdiv(t, _FWD_BT)
    bnt = triton.next_power_of_2(nt) * _FWD_BNT_MULTIPLIER
    rows = b * n

    sum_r = torch.empty((b, nt, n, 4), device=r.device, dtype=torch.float32)
    cond_r = torch.empty((b, nt, n), device=r.device, dtype=torch.float32)
    sum_rho = torch.empty_like(sum_r)
    cond_rho = torch.empty_like(cond_r)
    p_bar_r = torch.empty((rows, 4), device=r.device, dtype=torch.float32)
    p_bar_rho = torch.empty_like(p_bar_r)
    stats_rows = torch.empty((rows, 4), device=r.device, dtype=torch.float32)
    stats_out = torch.empty((8,), device=r.device, dtype=torch.float32)
    aux_loss = torch.empty((), device=r.device, dtype=torch.float32)

    _mi_fwd_partials_e4_kernel[(b, triton.cdiv(n, _FWD_BR), nt)](
        r,
        rho,
        sum_r,
        cond_r,
        sum_rho,
        cond_rho,
        t,
        n,
        nt,
        _FWD_BT,
        _FWD_BR,
        num_warps=_FWD_PARTIAL_WARPS,
        num_stages=_FWD_PARTIAL_STAGES,
    )
    _mi_fwd_finalize_e4_kernel[(b, triton.cdiv(n, _FWD_FINALIZE_BR))](
        sum_r,
        cond_r,
        sum_rho,
        cond_rho,
        p_bar_r,
        p_bar_rho,
        stats_rows,
        t,
        n,
        nt,
        bnt,
        _FWD_FINALIZE_BR,
        num_warps=_FWD_FINALIZE_WARPS,
        num_stages=_FWD_FINALIZE_STAGES,
    )
    stats_sums = _reduce_mi_stats(stats_rows)
    _mi_fwd_loss_e4_kernel[(1,)](
        stats_sums,
        aux_loss,
        stats_out,
        rows,
        float(mi_loss_weight),
        num_warps=1,
    )
    return aux_loss, p_bar_r, p_bar_rho, stats_out


class _FusedDenseE4MILoss(torch.autograd.Function):
    @staticmethod
    @_input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        r: torch.Tensor,
        rho: torch.Tensor,
        mi_loss_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not r.is_cuda or torch.version.hip is not None:
            raise RuntimeError("The fused SwiLA MI path requires NVIDIA CUDA")
        b, t, n, e = r.shape
        aux_loss, p_bar_r, p_bar_rho, stats_out = _triton_dense_e4_forward(
            r,
            rho,
            mi_loss_weight=float(mi_loss_weight),
        )
        ctx.shape = (b, t, n, e)
        ctx.mi_loss_weight = float(mi_loss_weight)
        ctx.save_for_backward(r, rho, p_bar_r, p_bar_rho)
        ctx.mark_non_differentiable(p_bar_r, p_bar_rho, stats_out)
        ctx.set_materialize_grads(False)
        return aux_loss, p_bar_r, p_bar_rho, stats_out

    @staticmethod
    @_input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        d_aux: torch.Tensor | None,
        _d_p_bar_r: torch.Tensor | None,
        _d_p_bar_rho: torch.Tensor | None,
        _d_stats_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        if d_aux is None:
            return None, None, None

        r, rho, p_bar_r, p_bar_rho = ctx.saved_tensors
        b, t, n, _e = ctx.shape
        rows = b * n
        side_scale = (0.5 * ctx.mi_loss_weight) / (rows * t * _LOG_NUM_EXPERTS)
        grid = (b, triton.cdiv(n, _BWD_BR), triton.cdiv(t, _BWD_BT))
        d_aux = d_aux.contiguous()

        dr = None
        if ctx.needs_input_grad[0]:
            dr = torch.empty_like(r)
            _mi_prob_bwd_e4_kernel[grid](
                r,
                p_bar_r,
                dr,
                d_aux,
                t,
                n,
                _BWD_BT,
                _BWD_BR,
                _BWD_BR * 4,
                side_scale,
                num_warps=_BWD_WARPS,
                num_stages=_BWD_STAGES,
            )

        drho = None
        if ctx.needs_input_grad[1]:
            drho = torch.empty_like(rho)
            _mi_prob_bwd_e4_kernel[grid](
                rho,
                p_bar_rho,
                drho,
                d_aux,
                t,
                n,
                _BWD_BT,
                _BWD_BR,
                _BWD_BR * 4,
                side_scale,
                num_warps=_BWD_WARPS,
                num_stages=_BWD_STAGES,
            )
        return dr, drho, None


def can_use_fused_swila_mi_loss_from_probs(r: torch.Tensor, rho: torch.Tensor) -> bool:
    """Return whether ``r`` and ``rho`` can use the dense E4 Triton path."""
    if triton is None or not isinstance(r, torch.Tensor) or not isinstance(rho, torch.Tensor):
        return False
    if r.ndim != 5 or rho.ndim != 5 or r.shape != rho.shape:
        return False
    if r.shape[-1] != _NUM_EXPERTS or any(size <= 0 for size in r.shape):
        return False
    if torch.version.hip is not None:
        return False
    if r.device != rho.device or not r.is_cuda or not rho.is_cuda:
        return False
    if r.dtype not in _SUPPORTED_DTYPES or rho.dtype not in _SUPPORTED_DTYPES:
        return False
    return r.is_contiguous() and rho.is_contiguous()


@torch.compiler.disable
def fused_swila_mi_loss_from_probs(
    r: torch.Tensor,
    rho: torch.Tensor,
    *,
    mi_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the dense SwiLA MI auxiliary loss from two probability tensors.

    ``r`` and ``rho`` must have shape ``[B, T, H, D_v, E]``.  Eligible
    contiguous NVIDIA CUDA E4 inputs use the custom-autograd Triton implementation;
    CPU, non-contiguous, and non-E4 inputs transparently use the exact eager
    formula. Mixed supported floating dtypes are accepted so the non-temporal
    FP32-``r``/BF16-``rho`` contract stays on the fused path. Packed inputs are
    intentionally outside this API.
    """
    _validate_dense_probability_inputs(r, rho)
    weight = float(mi_loss_weight)
    if not can_use_fused_swila_mi_loss_from_probs(r, rho):
        return _eager_dense_mi_loss_from_probs(r, rho, mi_loss_weight=weight)

    b, t, h, d_v, e = r.shape
    n = h * d_v
    r_flat = r.view(b, t, n, e)
    rho_flat = rho.view(b, t, n, e)
    aux_loss, p_bar_r, p_bar_rho, stats_out = _FusedDenseE4MILoss.apply(
        r_flat,
        rho_flat,
        weight,
    )
    stats = {
        "H_marginal_r": stats_out[0].detach(),
        "H_marginal_rho": stats_out[1].detach(),
        "H_cond_r": stats_out[2].detach(),
        "H_cond_rho": stats_out[3].detach(),
        "MI_r": stats_out[4].detach(),
        "MI_rho": stats_out[5].detach(),
        "expert_util_r": p_bar_r.view(b, h, d_v, e).detach(),
        "expert_util_rho": p_bar_rho.view(b, h, d_v, e).detach(),
        "MI_r_normalized": stats_out[6].detach(),
        "MI_rho_normalized": stats_out[7].detach(),
    }
    return aux_loss, stats


__all__ = [
    "can_use_fused_swila_mi_loss_from_probs",
    "fused_swila_mi_loss_from_probs",
]
