# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Fused transforms for SwiLA's input-dependent state-decay gate.

The low-level autograd functions operate on a flat channel axis.  The public
``fused_swila_log_decay_gate`` adapter additionally understands the current SwiLA
parameter layout: one ``A_log`` scalar per head and a ``dt_bias`` over
``[H, D_k]``. The stored parameters are FP32, while mixed
precision runtime views are promoted back to FP32 for the gate transform.

The NVIDIA CUDA primitives use dense pointer arithmetic. All three inputs are made
contiguous before launch and those same tensors are saved for backward, so
broadcast, sliced and transposed views retain their ordinary autograd meaning.
"""

import torch
import torch.nn.functional as F

from fla.ops.swila.autograd import autocast_custom_bwd, autocast_custom_fwd

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:  # pragma: no cover - exercised only on installs without Triton.
    triton = None
    tl = None


if triton is not None:
    from fla.utils import input_guard

    _input_guard = input_guard(no_guard_contiguous=True)
else:
    def _input_guard(fn):
        return fn


_BLOCK_M = 64
_BLOCK_D = 64


def _fallback_autograd_grads(
    output: torch.Tensor,
    references: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    needs_input_grad: tuple[bool, ...],
) -> tuple[torch.Tensor | None, ...]:
    """Differentiate only fallback inputs that actually require gradients.

    ``allow_unused=True`` does not permit a non-requires-grad tensor in
    ``torch.autograd.grad``'s input list.  Filtering first preserves the public
    fallback contract when, for example, ``A_log`` is frozen while the gate
    projection remains trainable.
    """
    active_indices = [
        index for index, needed in enumerate(needs_input_grad) if needed
    ]
    if not active_indices:
        return (None,) * len(references)
    active_gradients = torch.autograd.grad(
        output,
        tuple(references[index] for index in active_indices),
        grad_output,
        retain_graph=False,
        allow_unused=True,
    )
    gradients: list[torch.Tensor | None] = [None] * len(references)
    for index, gradient in zip(active_indices, active_gradients, strict=True):
        gradients[index] = gradient
    return tuple(gradients)


if triton is not None:
    @triton.jit
    def _alpha_gate_fwd_kernel(
        raw,
        a_scale,
        dt_bias,
        alpha,
        M: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0).to(tl.int64)
        pid_d = tl.program_id(1).to(tl.int64)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
        mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
        offsets = offs_m[:, None] * D + offs_d[None, :]

        z = tl.load(raw + offsets, mask=mask, other=0.0).to(tl.float32)
        z += tl.load(dt_bias + offs_d, mask=offs_d < D, other=0.0)[None, :].to(tl.float32)
        z_abs = tl.where(z >= 0.0, z, -z)
        # log1p preserves negative tails that round away in log(1 + exp(z)).
        softplus_z = tl.maximum(z, 0.0) + libdevice.log1p(tl.exp(-z_abs))
        scale = tl.load(a_scale + offs_d, mask=offs_d < D, other=0.0)[None, :].to(tl.float32)
        tl.store(alpha + offsets, -scale * softplus_z, mask=mask)

    @triton.jit
    def _alpha_gate_bwd_kernel(
        d_alpha,
        raw,
        a_scale,
        dt_bias,
        d_raw,
        d_a_scale_partial,
        d_dt_partial,
        M: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0).to(tl.int64)
        pid_d = tl.program_id(1).to(tl.int64)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
        mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
        offsets = offs_m[:, None] * D + offs_d[None, :]

        d_alpha_vals = tl.load(d_alpha + offsets, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(raw + offsets, mask=mask, other=0.0).to(tl.float32)
        z += tl.load(dt_bias + offs_d, mask=offs_d < D, other=0.0)[None, :].to(tl.float32)
        z_abs = tl.where(z >= 0.0, z, -z)
        softplus_z = tl.maximum(z, 0.0) + libdevice.log1p(tl.exp(-z_abs))
        scale = tl.load(a_scale + offs_d, mask=offs_d < D, other=0.0)[None, :].to(tl.float32)

        d_z = d_alpha_vals * (-scale) * tl.sigmoid(z)
        tl.store(d_raw + offsets, d_z, mask=mask)

        d_scale = -d_alpha_vals * softplus_z
        tl.store(d_a_scale_partial + pid_m * D + offs_d, tl.sum(d_scale, axis=0), mask=offs_d < D)
        tl.store(d_dt_partial + pid_m * D + offs_d, tl.sum(d_z, axis=0), mask=offs_d < D)


class _FusedAlphaGateFunction(torch.autograd.Function):
    @staticmethod
    @_input_guard
    @autocast_custom_fwd
    def forward(ctx, raw: torch.Tensor, a_scale: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
        if triton is None or not raw.is_cuda or torch.version.hip is not None:
            alpha = -a_scale * F.softplus(raw.float() + dt_bias)
            ctx.save_for_backward(raw, a_scale, dt_bias)
            ctx.fallback = True
            return alpha

        raw, a_scale, dt_bias = raw.contiguous(), a_scale.contiguous(), dt_bias.contiguous()
        m = raw.numel() // raw.shape[-1]
        d = raw.shape[-1]
        alpha = torch.empty((m, d), device=raw.device, dtype=torch.float32)
        grid = (triton.cdiv(m, _BLOCK_M), triton.cdiv(d, _BLOCK_D))
        _alpha_gate_fwd_kernel[grid](raw, a_scale, dt_bias, alpha, m, d, _BLOCK_M, _BLOCK_D)
        ctx.save_for_backward(raw, a_scale, dt_bias)
        ctx.shape = raw.shape
        ctx.m = m
        ctx.d = d
        ctx.grid = grid
        ctx.fallback = False
        return alpha.view(raw.shape)

    @staticmethod
    @_input_guard
    @autocast_custom_bwd
    def backward(ctx, d_alpha: torch.Tensor):
        raw, a_scale, dt_bias = ctx.saved_tensors
        if d_alpha is None:
            return tuple(
                torch.zeros_like(tensor) if needed else None
                for tensor, needed in zip(
                    (raw, a_scale, dt_bias),
                    ctx.needs_input_grad,
                    strict=True,
                )
            )

        if ctx.fallback:
            with torch.enable_grad():
                raw_ref = raw.detach().requires_grad_(ctx.needs_input_grad[0])
                a_ref = a_scale.detach().requires_grad_(ctx.needs_input_grad[1])
                dt_ref = dt_bias.detach().requires_grad_(ctx.needs_input_grad[2])
                alpha_ref = -a_ref * F.softplus(raw_ref.float() + dt_ref)
            return _fallback_autograd_grads(
                alpha_ref,
                (raw_ref, a_ref, dt_ref),
                d_alpha,
                ctx.needs_input_grad,
            )

        m = ctx.m
        d = ctx.d
        num_m_blocks = triton.cdiv(m, _BLOCK_M)
        d_alpha = d_alpha.contiguous()
        d_raw = torch.empty_like(raw)
        d_a_scale_partial = torch.empty((num_m_blocks, d), device=raw.device, dtype=torch.float32)
        d_dt_partial = torch.empty((num_m_blocks, d), device=raw.device, dtype=torch.float32)
        _alpha_gate_bwd_kernel[ctx.grid](
            d_alpha,
            raw,
            a_scale,
            dt_bias,
            d_raw,
            d_a_scale_partial,
            d_dt_partial,
            m,
            d,
            _BLOCK_M,
            _BLOCK_D,
        )
        d_a_scale = d_a_scale_partial.sum(dim=0)
        d_dt_bias = d_dt_partial.sum(dim=0)
        return d_raw, d_a_scale, d_dt_bias


def fused_swila_alpha_gate(raw: torch.Tensor, a_scale: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    return _FusedAlphaGateFunction.apply(raw, a_scale, dt_bias)


def fused_swila_log_decay_gate(
    raw: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    """Compute the current-layout SwiLA log decay with the fused alpha gate.

    Args:
        raw: Projected gate values with trailing shape ``[H, D_k]``.
        a_log: Log decay magnitudes with shape ``[H]``. Mixed precision
            runtime views are promoted to FP32 before exponentiation.
        dt_bias: Time-step biases whose shape matches the trailing gate
            layout. Mixed precision runtime views are promoted to FP32 before
            softplus.

    Returns:
        FP32 log decay with the same shape as ``raw``.

    The adapter flattens the structural gate axes only for the existing Triton
    primitive, then restores the caller-visible layout. Gradients through the
    expanded per-head scale reduce back into ``a_log`` through ordinary
    autograd, preserving the current parameter contract.
    """
    trailing_ndim = 2

    if dt_bias.ndim != 2:
        raise ValueError(f"dt_bias must have shape [H, D_k], got {tuple(dt_bias.shape)}")
    if raw.ndim < trailing_ndim:
        raise ValueError(
            f"raw must have at least {trailing_ndim} dimensions, got shape {tuple(raw.shape)}"
        )
    if tuple(raw.shape[-trailing_ndim:]) != tuple(dt_bias.shape):
        raise ValueError(
            "raw's trailing gate layout must match dt_bias: "
            f"raw={tuple(raw.shape)}, dt_bias={tuple(dt_bias.shape)}"
        )
    if a_log.shape != (dt_bias.shape[0],):
        raise ValueError(
            "a_log must contain one scalar per head: "
            f"expected {(dt_bias.shape[0],)}, got {tuple(a_log.shape)}"
        )
    if raw.device != a_log.device or raw.device != dt_bias.device:
        raise ValueError(
            "raw, a_log, and dt_bias must be on the same device, got "
            f"raw={raw.device}, a_log={a_log.device}, dt_bias={dt_bias.device}"
        )

    leading_shape = raw.shape[:-trailing_ndim]
    flat_width = dt_bias.numel()
    flat_raw = raw.reshape(*leading_shape, flat_width)
    a_log = a_log.float()
    dt_bias = dt_bias.float()
    scale_shape = (a_log.shape[0],) + (1,) * (dt_bias.ndim - 1)
    flat_scale = a_log.exp().view(scale_shape).expand_as(dt_bias).reshape(flat_width)
    flat_bias = dt_bias.reshape(flat_width)
    return fused_swila_alpha_gate(flat_raw, flat_scale, flat_bias).view_as(raw)
