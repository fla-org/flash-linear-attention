# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Fused recurrent SwiLA with key-wise exponential state decay.

This is the internal implementation used by :func:`fused_recurrent_swila` when
``log_decay`` is provided. Its shape is ``[B, T, H, D_k]``, shared over
values and experts, matching ``SwitchingLinearAttention``.

The retired output-wise ``alpha`` layout ``[B, T, H, D_v, E]`` is deliberately
not accepted because it represents a different recurrence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fla.modules.norm.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.swila.gated_bwd import fused_recurrent_gated_swila_bwd
from fla.ops.swila.gated_fwd import fused_recurrent_gated_swila_fwd
from fla.utils import IS_NVIDIA, autocast_custom_bwd, autocast_custom_fwd, input_guard


class FusedRecurrentGatedSwiLAFunction(torch.autograd.Function):
    """Autograd function for fused recurrent SwiLA with key-wise decay."""

    @staticmethod
    @input_guard(
        no_guard_contiguous=[
            "q", "k", "v", "beta", "pi_q", "pi_k", "log_decay",
            "initial_state",
        ]
    )
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        pi_q: torch.Tensor,
        pi_k: torch.Tensor,
        log_decay: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        if not IS_NVIDIA or q.device.type != "cuda":
            raise NotImplementedError("Fused SwiLA requires NVIDIA CUDA; use swila_recurrence for a PyTorch reference.")
        assert 1 <= beta.shape[-1] <= 8, f"fused_recurrent SwiLA requires 1 <= num_experts <= 8, got E={beta.shape[-1]}"
        ctx.set_materialize_grads(False)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        beta = beta.contiguous()
        pi_q = pi_q.contiguous()
        pi_k = pi_k.contiguous()
        log_decay = log_decay.contiguous()
        if initial_state is not None:
            initial_state = initial_state.contiguous()

        # Apply L2 normalization to q and k if requested
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        from fla.ops.swila.checkpoint_context import get_swila_checkpoint_phase

        checkpoint_prepass = get_swila_checkpoint_phase() == "prepass"

        o, final_state, intermediates = fused_recurrent_gated_swila_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            pi_q=pi_q,
            pi_k=pi_k,
            scale=scale,
            log_decay=log_decay,
            initial_state=initial_state,
            output_final_state=output_final_state,
            output_intermediates=True,
            cu_seqlens=cu_seqlens,
            checkpoint_prepass=checkpoint_prepass,
        )
        r, u = intermediates

        # Non-reentrant checkpointing associates recomputed saved tensors
        # with the tensor slots created during the prepass.  Omitting the
        # backward-only ``u`` tensor from the prepass store is intentional,
        # but saving ``None`` here would remove that slot entirely and leave
        # backward with the prepass value instead of the recomputed tensor.
        # ``r`` has the same shape/dtype/device as ``u`` and is only a slot
        # placeholder; the checkpoint saved-tensor hook replaces it during
        # recomputation before backward reads ``ctx.saved_tensors``.
        saved_u = r if u is None else u

        # Save tensors for backward
        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, beta, pi_q, pi_k, r, saved_u,
            initial_state, log_decay,
            *([final_state] if final_state is not None else []),
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cu_seqlens = cu_seqlens
        ctx.has_final_state = final_state is not None
        return o, final_state, r

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht, d_r_seq):
        has_final_state = ctx.has_final_state

        # Unpack saved tensors
        saved = list(ctx.saved_tensors)
        (
            q, q_rstd, k, k_rstd, v, beta, pi_q, pi_k, r, u,
            initial_state, log_decay,
        ) = saved[:12]
        idx = 12
        final_state = saved[idx] if has_final_state else None

        scale = ctx.scale
        use_qk_l2norm_in_kernel = ctx.use_qk_l2norm_in_kernel

        if do is None:
            do = torch.zeros_like(v)

        # WORKAROUND: When output_final_state=True but final state is not used in loss,
        # PyTorch passes a zero tensor for dht instead of None. The Triton kernel's
        # USE_FINAL_STATE_GRADIENT=True code path has a bug that produces incorrect
        # gradients even when dht is all zeros. Detect this case and pass None instead.
        if dht is not None and (not dht.requires_grad and dht.abs().max() == 0):
            dht = None

        dq, dk, dv, dbeta, dpi_q, dpi_k, dh0, dlog_decay = fused_recurrent_gated_swila_bwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            pi_q=pi_q,
            pi_k=pi_k,
            r=r,
            u=u,
            initial_state=initial_state,
            do=do.contiguous(),
            dht=dht,
            log_decay=log_decay,
            final_state=final_state,
            # With output-gradient materialization disabled, an unused r
            # already arrives as None. Do not scan a real MI VJP for zeros:
            # that would add a full responsibility reduction and host sync to
            # every FineWeb backward. Explicit zero VJPs are safe to execute.
            d_r_seq=(
                d_r_seq.contiguous() if d_r_seq is not None else None
            ),
            scale=scale,
            cu_seqlens=ctx.cu_seqlens,
        )

        # Handle L2 normalization backward if it was used in forward
        # l2norm_bwd signature: (y, rstd, dy) -> dx
        if use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        # Return gradients in same order as forward inputs:
        # q, k, v, beta, pi_q, pi_k, log_decay,
        # scale, initial_state, output_final_state, use_qk_l2norm_in_kernel,
        # cu_seqlens
        return (
            dq,
            dk,
            dv,
            dbeta,
            dpi_q,
            dpi_k,
            dlog_decay,
            None,
            dh0,
            None,
            None,
            None,
        )


@torch.compiler.disable
def fused_recurrent_gated_swila(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    pi_q: torch.Tensor,
    pi_k: torch.Tensor,
    log_decay: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    return_rho_seq: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    """
    Fused recurrent SwiLA with key-wise exponential forgetting.

    Prediction and routing use the decayed state (Equation 41):
        S_decayed = exp(log_decay) * S_{t-1}
        pred_k = k @ S_decayed              # prediction per expert
        u_k = v - pred_k                    # error per expert
        r = softmax(pi_k - 0.5 * u_k^2)     # responsibilities
        S_t = S_decayed + beta * r * u * k
        o = sum_k(pi_q_star_k * q @ S_k)    # weighted output
        where pi_q_star = softmax(pi_q)

    For E=1 (single expert), this reduces to gated DeltaNet.

    Args:
        q: Queries [B, T, H, D_k]
        k: Keys [B, T, H, D_k]
        v: Values [B, T, H, D_v]
        beta: Per-dimension learning rate [B, T, H, D_v, E]
        pi_q: Query-side log-prior responsibilities [B, T, H, D_v, E]
        pi_k: Key-side log-prior responsibilities [B, T, H, D_v, E]
        log_decay: [B, T, H, D_k], shared over values and experts.
        scale: Scaling factor for queries. Default: 1/sqrt(D_k)
        initial_state: Initial hidden state [B, H, D_v, E, D_k]. Default: zeros
        output_final_state: Whether to return final hidden state
        use_qk_l2norm_in_kernel: Whether to apply L2 normalization to q and k
            inside the kernel. Default: False.
        return_rho_seq: Whether to materialize Q-side responsibilities. Set false
            when no auxiliary diagnostic/loss path consumes them.

    Returns:
        o: Output [B, T, H, D_v]
        final_state: Final hidden state [B, H, D_v, E, D_k] if output_final_state else None
        r_seq: KV-side responsibilities [B, T, H, D_v, E]
        rho_seq: Q-side responsibilities [B, T, H, D_v, E], or None when
            return_rho_seq is false.
    """
    assert 1 <= beta.shape[-1] <= 8, f"fused_recurrent SwiLA requires 1 <= num_experts <= 8, got E={beta.shape[-1]}"
    if log_decay is None:
        raise ValueError(
            "fused_recurrent_gated_swila requires key-wise `log_decay`; use "
            "fused_recurrent_swila for the ungated recurrence."
        )
    expected_decay_shape = q.shape
    if log_decay.shape != expected_decay_shape:
        raise ValueError(
            f"log_decay must have shape {tuple(expected_decay_shape)}, got "
            f"{tuple(log_decay.shape)}"
        )

    if scale is None:
        scale = q.shape[-1] ** -0.5

    if cu_seqlens is not None:
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor")
        if cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be one-dimensional")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError("cu_seqlens must have dtype torch.int32 or torch.int64")
        if cu_seqlens.device != q.device:
            raise ValueError("cu_seqlens must be on the same device as q")
        if not cu_seqlens.is_contiguous():
            raise ValueError("cu_seqlens must be contiguous")
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing."
            )
        if cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must describe at least one sequence")
        if cu_seqlens[0].item() != 0 or cu_seqlens[-1].item() != q.shape[1]:
            raise ValueError("cu_seqlens must start at 0 and end at the packed sequence length")
        if (cu_seqlens[1:] <= cu_seqlens[:-1]).any().item():
            raise ValueError("cu_seqlens must be strictly increasing; empty sequences are not supported")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )

    o, final_state, r_seq = FusedRecurrentGatedSwiLAFunction.apply(
        q, k, v, beta, pi_q, pi_k, log_decay, scale,
        initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens,
    )
    if return_rho_seq:
        rho_seq = F.softmax(pi_q, dim=-1)
    else:
        rho_seq = None
    return o, final_state, r_seq, rho_seq
