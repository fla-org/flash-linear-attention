# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Public exact-identity temporal SwiLA dispatch and autograd integration."""

from __future__ import annotations

import torch

from fla.modules.norm.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.swila.temporal_bwd import fused_recurrent_swila_temporal_bwd
from fla.ops.swila.temporal_fwd import fused_recurrent_swila_temporal_fwd
from fla.utils import IS_NVIDIA, autocast_custom_bwd, autocast_custom_fwd, input_guard


def _validate_log_decay_shape(
    log_decay: torch.Tensor | None,
    q: torch.Tensor,
) -> None:
    expected = (*q.shape[:-1], q.shape[-1])
    if log_decay is not None and tuple(log_decay.shape) != expected:
        raise ValueError(
            f"log_decay must have shape {expected}, "
            f"got {tuple(log_decay.shape)}"
        )


class FusedRecurrentSwiLATemporalFunction(torch.autograd.Function):
    """Autograd function for fused recurrent SwiLA with temporal transitions."""

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        pi_q: torch.Tensor,
        pi_k: torch.Tensor,
        log_decay: torch.Tensor | None,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = False,
        r_prev_init: torch.Tensor | None = None,
        rho_prev_init: torch.Tensor | None = None,
        gate_kv: torch.Tensor | None = None,
        gate_q: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ):
        if not IS_NVIDIA or q.device.type != "cuda":
            raise NotImplementedError("Fused SwiLA requires NVIDIA CUDA; use swila_recurrence for a PyTorch reference.")
        assert 1 <= beta.shape[-1] <= 8, f"fused_recurrent SwiLA requires 1 <= num_experts <= 8, got E={beta.shape[-1]}"
        # Most callers optimize only ``o``.  In particular, the diagnostic
        # ``u`` and ``pi_bar`` outputs are each E times wider than ``o``;
        # materializing zero gradients for them wastes both an allocation and
        # the full-tensor scans that were previously needed to recognize those
        # zeros in backward.  Unused outputs now arrive as ``None`` instead.
        ctx.set_materialize_grads(False)

        if log_decay is not None:
            log_decay = log_decay.contiguous()

        # Apply L2 normalization to q and k if requested
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        from fla.ops.swila.checkpoint_context import get_swila_checkpoint_phase

        checkpoint_prepass = get_swila_checkpoint_phase() == "prepass"

        o, final_state, intermediates, r_prev_final, rho_prev_final = fused_recurrent_swila_temporal_fwd(
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
            r_prev_init=r_prev_init,
            rho_prev_init=rho_prev_init,
            gate_kv=gate_kv,
            gate_q=gate_q,
            cu_seqlens=cu_seqlens,
            checkpoint_prepass=checkpoint_prepass,
        )

        # Unpack intermediates
        r, u, pi_bar, rho, qS = intermediates

        # Preserve every saved-tensor slot during a non-reentrant checkpoint
        # prepass even though the expensive backward-only intermediates were
        # deliberately not materialized.  The checkpoint hook replaces these
        # same-metadata placeholders with the tensors saved by recomputation;
        # saving ``None`` would drop the slots and make backward observe the
        # prepass values.  These substitutions affect only ctx storage—the
        # public diagnostic outputs below retain their original ``None``
        # values during the prepass.
        saved_u = r if u is None else u
        saved_pi_bar = r if pi_bar is None else pi_bar
        saved_qS = rho if qS is None else qS

        # Save tensors for backward
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, beta, pi_q, pi_k,
                              r, saved_u, saved_pi_bar, rho, saved_qS,
                              initial_state, r_prev_init, rho_prev_init,
                              gate_kv, gate_q, final_state, log_decay)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cu_seqlens = cu_seqlens

        # Only return final state to user if they requested it
        user_final_state = final_state if output_final_state else None
        user_r_prev_final = r_prev_final if output_final_state else None
        user_rho_prev_final = rho_prev_final if output_final_state else None

        # Return r and rho sequences [B, T, H, D_v, E] for logging/analysis
        return o, user_final_state, user_r_prev_final, user_rho_prev_final, r, rho, None, u, pi_bar

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht, d_r_prev_final, d_rho_prev_final, d_r_seq, d_rho_seq, _d_unused_seq, d_u_seq, d_pi_bar_seq):
        (q, q_rstd, k, k_rstd, v, beta, pi_q, pi_k,
         r, u, pi_bar, rho, qS, initial_state, r_prev_init, rho_prev_init,
         gate_kv, gate_q, final_state, log_decay) = ctx.saved_tensors
        scale = ctx.scale
        use_qk_l2norm_in_kernel = ctx.use_qk_l2norm_in_kernel
        cu_seqlens = ctx.cu_seqlens

        # Preserve explicit-zero upstream semantics.  Ordinarily unused
        # outputs are None because forward disables grad materialization, so
        # these defensive scans run only when a caller explicitly supplies a
        # zero tensor.
        if dht is not None and (not dht.requires_grad and dht.abs().max() == 0):
            dht = None
        if (
            d_r_prev_final is not None
            and not d_r_prev_final.requires_grad
            and d_r_prev_final.abs().max() == 0
        ):
            d_r_prev_final = None
        if (
            d_rho_prev_final is not None
            and not d_rho_prev_final.requires_grad
            and d_rho_prev_final.abs().max() == 0
        ):
            d_rho_prev_final = None
        if (
            d_u_seq is not None
            and not d_u_seq.requires_grad
            and d_u_seq.abs().max() == 0
        ):
            d_u_seq = None
        if (
            d_pi_bar_seq is not None
            and not d_pi_bar_seq.requires_grad
            and d_pi_bar_seq.abs().max() == 0
        ):
            d_pi_bar_seq = None

        # The Triton kernel always receives an output-gradient pointer.  An
        # isolated state/responsibility/auxiliary loss leaves ``o`` unused, so
        # provide the much smaller [B,T,H,D_v] zero buffer in that case.  All
        # other unused output gradients stay as None and select the existing
        # constexpr-disabled paths without allocating or scanning their full
        # tensors.
        if do is None:
            do = torch.zeros_like(v)
        else:
            do = do.contiguous()

        # The main temporal kernel currently handles r/rho auxiliary VJPs as
        # one paired specialization. Preserve isolated r-only and rho-only
        # losses by supplying a zero counterpart only for that uncommon case;
        # the FineWeb MI loss uses both tensors and allocates nothing here.
        if d_r_seq is None and d_rho_seq is not None:
            d_r_seq = torch.zeros_like(r)
        elif d_r_seq is not None and d_rho_seq is None:
            d_rho_seq = torch.zeros_like(rho)

        bwd_kwargs = dict(
            q=q,
            k=k,
            v=v,
            beta=beta,
            pi_q=pi_q,
            pi_k=pi_k,
            r=r,
            u=u,
            pi_bar=pi_bar,
            rho=rho,
            qS=qS,
            log_decay=log_decay,
            initial_state=initial_state,
            final_state=final_state,
            r_prev_init=r_prev_init,
            rho_prev_init=rho_prev_init,
            do=do,
            dht=dht,
            scale=scale,
            d_r_prev_final=(
                d_r_prev_final.contiguous()
                if d_r_prev_final is not None
                else None
            ),
            d_rho_prev_final=(
                d_rho_prev_final.contiguous()
                if d_rho_prev_final is not None
                else None
            ),
            d_r_seq=d_r_seq.contiguous() if d_r_seq is not None else None,
            d_rho_seq=d_rho_seq.contiguous() if d_rho_seq is not None else None,
            d_u_seq=d_u_seq.contiguous() if d_u_seq is not None else None,
            d_pi_bar_seq=(
                d_pi_bar_seq.contiguous()
                if d_pi_bar_seq is not None
                else None
            ),
            gate_kv=gate_kv,
            gate_q=gate_q,
            cu_seqlens=cu_seqlens,
        )
        (
            dq, dk, dv, dbeta, dpi_q, dpi_k, dh0,
            dr_prev_init, drho_prev_init, dlog_decay, d_gate_kv, d_gate_q,
        ) = fused_recurrent_swila_temporal_bwd(**bwd_kwargs)

        # Handle L2 normalization backward if it was used in forward
        if use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        # Return gradients in same order as forward inputs:
        # q, k, v, beta, pi_q, pi_k, log_decay,
        # scale, initial_state,
        # output_final_state, use_qk_l2norm_in_kernel,
        # r_prev_init, rho_prev_init, gate_kv, gate_q, cu_seqlens,
        return (
            dq, dk, dv, dbeta, dpi_q, dpi_k,
            dlog_decay, None, dh0, None, None,
            dr_prev_init, drho_prev_init, d_gate_kv, d_gate_q,
            None,
        )


@torch.compiler.disable
def fused_recurrent_swila_temporal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    pi_q: torch.Tensor,
    pi_k: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    r_prev_init: torch.Tensor | None = None,
    rho_prev_init: torch.Tensor | None = None,
    gate_kv: torch.Tensor | None = None,
    gate_q: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    log_decay: torch.Tensor | None = None,
):
    """
    Fused recurrent SwiLA with exact identity temporal transitions.

    When supplied, ``log_decay`` has shape ``[B,T,H,D_k]`` and is applied
    before prediction, routing, and the write, shared over values and experts.
    Temporal responsibilities carry directly between reset-gate updates.
    """
    assert 1 <= beta.shape[-1] <= 8, f"fused_recurrent SwiLA requires 1 <= num_experts <= 8, got E={beta.shape[-1]}"
    if scale is None:
        scale = q.shape[-1] ** -0.5

    _validate_log_decay_shape(log_decay, q)
    if (r_prev_init is None) != (rho_prev_init is None):
        raise ValueError(
            "r_prev_init and rho_prev_init must either both be provided or both be None"
        )

    # Validate cu_seqlens
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
        B = q.shape[0]
        N = len(cu_seqlens) - 1
        if B != 1:
            raise ValueError(f"batch size must be 1 when using cu_seqlens, got B={B}")
        if cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must describe at least one sequence")
        if cu_seqlens[0].item() != 0 or cu_seqlens[-1].item() != q.shape[1]:
            raise ValueError("cu_seqlens must start at 0 and end at the packed sequence length")
        if (cu_seqlens[1:] <= cu_seqlens[:-1]).any().item():
            raise ValueError("cu_seqlens must be strictly increasing; empty sequences are not supported")
        if initial_state is not None and initial_state.shape[0] != N:
            raise ValueError(f"initial_state has {initial_state.shape[0]} sequences but cu_seqlens has {N}")
        if r_prev_init is not None and r_prev_init.shape[0] != N:
            raise ValueError(f"r_prev_init has {r_prev_init.shape[0]} sequences but cu_seqlens has {N}")
        if rho_prev_init is not None and rho_prev_init.shape[0] != N:
            raise ValueError(f"rho_prev_init has {rho_prev_init.shape[0]} sequences but cu_seqlens has {N}")

    return FusedRecurrentSwiLATemporalFunction.apply(
        q, k, v, beta, pi_q, pi_k,
        log_decay, scale, initial_state,
        output_final_state, use_qk_l2norm_in_kernel,
        r_prev_init, rho_prev_init, gate_kv, gate_q, cu_seqlens,
    )
