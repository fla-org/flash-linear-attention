# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Public recurrent SwiLA API and autograd integration.

Forward and backward execution live in their phase modules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fla.modules.norm.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.swila.nongated_bwd import fused_recurrent_swila_bwd
from fla.ops.swila.nongated_fwd import fused_recurrent_swila_fwd
from fla.utils import IS_NVIDIA, autocast_custom_bwd, autocast_custom_fwd, input_guard


class FusedRecurrentSwiLAFunction(torch.autograd.Function):
    """Autograd function for fused recurrent SwiLA."""

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

        # Apply L2 normalization to q and k if requested
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        o, final_state, intermediates = fused_recurrent_swila_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            pi_q=pi_q,
            pi_k=pi_k,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            output_intermediates=True,  # Save for backward
            cu_seqlens=cu_seqlens,
        )

        # Unpack intermediates (r, u)
        r, u = intermediates

        # Save tensors for backward
        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, beta, pi_q, pi_k, r, u, initial_state,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cu_seqlens = cu_seqlens

        return o, final_state, r

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht, d_r_seq):
        q, q_rstd, k, k_rstd, v, beta, pi_q, pi_k, r, u, initial_state = ctx.saved_tensors
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

        # With materialize_grads(False), unused responsibility outputs arrive as
        # None. Avoid scanning the full [B,T,H,Dv,E] tensor just to detect zeros.

        # Call backward kernel with saved intermediates.
        dq, dk, dv, dbeta, dpi_q, dpi_k, dh0 = fused_recurrent_swila_bwd(
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
            d_r_seq=d_r_seq,
            scale=scale,
            cu_seqlens=ctx.cu_seqlens,
        )

        # Handle L2 normalization backward if it was used in forward
        # l2norm_bwd signature: (y, rstd, dy) -> dx
        if use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        # Return gradients in same order as forward inputs:
        # q, k, v, beta, pi_q, pi_k, scale, initial_state, output_final_state,
        # use_qk_l2norm_in_kernel, cu_seqlens
        return (
            dq,
            dk,
            dv,
            dbeta,
            dpi_q,
            dpi_k,
            None,
            dh0,
            None,
            None,
            None,
        )


@torch.compiler.disable
def fused_recurrent_swila(
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
    cu_seqlens: torch.LongTensor | None = None,
    return_rho_seq: bool = True,
    log_decay: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    """
    Fused recurrent SwiLA (Switching Linear Attention) with optional state decay.

    Prediction and routing use the state after optional decay:
        S_minus = exp(log_decay) * S_prev   # omitted when log_decay is None
        pred_k = k @ S_minus                # prediction per expert
        u_k = v - pred_k                    # error per expert
        r = softmax(pi_k - 0.5 * u^2)       # responsibilities
        S_k = S_minus + beta_k * r_k * u_k * k
        o = sum_k(pi_q_star_k * q @ S_k)    # weighted output
        where pi_q_star = softmax(pi_q)

    With ``log_decay=None`` the ungated implementation is used. For E=1, r=1, the post-decay
    convention reduces to DeltaNet (or gated DeltaNet when decay is supplied).

    Args:
        q: Queries [B, T, H, D_k]
        k: Keys [B, T, H, D_k]
        v: Values [B, T, H, D_v]
        beta: Per-dimension learning rate [B, T, H, D_v, E]
        pi_q: Query-side log-prior responsibilities [B, T, H, D_v, E]
        pi_k: Key-side log-prior responsibilities [B, T, H, D_v, E]
        scale: Scaling factor for queries. Default: 1/sqrt(D_k)
        initial_state: Initial hidden state [B, H, D_v, E, D_k]. Default: zeros
        output_final_state: Whether to return final hidden state
        use_qk_l2norm_in_kernel: Whether to apply L2 normalization to q and k
            inside the kernel. Default: False.
        return_rho_seq: Whether to materialize Q-side responsibilities. Set false
            when an auxiliary loss path consumes pi_q directly and does not need
            rho_seq.
        log_decay: Optional logarithmic state decay with shape [B, T, H, D_k],
            shared over values and experts and applied before prediction.
    Returns:
        o: Output [B, T, H, D_v]
        final_state: Final hidden state [B, H, D_v, E, D_k] if output_final_state else None
        r_seq: KV-side responsibilities [B, T, H, D_v, E]
        rho_seq: Q-side responsibilities [B, T, H, D_v, E], or None when
            return_rho_seq is false.
    """

    assert 1 <= beta.shape[-1] <= 8, f"fused_recurrent SwiLA requires 1 <= num_experts <= 8, got E={beta.shape[-1]}"
    if log_decay is not None:
        # Decayed state uses a separate Triton implementation because its state
        # layout and coordinatewise backward differ from the old scalar gate.
        from fla.ops.swila.fused_recurrent_gated import fused_recurrent_gated_swila

        return fused_recurrent_gated_swila(
            q=q,
            k=k,
            v=v,
            beta=beta,
            pi_q=pi_q,
            pi_k=pi_k,
            log_decay=log_decay,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            return_rho_seq=return_rho_seq,
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

    recurrent_outputs = FusedRecurrentSwiLAFunction.apply(
        q, k, v, beta, pi_q, pi_k, scale, initial_state, output_final_state,
        use_qk_l2norm_in_kernel, cu_seqlens,
    )
    o, final_state, r_seq = recurrent_outputs
    if return_rho_seq:
        rho_seq = F.softmax(pi_q, dim=-1)
    else:
        rho_seq = None
    return o, final_state, r_seq, rho_seq
