# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Compatibility entry point for temporally filtered SwiLA with state decay.

The authoritative temporal implementation lives in
``fused_recurrent_swila_temporal`` and specializes at compile time depending on
whether ``log_decay`` is present. Keeping this module as a thin entry point
avoids maintaining a second copy of the temporal routing, reset-gate, packed
sequence, and backward logic.

Unlike the removed legacy implementation, decay is key-wise and shared over
value dimensions:

``log_decay`` has shape ``[B, T, H, D_k]`` and is shared over experts.
"""

from __future__ import annotations

import torch

from fla.ops.swila.fused_recurrent_temporal import fused_recurrent_swila_temporal


def fused_recurrent_gated_swila_temporal(
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
    log_decay: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
):
    """Run temporal SwiLA with key-wise decay and exact temporal carry.

    This function keeps the historical gated import path, but it intentionally
    has no legacy output-wise ``alpha`` argument.
    """
    assert 1 <= beta.shape[-1] <= 8, f"fused_recurrent SwiLA requires 1 <= num_experts <= 8, got E={beta.shape[-1]}"
    if log_decay is None:
        raise ValueError(
            "fused_recurrent_gated_swila_temporal requires key-wise log_decay"
        )
    return fused_recurrent_swila_temporal(
        q=q,
        k=k,
        v=v,
        beta=beta,
        pi_q=pi_q,
        pi_k=pi_k,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        r_prev_init=r_prev_init,
        rho_prev_init=rho_prev_init,
        gate_kv=gate_kv,
        gate_q=gate_q,
        log_decay=log_decay,
        cu_seqlens=cu_seqlens,
    )


__all__ = [
    "fused_recurrent_gated_swila_temporal",
]
