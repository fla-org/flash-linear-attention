# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Pure router reductions and graph-connected SwiLA mutual-information loss."""

from __future__ import annotations

import torch


def mean_over_sequences(
    values: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Average tokens independently for dense or concatenated sequences."""
    if cu_seqlens is None:
        return values.mean(dim=1)

    boundaries = cu_seqlens.reshape(-1).to(
        device=values.device,
        dtype=torch.long,
    )
    if boundaries.numel() < 2:
        raise ValueError("cu_seqlens must contain at least two boundaries")

    lengths = boundaries[1:] - boundaries[:-1]
    if torch.any(lengths < 0):
        raise ValueError("cu_seqlens must be monotonically nondecreasing")

    flat = values.reshape(-1, *values.shape[2:])
    sequence_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=values.device),
        lengths,
    )
    if sequence_ids.numel() != flat.shape[0]:
        raise ValueError(
            "cu_seqlens boundaries do not match the packed token count: "
            f"{sequence_ids.numel()} != {flat.shape[0]}"
        )

    sums = values.new_zeros(
        lengths.numel(),
        *values.shape[2:],
    ).index_add(0, sequence_ids, flat)
    nonempty = lengths > 0
    if not torch.any(nonempty):
        raise ValueError("cu_seqlens must describe at least one token")
    denominator = lengths[nonempty].view(
        -1,
        *([1] * (values.ndim - 2)),
    )
    return sums[nonempty] / denominator


def swila_router_statistics(
    r: torch.Tensor,
    rho: torch.Tensor,
    num_experts: int,
    cu_seqlens: torch.Tensor | None = None,
    *,
    compute_statistics: bool = True,
):
    """Evaluate the existing sequence-aware entropy/MI formulas once."""
    eps = 1e-8
    p_bar_r = mean_over_sequences(r, cu_seqlens)
    p_bar_rho = mean_over_sequences(rho, cu_seqlens)
    H_marginal_r = -(p_bar_r * torch.log(p_bar_r + eps)).sum(dim=-1)
    H_marginal_rho = -(p_bar_rho * torch.log(p_bar_rho + eps)).sum(dim=-1)

    token_entropy_r = -(r * torch.log(r + eps)).sum(dim=-1)
    token_entropy_rho = -(rho * torch.log(rho + eps)).sum(dim=-1)
    H_cond_r_by_sequence = mean_over_sequences(token_entropy_r, cu_seqlens)
    H_cond_rho_by_sequence = mean_over_sequences(token_entropy_rho, cu_seqlens)

    if num_experts == 1:
        graph_zero = r.sum() * 0.0 + rho.sum() * 0.0
        mi_loss_r = graph_zero
        mi_loss_rho = graph_zero
    else:
        log_K = r.new_tensor(float(num_experts)).log()
        mi_loss_r = (-H_marginal_r + H_cond_r_by_sequence).mean() / log_K
        mi_loss_rho = (-H_marginal_rho + H_cond_rho_by_sequence).mean() / log_K
    if not compute_statistics:
        return mi_loss_r, mi_loss_rho, None

    # Optional logging must not add checkpoint-saved autograd tensors.
    with torch.no_grad():
        MI_r = (H_marginal_r - H_cond_r_by_sequence).mean()
        MI_rho = (H_marginal_rho - H_cond_rho_by_sequence).mean()
        if num_experts == 1:
            MI_r = graph_zero
            MI_rho = graph_zero
            normalized_MI_r = graph_zero
            normalized_MI_rho = graph_zero
        else:
            normalized_MI_r = MI_r / log_K
            normalized_MI_rho = MI_rho / log_K

        statistics = {
            "H_marginal_r": H_marginal_r.mean().detach(),
            "H_marginal_rho": H_marginal_rho.mean().detach(),
            "H_cond_r": H_cond_r_by_sequence.mean().detach(),
            "H_cond_rho": H_cond_rho_by_sequence.mean().detach(),
            "MI_r": MI_r.detach(),
            "MI_rho": MI_rho.detach(),
            "expert_util_r": p_bar_r.detach(),
            "expert_util_rho": p_bar_rho.detach(),
            "MI_r_normalized": normalized_MI_r.detach(),
            "MI_rho_normalized": normalized_MI_rho.detach(),
        }
    return mi_loss_r, mi_loss_rho, statistics


@torch.no_grad()
def detached_swila_router_stats(
    r: torch.Tensor,
    rho: torch.Tensor,
    num_experts: int,
    cu_seqlens: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute optional metrics without retaining or extending the graph."""
    _, _, statistics = swila_router_statistics(
        r.detach(), rho.detach(), num_experts, cu_seqlens
    )
    return statistics


def compute_swila_mi_loss(
    r: torch.Tensor | None,
    rho: torch.Tensor | None,
    num_experts: int,
    weight: float,
    training: bool,
    cu_seqlens: torch.Tensor | None = None,
    *,
    return_stats: bool = False,
):
    """Return graph-connected MI loss independently of diagnostic capture.

    Dense CUDA training retains the established lazy fused dispatch. Request
    detached statistics only for an active collector; disabled/evaluation calls
    return no loss or statistics and require no diagnostic reductions.
    """
    if not training or weight <= 0.0 or r is None or rho is None:
        return (None, None) if return_stats else None
    if cu_seqlens is None and r.is_cuda and rho.is_cuda:
        from fla.ops.swila.mi_loss import fused_swila_mi_loss_from_probs

        loss, statistics = fused_swila_mi_loss_from_probs(
            r, rho, mi_loss_weight=weight
        )
    else:
        loss_r, loss_rho, statistics = swila_router_statistics(
            r, rho, num_experts, cu_seqlens, compute_statistics=return_stats
        )
        loss = weight * 0.5 * (loss_r + loss_rho)
    return (loss, statistics) if return_stats else loss
