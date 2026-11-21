# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Detached model metrics for opt-in diagnostics and training integrations."""

from __future__ import annotations

import math

import torch

ROUTER_DEAD_EXPERT_UNIFORM_FRACTION = 0.1
ROUTER_METRIC_KEYS = (
    "router/r_routing_entropy_norm",
    "router/rho_routing_entropy_norm",
    "router/combined_routing_entropy_norm",
    "router/r_utilization_entropy_norm",
    "router/rho_utilization_entropy_norm",
    "router/combined_utilization_entropy_norm",
    "router/r_dead_expert_frac",
    "router/rho_dead_expert_frac",
    "router/combined_dead_expert_frac",
)
TEMPORAL_GATE_STAT_KEYS = ("mean", "std", "saturated_frac")
TEMPORAL_GATE_MOMENT_WIDTH = 4


def router_dead_expert_threshold(num_experts: int) -> float:
    return ROUTER_DEAD_EXPERT_UNIFORM_FRACTION / float(num_experts)


def expert_util_by_unit(util: torch.Tensor, num_experts: int, device: torch.device) -> torch.Tensor:
    util = util.detach().float().to(device)
    if util.numel() == 0:
        return torch.zeros(1, num_experts, device=device)
    if util.shape[-1] != num_experts:
        return torch.zeros(1, num_experts, device=device)
    if util.ndim == 1:
        out = util.reshape(1, num_experts)
    else:
        # Diagnostics are [B, H, D_v, K] or [B, H, K] after averaging over time.
        # Average over batch only: each head/value slot has its own expert set.
        out = util.mean(dim=0).reshape(-1, num_experts)
    total = out.sum(dim=-1, keepdim=True)
    out = torch.where(total > 0, out / total.clamp_min(1e-8), torch.zeros_like(out))
    return out


def normalized_entropy(probs: torch.Tensor, log_k: float) -> torch.Tensor:
    probs = probs.float()
    if log_k == 0.0:
        return probs.new_zeros(probs.shape[:-1])
    return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1) / log_k


def router_metric_vector(diagnostics, num_experts: int, log_k: float, device: torch.device) -> torch.Tensor | None:
    h_cond_r = getattr(diagnostics, "H_cond_r", None)
    h_cond_rho = getattr(diagnostics, "H_cond_rho", None)
    expert_util_r = getattr(diagnostics, "expert_util_r", None)
    expert_util_rho = getattr(diagnostics, "expert_util_rho", None)
    if any(
        value is None
        for value in (h_cond_r, h_cond_rho, expert_util_r, expert_util_rho)
    ):
        return None

    r_util = expert_util_by_unit(expert_util_r, num_experts, device)
    rho_util = expert_util_by_unit(expert_util_rho, num_experts, device)
    combined_util = 0.5 * (r_util + rho_util)
    dead_threshold = router_dead_expert_threshold(num_experts)

    if num_experts == 1:
        r_routing_entropy = torch.zeros((), device=device)
        rho_routing_entropy = torch.zeros((), device=device)
    else:
        r_routing_entropy = h_cond_r.detach().float().to(device) / log_k
        rho_routing_entropy = h_cond_rho.detach().float().to(device) / log_k
    combined_routing_entropy = 0.5 * (r_routing_entropy + rho_routing_entropy)

    values = [
        r_routing_entropy.reshape(()),
        rho_routing_entropy.reshape(()),
        combined_routing_entropy.reshape(()),
        normalized_entropy(r_util, log_k).mean(),
        normalized_entropy(rho_util, log_k).mean(),
        normalized_entropy(combined_util, log_k).mean(),
        (r_util < dead_threshold).float().mean(),
        (rho_util < dead_threshold).float().mean(),
        (combined_util < dead_threshold).float().mean(),
    ]
    return torch.stack(values)


def temporal_gate_moment_vector(
    diagnostics,
    diagnostics_name: str,
    device: torch.device,
) -> torch.Tensor | None:
    stats = getattr(diagnostics, diagnostics_name, None)
    if (
        not isinstance(stats, torch.Tensor)
        or stats.numel() != TEMPORAL_GATE_MOMENT_WIDTH
    ):
        return None
    return stats.detach().float().to(device).reshape(
        TEMPORAL_GATE_MOMENT_WIDTH
    )


def collect_router_metrics(
    records,
    num_experts: int,
    device: torch.device,
) -> dict[str, float]:
    records = tuple(records)
    log_k = math.log(num_experts)
    router_vectors = [
        vec for record in records
        if (vec := router_metric_vector(record, num_experts, log_k, device)) is not None
    ]
    temporal_q_vectors = [
        vec for record in records
        if (
            vec := temporal_gate_moment_vector(
                record,
                "temporal_gate_q_stats",
                device,
            )
        ) is not None
    ]
    temporal_kv_vectors = [
        vec for record in records
        if (
            vec := temporal_gate_moment_vector(
                record,
                "temporal_gate_kv_stats",
                device,
            )
        ) is not None
    ]

    def sum_and_count(vectors: list[torch.Tensor], width: int) -> torch.Tensor:
        values = (
            torch.stack(vectors).sum(dim=0)
            if vectors
            else torch.zeros(width, device=device)
        )
        return torch.cat((values, values.new_tensor([len(vectors)])))

    # Reducing fixed-width sum/count groups lets each metric family be absent
    # locally while still being contributed by another distributed rank.
    reduced = torch.cat(
        (
            sum_and_count(router_vectors, len(ROUTER_METRIC_KEYS)),
            (
                torch.stack(temporal_q_vectors).sum(dim=0)
                if temporal_q_vectors
                else torch.zeros(TEMPORAL_GATE_MOMENT_WIDTH, device=device)
            ),
            (
                torch.stack(temporal_kv_vectors).sum(dim=0)
                if temporal_kv_vectors
                else torch.zeros(TEMPORAL_GATE_MOMENT_WIDTH, device=device)
            ),
        )
    )
    import torch.distributed as dist
    if dist.is_initialized():
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

    metrics: dict[str, float] = {}
    offset = 0

    def add_averaged_metrics(keys: tuple[str, ...]) -> None:
        nonlocal offset
        values = reduced[offset:offset + len(keys)]
        count = reduced[offset + len(keys)]
        offset += len(keys) + 1
        if count.item() > 0:
            averages = values / count
            metrics.update(
                {
                    key: averages[index].item()
                    for index, key in enumerate(keys)
                }
            )

    add_averaged_metrics(ROUTER_METRIC_KEYS)

    def add_temporal_gate_metrics(side: str) -> None:
        nonlocal offset
        total, square_total, saturated_count, count = reduced[
            offset:offset + TEMPORAL_GATE_MOMENT_WIDTH
        ]
        offset += TEMPORAL_GATE_MOMENT_WIDTH
        if count.item() <= 0:
            return
        mean = total / count
        variance = (square_total / count - mean.square()).clamp_min(0)
        values = (mean, variance.sqrt(), saturated_count / count)
        metrics.update(
            {
                f"temporal_gate/{side}_{stat}": value.item()
                for stat, value in zip(
                    TEMPORAL_GATE_STAT_KEYS,
                    values,
                    strict=True,
                )
            }
        )

    add_temporal_gate_metrics("q")
    add_temporal_gate_metrics("kv")
    return metrics


def collect_training_metrics(model, loss, diagnostics, *, router_stats: bool) -> dict[str, float]:
    """Collect detached metrics; preserve the training AVG-then-SUM order.

    Every training rank calls this even without W&B or locally available
    records. Collective reductions intentionally use the default process group.
    """
    records = [
        (name, diagnostics.records[module])
        for name, module in model.named_modules()
        if diagnostics is not None and module in diagnostics.records
    ] if router_stats else []
    total_aux_loss = getattr(model.model, "_total_aux_loss", 0.0)
    total_aux = total_aux_loss.detach().item() if isinstance(total_aux_loss, torch.Tensor) else 0.0
    ce = loss.detach().item()
    log_k = math.log(model.config.num_experts)
    mi_sum, count = 0.0, 0
    per_layer = {}
    for name, record in records:
        if record.MI_r is None:
            continue
        if record.MI_r_normalized is not None:
            mi_r = record.MI_r_normalized.item()
            mi_rho = record.MI_rho_normalized.item()
        elif log_k == 0.0:
            mi_r, mi_rho = 0.0, 0.0
        else:
            mi_r = record.MI_r.item() / log_k
            mi_rho = record.MI_rho.item() / log_k
        mi_sum += 0.5 * (mi_r + mi_rho)
        count += 1
        index = next((part for part in name.split(".") if part.isdigit()), name)
        per_layer[f"mi_per_layer/L{index}_mi_norm"] = 0.5 * (mi_r + mi_rho)
        if isinstance(record.aux_loss, torch.Tensor):
            per_layer[f"mi_per_layer/L{index}_aux_loss"] = record.aux_loss.item()
    mi_norm = mi_sum / count if count else 0.0

    import torch.distributed as dist
    if dist.is_initialized():
        values = torch.tensor([ce, total_aux, mi_norm], device=loss.device)
        dist.all_reduce(values, op=dist.ReduceOp.AVG)
        global_ce, global_mi, global_mi_norm = values[0].item(), values[1].item(), values[2].item()
    else:
        global_ce, global_mi, global_mi_norm = ce, total_aux, mi_norm
    router_metrics = collect_router_metrics(
        (record for _, record in records), model.config.num_experts, loss.device,
    ) if router_stats else {}
    return {
        "loss_metrics/ce_loss": ce,
        "loss_metrics/mi_aux_loss": total_aux,
        "loss_metrics/mi_normalized": mi_norm,
        "loss_metrics/global_avg_total_loss": global_ce + global_mi,
        "loss_metrics/global_avg_mi_aux_loss": global_mi,
        "loss_metrics/global_avg_mi_normalized": global_mi_norm,
        **router_metrics,
        **per_layer,
    }
