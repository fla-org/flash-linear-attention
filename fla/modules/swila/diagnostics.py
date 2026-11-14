# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).


"""Opt-in detached SwiLA diagnostics; no diagnostic state is stored on layers."""

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.ops.swila.checkpoint_context import get_swila_checkpoint_phase
from fla.ops.swila.router_stats import detached_swila_router_stats

TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str) -> bool:
    """Parse a flag for explicit model/trainer policy, not collector activation."""
    return os.environ.get(name, "").strip().lower() in TRUE_ENV_VALUES


@dataclass
class SwiLADiagnostics:
    """Detached latest-forward values owned by an explicit capture scope."""
    # Expert logits & responsibilities
    pi_q: torch.Tensor | None = None
    pi_k: torch.Tensor | None = None
    responsibilities: torch.Tensor | None = None
    # Temporal states (temporal mode only)
    r: torch.Tensor | None = None       # KV-side filtered [B, T, H, D_v, E]
    rho: torch.Tensor | None = None     # Q-side filtered [B, T, H, D_v, E]
    u: torch.Tensor | None = None       # Prediction error
    pi_bar: torch.Tensor | None = None  # Prior (before Bayesian update)
    beta: torch.Tensor | None = None    # Learning rate
    valid_token_mask: torch.Tensor | None = None  # [B, T] after repadding
    # Gate values (reset_gate mode only)
    gate_kv: torch.Tensor | None = None
    gate_q: torch.Tensor | None = None
    # Detached [sum, sum of squares, saturated count, element count] moments.
    # These permit exact aggregation without retaining full gate tensors.
    temporal_gate_kv_stats: torch.Tensor | None = None
    temporal_gate_q_stats: torch.Tensor | None = None
    # Noise tracking (scalars)
    noise_std_k: torch.Tensor | None = None
    noise_std_q: torch.Tensor | None = None
    gate_noise_std_kv: torch.Tensor | None = None
    gate_noise_std_q: torch.Tensor | None = None
    # Expert utilization [H, D_v, K] or [H, K]
    expert_util_r: torch.Tensor | None = None
    expert_util_rho: torch.Tensor | None = None
    # MI / Entropy metrics (scalars)
    H_marginal_r: torch.Tensor | None = None
    H_marginal_rho: torch.Tensor | None = None
    H_cond_r: torch.Tensor | None = None
    H_cond_rho: torch.Tensor | None = None
    MI_r: torch.Tensor | None = None
    MI_rho: torch.Tensor | None = None
    MI_r_normalized: torch.Tensor | None = None
    MI_rho_normalized: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None


@dataclass
class SwiLAGradNorms:
    """Detached scalar norms, converted to Python numbers only by consumers."""
    output: torch.Tensor | None = None
    q: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    beta: torch.Tensor | None = None
    pi_q: torch.Tensor | None = None
    pi_k: torch.Tensor | None = None
    gate_kv: torch.Tensor | None = None
    gate_q: torch.Tensor | None = None


def summarize_temporal_gate(
    gate_logits: torch.Tensor | None,
) -> torch.Tensor | None:
    """Return detached moments for exact aggregate gate statistics."""
    if gate_logits is None:
        return None
    probabilities = torch.sigmoid(gate_logits.detach().float())
    saturated = (
        (probabilities < 0.01)
        | (probabilities > 0.99)
    ).float()
    return torch.stack(
        (
            probabilities.sum(),
            probabilities.square().sum(),
            saturated.sum(),
            probabilities.new_tensor(probabilities.numel()),
        )
    )


def repad_diagnostics(
    diagnostics: SwiLADiagnostics,
    *,
    indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
    valid_token_mask: torch.Tensor,
    pad_fn,
) -> None:
    """Repad only retained token fields using the caller's padding function."""
    from fla.modules.swila.packing import repad_packed_tensor

    retained_any = False
    for name in (
        "pi_q", "pi_k", "responsibilities", "r", "rho", "u",
        "pi_bar", "beta", "gate_kv", "gate_q",
    ):
        tensor = getattr(diagnostics, name)
        if tensor is None:
            continue
        setattr(diagnostics, name, repad_packed_tensor(
            tensor, indices, batch_size, seq_len, pad_fn=pad_fn
        ))
        retained_any = True
    diagnostics.valid_token_mask = valid_token_mask.detach() if retained_any else None


SWILA_CAPTURES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def active_swila_capture(layer) -> SwiLADiagnosticCapture | None:
    """Find the current scope without attaching anything to a model layer."""
    return SWILA_CAPTURES.get(layer)


def diagnostic_recomputation() -> bool:
    """Recognize marked checkpoints and ordinary PyTorch backward execution.

    The pinned Torch/Flame checkpoint stack uses this internal engine task ID.
    It also classifies a deliberately nested forward during backward as
    recomputation: such work must not replace the user's forward observation.
    Kernel checkpoint phases and launch policies are not changed here.
    """
    return (
        get_swila_checkpoint_phase() == "recompute"
        or torch._C._current_graph_task_id() >= 0
    )


def find_active_swila_diagnostics(model_or_layer: nn.Module) -> SwiLADiagnosticCapture | None:
    """Find an intersecting scope so a caller can respect external capture."""
    captures = {
        capture for module in model_or_layer.modules()
        if (capture := active_swila_capture(module)) is not None
    }
    if len(captures) > 1:
        raise RuntimeError("model intersects multiple active SwiLA diagnostic captures")
    return next(iter(captures), None)


class SwiLADiagnosticCapture:
    """Latest-per-layer detached records and hooks with explicit lifetimes.

    Keep the scope open through backward to collect gradient norms. Marked
    checkpoint recomputation never replaces forward records. Ordinary forward
    calls replace the previous record and discard its hooks, so a late backward
    cannot overwrite the latest forward's norms. Closing removes all hooks and
    registry entries; detached records remain readable until clear() or deletion.
    """

    def __init__(
        self,
        model_or_layer: nn.Module,
        *,
        grad_norms: bool = False,
        retain_tensors: bool = False,
        retain_gates: bool = False,
        router_stats: bool = True,
    ):
        if not isinstance(model_or_layer, nn.Module):
            raise TypeError("capture_swila_diagnostics expects a torch.nn.Module")
        named_layers = [
            (name, module) for name, module in model_or_layer.named_modules()
            if getattr(module, "requires_swila_reset_after_generic_init", False)
        ]
        if not named_layers:
            raise ValueError("no SwiLA layers found for diagnostic capture")
        self.layer_names = weakref.WeakKeyDictionary(
            (layer, name or "<root>") for name, layer in named_layers
        )
        self.collect_grad_norms = grad_norms
        self.retain_tensors = retain_tensors
        self.retain_gates = retain_gates
        self.collect_router_stats = router_stats
        self.records = weakref.WeakKeyDictionary()
        self.grad_norms = weakref.WeakKeyDictionary()
        self.hook_handles = weakref.WeakKeyDictionary()
        self.recorded_with_grad = weakref.WeakKeyDictionary()
        self.active = False

    def __enter__(self):
        if self.active:
            raise RuntimeError("SwiLA diagnostic capture is already active")
        if not callable(getattr(torch._C, "_current_graph_task_id", None)):
            raise RuntimeError(
                "SwiLA diagnostic capture requires torch._C._current_graph_task_id "
                "to identify checkpoint recomputation safely"
            )
        if any(active_swila_capture(layer) is not None for layer in self.layer_names):
            raise RuntimeError("overlapping SwiLA diagnostic captures are not supported")
        self.clear()
        for layer in self.layer_names:
            SWILA_CAPTURES[layer] = self
        self.active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def remove_hooks(self, layer):
        for handle in self.hook_handles.pop(layer, ()):
            handle.remove()

    def close(self):
        for layer in list(self.hook_handles):
            self.remove_hooks(layer)
        for layer in self.layer_names:
            if active_swila_capture(layer) is self:
                del SWILA_CAPTURES[layer]
        self.active = False

    def clear(self):
        """Release detached records and hooks without changing model state."""
        for layer in list(self.hook_handles):
            self.remove_hooks(layer)
        self.records.clear()
        self.grad_norms.clear()
        self.recorded_with_grad.clear()

    def named_records(self):
        for layer, record in self.records.items():
            yield self.layer_names[layer], record

    def named_grad_norms(self):
        for layer, norms in self.grad_norms.items():
            yield self.layer_names[layer], norms

    def summarize(self):
        """Export small records; scalar synchronization happens only on request."""
        token_fields = {
            "pi_q", "pi_k", "responsibilities", "r", "rho", "u", "pi_bar",
            "beta", "valid_token_mask", "gate_kv", "gate_q",
        }

        def scalar_or_tensor(value):
            if isinstance(value, torch.Tensor):
                return value.item() if value.numel() == 1 else value.detach()
            return value

        result = {}
        for layer, record in self.records.items():
            result[self.layer_names[layer]] = {
                name: scalar_or_tensor(value)
                for name, value in vars(record).items()
                if name not in token_fields and value is not None
            }
            result[self.layer_names[layer]]["grad_norms"] = {
                name: scalar_or_tensor(value)
                for name, value in vars(self.grad_norms[layer]).items()
                if value is not None
            }
        return result

    def record(
        self,
        layer,
        *,
        pi_q,
        pi_k,
        r,
        rho,
        responsibilities=None,
        gate_kv=None,
        gate_q=None,
        u=None,
        pi_bar=None,
        beta=None,
        cu_seqlens=None,
        aux_loss=None,
        mi_stats=None,
        noise_std_k=None,
        noise_std_q=None,
        gate_noise_std_kv=None,
        gate_noise_std_q=None,
    ) -> SwiLADiagnostics | None:
        if not self.active or active_swila_capture(layer) is not self:
            raise RuntimeError("layer is not in this active SwiLA diagnostic capture")
        if diagnostic_recomputation():
            return self.records.get(layer)
        self.remove_hooks(layer)
        record = SwiLADiagnostics()
        self.records[layer] = record
        self.grad_norms[layer] = SwiLAGradNorms()
        self.recorded_with_grad[layer] = torch.is_grad_enabled()
        if self.retain_tensors:
            for name, tensor in (
                ("pi_q", pi_q), ("pi_k", pi_k), ("r", r), ("rho", rho),
                ("u", u), ("pi_bar", pi_bar), ("beta", beta),
            ):
                setattr(record, name, tensor.detach() if tensor is not None else None)
            responsibility_tensor = (
                responsibilities.get("r") if isinstance(responsibilities, dict)
                else responsibilities
            )
            record.responsibilities = (
                responsibility_tensor.detach() if responsibility_tensor is not None else None
            )
        if self.retain_tensors or self.retain_gates:
            record.gate_kv = torch.sigmoid(gate_kv.detach()) if gate_kv is not None else None
            record.gate_q = torch.sigmoid(gate_q.detach()) if gate_q is not None else None
        if self.collect_router_stats:
            record.temporal_gate_kv_stats = summarize_temporal_gate(gate_kv)
            record.temporal_gate_q_stats = summarize_temporal_gate(gate_q)
        for name, value in (
            ("noise_std_k", noise_std_k), ("noise_std_q", noise_std_q),
            ("gate_noise_std_kv", gate_noise_std_kv),
            ("gate_noise_std_q", gate_noise_std_q),
        ):
            setattr(record, name, value.detach().mean() if isinstance(value, torch.Tensor) else value)
        if self.collect_router_stats and mi_stats is None and r is not None and rho is not None:
            mi_stats = detached_swila_router_stats(r, rho, layer.num_experts, cu_seqlens)
        if self.collect_router_stats and mi_stats is not None:
            for name, value in mi_stats.items():
                setattr(record, name, value.detach())
        record.aux_loss = aux_loss.detach() if aux_loss is not None else None
        return record

    def register_gradients(
        self, layer, *, q, k, v, beta, pi_q, pi_k,
        gate_kv=None, gate_q=None, output=None,
    ):
        if not self.active or active_swila_capture(layer) is not self:
            raise RuntimeError("layer is not in this active SwiLA diagnostic capture")
        if not (self.collect_grad_norms and layer.training):
            return
        if layer not in self.records:
            return
        if diagnostic_recomputation():
            # Non-reentrant checkpoints retain the original autograd graph and
            # hooks. Reentrant no-grad prepasses require hooks on recomputation.
            if self.recorded_with_grad.get(layer, True):
                return
        if layer in self.hook_handles:
            return
        norms = self.grad_norms[layer]
        layer_ref, capture_ref = weakref.ref(layer), weakref.ref(self)
        handles = []
        for name, tensor in (
            ("q", q), ("k", k), ("v", v), ("beta", beta),
            ("pi_q", pi_q), ("pi_k", pi_k),
            ("gate_kv", gate_kv), ("gate_q", gate_q), ("output", output),
        ):
            if tensor is None or not tensor.requires_grad:
                continue

            def hook(gradient, field=name, expected=norms):
                capture, current_layer = capture_ref(), layer_ref()
                if (gradient is not None and capture is not None and current_layer is not None
                        and capture.active and capture.grad_norms.get(current_layer) is expected):
                    # Preserve the original dtype and norm formula; never
                    # synchronize a device scalar to Python inside backward.
                    setattr(expected, field, gradient.norm().detach())

            handles.append(tensor.register_hook(hook))
        if handles:
            self.hook_handles[layer] = handles

    def repad(self, layer, **kwargs):
        if diagnostic_recomputation():
            return
        if layer in self.records:
            repad_diagnostics(self.records[layer], **kwargs)


def capture_swila_diagnostics(
    model_or_layer: nn.Module,
    *,
    grad_norms: bool = False,
    retain_tensors: bool = False,
    retain_gates: bool = False,
    router_stats: bool = True,
) -> SwiLADiagnosticCapture:
    """Create an opt-in diagnostic scope; no environment flags are read here."""
    return SwiLADiagnosticCapture(
        model_or_layer, grad_norms=grad_norms,
        retain_tensors=retain_tensors, retain_gates=retain_gates,
        router_stats=router_stats,
    )


def get_expert_metrics(
    diagnostics: SwiLADiagnostics,
    targets: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> dict:
    """Compute expert usage metrics from the last forward pass.

        Full token-level routing state is intentionally not retained by
        default. Use ``capture_swila_diagnostics(..., retain_tensors=True)``
        for metrics requiring token states; otherwise this returns an empty dict.

        Args:
            targets: [B, T] target labels. If provided, query-side metrics are computed
                only at positions where targets != ignore_index (prediction positions).
                Key-side metrics always use all positions.
            ignore_index: Label value to ignore for query-side filtering. Defaults to -100.

        Returns a dict with metrics for monitoring expert specialization:
        - expert_usage_query: [K] mean query-side weight per expert at prediction positions
        - expert_usage_key: [K] mean key-side weight per expert (all positions)
        - entropy_query: scalar, entropy of query distribution at prediction positions
        - entropy_key: scalar, entropy of key-side distribution (all positions)
        - max_prob_query: scalar, mean max expert probability at prediction positions
        - max_prob_key: scalar, mean max expert probability (all positions)
        - load_balance_query: scalar, load balance factor at prediction positions
        - load_balance_key: scalar, load balance factor (all positions)

        Returns:
            Dict of metrics, or empty dict if no forward pass has been done.
        """
    metrics = {}
    eps = 1e-8

    if diagnostics.pi_q is None:
        return metrics

    query_probabilities = diagnostics.rho
    if query_probabilities is None:
        query_probabilities = F.softmax(diagnostics.pi_q, dim=-1)
    K = query_probabilities.shape[-1]

    valid_mask = diagnostics.valid_token_mask
    query_mask = valid_mask
    if targets is not None:
        target_mask = targets != ignore_index
        query_mask = (
            target_mask
            if query_mask is None
            else query_mask.to(target_mask.device) & target_mask
        )

    def summarize(
        probabilities: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sum_dims = (0, 1, 2, 3)
        if token_mask is None:
            usage = probabilities.mean(dim=sum_dims)
            entropy = -(
                probabilities * torch.log(probabilities + eps)
            ).sum(dim=-1).mean()
            max_probability = probabilities.max(dim=-1).values.mean()
            return usage, entropy, max_probability

        token_mask = token_mask.to(
            device=probabilities.device,
            dtype=torch.bool,
        )
        if tuple(token_mask.shape) != tuple(probabilities.shape[:2]):
            raise ValueError(
                "metric mask shape must match router batch/time axes, "
                f"got {tuple(token_mask.shape)} and "
                f"{tuple(probabilities.shape[:2])}"
            )
        if not torch.any(token_mask):
            zero = probabilities.new_zeros(())
            return probabilities.new_zeros(K), zero, zero

        mask = token_mask.to(probabilities.dtype)[..., None, None]
        normalizer = (
            token_mask.sum().to(probabilities.dtype)
            * probabilities.shape[2]
            * probabilities.shape[3]
        )
        usage = (
            probabilities * mask[..., None]
        ).sum(dim=sum_dims) / normalizer
        entropy = (
            -(
                probabilities * torch.log(probabilities + eps)
            ).sum(dim=-1)
            * mask
        ).sum() / normalizer
        max_probability = (
            probabilities.max(dim=-1).values * mask
        ).sum() / normalizer
        return usage, entropy, max_probability

    expert_usage_q, entropy_q, max_prob_q = summarize(
        query_probabilities,
        query_mask,
    )

    metrics['expert_usage_query'] = expert_usage_q
    metrics['entropy_query'] = entropy_q.item()
    metrics['max_prob_query'] = max_prob_q.item()
    load_balance_q = 1.0 / (K * (expert_usage_q ** 2).sum() + eps)
    metrics['load_balance_query'] = load_balance_q.item()

    # Key-side metrics: responsibilities at all positions, [B, T, H, D_v, K].
    r = diagnostics.responsibilities
    if r is None:
        r = diagnostics.r
    if r is not None:
        expert_usage_k, entropy_k, max_prob_k = summarize(r, valid_mask)

        metrics['expert_usage_key'] = expert_usage_k
        metrics['entropy_key'] = entropy_k.item()
        metrics['max_prob_key'] = max_prob_k.item()
        load_balance_k = 1.0 / (K * (expert_usage_k ** 2).sum() + eps)
        metrics['load_balance_key'] = load_balance_k.item()

    return metrics
