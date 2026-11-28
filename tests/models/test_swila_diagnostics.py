# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Detached model metrics and fixed distributed participation on CPU."""

import math

import pytest
import torch
import torch.distributed as dist

from fla.models.swila import SwiLAForCausalLM
from fla.models.swila.diagnostics import (
    collect_router_metrics,
    collect_training_metrics,
    router_metric_vector,
    temporal_gate_moment_vector,
)
from fla.modules.swila import SwiLADiagnosticCapture, SwiLADiagnostics
from tests.models.swila_helpers import model_config

CPU = torch.device("cpu")
ROUTER_KEYS = (
    "router/r_routing_entropy_norm", "router/rho_routing_entropy_norm",
    "router/combined_routing_entropy_norm", "router/r_utilization_entropy_norm",
    "router/rho_utilization_entropy_norm", "router/combined_utilization_entropy_norm",
    "router/r_dead_expert_frac", "router/rho_dead_expert_frac", "router/combined_dead_expert_frac",
)


@pytest.fixture(autouse=True)
def no_distributed_process_group(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: False)


def routing_record():
    return SwiLADiagnostics(
        H_cond_r=torch.tensor(math.log(4), requires_grad=True),
        H_cond_rho=torch.tensor(0., requires_grad=True),
        expert_util_r=torch.full((4,), .25, requires_grad=True),
        expert_util_rho=torch.tensor([1., 0., 0., 0.], requires_grad=True),
    )


def small_model(**options):
    return SwiLAForCausalLM(model_config(use_short_conv=False, use_pi_conv=False, **options))


def assert_finite_scalars(metrics):
    assert metrics
    assert all(isinstance(value, float) and math.isfinite(value) for value in metrics.values())


def test_router_metrics_are_detached_and_respect_each_router_unit_cpu():
    record = routing_record()
    combined_entropy = -sum(p * math.log(p) for p in (.625, .125, .125, .125)) / math.log(4)
    expected = dict(zip(ROUTER_KEYS, (1., 0., .5, 1., 0., combined_entropy, 0., .75, 0.), strict=True))
    metrics = collect_router_metrics([record], 4, CPU)
    assert metrics == pytest.approx(expected)
    assert_finite_scalars(metrics)
    vector = router_metric_vector(record, 4, math.log(4), CPU)
    assert not vector.requires_grad and vector.grad_fn is None
    assert record.H_cond_r.requires_grad and record.expert_util_r.requires_grad
    assert torch.autograd.grad(record.H_cond_r * 2, record.H_cond_r) == (torch.tensor(2.),)

    # Averaging heads together would hide two different sets of unused experts.
    utilization = torch.tensor([
        [[.98, .02, 0., 0.], [0., .02, .98, 0.]],
        [[.98, .02, 0., 0.], [0., .02, .98, 0.]],
    ], requires_grad=True)
    record.expert_util_r = record.expert_util_rho = utilization
    metrics = collect_router_metrics([record], 4, CPU)
    for side in ("r", "rho", "combined"):
        assert metrics[f"router/{side}_dead_expert_frac"] == pytest.approx(.75)


def test_temporal_metrics_use_element_weighted_detached_moments_cpu():
    first = SwiLADiagnostics(
        temporal_gate_q_stats=torch.tensor([.4, .10, 0., 2.], dtype=torch.float64, requires_grad=True),
        temporal_gate_kv_stats=torch.tensor([1.6, .84, 1., 4.], requires_grad=True),
    )
    second = SwiLADiagnostics(temporal_gate_q_stats=torch.tensor([2.1, 1.55, 2., 3.]))
    malformed = SwiLADiagnostics(temporal_gate_q_stats=torch.ones(3))
    metrics = collect_router_metrics([first, second, malformed], 4, CPU)
    assert metrics == pytest.approx({
        "temporal_gate/q_mean": .5, "temporal_gate/q_std": math.sqrt(.08),
        "temporal_gate/q_saturated_frac": .4,
        "temporal_gate/kv_mean": .4, "temporal_gate/kv_std": math.sqrt(.05),
        "temporal_gate/kv_saturated_frac": .25,
    })
    assert_finite_scalars(metrics)
    moments = temporal_gate_moment_vector(first, "temporal_gate_q_stats", CPU)
    assert moments.dtype == torch.float32 and not moments.requires_grad and moments.grad_fn is None
    assert first.temporal_gate_q_stats.requires_grad
    assert temporal_gate_moment_vector(malformed, "temporal_gate_q_stats", CPU) is None


def test_missing_records_and_single_expert_metrics_are_finite_cpu():
    assert collect_router_metrics([], 4, CPU) == {}
    assert collect_router_metrics([SwiLADiagnostics()], 4, CPU) == {}
    record = SwiLADiagnostics(
        H_cond_r=torch.tensor(0.), H_cond_rho=torch.tensor(0.),
        expert_util_r=torch.ones(1), expert_util_rho=torch.ones(1),
        MI_r=torch.tensor(0.), MI_rho=torch.tensor(0.),
    )
    assert collect_router_metrics([record], 1, CPU) == dict.fromkeys(ROUTER_KEYS, 0.)
    model = small_model(num_experts=1)
    model.model._total_aux_loss = torch.tensor(-.25, requires_grad=True)
    capture = SwiLADiagnosticCapture(model)
    capture.records[model.model.layers[0].attn] = record
    metrics = collect_training_metrics(model, torch.tensor(2., requires_grad=True), capture, router_stats=True)
    assert_finite_scalars(metrics)
    assert metrics["loss_metrics/mi_normalized"] == 0.
    assert metrics["mi_per_layer/L0_mi_norm"] == 0.
    assert metrics["loss_metrics/global_avg_total_loss"] == 1.75


def test_empty_rank_contributes_fixed_width_default_group_router_reduction_cpu(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    remote = torch.tensor([
        2., 0., 1., 2., 0., 1.5, 0., 1.5, 0., 2.,
        1.2, .74, 1., 2., 1.2, .56, 0., 3.,
    ])
    calls = []

    def reduce(values, op, group=None, async_op=False):
        calls.append((op, group, async_op, tuple(values.shape)))
        assert values.device == CPU and values.dtype == torch.float32 and not values.requires_grad
        torch.testing.assert_close(values, torch.zeros(18), rtol=0, atol=0)
        values.add_(remote)

    monkeypatch.setattr(dist, "all_reduce", reduce)
    metrics = collect_router_metrics([], 4, CPU)
    assert calls == [(dist.ReduceOp.SUM, None, False, (18,))]
    expected = dict(zip(ROUTER_KEYS, (1., 0., .5, 1., 0., .75, 0., .75, 0.), strict=True))
    expected.update({
        "temporal_gate/q_mean": .6, "temporal_gate/q_std": .1,
        "temporal_gate/q_saturated_frac": .5,
        "temporal_gate/kv_mean": .4, "temporal_gate/kv_std": math.sqrt(.08 / 3),
        "temporal_gate/kv_saturated_frac": 0.,
    })
    assert metrics == pytest.approx(expected)


def test_training_metrics_keep_local_values_and_average_before_router_sum_cpu(monkeypatch):
    model = small_model(num_hidden_layers=2)
    loss = torch.tensor(2., requires_grad=True)
    auxiliary = torch.tensor(-.375, requires_grad=True)
    model.model._total_aux_loss = auxiliary
    capture = SwiLADiagnosticCapture(model)
    for index, layer in enumerate(model.model.layers):
        record = routing_record()
        record.MI_r = torch.tensor((.2 if index == 0 else .8) * math.log(4), requires_grad=True)
        record.MI_rho = torch.tensor(.4 * math.log(4), requires_grad=True)
        record.aux_loss = torch.tensor(-.125 * (index + 1), requires_grad=True)
        if index == 0:
            record.MI_r_normalized, record.MI_rho_normalized = torch.tensor(.2), torch.tensor(.4)
        capture.records[layer.attn] = record
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    calls = []

    def reduce(values, op, group=None, async_op=False):
        calls.append((op, group, async_op, tuple(values.shape)))
        assert not values.requires_grad and values.device == CPU
        if op == dist.ReduceOp.AVG:
            torch.testing.assert_close(values, torch.tensor([2., -.375, .45]), rtol=1e-6, atol=1e-8)
            values.copy_(torch.tensor([4., -.125, .75]))
        else:
            assert op == dist.ReduceOp.SUM and values.shape == (18,)
            assert values[9] == 2 and torch.count_nonzero(values[10:]) == 0
            values.mul_(2)

    monkeypatch.setattr(dist, "all_reduce", reduce)
    metrics = collect_training_metrics(model, loss, capture, router_stats=True)
    assert calls == [(dist.ReduceOp.AVG, None, False, (3,)), (dist.ReduceOp.SUM, None, False, (18,))]
    assert_finite_scalars(metrics)
    expected = {
        "loss_metrics/ce_loss": 2., "loss_metrics/mi_aux_loss": -.375,
        "loss_metrics/mi_normalized": .45, "loss_metrics/global_avg_total_loss": 3.875,
        "loss_metrics/global_avg_mi_aux_loss": -.125, "loss_metrics/global_avg_mi_normalized": .75,
        "mi_per_layer/L0_mi_norm": .3, "mi_per_layer/L0_aux_loss": -.125,
        "mi_per_layer/L1_mi_norm": .6, "mi_per_layer/L1_aux_loss": -.25,
        "router/r_routing_entropy_norm": 1., "router/rho_dead_expert_frac": .75,
    }
    assert {key: metrics[key] for key in expected} == pytest.approx(expected)
    assert torch.autograd.grad(loss + auxiliary, (loss, auxiliary)) == (torch.tensor(1.), torch.tensor(1.))


def test_training_rank_without_records_still_reduces_and_disabled_stats_skip_only_sum_cpu(monkeypatch):
    model = small_model()
    model.model._total_aux_loss = 0.
    capture = SwiLADiagnosticCapture(model)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    calls = []

    def reduce(values, op, group=None, async_op=False):
        calls.append((op, group, async_op, tuple(values.shape)))
        assert not values.requires_grad
        if op == dist.ReduceOp.AVG:
            torch.testing.assert_close(values, torch.tensor([2., 0., 0.]), rtol=0, atol=0)
        else:
            torch.testing.assert_close(values, torch.zeros(18), rtol=0, atol=0)
            values[10:14] = torch.tensor([1., .5, 0., 2.])

    monkeypatch.setattr(dist, "all_reduce", reduce)
    metrics = collect_training_metrics(model, torch.tensor(2.), capture, router_stats=True)
    assert calls == [(dist.ReduceOp.AVG, None, False, (3,)), (dist.ReduceOp.SUM, None, False, (18,))]
    assert metrics["temporal_gate/q_mean"] == .5 and metrics["temporal_gate/q_std"] == 0.
    assert not any(key.startswith(("router/", "mi_per_layer/")) for key in metrics)
    calls.clear()
    metrics = collect_training_metrics(model, torch.tensor(2.), None, router_stats=False)
    assert calls == [(dist.ReduceOp.AVG, None, False, (3,))]
    assert len(metrics) == 6 and metrics["loss_metrics/global_avg_total_loss"] == 2.
    assert_finite_scalars(metrics)
