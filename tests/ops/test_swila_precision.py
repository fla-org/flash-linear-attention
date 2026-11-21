# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Reference checks for temporal FP16, long sequences, and full-width E8 SwiLA."""

import json
from importlib import import_module

import pytest
import torch

from fla.utils import IS_NVIDIA, device
from tests.ops.swila_dispatch_helpers import connected_to_objective, observe_routes, record_case
from tests.ops.swila_helpers import (
    clone_inputs,
    compare_gradients,
    compare_outputs,
    cotangents,
    gradients,
    make_inputs,
    objective,
    reference_graph,
    run_fused,
)

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason="SwiLA numerical qualification requires NVIDIA CUDA")


def make_precision_inputs(family, **kwargs):
    data = make_inputs(family, device=device, seed=42, **kwargs)
    for name in ("initial_state", "initial_r", "initial_rho"):
        if name in data:
            data[name] = data[name].float()
    return data


def check_generic_reference(data, family, monkeypatch, record_property):
    reference_inputs, reference = reference_graph(data, family)
    names = tuple(reference)
    weights = cotangents(reference, seed=42)
    expected_grad = gradients(objective(reference, weights, names), reference_inputs)
    inputs = clone_inputs(data)
    with observe_routes(monkeypatch, family, mode="auto") as trace:
        actual = run_fused(inputs, family)
        actual_grad = gradients(objective(actual, weights, names), inputs)
    record_case(
        record_property,
        name="generic-reference",
        inputs=inputs,
        names=names,
        trace=trace,
        reference=reference,
        actual=actual,
        expected_grad=expected_grad,
        actual_grad=actual_grad,
    )
    record_property("comparison_policy", "per-returned-output-and-leaf-dtype")
    trace.assert_route(expected="generic", deferred=False if family == "gated" else None)
    compare_outputs(reference, actual)
    compare_gradients(
        expected_grad,
        actual_grad,
        connected=connected_to_objective(reference_inputs, names),
    )
    return trace


def check_e8_backward_pruning(data, family, record_property):
    policies = import_module("fla.ops.swila.autotune")
    if family == "gated":
        kernel = import_module("fla.ops.swila.gated_bwd").fused_recurrent_gated_swila_bwd_kernel
        expected_pruner = policies.prune_gated_backward_configs
    else:
        kernel = import_module("fla.ops.swila.temporal_bwd").fused_recurrent_swila_temporal_bwd_kernel
        expected_pruner = policies.prune_temporal_backward_configs
    assert kernel.early_config_prune is expected_pruner
    arguments = {
        "q": data["q"], "D_k": 64, "D_v": 64, "E": 8, "BK": 64, "BE": 8,
        "RESOURCE_SAFE": False, "PRODUCTION_FAST": False,
    }
    configs = list(kernel.configs)
    original = [(config.kwargs["BV"], config.num_warps, config.num_stages) for config in configs]
    selected = kernel.early_config_prune(configs, arguments)
    retained = [(config.kwargs["BV"], config.num_warps, config.num_stages) for config in selected]
    sm90 = torch.cuda.get_device_capability(data["q"].device) == (9, 0)
    if family == "gated":
        expected = [triple for triple in original if not (sm90 and triple[:2] == (64, 1))]
    else:
        expected = [triple for triple in original if triple[0] in (8, 16)]
    record_property("backward_pruning", json.dumps({
        "original": original, "retained": retained, "expected": expected, "sm90": sm90,
        "pruner": expected_pruner.__name__,
    }, sort_keys=True))
    assert original and expected
    assert retained == expected
    if family != "gated" or sm90:
        assert len(retained) < len(original)
    return retained


@pytest.mark.parametrize(("family", "initial"), [
    pytest.param("temporal", True, id="temporal-carried"),
    pytest.param("temporal_gated", False, id="temporal-gated-default"),
])
def test_fused_recurrent_temporal_fp16(family, initial, monkeypatch, record_property):
    data = make_precision_inputs(
        family, dtype=torch.float16, batch=1, length=17, heads=1, keys=8, values=4, experts=4, initial=initial,
    )
    check_generic_reference(data, family, monkeypatch, record_property)


@pytest.mark.parametrize("family", [
    pytest.param("plain", id="plain"),
    pytest.param("temporal_gated", id="temporal-gated"),
])
def test_fused_recurrent_long_reference(family, monkeypatch, record_property):
    data = make_precision_inputs(
        family, dtype=torch.float32, batch=1, length=1025, heads=1, keys=8, values=4, experts=4, initial=True,
    )
    check_generic_reference(data, family, monkeypatch, record_property)


@pytest.mark.parametrize("family", [
    pytest.param("gated", id="gated"),
    pytest.param("temporal_gated", id="temporal-gated"),
])
def test_fused_recurrent_full_width_e8(family, monkeypatch, record_property):
    data = make_precision_inputs(
        family, dtype=torch.bfloat16, batch=1, length=33, heads=1, keys=64, values=64, experts=8, initial=True,
    )
    retained = check_e8_backward_pruning(data, family, record_property)
    trace = check_generic_reference(data, family, monkeypatch, record_property)
    reverse = trace.one("backward" if family == "gated" else "memory_reverse")
    assert tuple(reverse["arguments"][key] for key in ("D_k", "D_v", "E", "BK", "BE")) == (64, 64, 8, 64, 8)
    assert tuple(reverse["metadata"][key] for key in ("BV", "num_warps", "num_stages")) in retained
    if family == "temporal_gated":
        for flag in ("RESOURCE_SAFE", "PRODUCTION_FAST", "REUSE_DK_COEF", "SKIP_REPLAY", "SPLIT_Q_TEMPORAL"):
            assert reverse["arguments"][flag] is False
