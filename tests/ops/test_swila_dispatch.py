# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Numerical qualification of the real generic, split, and automatic SwiLA routes."""

import json

import pytest
import torch

from fla.utils import IS_NVIDIA, device
from tests.ops.swila_dispatch_helpers import connected_to_objective, observe_routes, record_case
from tests.ops.swila_helpers import (
    FINAL_KEYS,
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

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason="SwiLA production-route qualification requires NVIDIA CUDA")


def make_route_inputs(family, *, length, heads, width):
    data = make_inputs(
        family,
        dtype=torch.bfloat16,
        device=device,
        length=length,
        heads=heads,
        keys=width,
        values=width,
        experts=4,
        initial=True,
    )
    for name in ("initial_state", "initial_r", "initial_rho"):
        if name in data:
            data[name] = data[name].float()
    if family == "temporal_gated":
        del data["initial_r"], data["initial_rho"]
    return data


@pytest.mark.parametrize(("family", "width", "routes"), [
    pytest.param("plain", 32, (("auto", "generic"),), id="plain-generic-d32-e4"),
    pytest.param("gated", 64, (("native", "split_native"), ("generic", "generic")), id="gated-native-and-generic-d64-e4"),
    pytest.param("temporal", 64, (("auto", "generic"),), id="temporal-generic-carried-priors-d64-e4"),
    pytest.param("temporal_gated", 64, (("native", "split_native"), ("generic", "generic")),
                 id="temporal-gated-split-and-final-state-fallback-d64-e4"),
])
def test_swila_production_routes_moderate(family, width, routes, monkeypatch, record_property):
    torch.manual_seed(42)
    data = make_route_inputs(family, length=65, heads=1, width=width)
    reference_inputs, reference = reference_graph(data, family)
    names = tuple(name for name in reference if family != "temporal_gated" or name not in FINAL_KEYS)
    weights = cotangents(reference)
    connected = connected_to_objective(reference_inputs, names)
    expected_grad = gradients(objective(reference, weights, names), reference_inputs)

    for mode, expected_route in routes:
        inputs = clone_inputs(data)
        with observe_routes(monkeypatch, family, mode=mode) as trace:
            actual = run_fused(inputs, family)
            actual_grad = gradients(objective(actual, weights, names), inputs)
        record_case(
            record_property,
            name=mode,
            inputs=inputs,
            names=names,
            trace=trace,
            reference=reference,
            actual=actual,
            expected_grad=expected_grad,
            actual_grad=actual_grad,
        )
        trace.assert_route(expected=expected_route, deferred=False if family == "gated" else None)
        compare_outputs(reference, actual, prefix=f"{mode}/")
        compare_gradients(expected_grad, actual_grad, connected=connected, prefix=f"{mode}/")

    if family == "temporal_gated":
        # nonzero final-state/prior cotangents must independently select the generic backward.
        reference_inputs, reference = reference_graph(data, family)
        names = tuple(name for name in reference if name in FINAL_KEYS)
        expected_grad = gradients(objective(reference, weights, names), reference_inputs)
        inputs = clone_inputs(data)
        with observe_routes(monkeypatch, family) as trace:
            actual = run_fused(inputs, family)
            actual_grad = gradients(objective(actual, weights, names), inputs)
        record_case(
            record_property,
            name="final-only-natural-fallback",
            inputs=inputs,
            names=names,
            trace=trace,
            reference=reference,
            actual=actual,
            expected_grad=expected_grad,
            actual_grad=actual_grad,
        )
        trace.assert_route(expected="generic")
        compare_outputs(reference, actual, prefix="final-only/")
        compare_gradients(expected_grad, actual_grad, connected=connected_to_objective(inputs, names), prefix="final-only/")


@pytest.mark.parametrize("family", ("gated", "temporal_gated"), ids=("gated-deferred", "temporal-gated"))
def test_swila_production_routes_automatic(family, monkeypatch, record_property):
    if torch.cuda.get_device_capability(device) != (9, 0) or "H100" not in torch.cuda.get_device_name(device):
        pytest.skip("The fixed automatic SwiLA schedule is defined only for H100 SM90")
    torch.manual_seed(42)
    data = make_route_inputs(family, length=4096, heads=16, width=64)
    names = ("o", "r", "rho")
    automatic_inputs = clone_inputs(data)
    with observe_routes(monkeypatch, family) as automatic_trace:
        automatic = run_fused(automatic_inputs, family)
        weights = cotangents({name: automatic[name] for name in names})
        automatic_grad = gradients(objective(automatic, weights, names), automatic_inputs)
    record_property("automatic_launches", json.dumps(automatic_trace.events, sort_keys=True))
    automatic_trace.assert_route(expected="split_auto", deferred=True if family == "gated" else None)

    native_inputs = clone_inputs(data)
    with observe_routes(monkeypatch, family, mode="native", disable_deferred=family == "gated") as native_trace:
        native = run_fused(native_inputs, family)
        native_grad = gradients(objective(native, weights, names), native_inputs)
    record_property("native_launches", json.dumps(native_trace.events, sort_keys=True))
    record_case(
        record_property,
        name="automatic-versus-native",
        inputs=automatic_inputs,
        names=names,
        trace=automatic_trace,
        reference=native,
        actual=automatic,
        expected_grad=native_grad,
        actual_grad=automatic_grad,
    )
    native_trace.assert_route(expected="split_native", deferred=False if family == "gated" else None)
    compare_outputs(native, automatic, prefix="automatic/")
    compare_gradients(native_grad, automatic_grad, connected=connected_to_objective(data, names), prefix="automatic/")
