# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Public strided inputs, omitted outputs, and full-width packed SwiLA gradients."""

import json

import pytest
import torch

from fla.utils import IS_NVIDIA, device
from tests.ops.swila_dispatch_helpers import connected_to_objective, observe_routes, record_case
from tests.ops.swila_helpers import (
    FAMILIES,
    FINAL_KEYS,
    STATE_KEYS,
    clone_inputs,
    compare_gradients,
    compare_outputs,
    cotangents,
    make_inputs,
    reference_graph,
    run_fused,
)

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason="SwiLA layout qualification requires NVIDIA CUDA")


def strided_views(tensors, *, requires_grad):
    backing, views = {}, {}
    for name, tensor in tensors.items():
        storage = torch.full(
            (*tensor.shape[:-1], 2 * tensor.shape[-1]), 17., dtype=tensor.dtype, device=tensor.device,
        )
        storage[..., 1::2].copy_(tensor.detach())
        backing[name] = storage.requires_grad_(requires_grad)
        views[name] = backing[name][..., 1::2]
    return backing, views


def output_vjp(outputs, inputs, weights):
    values = torch.autograd.grad(
        outputs=tuple(outputs[name] for name in weights),
        inputs=tuple(inputs.values()),
        grad_outputs=tuple(weights.values()),
        allow_unused=True,
    )
    return dict(zip(inputs, values))


@pytest.mark.parametrize("family", FAMILIES)
def test_fused_strided_optional_outputs(family, monkeypatch, record_property):
    data = make_inputs(
        family, dtype=torch.float32, device=device, seed=42, batch=2, length=7,
        heads=2, keys=5, values=3, experts=3, initial=True,
    )
    omitted = FINAL_KEYS | ({"rho"} if "temporal" not in family else set())
    for objective_name in ("responsibilities", "combined"):
        reference_inputs, full_reference = reference_graph(data, family, scale=.7)
        reference = {name: tensor for name, tensor in full_reference.items() if name not in omitted}
        names = ("r",) if objective_name == "responsibilities" else tuple(reference)
        backing, inputs = strided_views(clone_inputs(data, requires_grad=False), requires_grad=True)
        _, weights = strided_views(cotangents(reference, seed=42), requires_grad=False)
        weights = {name: weights[name] for name in names}
        record_property(f"{objective_name}_layout", json.dumps({
            "input_seed": 42,
            "cotangent_seed": 42,
            "output_final_state": False,
            "return_rho_seq": "temporal" in family,
            "inputs": {name: {"stride": list(x.stride()), "offset": x.storage_offset()} for name, x in inputs.items()},
            "cotangents": {name: {"stride": list(x.stride()), "offset": x.storage_offset()} for name, x in weights.items()},
        }, sort_keys=True))
        assert all(not tensor.is_contiguous() and tensor.stride(-1) == 2 for tensor in inputs.values())
        assert all(not tensor.is_contiguous() and tensor.stride(-1) == 2 for tensor in weights.values())
        assert all(tensor.is_contiguous() for tensor in reference_inputs.values())
        expected_grad = output_vjp(reference, reference_inputs, weights)
        targets = {**inputs, **{f"backing/{name}": tensor for name, tensor in backing.items()}}
        with observe_routes(monkeypatch, family) as trace:
            actual = run_fused(
                inputs, family, scale=.7, output_final_state=False, return_rho_seq="temporal" in family,
            )
            all_grad = output_vjp(actual, targets, weights)
        actual_grad = {name: all_grad[name] for name in inputs}
        record_case(
            record_property,
            name=objective_name,
            inputs=inputs,
            names=names,
            trace=trace,
            reference={name: None if name in omitted else tensor for name, tensor in full_reference.items()},
            actual=actual,
            expected_grad=expected_grad,
            actual_grad=actual_grad,
        )
        record_property(f"{objective_name}_backing", json.dumps({
            name: None if all_grad[f"backing/{name}"] is None else {
                "finite": bool(torch.isfinite(all_grad[f"backing/{name}"]).all()),
                "unused_nonzero": torch.count_nonzero(all_grad[f"backing/{name}"][..., ::2]).item(),
            } for name in inputs
        }, sort_keys=True))
        trace.assert_route(expected="generic", deferred=False if family == "gated" else None)
        assert actual.keys() == full_reference.keys()
        assert all(actual[name] is None for name in omitted & actual.keys())
        compare_outputs(reference, {name: actual[name] for name in reference}, prefix=f"{objective_name}/")
        compare_gradients(
            expected_grad,
            actual_grad,
            connected=connected_to_objective(reference_inputs, names),
            prefix=f"{objective_name}/",
        )
        for name, tensor in inputs.items():
            torch.testing.assert_close(tensor, data[name], atol=0, rtol=0)
            assert torch.all(backing[name][..., ::2] == 17.), f"{name}: unused backing storage changed"
            storage_grad = all_grad[f"backing/{name}"]
            if storage_grad is None:
                assert actual_grad[name] is None
            else:
                assert torch.isfinite(storage_grad).all()
                assert torch.count_nonzero(storage_grad[..., ::2]) == 0, f"{name}: gradient in unused storage"
                assert actual_grad[name] is not None
                torch.testing.assert_close(storage_grad[..., 1::2], actual_grad[name], atol=0, rtol=0)


def test_fused_temporal_gated_packed_full_width(monkeypatch, record_property):
    family, lengths = "temporal_gated", (1, 33, 31)
    data = make_inputs(
        family, dtype=torch.bfloat16, device=device, seed=42, length=sum(lengths), heads=1,
        keys=64, values=64, experts=4, initial=True, sequences=len(lengths),
    )
    for name in STATE_KEYS:
        data[name] = data[name].float()
    boundaries = torch.tensor([0, 1, 34, 65], dtype=torch.int32, device=device)
    record_property("packed_fixture", json.dumps({
        "lengths": list(lengths), "cu_seqlens": [0, 1, 34, 65], "input_seed": 42, "cotangent_seed": 42,
        "first_sequence_cotangent": "ones on the first output sequence",
        "backward_graphs": "fresh per objective",
    }, sort_keys=True))

    for objective_name in ("combined", "u", "pi_bar", "first_sequence"):
        reference_inputs, reference = reference_graph(data, family, lengths=lengths)
        inputs = clone_inputs(data)
        weights = cotangents(reference, seed=42)
        names = tuple(reference) if objective_name == "combined" else (
            "o" if objective_name == "first_sequence" else objective_name,
        )
        weights = {name: weights[name] for name in names}
        reference_objective = reference
        if objective_name == "first_sequence":
            reference_objective = {"o": reference["o"][:, :lengths[0]]}
            weights = {"o": torch.ones_like(reference_objective["o"])}
        expected_grad = output_vjp(reference_objective, reference_inputs, weights)
        with observe_routes(monkeypatch, family) as trace:
            actual = run_fused(inputs, family, cu_seqlens=boundaries)
            actual_objective = {"o": actual["o"][:, :lengths[0]]} if objective_name == "first_sequence" else actual
            actual_grad = output_vjp(actual_objective, inputs, weights)
        record_case(
            record_property,
            name=objective_name,
            inputs=inputs,
            names=names,
            trace=trace,
            reference=reference,
            actual=actual,
            expected_grad=expected_grad,
            actual_grad=actual_grad,
        )
        if objective_name == "first_sequence":
            record_property("boundary_gradients", json.dumps({
                label: {
                    name: None if tensor is None else {
                        "finite": bool(torch.isfinite(tensor).all()),
                        "outside_nonzero": torch.count_nonzero(
                            tensor[1:] if name in STATE_KEYS else tensor[:, lengths[0]:],
                        ).item(),
                    } for name, tensor in grads.items()
                } for label, grads in (("reference", expected_grad), ("actual", actual_grad))
            }, sort_keys=True))
        trace.assert_route(expected="generic")
        for event in trace.events:
            assert event["arguments"]["IS_VARLEN"] is True
            assert event["arguments"]["cu_seqlens"]["shape"] == [4]
            assert event["arguments"]["cu_seqlens"]["dtype"] == "torch.int32"
        for stage in ("forward", "memory_reverse"):
            arguments = trace.one(stage)["arguments"]
            assert (arguments["D_k"], arguments["D_v"], arguments["E"]) == (64, 64, 4)
            assert arguments["USE_INITIAL_STATE"] and arguments["USE_INITIAL_TEMPORAL_STATE"]
        reverse = trace.one("memory_reverse")["arguments"]
        assert reverse["REUSE_DK_COEF"] is False and reverse["RESOURCE_SAFE"] is False
        compare_outputs(reference, actual, prefix=f"{objective_name}/")
        compare_gradients(
            expected_grad,
            actual_grad,
            connected=connected_to_objective(reference_inputs, names),
            prefix=f"{objective_name}/",
        )
        if objective_name == "first_sequence":
            for grads in (expected_grad, actual_grad):
                for name, tensor in grads.items():
                    assert tensor is not None and torch.isfinite(tensor).all(), name
                    outside = tensor[1:] if name in STATE_KEYS else tensor[:, lengths[0]:]
                    assert torch.count_nonzero(outside) == 0, f"{name}: gradient crossed a packed boundary"
