# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Temporal reverse-state capture: source offsets, guarded storage, and numerical isolation."""

import ast
import json
import math
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import pytest
import torch

from fla.utils import IS_NVIDIA, device
from tests.ops.swila_dispatch_helpers import comparison_summary, observe_routes, record_case
from tests.ops.swila_helpers import (
    FINAL_KEYS,
    STATE_KEYS,
    assert_swila_close,
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


def debug_address_expression():
    source = Path(__file__).resolve().parents[2] / "fla/ops/swila/temporal_bwd_state.py"
    tree = ast.parse(source.read_text())
    definitions = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "debug_dh_base" for target in node.targets)]
    assert len(definitions) == 1, "audit every debug capture store when its addressing changes"
    value = definitions[0].value
    assert all(isinstance(node, (ast.BinOp, ast.Add, ast.Mult, ast.Name, ast.Load)) for node in ast.walk(value))
    return compile(ast.Expression(value), str(source), "eval")


@pytest.mark.parametrize(("lengths", "heads", "packed"), [
    pytest.param((3, 3), 2, False, id="dense-b2-t3-h2"),
    pytest.param((1, 4), 2, True, id="packed-short-long-h2"),
    pytest.param((4, 1), 2, True, id="packed-long-short-h2"),
])
def test_swila_debug_address_layout(lengths, heads, packed):
    expression = debug_address_expression()
    values, experts, keys = 3, 3, 5
    slab = values * experts * keys
    total = sum(lengths) if packed else lengths[0]
    batch = 1 if packed else len(lengths)
    actual, expected = [], []
    bos = 0
    for sequence, length in enumerate(lengths):
        for head in range(heads):
            for time in range(length):
                actual.append(eval(expression, {"__builtins__": {}}, {
                    "debug_dh": 0, "bos": bos, "remaining": time, "H": heads, "i_h": head,
                    "i_nh": sequence * heads + head, "T": length,
                    "D_v": values, "E": experts, "D_k": keys,
                }))
                token = bos + time if packed else sequence * total + time
                expected.append((token * heads + head) * slab)
        bos += length
    assert actual == expected
    assert sorted(actual) == list(range(0, batch * total * heads * slab, slab))


@contextmanager
def guarded_debug_capture(monkeypatch, data, lengths):
    """Keep even the historical bad offsets inside allocated sentinel storage."""
    consumer = import_module("fla.ops.swila.fused_recurrent_temporal")
    backward = consumer.fused_recurrent_swila_temporal_bwd
    new_zeros = torch.Tensor.new_zeros
    batch, length, heads, keys = data["q"].shape
    values, experts = data["beta"].shape[-2:]
    shape = (batch, length, heads, values, experts, keys)
    slab = values * experts * keys
    segments = (length,) * batch if lengths is None else lengths
    historical_extent = max((sequence * heads + heads) * size for sequence, size in enumerate(segments)) * slab
    payload_size = math.prod(shape)
    prefix = slab
    suffix = max(slab, historical_extent - payload_size + slab)
    sentinel = 12345.0
    allocations, captures = [], []

    def allocate(tensor, *size, **kwargs):
        if size != shape:
            return new_zeros(tensor, *size, **kwargs)
        assert kwargs == {"dtype": torch.float32}
        storage = torch.full((prefix + payload_size + suffix,), sentinel, dtype=torch.float32, device=tensor.device)
        payload = storage[prefix:prefix + payload_size].view(shape)
        payload.zero_()
        allocations.append((storage, payload))
        return payload

    def capture(*args, **kwargs):
        assert "debug_intermediates" not in kwargs
        result = backward(*args, **kwargs, debug_intermediates=True)
        assert len(result) == 14
        captures.append(result[-2])
        return result[:-2]

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "new_zeros", allocate)
        patch.setattr(consumer, "fused_recurrent_swila_temporal_bwd", capture)
        yield captures
    assert len(captures) == len(allocations) == 1
    storage, payload = allocations[0]
    assert captures[0] is payload
    assert torch.isfinite(payload).all()
    assert torch.all(storage[:prefix] == sentinel), "debug capture overwrote its prefix guard"
    assert torch.all(storage[prefix + payload_size:] == sentinel), "debug capture overwrote its suffix guard"


def select_sequence_head(tensors, state_names, *, batch, sequence, head, start, end):
    return {
        name: (tensor[sequence:sequence + 1, head:head + 1] if name in state_names
               else tensor[batch:batch + 1, start:end, head:head + 1]).contiguous()
        for name, tensor in tensors.items()
    }


def capture_graph(monkeypatch, data, weights, names, lengths):
    inputs = clone_inputs(data)
    cu_seqlens = None if lengths is None else torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64, device=inputs["q"].device,
    )
    with guarded_debug_capture(monkeypatch, inputs, lengths) as captures:
        outputs = run_fused(inputs, "temporal_gated", cu_seqlens=cu_seqlens)
        actual_grad = gradients(objective(outputs, weights, names), inputs)
    return inputs, outputs, actual_grad, captures[0]


@pytest.mark.skipif(not IS_NVIDIA, reason="Temporal debug capture requires NVIDIA CUDA")
@pytest.mark.parametrize(("lengths", "batch", "dtype", "width", "expected_route"), [
    pytest.param(None, 2, torch.float32, 5, "generic", id="dense-b2-h2-fp32"),
    pytest.param((1, 4), 1, torch.float32, 5, "generic", id="packed-short-long-h2-fp32"),
    pytest.param((4, 1), 1, torch.float32, 5, "generic", id="packed-long-short-h2-fp32"),
    pytest.param(None, 1, torch.bfloat16, 64, "split_native", id="dense-native-split-h2-bf16"),
])
def test_swila_temporal_debug_capture(lengths, batch, dtype, width, expected_route, monkeypatch, record_property):
    torch.manual_seed(42)
    data = make_inputs(
        "temporal_gated", dtype=dtype, device=device, batch=batch, length=5, heads=2,
        keys=width, values=64 if width == 64 else 3, experts=4 if width == 64 else 3,
        initial=True, sequences=None if lengths is None else len(lengths),
    )
    for name in STATE_KEYS & data.keys():
        data[name] = data[name].float()
    if expected_route == "split_native":
        del data["initial_r"], data["initial_rho"]
    reference_inputs, reference = reference_graph(data, "temporal_gated", lengths=lengths)
    names = tuple(name for name in reference if expected_route != "split_native" or name not in FINAL_KEYS)
    weights = cotangents(reference)
    expected_grad = gradients(objective(reference, weights, names), reference_inputs)
    cu_seqlens = None if lengths is None else torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64, device=device,
    )

    control_inputs = clone_inputs(data)
    with observe_routes(monkeypatch, "temporal_gated") as control_trace:
        control = run_fused(control_inputs, "temporal_gated", cu_seqlens=cu_seqlens)
        control_grad = gradients(objective(control, weights, names), control_inputs)
    control_trace.assert_route(expected=expected_route)
    assert control_trace.one("memory_reverse")["arguments"]["DEBUG_INTERMEDIATES"] is False
    with observe_routes(monkeypatch, "temporal_gated") as trace:
        inputs, actual, actual_grad, captured = capture_graph(monkeypatch, data, weights, names, lengths)
    record_case(
        record_property, name="debug-capture", inputs=inputs, names=names, trace=trace,
        reference=reference, actual=actual, expected_grad=expected_grad, actual_grad=actual_grad,
    )
    trace.assert_route(expected=expected_route)
    assert trace.one("memory_reverse")["arguments"]["DEBUG_INTERMEDIATES"] is True
    for label, outputs, observed_grad in (("capture", actual, actual_grad), ("control", control, control_grad)):
        compare_outputs(reference, outputs, prefix=label + "/")
        compare_gradients(expected_grad, observed_grad, prefix=label + "/")
    compare_outputs(control, actual, prefix="capture-preserves/")
    compare_gradients(control_grad, actual_grad, prefix="capture-preserves/")

    independent = torch.empty_like(captured)
    start = 0
    for sequence, length in enumerate((5,) * batch if lengths is None else lengths):
        current_batch = sequence if lengths is None else 0
        current_start = 0 if lengths is None else start
        end = current_start + length
        for head in range(2):
            selection = dict(batch=current_batch, sequence=sequence, head=head, start=current_start, end=end)
            part = select_sequence_head(data, STATE_KEYS, **selection)
            part_weights = select_sequence_head(weights, FINAL_KEYS, **selection)
            with observe_routes(monkeypatch, "temporal_gated") as part_trace:
                _, _, _, part_capture = capture_graph(monkeypatch, part, part_weights, names, None)
            part_trace.assert_route(expected=expected_route)
            independent[current_batch:current_batch + 1, current_start:end, head:head + 1] = part_capture
        start += length
    record_property("debug-state-comparison", json.dumps(comparison_summary({"dh": independent}, {"dh": captured})))
    assert_swila_close("debug/per-sequence-per-head", independent, captured, gradient=True)
