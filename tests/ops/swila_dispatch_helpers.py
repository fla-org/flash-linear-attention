# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Observe real SwiLA launches while controlling only route selection."""

import json
from contextlib import contextmanager
from functools import wraps
from importlib import import_module

import torch

from tests.ops.swila_helpers import connected_inputs


def launch_value(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [launch_value(item) for item in value]
    return type(value).__name__


class KernelObserver:
    def __init__(self, kernel, trace, stage, binding, direct=False):
        self.kernel, self.trace, self.stage, self.binding, self.direct = kernel, trace, stage, binding, direct

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    @property
    def fn(self):
        underlying = self.kernel.fn
        if hasattr(underlying, "__getitem__"):
            return KernelObserver(underlying, self.trace, self.stage, self.binding, direct=True)
        return underlying

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def observe(*args, **kwargs):
            bound = dict(zip(self.kernel.arg_names, args))
            bound.update(kwargs)
            event = {
                "stage": self.stage,
                "binding": self.binding,
                "direct": self.direct,
                "arguments": {key: launch_value(value) for key, value in bound.items()},
                "completed": False,
            }
            self.trace.events.append(event)
            result = launch(*args, **kwargs)
            selected = getattr(self.kernel, "best_config", None)
            meta = {} if selected is None else {
                **selected.kwargs, "num_warps": selected.num_warps, "num_stages": selected.num_stages,
            }
            meta.update({key: value for key, value in kwargs.items() if not isinstance(value, torch.Tensor)})
            event["metadata"] = {key: launch_value(value) for key, value in meta.items()}
            event["grid"] = launch_value(grid(meta) if callable(grid) else grid)
            event["completed"] = True
            return result

        return observe


class RouteTrace:
    def __init__(self, family):
        self.family, self.events = family, []

    def one(self, stage):
        events = [event for event in self.events if event["stage"] == stage]
        assert len(events) == 1, f"expected one {stage} invocation, observed {events}"
        return events[0]

    def assert_route(self, expected, deferred=None):
        assert expected in ("generic", "split_native", "split_auto")
        assert self.events and all(event["completed"] for event in self.events), self.events
        forward = self.one("forward")
        if deferred is not None:
            assert forward["arguments"]["DEFER_READOUT"] is deferred
        stages = [event["stage"] for event in self.events]
        if self.family == "plain":
            assert expected == "generic" and stages == ["forward", "backward"]
            return
        if self.family == "gated":
            required = ["forward", "backward", "decay_scan"] if expected == "generic" else [
                "forward", "memory_reverse", "replay", "decay_scan",
            ]
            assert stages == required, self.events
            if expected != "generic":
                reverse, replay = self.one("memory_reverse"), self.one("replay")
                assert reverse["direct"] is (expected == "split_auto")
                assert replay["direct"] is (expected == "split_auto")
                if expected == "split_auto":
                    assert launch_triple(reverse) == (8, 1, 2)
                    assert launch_triple(replay) == (8, 1, 1)
                else:
                    assert launch_triple(reverse) in ((16, 1, 2), (32, 1, 2))
                    assert launch_triple(replay) in ((16, 1, 1), (32, 1, 1))
            return
        required = ["forward", "memory_reverse"] if expected == "generic" else [
            "forward", "query_reverse", "memory_reverse", "replay",
        ]
        if self.family == "temporal_gated":
            required.append("decay_scan")
        assert stages == required, self.events
        reverse = self.one("memory_reverse")
        split = expected != "generic"
        assert reverse["arguments"]["SPLIT_Q_TEMPORAL"] is split
        assert reverse["arguments"]["SKIP_REPLAY"] is split
        assert reverse["arguments"]["PRODUCTION_FAST"] is split
        assert reverse["direct"] is (expected == "split_auto")
        if expected == "split_auto":
            assert launch_triple(self.one("query_reverse")) == (32, 1, 2)
            assert launch_triple(reverse) == (8, 1, 2)
            assert launch_triple(self.one("replay")) == (16, 1, 1)
        elif expected == "split_native":
            assert launch_triple(self.one("query_reverse")) == (16, 1, 2)
            assert launch_triple(reverse) in ((16, 1, 2), (32, 1, 2))
            assert launch_triple(self.one("replay")) == (16, 1, 1)


def launch_triple(event):
    return tuple(event["metadata"][key] for key in ("BV", "num_warps", "num_stages"))


@contextmanager
def observe_routes(monkeypatch, family, mode="auto", disable_deferred=False):
    assert family in ("plain", "gated", "temporal", "temporal_gated")
    assert mode in ("auto", "native", "generic")
    trace = RouteTrace(family)
    with monkeypatch.context() as patch:
        def observe(module_name, name, stage):
            module = import_module(f"fla.ops.swila.{module_name}")
            patch.setattr(module, name, KernelObserver(getattr(module, name), trace, stage, f"{module_name}.{name}"))

        if family == "plain":
            observe("nongated_fwd", "fused_recurrent_swila_fwd_kernel", "forward")
            observe("nongated_bwd", "fused_recurrent_swila_bwd_kernel", "backward")
        elif family == "gated":
            observe("gated_fwd", "fused_recurrent_gated_swila_fwd_kernel", "forward")
            observe("gated_bwd", "fused_recurrent_gated_swila_bwd_kernel", "backward")
            for module_name in ("gated_bwd", "gated_split_bwd"):
                observe(module_name, "_fused_recurrent_gated_swila_dense_d64e4_bwd_reverse_kernel", "memory_reverse")
                observe(module_name, "_fused_recurrent_gated_swila_dense_d64e4_bwd_replay_kernel", "replay")
                observe(module_name, "_reverse_cumsum_log_decay_kernel", "decay_scan")
        else:
            observe("temporal_fwd", "fused_recurrent_swila_temporal_fwd_kernel", "forward")
            observe("temporal_bwd", "fused_recurrent_swila_temporal_bwd_kernel", "memory_reverse")
            observe("temporal_bwd", "fused_recurrent_swila_temporal_bwd_q_temporal_kernel", "query_reverse")
            observe("temporal_bwd", "fused_recurrent_swila_temporal_bwd_replay_kernel", "replay")
            observe("temporal_bwd", "_reverse_cumsum_log_decay_kernel", "decay_scan")
        if family != "plain":
            suffix = "gated" if family == "gated" else "temporal"
            if mode == "generic":
                module = import_module(f"fla.ops.swila.{suffix}_bwd")
                patch.setattr(module, "_use_dense_d64e4_key_bwd_fastpath", lambda *args, **kwargs: False)
            if mode != "auto":
                consumer = import_module(f"fla.ops.swila.fused_recurrent_{suffix}")
                name = "fused_recurrent_gated_swila_bwd" if family == "gated" else "fused_recurrent_swila_temporal_bwd"
                original = getattr(consumer, name)
                schedule_key = "dense_split_schedule" if family == "gated" else "split_schedule"

                @wraps(original)
                def native_backward(*args, **kwargs):
                    kwargs[schedule_key] = None
                    return original(*args, **kwargs)

                patch.setattr(consumer, name, native_backward)
        if disable_deferred:
            assert family == "gated", "only gated forward has the deferred-readout control"
            module = import_module("fla.ops.swila.gated_fwd")
            patch.setattr(module, "_use_dense_d64e4_key_deferred_readout", lambda *args, **kwargs: False)
        yield trace


def connected_to_objective(inputs, names):
    return set().union(*(connected_inputs(inputs, name) for name in names))


def comparison_summary(reference, actual):
    summary = {}
    for name, expected in reference.items():
        observed = actual[name]
        if expected is None or observed is None:
            summary[name] = {"reference_none": expected is None, "actual_none": observed is None}
            continue
        difference = observed.detach().float() - expected.detach().float()
        rms = difference.square().mean().sqrt()
        reference_rms = expected.detach().float().square().mean().sqrt()
        summary[name] = {
            "maximum_absolute": difference.abs().max().item(),
            "relative_rms": (rms / (reference_rms + 1e-8)).item(),
        }
    return summary


def record_case(record_property, *, name, inputs, names, trace, reference, actual, expected_grad, actual_grad):
    record_property(name, json.dumps({
        "seed": 42,
        "inputs": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in inputs.items()},
        "objective": list(names),
        "launches": trace.events,
        "outputs": comparison_summary(reference, actual),
        "gradients": comparison_summary(expected_grad, actual_grad),
    }, sort_keys=True))
