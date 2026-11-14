# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Numerical and real compilation-reuse checks for dynamic-length SwiLA MI."""

import json
import math
import re

import pytest
import torch

from fla.ops.swila import mi_loss
from fla.utils import IS_NVIDIA, device
from tests.ops.swila_dispatch_helpers import comparison_summary
from tests.ops.swila_helpers import assert_swila_grad, check_mi_result, dense_mi_gradient, mi_reference

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason="SwiLA MI Triton kernels require NVIDIA CUDA")
LENGTHS = (1, 63, 65, 95, 129, 193)


def mi_inputs(length, dtype, *, logits):
    values = []
    for seed in (42, 43):
        generator = torch.Generator(device=device).manual_seed(seed)
        value = torch.randn((2, length, 1, 3, 4), generator=generator, device=device, dtype=torch.float32)
        if not logits:
            value = value.softmax(-1)
            # Exercise epsilon derivatives and masked tails, including both sides of a tile boundary.
            edge = value.new_tensor([0., 1e-12, .25, .75])
            for index in {0, length - 1, min(63, length - 1), min(64, length - 1)}:
                value[0, index, 0, 0] = edge
        values.append(value.to(dtype).detach().requires_grad_())
    return tuple(values)


def check_mi_runtime(length, dtype, *, logits, record_property):
    leaves = mi_inputs(length, dtype, logits=logits)
    references = tuple(value.detach().clone().requires_grad_() for value in leaves)
    probs = tuple(value.float().softmax(-1).to(dtype) for value in leaves) if logits else leaves
    ref_probs = tuple(value.float().softmax(-1).to(dtype) for value in references) if logits else references
    expected = mi_reference(*(value.float() for value in ref_probs), .73)
    assert mi_loss.can_use_fused_swila_mi_loss_from_probs(*probs)
    actual = mi_loss.fused_swila_mi_loss_from_probs(*probs, mi_loss_weight=.73)
    upstream = (-1.37 if length in (63, 95, 193) else 1.37) * math.prod(probs[0].shape[:-1])
    expected_grad = torch.autograd.grad(expected[0], references, expected[0].new_tensor(upstream))
    actual_grad = torch.autograd.grad(actual[0], leaves, actual[0].new_tensor(upstream))
    record_property(f"mi_T{length}", json.dumps({
        "seeds": [42, 43], "shape": list(probs[0].shape), "dtype": str(dtype),
        "input": "logits" if logits else "probabilities", "upstream": upstream,
        "loss": comparison_summary({"loss": expected[0]}, {"loss": actual[0]})["loss"],
        "stats": comparison_summary(expected[1], actual[1]),
        "gradients": comparison_summary(dict(zip(("r", "rho"), expected_grad)), dict(zip(("r", "rho"), actual_grad))),
    }, sort_keys=True))
    assert "FusedDenseE4MILossBackward" in type(actual[0].grad_fn).__name__
    assert actual[0].dtype == torch.float32 and torch.isfinite(actual[0])
    assert all(torch.isfinite(value).all() for value in actual[1].values())
    # Keep the existing MI loss/stat and per-leaf VJP thresholds unchanged.
    rtol, atol = (2e-5, 1e-5) if dtype == torch.float32 else (2e-2, 2e-3)
    check_mi_result(actual, expected, rtol=rtol, atol=atol)
    for name, reference, observed in zip(("r", "rho"), expected_grad, actual_grad, strict=True):
        assert_swila_grad(name, reference, observed, require_relative=True, slice_dims=(0, 1, 2, 3))
        if length == 1:
            assert torch.count_nonzero(observed) == 0
        else:
            assert torch.count_nonzero(observed) > 0
    if not logits:
        for name, value, observed in zip(("r", "rho"), probs, actual_grad, strict=True):
            assert_swila_grad(
                f"analytic-{name}", dense_mi_gradient(value, .73, upstream), observed,
                require_relative=True, slice_dims=(0, 1, 2, 3),
            )
    # At T=1 the separate entropy reductions can leave a tiny forward residue;
    # the baseline does too. The existing loss/stat limits apply, while VJPs are exactly zero.


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16), ids=("fp32", "bf16"))
def test_swila_mi_runtime_probability_vjp(dtype, record_property):
    for length in LENGTHS:
        check_mi_runtime(length, dtype, logits=False, record_property=record_property)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16), ids=("fp32", "bf16"))
def test_swila_mi_runtime_logit_vjp(dtype, record_property):
    for length in LENGTHS:
        check_mi_runtime(length, dtype, logits=True, record_property=record_property)


class MILaunchObserver:
    """Forward every actual launch and retain the returned compiled-kernel object."""

    def __init__(self, kernel):
        self.kernel, self.calls = kernel, []

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def observe(*args, **kwargs):
            compiled = launch(*args, **kwargs)
            assert compiled is not None and isinstance(compiled.hash, str)
            arguments = dict(zip(self.kernel.arg_names, args))
            arguments.update(kwargs)
            self.calls.append((arguments, compiled))
            return compiled

        return observe


def check_mi_runtime_ir(compiled, runtime_names, *, address_names):
    ir = compiled.asm["ttir"]
    assert not re.search(r"(?:\bf64\b|xf64\b)", ir), "MI arithmetic must remain FP32"
    signature = compiled.src.signature
    names = [name for name, kind in signature.items() if kind != "constexpr"]
    header = next(line for line in ir.splitlines() if "tt.func public" in line)
    arguments = re.findall(r"(%\w+):", header)
    for name in runtime_names:
        assert signature[name] in ("i32", "i64"), f"{name} must remain a runtime integer"
        if name in address_names and signature[name] == "i32":
            value = re.escape(arguments[names.index(name)])
            assert re.search(rf"arith\.extsi\s+{value}\s*:\s*i32 to i64", ir), f"{name} must widen before addressing"
    additions = [line for line in ir.splitlines() if "tt.addptr" in line]
    assert additions, ir
    # LLVM may keep fixed statistics-channel offsets narrow; dynamic offsets must remain int64.
    fixed_channels = set(re.findall(r"^\s*(%\w+)\s*= arith\.constant [123] : i32\b", ir, flags=re.MULTILINE))
    for line in additions:
        if re.search(r"(?:\bi64\b|xi64\b)", line):
            continue
        offset = re.search(r"tt\.addptr %\w+, (%\w+) : !tt\.ptr<[^>]+>, i32\b", line)
        assert offset is not None and offset.group(1) in fixed_channels, line


def test_swila_mi_runtime_compile_reuse(monkeypatch, record_property):
    observers = {}
    for stage, name in (
        ("partials", "_mi_fwd_partials_e4_kernel"),
        ("finalize", "_mi_fwd_finalize_e4_kernel"),
        ("backward", "_mi_prob_bwd_e4_kernel"),
    ):
        observers[stage] = MILaunchObserver(getattr(mi_loss, name))
        monkeypatch.setattr(mi_loss, name, observers[stage])
    # NT1/BNT2, NT2/BNT4, then NT3/4 sharing BNT8. Weight changes exercise runtime SCALE independently of T.
    cases = [(length, .73) for length in LENGTHS] + [(193, -.41), (193, 0.)]
    launches = []
    for length, weight in cases:
        inputs = mi_inputs(length, torch.float32, logits=False)
        actual = mi_loss.fused_swila_mi_loss_from_probs(*inputs, mi_loss_weight=weight)
        loss = actual[0]
        upstream = 1.37 * math.prod(inputs[0].shape[:-1])
        gradients = torch.autograd.grad(loss, inputs, loss.new_tensor(upstream))
        assert torch.isfinite(loss) and all(torch.isfinite(value).all() for value in gradients)
        if weight != .73:
            check_mi_result(actual, mi_reference(*inputs, weight), rtol=2e-5, atol=1e-5)
            for name, value, observed in zip(("r", "rho"), inputs, gradients, strict=True):
                assert_swila_grad(
                    f"weight-{weight}-{name}", dense_mi_gradient(value, weight, upstream), observed,
                    require_relative=True, slice_dims=(0, 1, 2, 3),
                )
        if weight == 0:
            assert loss.item() == 0 and all(torch.count_nonzero(value) == 0 for value in gradients)
        launches.append({"T": length, "weight": weight, "kernels": {
            stage: {"object_id": id(observer.calls[-1][1]), "hash": observer.calls[-1][1].hash}
            for stage, observer in observers.items()
        }})
    torch.cuda.synchronize()
    record_property("actual_mi_compiled_launches", json.dumps(launches, sort_keys=True))
    for stage, observer in observers.items():
        assert len(observer.calls) == len(cases) * (2 if stage == "backward" else 1)
        if stage != "finalize":
            assert all(compiled is observer.calls[0][1] for _, compiled in observer.calls), stage
        else:
            groups = {}
            for arguments, compiled in observer.calls:
                bnt = arguments["BNT"]
                assert compiled is groups.setdefault(bnt, compiled), "finalizer recompiled within the same BNT"
            assert set(groups) == {2, 4, 8}
            assert len({compiled.hash for compiled in groups.values()}) == 3, "BNT remains compile-time metadata"
        for compiled in {id(value): value for _, value in observer.calls}.values():
            runtime_names = ("T",) if stage == "backward" else ("T", "NT")
            check_mi_runtime_ir(
                compiled, runtime_names, address_names=("NT",) if stage == "finalize" else runtime_names,
            )
            if stage == "backward":
                assert compiled.src.signature["SCALE"] == "fp32"
