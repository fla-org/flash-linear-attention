# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Autograd context, mixed precision, and address-width integration contracts."""

import ast
import inspect
import math
import re
from contextlib import contextmanager
from importlib import import_module

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from fla.ops.swila import alpha_gate, mi_loss
from fla.ops.swila.checkpoint_context import swila_checkpoint_context_fn
from fla.ops.swila.temporal_math import mix_exact_initial_temporal_prior
from fla.utils import IS_NVIDIA, decorators, device
from tests.ops.swila_helpers import (
    FAMILIES,
    assert_swila_close,
    assert_swila_grad,
    check_mi_result,
    clone_inputs,
    compare_gradients,
    compare_outputs,
    connected_inputs,
    cotangents,
    dense_mi_gradient,
    gradients,
    make_inputs,
    mi_reference,
    objective,
    probabilities,
    run_fused,
    run_reference,
)

requires_cuda = pytest.mark.skipif(not IS_NVIDIA, reason="SwiLA Triton kernels require NVIDIA CUDA")


@pytest.mark.parametrize(("target", "frozen", "forward_amp"), [
    pytest.param(device, False, True, marks=requires_cuda, id="cuda-all-grad-autocast-forward"),
    pytest.param(device, True, False, marks=requires_cuda, id="cuda-frozen-autocast-backward"),
    pytest.param("cpu", True, True, id="cpu-frozen-autocast-forward"),
])
def test_swila_alpha_partial_tiles(target, frozen, forward_amp):
    torch.manual_seed(42)
    rows = torch.linspace(-.25, .25, 65, device=target)[:, None]
    columns = torch.linspace(-40, 8, 130, device=target)[None, :]
    raw = (rows + columns)[:, ::2].detach().requires_grad_()
    scale = torch.linspace(.25, .75, 130, device=target)[::2].requires_grad_(not frozen)
    bias = torch.linspace(-.125, .125, 130, device=target)[::2].requires_grad_()
    inputs = (raw, scale, bias)
    assert all(not value.is_contiguous() for value in inputs)
    upstream = torch.linspace(.5, 1.5, raw.numel(), device=target).reshape_as(raw)
    z = raw.detach().double() + bias.detach().double()
    softplus = torch.logaddexp(torch.zeros_like(z), z)
    expected = -scale.detach().double() * softplus
    d_raw = -upstream.double() * scale.detach().double() * z.sigmoid()
    expected_grad = (d_raw, (-upstream.double() * softplus).sum(0), d_raw.sum(0))
    with torch.autocast(device_type=raw.device.type, dtype=torch.bfloat16, enabled=forward_amp):
        actual = alpha_gate.fused_swila_alpha_gate(raw, scale, bias)
    if raw.device.type == "cpu":
        assert actual.grad_fn.saved_tensors[0].stride() == raw.stride()
    assert actual.dtype == torch.float32 and actual.device == raw.device
    assert_swila_close("alpha-partial", expected.float(), actual)
    torch.testing.assert_close(actual.double(), expected, rtol=5e-6, atol=0)
    active = tuple(value for value in inputs if value.requires_grad)
    with torch.autocast(device_type=raw.device.type, dtype=torch.bfloat16, enabled=not forward_amp):
        actual_grad = torch.autograd.grad(actual, active, upstream)
    expected_grad = tuple(gradient for gradient, value in zip(expected_grad, inputs, strict=True) if value.requires_grad)
    for reference, observed, value in zip(expected_grad, actual_grad, active, strict=True):
        assert observed.dtype == value.dtype and observed.device == value.device
        assert torch.isfinite(observed).all() and torch.count_nonzero(observed) == observed.numel()
        assert_swila_grad("alpha-partial-gradient", reference.float(), observed)
        torch.testing.assert_close(observed.double(), reference, rtol=5e-6, atol=0)
    if frozen:
        assert scale.grad is None


@pytest.mark.parametrize(("target", "noncontiguous", "forward_amp"), [
    pytest.param(device, False, True, marks=requires_cuda, id="cuda-fused-partial-autocast-forward"),
    pytest.param(device, True, False, marks=requires_cuda, id="cuda-eager-noncontiguous-autocast-backward"),
    pytest.param("cpu", True, True, id="cpu-eager-noncontiguous-autocast-forward"),
])
def test_swila_mi_partial_mixed_precision(target, noncontiguous, forward_amp):
    torch.manual_seed(42)
    shape = (1, 65, 1, 130 if noncontiguous else 65, 4)
    r = probabilities(shape, target, torch.float32, 42)
    rho = probabilities(shape, target, torch.bfloat16, 43)
    if noncontiguous:
        r, rho = (value[:, :, :, ::2].detach().requires_grad_() for value in (r, rho))
    fused = r.is_cuda and not noncontiguous
    expected = mi_reference(r.float(), rho.float(), .73) if fused else mi_reference(r, rho, .73)
    with torch.autocast(device_type=r.device.type, dtype=torch.bfloat16, enabled=forward_amp):
        actual = mi_loss.fused_swila_mi_loss_from_probs(r, rho, mi_loss_weight=.73)
    assert actual[0].dtype == torch.float32
    assert mi_loss.can_use_fused_swila_mi_loss_from_probs(r, rho) is fused
    assert ("FusedDenseE4MILossBackward" in type(actual[0].grad_fn).__name__) is fused
    check_mi_result(actual, expected, rtol=2e-2, atol=2e-3)
    upstream = 1.37 * math.prod(r.shape[:-1])
    expected_grad = tuple(dense_mi_gradient(value, .73, upstream) for value in (r, rho)) if fused else (
        torch.autograd.grad(expected[0], (r, rho), expected[0].new_tensor(upstream))
    )
    with torch.autocast(device_type=r.device.type, dtype=torch.bfloat16, enabled=not forward_amp):
        actual_grad = torch.autograd.grad(actual[0], (r, rho), actual[0].new_tensor(upstream))
    for name, value, reference, observed in zip(("r", "rho"), (r, rho), expected_grad, actual_grad, strict=True):
        assert observed.dtype == value.dtype and observed.device == value.device
        assert torch.isfinite(observed).all() and torch.count_nonzero(observed) > 0
        assert_swila_grad(name, reference, observed,
                          require_relative=True, slice_dims=(0, 1, 2, 3))


def check_recurrence_autocast(family, forward_amp, monkeypatch):
    torch.manual_seed(42)
    data = make_inputs(family, dtype=torch.bfloat16, device=device, keys=8, values=4, experts=4)
    for name in ("initial_state", "initial_r", "initial_rho"):
        if name in data:
            data[name] = data[name].float()
    ref_inputs, control_inputs, actual_inputs = (clone_inputs(data) for _ in range(3))
    with torch.autocast(device_type="cuda", enabled=False):
        # shared FP32 casts combine main and oracle-auxiliary VJPs before BF16 rounding.
        reference_inputs = {name: value.float() for name, value in ref_inputs.items()} if "temporal" in family else ref_inputs
        ref = run_reference(reference_inputs, family)
        ref["o"] = ref["o"].to(data["q"].dtype)
        control = run_fused(control_inputs, family)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=forward_amp):
        if "temporal" not in family:
            ref["rho"] = torch.softmax(ref_inputs["pi_q"], dim=-1)
            control["rho"] = torch.softmax(control_inputs["pi_q"], dim=-1)
        actual = run_fused(actual_inputs, family)
    compare_outputs(ref, actual)
    compare_outputs(control, actual, prefix="explicit-dtype/")
    for name, output in actual.items():
        expected_dtype = data["q"].dtype if name == "o" else torch.float32
        if name == "rho" and "temporal" not in family:
            expected_dtype = torch.float32 if forward_amp else data["pi_q"].dtype
        assert output.dtype == expected_dtype and output.device == data["q"].device
    weights = cotangents(ref)
    auxiliary = ("r", "rho")
    connected = connected_inputs(data, "r") | connected_inputs(data, "rho")
    with torch.autocast(device_type="cuda", enabled=False):
        reference_grad = gradients(objective(ref, weights), ref_inputs)
        control_grad = gradients(objective(control, weights), control_inputs, retain_graph=True)
        control_aux_grad = gradients(objective(control, weights, auxiliary), control_inputs)
    module_name, backward_name = {
        "plain": ("fused_recurrent", "fused_recurrent_swila_bwd"),
        "gated": ("fused_recurrent_gated", "fused_recurrent_gated_swila_bwd"),
        "temporal": ("fused_recurrent_temporal", "fused_recurrent_swila_temporal_bwd"),
        "temporal_gated": ("fused_recurrent_temporal", "fused_recurrent_swila_temporal_bwd"),
    }[family]
    module = import_module(f"fla.ops.swila.{module_name}")
    original_backward = getattr(module, backward_name)
    backward_states = []

    def observe_backward(*args, **kwargs):
        backward_states.append(torch.is_autocast_enabled("cuda"))
        return original_backward(*args, **kwargs)

    monkeypatch.setattr(module, backward_name, observe_backward)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=not forward_amp):
        actual_grad = gradients(objective(actual, weights), actual_inputs, retain_graph=True)
        actual_aux_grad = gradients(objective(actual, weights, auxiliary), actual_inputs)
    assert backward_states == [forward_amp, forward_amp]
    compare_gradients(reference_grad, actual_grad)
    compare_gradients(control_grad, actual_grad, prefix="explicit-dtype/")
    compare_gradients(control_aux_grad, actual_aux_grad, connected=connected, prefix="auxiliary-only/")
    for name, value in actual_inputs.items():
        for gradient in (actual_grad[name], actual_aux_grad[name]):
            if gradient is not None:
                assert gradient.dtype == value.dtype and gradient.device == value.device


@requires_cuda
@pytest.mark.parametrize("family", FAMILIES, ids=FAMILIES)
def test_swila_recurrence_autocast_forward(family, monkeypatch):
    check_recurrence_autocast(family, forward_amp=True, monkeypatch=monkeypatch)


@requires_cuda
@pytest.mark.parametrize("family", FAMILIES, ids=FAMILIES)
def test_swila_recurrence_autocast_backward(family, monkeypatch):
    check_recurrence_autocast(family, forward_amp=False, monkeypatch=monkeypatch)


@requires_cuda
@pytest.mark.parametrize("family", ("gated", "temporal_gated"), ids=("gated", "temporal-gated"))
def test_swila_recurrence_autocast_checkpoint(family):
    torch.manual_seed(42)
    data = make_inputs(family, dtype=torch.bfloat16, device=device, keys=8, values=4, experts=4)
    control_inputs, actual_inputs = clone_inputs(data), clone_inputs(data)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        control = run_fused(control_inputs, family)
    weights = cotangents(control)

    def loss_fn(*values):
        outputs = run_fused(dict(zip(actual_inputs, values, strict=True)), family)
        return objective(outputs, weights, ("o", "r", "rho"))

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actual = checkpoint(loss_fn, *actual_inputs.values(), use_reentrant=False, context_fn=swila_checkpoint_context_fn)
    expected = objective(control, weights, ("o", "r", "rho"))
    assert_swila_close("autocast-checkpoint", expected, actual)
    with torch.autocast(device_type="cuda", enabled=False):
        compare_gradients(gradients(expected, control_inputs), gradients(actual, actual_inputs))


@pytest.mark.parametrize(("forward_amp", "log_only"), [
    pytest.param(True, False, id="autocast-forward-both-outputs"),
    pytest.param(False, True, id="autocast-backward-log-only"),
])
def test_swila_initial_prior_cpu_autocast(forward_amp, log_only):
    torch.manual_seed(42)
    initial = torch.tensor([[0., .2, .8], [.3, 0., .7]], dtype=torch.float64)
    gate = torch.tensor([[-4.], [2.]], dtype=torch.float64)
    router = torch.randn(2, 3, dtype=torch.float64)
    control_inputs = tuple(value.detach().clone().requires_grad_() for value in (initial, gate, router))
    actual_inputs = tuple(value.detach().clone().requires_grad_() for value in (initial, gate, router))
    probability = (1 - control_inputs[1].sigmoid()) * control_inputs[0]
    probability = probability + control_inputs[1].sigmoid() * control_inputs[2].softmax(-1)
    reference = (probability, probability.log())
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=forward_amp):
        actual = mix_exact_initial_temporal_prior(*actual_inputs)
    weights = tuple(torch.randn_like(value) for value in reference)
    for expected, observed in zip(reference, actual, strict=True):
        assert observed.dtype == torch.float64 and observed.device.type == "cpu"
        torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)
    chosen = (1,) if log_only else (0, 1)
    expected_loss = sum((reference[index] * weights[index]).sum() for index in chosen)
    actual_loss = sum((actual[index] * weights[index]).sum() for index in chosen)
    expected_grad = torch.autograd.grad(expected_loss, control_inputs)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=not forward_amp):
        actual_grad = torch.autograd.grad(actual_loss, actual_inputs)
    for expected, observed in zip(expected_grad, actual_grad, strict=True):
        assert observed.dtype == torch.float64 and observed.device.type == "cpu"
        torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)


@requires_cuda
@pytest.mark.parametrize("helper", ("alpha", "mi"), ids=("alpha", "mi"))
def test_swila_helper_device_context(monkeypatch, helper):
    torch.manual_seed(42)
    entries, exits, active = [], [], []
    original_context = decorators.custom_device_ctx

    @contextmanager
    def record_context(index):
        entries.append(index)
        active.append(index)
        try:
            with original_context(index):
                yield
        finally:
            exits.append(active.pop())

    monkeypatch.setattr(decorators, "custom_device_ctx", record_context)
    if helper == "alpha":
        inputs = tuple(torch.ones(shape, device=device, requires_grad=True) for shape in ((3, 5), (5,), (5,)))

        def invoke():
            return alpha_gate.fused_swila_alpha_gate(*inputs)
        function = alpha_gate._FusedAlphaGateFunction
    else:
        inputs = tuple(probabilities((1, 3, 1, 5, 4), device, torch.float32, seed) for seed in (42, 43))

        def invoke():
            return mi_loss.fused_swila_mi_loss_from_probs(*inputs, mi_loss_weight=.73)[0]
        function = mi_loss._FusedDenseE4MILoss
    output = invoke()
    assert entries == exits == [inputs[0].device.index] and not active
    absent = (None,) if helper == "alpha" else (None,) * 4
    missing_gradients = function.backward(output.grad_fn, *absent)
    assert all(value is None or torch.count_nonzero(value) == 0 for value in missing_gradients)
    assert entries == exits == [inputs[0].device.index] and not active
    torch.autograd.grad(output.sum(), inputs)
    assert entries == exits == [inputs[0].device.index] * 2 and not active

    def fail(*args, **kwargs):
        assert active == [inputs[0].device.index]
        raise RuntimeError("injected launch failure")

    if helper == "alpha":
        class FailedKernel:
            def __getitem__(self, grid):
                return fail

        monkeypatch.setattr(alpha_gate, "_alpha_gate_fwd_kernel", FailedKernel())
    else:
        monkeypatch.setattr(mi_loss, "_triton_dense_e4_forward", fail)
    with pytest.raises(RuntimeError, match="injected launch failure"):
        invoke()
    assert entries == exits == [inputs[0].device.index] * 3 and not active


@pytest.mark.parametrize("module", [
    pytest.param(alpha_gate, id="alpha"),
    pytest.param(mi_loss, id="mi"),
    pytest.param("nongated_fwd", id="nongated_fwd"),
    pytest.param("nongated_bwd", id="nongated_bwd"),
    pytest.param("gated_fwd", id="gated_fwd"),
    pytest.param("gated_bwd", id="gated_bwd"),
    pytest.param("gated_split_bwd", id="gated_split_bwd"),
    pytest.param("temporal_fwd", id="temporal_fwd"),
    pytest.param("temporal_bwd_state", id="temporal_bwd_state"),
    pytest.param("temporal_split_bwd", id="temporal_split_bwd"),
    pytest.param("temporal_bwd", id="temporal_bwd"),
])
def test_swila_address_sources_widen_before_arithmetic(module):
    if isinstance(module, str):
        module = import_module(f"fla.ops.swila.{module}")
    tree = ast.parse(inspect.getsource(module))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    indices = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
               and node.func.value.id == "tl" and node.func.attr in ("program_id", "arange")]
    assert indices
    for index in indices:
        conversion = parents[index]
        assert isinstance(conversion, ast.Attribute) and conversion.attr == "to", ast.unparse(index)
        call = parents[conversion]
        assert isinstance(call, ast.Call) and len(call.args) == 1
        assert ast.unparse(call.args[0]) == "tl.int64", ast.unparse(call)


def check_address_ir(compiled, tmp_path, name):
    ir = compiled.asm["ttir"]
    (tmp_path / f"{name}.ttir").write_text(ir)
    pointer_additions = [line for line in ir.splitlines() if "tt.addptr" in line]
    assert pointer_additions, ir
    assert any(re.search(r"(?:\bi64\b|xi64\b)", line) for line in pointer_additions), ir
    assert re.search(r"arith\.(?:muli|shli).*?(?:\bi64\b|xi64\b)", ir), ir


@requires_cuda
@pytest.mark.parametrize("backward", (False, True), ids=("forward", "backward"))
def test_swila_alpha_large_address_ir(backward, tmp_path):
    buffer = torch.empty(1, device=device, dtype=torch.float32)
    kernel = alpha_gate._alpha_gate_bwd_kernel if backward else alpha_gate._alpha_gate_fwd_kernel
    count = 7 if backward else 4
    compiled = kernel.warmup(*([buffer] * count), M=2**31 + 65, D=2**29 + 1,
                             BLOCK_M=64, BLOCK_D=64, grid=(2, 2))
    check_address_ir(compiled, tmp_path, "alpha_backward" if backward else "alpha_forward")


@requires_cuda
@pytest.mark.parametrize("phase", ("partials", "finalize", "reduce", "backward"),
                         ids=("partials", "finalize", "reduce", "backward"))
def test_swila_mi_large_address_ir(phase, tmp_path):
    buffer = torch.empty(1, device=device, dtype=torch.float32)
    n = 2**29 + 1
    cases = {
        "partials": (mi_loss._mi_fwd_partials_e4_kernel, 6, dict(T=65, N=n, NT=2, BT=64, BR=1)),
        "finalize": (mi_loss._mi_fwd_finalize_e4_kernel, 7, dict(T=65, N=n, NT=2, BNT=4, BR=1)),
        "reduce": (mi_loss._mi_reduce_stats_kernel, 2, dict(ROWS=2**31 + 1, BLOCK_ROWS=256)),
        "backward": (mi_loss._mi_prob_bwd_e4_kernel, 4, dict(T=65, N=n, BT=64, BR=2, BRE=8, SCALE=1.)),
    }
    kernel, count, constants = cases[phase]
    compiled = kernel.warmup(*([buffer] * count), **constants, grid=(2, 2, 2))
    check_address_ir(compiled, tmp_path, f"mi_{phase}")


@requires_cuda
@pytest.mark.parametrize(("family", "module_name", "forward_name"), [
    pytest.param("plain", "fused_recurrent", "fused_recurrent_swila_fwd", id="plain"),
    pytest.param("gated", "fused_recurrent_gated", "fused_recurrent_gated_swila_fwd", id="gated"),
    pytest.param("temporal", "fused_recurrent_temporal", "fused_recurrent_swila_temporal_fwd", id="temporal"),
])
def test_swila_recurrence_rejects_unsupported_backend(monkeypatch, family, module_name, forward_name):
    module = import_module(f"fla.ops.swila.{module_name}")
    monkeypatch.setattr(module, "IS_NVIDIA", False)

    def fail(*args, **kwargs):
        pytest.fail("unsupported backend reached the kernel launcher")

    monkeypatch.setattr(module, forward_name, fail)
    with pytest.raises(NotImplementedError, match="NVIDIA CUDA"):
        run_fused(make_inputs(family, device=device), family)


@requires_cuda
def test_swila_helpers_unsupported_backend_eager(monkeypatch):
    torch.manual_seed(42)
    monkeypatch.setattr(torch.version, "hip", "mock-unsupported-backend")

    class FailedKernel:
        def __getitem__(self, grid):
            pytest.fail("unsupported backend reached the alpha kernel")

    def fail(*args, **kwargs):
        pytest.fail("unsupported backend reached the MI kernel")

    monkeypatch.setattr(alpha_gate, "_alpha_gate_fwd_kernel", FailedKernel())
    monkeypatch.setattr(mi_loss, "_triton_dense_e4_forward", fail)
    raw = torch.linspace(-40, 8, 15, device=device).reshape(3, 5).requires_grad_()
    scale = torch.ones(5, device=device, requires_grad=True)
    bias = torch.zeros(5, device=device, requires_grad=True)
    actual = alpha_gate.fused_swila_alpha_gate(raw, scale, bias)
    expected = -scale * torch.nn.functional.softplus(raw + bias)
    assert_swila_close("unsupported-eager-alpha", expected, actual)
    for reference, observed in zip(torch.autograd.grad(expected.sum(), (raw, scale, bias)),
                                   torch.autograd.grad(actual.sum(), (raw, scale, bias)), strict=True):
        assert_swila_grad("unsupported-eager-alpha", reference, observed)
    r, rho = (probabilities((1, 5, 1, 3, 4), device, torch.float32, seed) for seed in (42, 43))
    assert not mi_loss.can_use_fused_swila_mi_loss_from_probs(r, rho)
    expected_mi = mi_reference(r, rho, .73)
    actual_mi = mi_loss.fused_swila_mi_loss_from_probs(r, rho, mi_loss_weight=.73)
    check_mi_result(actual_mi, expected_mi, rtol=2e-5, atol=1e-5)
    for reference, observed in zip(torch.autograd.grad(expected_mi[0], (r, rho)),
                                   torch.autograd.grad(actual_mi[0], (r, rho)), strict=True):
        assert_swila_grad("unsupported-eager-mi", reference, observed, require_relative=True)
