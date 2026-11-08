# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from fla.ops.swila import compute_swila_mi_loss, fused_swila_log_decay_gate
from fla.ops.swila.alpha_gate import fused_swila_alpha_gate
from fla.ops.swila.checkpoint_context import get_swila_checkpoint_phase, swila_checkpoint_context_fn
from fla.ops.swila.mi_loss import can_use_fused_swila_mi_loss_from_probs, fused_swila_mi_loss_from_probs
from fla.utils import IS_NVIDIA, device
from tests.ops.swila_helpers import (
    assert_swila_close,
    assert_swila_grad,
    check_mi_result,
    dense_mi_gradient,
    mi_reference,
    probabilities,
)

requires_cuda = pytest.mark.skipif(not IS_NVIDIA, reason="SwiLA Triton kernels require NVIDIA CUDA")
GATE_CASES = [(False, False), (False, True), (True, False), (True, True)]


def gate_inputs(target, normalized):
    z = torch.tensor([-40., -20., -16., -8., -1., 0., 1., 8., 16., 20., 40.], device=target)
    offsets = torch.tensor([0., .25, -.25, .125, -.125, .5, -.5], device=target)
    bias = torch.tensor([-.25, .25], device=target)[:, None].expand(2, 11).clone()
    raw = z[None, None, None, :] + offsets[None, :, None, None] - bias
    leaves = tuple(value.requires_grad_() for value in (
        raw.clone(), torch.tensor([-.75, -1.25], device=target), bias,
    ))
    rows = torch.tensor([1., -.25, .5, 2., -.125, .75, 1.25], device=target)
    columns = torch.linspace(.5, 1.5, 22, device=target).reshape(2, 11)
    cotangent = rows[None, :, None, None] * columns
    if normalized:
        stored_z = raw.double() + bias.double()
        softplus = torch.logaddexp(torch.zeros_like(stored_z), stored_z)
        cotangent = (cotangent.double() / softplus).float()
    assert torch.isfinite(cotangent).all()
    return leaves, cotangent


def gate_reference(inputs, cotangent):
    raw, a_log, bias = (value.detach().double() for value in inputs)
    z = raw + bias
    scale = a_log.exp()[None, None, :, None]
    softplus = torch.logaddexp(torch.zeros_like(z), z)
    d_raw = -cotangent.double() * scale * z.sigmoid()
    d_a_log = (-cotangent.double() * scale * softplus).sum((0, 1, 3))
    return -scale * softplus, (d_raw, d_a_log, d_raw.sum((0, 1)))


def check_gate(target, low_level, normalized):
    inputs, cotangent = gate_inputs(target, normalized)
    expected, expected_grads = gate_reference(inputs, cotangent)
    raw, a_log, bias = inputs
    if low_level:
        scale = a_log.exp()[:, None].expand_as(bias).reshape(-1)
        actual = fused_swila_alpha_gate(raw.flatten(-2), scale, bias.reshape(-1)).reshape_as(raw)
    else:
        actual = fused_swila_log_decay_gate(raw, a_log, bias)
    gradients = torch.autograd.grad(actual, inputs, cotangent)
    assert actual.dtype == torch.float32 and actual.device == raw.device
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual) == actual.numel()
    assert_swila_close("gate", expected.float(), actual)
    # A zero absolute tolerance prevents the tiny negative tail from disappearing.
    torch.testing.assert_close(actual.double(), expected, rtol=5e-6, atol=0)
    for name, ref, observed, leaf in zip(
        ("raw", "A_log", "dt_bias"), expected_grads, gradients, inputs, strict=True,
    ):
        assert observed.dtype == leaf.dtype and observed.device == leaf.device
        assert torch.isfinite(observed).all() and torch.count_nonzero(observed) == observed.numel()
        assert_swila_grad(name, ref.float(), observed, connected=True)
        torch.testing.assert_close(observed.double(), ref, rtol=5e-6, atol=0)


@pytest.mark.parametrize(("low_level", "normalized"), GATE_CASES)
def test_swila_gate_tail_cpu(low_level, normalized):
    check_gate("cpu", low_level, normalized)


@requires_cuda
@pytest.mark.parametrize(("low_level", "normalized"), GATE_CASES)
def test_swila_gate_tail_cuda(low_level, normalized):
    check_gate(device, low_level, normalized)


@pytest.mark.parametrize(("experts", "boundaries"), [
    (1, None), (4, None), (4, (0, 1, 3, 7)), (3, (0, 1, 1, 7)),
])
def test_swila_mi_eager_fp64(experts, boundaries):
    shape = (2 if boundaries is None else 1, 7, 3, 5, experts)
    r, rho = (probabilities(shape, "cpu", torch.float64, seed) for seed in (101, 202))
    ref_r, ref_rho = (value.detach().clone().requires_grad_() for value in (r, rho))
    cu_seqlens = None if boundaries is None else torch.tensor(boundaries, dtype=torch.int32)
    expected = mi_reference(ref_r, ref_rho, .37, boundaries)
    actual = compute_swila_mi_loss(r, rho, experts, .37, True, cu_seqlens, return_stats=True)
    check_mi_result(actual, expected, rtol=1e-12, atol=1e-12)
    gradients = torch.autograd.grad(actual[0] * 1.73, (r, rho))
    ref_gradients = torch.autograd.grad(expected[0] * 1.73, (ref_r, ref_rho))
    for observed, reference in zip(gradients, ref_gradients, strict=True):
        torch.testing.assert_close(observed, reference, rtol=1e-12, atol=1e-12)
        if experts == 1:
            assert torch.count_nonzero(observed) == 0
    if boundaries is None:
        assert not can_use_fused_swila_mi_loss_from_probs(r, rho)
        check_mi_result(fused_swila_mi_loss_from_probs(r, rho, mi_loss_weight=.37), expected, rtol=1e-12, atol=1e-12)


def test_swila_mi_gradient_formula_fp64():
    r, rho = (probabilities((2, 7, 3, 5, 4), "cpu", torch.float64, seed) for seed in (9091, 9092))
    loss, _ = mi_reference(r, rho, .37)
    gradients = torch.autograd.grad(loss * -1.73, (r, rho))
    for prob, gradient in zip((r, rho), gradients, strict=True):
        torch.testing.assert_close(gradient, dense_mi_gradient(prob, .37, -1.73), rtol=1e-12, atol=1e-12)


@requires_cuda
@pytest.mark.parametrize(("shape", "r_dtype", "rho_dtype", "seeds", "atol", "rtol"), [
    ((2, 17, 3, 5, 4), torch.float32, torch.float32, (505, 606), 1e-5, 2e-5),
    ((1, 65, 2, 64, 4), torch.float16, torch.float16, (505, 606), 5e-4, 5e-3),
    ((2, 17, 3, 5, 4), torch.bfloat16, torch.bfloat16, (505, 606), 2e-3, 2e-2),
    ((1, 17, 2, 64, 4), torch.float32, torch.bfloat16, (909, 1001), 2e-3, 2e-2),
])
def test_swila_mi_fused_cuda(shape, r_dtype, rho_dtype, seeds, atol, rtol):
    r = probabilities(shape, device, r_dtype, seeds[0])
    rho = probabilities(shape, device, rho_dtype, seeds[1])
    assert can_use_fused_swila_mi_loss_from_probs(r, rho)
    expected = mi_reference(r, rho, .73)
    actual = compute_swila_mi_loss(r, rho, 4, .73, True, return_stats=True)
    assert "FusedDenseE4MILossBackward" in type(actual[0].grad_fn).__name__
    check_mi_result(actual, expected, rtol=rtol, atol=atol)
    # Remove the averaging scale so incorrect zero gradients cannot pass an absolute threshold.
    upstream = 1.37 * math.prod(shape[:-1])
    gradients = torch.autograd.grad(actual[0], (r, rho), actual[0].new_tensor(upstream))
    for name, prob, gradient in zip(("dr", "drho"), (r, rho), gradients, strict=True):
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
        assert_swila_grad(
            name, dense_mi_gradient(prob, .73, upstream), gradient,
            connected=True, require_relative=True, slice_dims=(0, 1, 2, 3),
        )


@requires_cuda
def test_swila_mi_frozen_input_cuda():
    r = probabilities((1, 9, 1, 7, 4), device, torch.float32, 707).detach()
    rho = probabilities((1, 9, 1, 7, 4), device, torch.float32, 808)
    loss, _ = fused_swila_mi_loss_from_probs(r, rho, mi_loss_weight=.41)
    loss.backward()
    assert r.grad is None and rho.grad is not None
    assert_swila_grad("drho_frozen_r", dense_mi_gradient(rho, .41, 1), rho.grad, connected=True, require_relative=True)


def test_swila_checkpoint_context_cleanup():
    assert get_swila_checkpoint_phase() == "normal"
    prepass, recompute = swila_checkpoint_context_fn()
    with prepass:
        assert get_swila_checkpoint_phase() == "prepass"
        with pytest.raises(RuntimeError, match="recompute failure"), recompute:
            assert get_swila_checkpoint_phase() == "recompute"
            raise RuntimeError("recompute failure")
        assert get_swila_checkpoint_phase() == "prepass"
    assert get_swila_checkpoint_phase() == "normal"
    with pytest.raises(RuntimeError, match="prepass failure"), swila_checkpoint_context_fn()[0]:
        raise RuntimeError("prepass failure")
    assert get_swila_checkpoint_phase() == "normal"


def test_swila_checkpoint_context_nonreentrant():
    phases = []

    def probe(x):
        phases.append(get_swila_checkpoint_phase())
        return x.sin() * x

    torch.manual_seed(42)
    x = torch.randn(8, dtype=torch.float64, requires_grad=True)
    checkpoint(probe, x, use_reentrant=False, context_fn=swila_checkpoint_context_fn).sum().backward()
    assert phases == ["prepass", "recompute"]
    assert get_swila_checkpoint_phase() == "normal"
    torch.testing.assert_close(x.grad, x.detach().cos() * x.detach() + x.detach().sin(), rtol=1e-12, atol=1e-12)


@requires_cuda
@pytest.mark.parametrize("temporal", [False, True])
def test_swila_operator_checkpoint_cuda(temporal):
    from fla.ops.swila import fused_recurrent_gated_swila, fused_recurrent_swila_temporal

    torch.manual_seed(8149 + int(temporal))
    shape, dtype = (1, 5, 1), torch.bfloat16
    values = {
        "q": F.normalize(torch.randn(*shape, 64, device=device, dtype=dtype) * .5, dim=-1),
        "k": F.normalize(torch.randn(*shape, 64, device=device, dtype=dtype) * .5, dim=-1),
        "v": torch.randn(*shape, 64, device=device, dtype=dtype) * .5,
        "beta": torch.randn(*shape, 64, 4, device=device, dtype=dtype).sigmoid(),
        "pi_q": torch.randn(*shape, 64, 4, device=device, dtype=dtype) * .5,
        "pi_k": torch.randn(*shape, 64, 4, device=device, dtype=dtype) * .5,
    }
    gate_kv = torch.randn(*shape, 64, device=device, dtype=dtype)
    gate_q = torch.randn(*shape, 64, device=device, dtype=dtype)
    values["log_decay"] = -(.02 + .3 * torch.rand(*shape, 64, device=device))
    values["initial_state"] = .05 * torch.randn(1, 1, 64, 4, 64, device=device, dtype=dtype)
    if temporal:
        values.update(gate_kv=gate_kv, gate_q=gate_q)
    names = tuple(values)
    normal_inputs = [value.detach().clone().requires_grad_() for value in values.values()]
    checkpoint_inputs = [value.detach().clone().requires_grad_() for value in values.values()]
    torch.manual_seed(8249 + int(temporal))
    do = torch.randn_like(values["v"])
    dr = torch.randn(*values["v"].shape, 4, device=device, dtype=torch.float32)
    drho = torch.randn_like(dr)

    def loss_fn(*args):
        kwargs = dict(zip(names, args, strict=True))
        if temporal:
            outputs = fused_recurrent_swila_temporal(**kwargs, scale=64 ** -.5, output_final_state=False)
            o, r, rho = outputs[0], outputs[4], outputs[5]
        else:
            o, _, r, rho = fused_recurrent_gated_swila(
                **kwargs, scale=64 ** -.5, output_final_state=False, return_rho_seq=True,
            )
        return (o * do).sum() + .17 * (r * dr).sum() + .13 * (rho * drho).sum()

    normal_loss = loss_fn(*normal_inputs)
    normal_loss.backward()
    checkpoint_loss = checkpoint(
        loss_fn, *checkpoint_inputs, use_reentrant=False,
        context_fn=swila_checkpoint_context_fn, preserve_rng_state=False,
    )
    checkpoint_loss.backward()
    assert get_swila_checkpoint_phase() == "normal"
    assert_swila_close("checkpoint_loss", normal_loss, checkpoint_loss, profile=torch.float32)
    for name, normal, actual in zip(names, normal_inputs, checkpoint_inputs, strict=True):
        assert_swila_grad(name, normal.grad, actual.grad, connected=True, profile=normal.dtype)
