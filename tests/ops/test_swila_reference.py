# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""CPU equation checks; independent FP64 oracle versus the public eager recurrence."""

import pytest
import torch

from tests.ops.swila_helpers import (
    FAMILIES,
    assert_swila_close,
    assert_swila_grad,
    clone_inputs,
    compare_gradients,
    gradients,
    make_historical_inputs,
    make_inputs,
    run_oracle,
    run_reference,
)

VARIANTS = (("plain", 1), ("gated", 2), ("temporal", 6), ("temporal_gated", 8))


@pytest.mark.parametrize(("family", "variant"), VARIANTS, ids=FAMILIES)
def test_reference_fp64_oracle(family, variant):
    # Historical seeds and draw order survive; inputs are rounded once before
    # the FP64 oracle, so this measures recurrence rather than input rounding.
    data = make_historical_inputs(seed=200 + variant, temporal="temporal" in family, gated="gated" in family)
    rounded = clone_inputs(data, dtype=torch.float32, requires_grad=False)
    oracle_inputs = clone_inputs(rounded, dtype=torch.float64)
    eager_inputs = clone_inputs(rounded)
    oracle = {name: value for name, value in run_oracle(oracle_inputs, family).items()
              if value is not None and name not in ("u", "pi_bar")}
    eager = {name: value for name, value in run_reference(eager_inputs, family).items()
             if name not in ("u", "pi_bar")}
    assert oracle.keys() == eager.keys()
    for name in eager:
        assert_swila_close(name, oracle[name].float(), eager[name], require_relative=True)
    generator = torch.Generator().manual_seed(400 + variant)
    coefficients = {"o": 1., "state": .31, "r": .17, "rho": .19, "r_final": .23, "rho_final": .29}
    weights = {name: torch.randn(value.shape, generator=generator, dtype=torch.float64) for name, value in oracle.items()}

    def loss(outputs):
        return torch.stack([coefficients[name] * (value * weights[name].to(value.dtype)).sum()
                            for name, value in outputs.items()]).sum()

    ref_grads, eager_grads = gradients(loss(oracle), oracle_inputs), gradients(loss(eager), eager_inputs)
    ref_grads = {name: value.float() if value is not None else None for name, value in ref_grads.items()}
    compare_gradients(ref_grads, eager_grads, gradient_ratio=2e-5, require_relative=True)


@pytest.mark.parametrize(("family", "variant"), VARIANTS, ids=FAMILIES)
def test_reference_oracle_gradcheck(family, variant):
    inputs = make_historical_inputs(
        seed=130 + variant, length=2, experts=2, temporal="temporal" in family, gated="gated" in family,
    )
    inputs = clone_inputs(inputs)

    def evaluate(*values):
        outputs = run_oracle(dict(zip(inputs, values)), family)
        return tuple(value for value in outputs.values() if value is not None)

    assert torch.autograd.gradcheck(
        evaluate, tuple(inputs.values()), eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=variant != 1,
    )


def test_reference_single_token_equations():
    data = make_historical_inputs(seed=100, length=1, temporal=False, gated=False)
    result = run_oracle(data, "plain")
    q, k, v = (data[name][:, 0] for name in ("q", "k", "v"))
    state = data["initial_state"]
    error = v.unsqueeze(-1) - torch.einsum("bhd,bhved->bhve", k, state)
    r = (data["pi_k"][:, 0] - .5 * error.square()).softmax(-1)
    expected = state + data["beta"][:, 0, ..., None] * r[..., None] * error[..., None] * k[:, :, None, None, :]
    rho = data["pi_q"][:, 0].softmax(-1)
    output = (rho * torch.einsum("bhd,bhved->bhve", q * q.shape[-1]**-.5, expected)).sum(-1)
    torch.testing.assert_close(result["state"], expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(result["o"][:, 0], output, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("family", ("plain", "gated"))
def test_reference_single_expert_delta_update(family):
    data = make_inputs(family, dtype=torch.float64, seed=103, length=1, experts=1, keys=3, values=2)
    output = run_oracle(data, family)
    k, v, beta = data["k"][:, 0], data["v"][:, 0], data["beta"][:, 0, ..., 0]
    state = data["initial_state"].squeeze(-2)
    if "log_decay" in data:
        state = state * data["log_decay"][:, 0, :, None, :].exp()
    error = v - torch.einsum("bhd,bhvd->bhv", k, state)
    expected = state + (beta * error)[..., None] * k[:, :, None, :]
    torch.testing.assert_close(output["state"].squeeze(-2), expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(output["r"], torch.ones_like(output["r"]), rtol=0, atol=0)


def test_reference_exact_initial_zero_mass():
    data = clone_inputs(make_inputs("temporal_gated", dtype=torch.float64, length=1))
    with torch.no_grad():
        data["initial_r"][..., 0] = 0
        data["initial_rho"][..., 0] = 0
    result = run_oracle(data, "temporal_gated")
    prior = ((1 - data["gate_kv"][:, 0].sigmoid()[..., None]) * data["initial_r"]
             + data["gate_kv"][:, 0].sigmoid()[..., None] * data["pi_k"][:, 0].softmax(-1))
    torch.testing.assert_close(result["pi_bar"][:, 0], prior, rtol=1e-10, atol=1e-12)
    grad = torch.autograd.grad(result["pi_bar"].sum(), data["initial_r"])[0]
    expected = (1 - data["gate_kv"][:, 0].sigmoid())[..., None].expand_as(grad)
    torch.testing.assert_close(grad, expected, rtol=1e-10, atol=1e-12)
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad[..., 0]) == grad[..., 0].numel()


@pytest.mark.parametrize("failure", ("coordinate", "slice", "nonfinite", "mandatory-relative"))
def test_reference_comparison_rejects_mutations(failure):
    ref = torch.ones(16, 64)
    actual = ref.clone()
    if failure == "coordinate":
        actual[0, 0] += 2e-4
    elif failure == "slice":
        actual[0] += 3e-5
    elif failure == "nonfinite":
        actual[0, 0] = float("nan")
    else:
        ref = torch.zeros(2, 3)
        actual = torch.full_like(ref, 1e-8)
    with pytest.raises(AssertionError):
        assert_swila_close(failure, ref, actual, require_relative=failure == "mandatory-relative")


@pytest.mark.parametrize("mismatch", ("shape", "dtype", "device"))
def test_reference_disconnected_gradient_requires_matching_metadata(mismatch):
    ref = torch.zeros(2, 3)
    if mismatch == "shape":
        actual = torch.zeros(3, 2)
    elif mismatch == "dtype":
        actual = torch.zeros(2, 3, dtype=torch.float64)
    else:
        # A device mismatch must fail before attempting to read tensor values.
        actual = torch.empty(2, 3, device="meta")
    with pytest.raises(AssertionError, match=mismatch):
        assert_swila_grad("disconnected", ref, actual, connected=False)
