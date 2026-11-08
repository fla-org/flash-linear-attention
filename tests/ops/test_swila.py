# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Compact direct-operator preservation gate; no layer or model adapters."""

import pytest
import torch

from fla.utils import IS_NVIDIA, device
from tests.ops.swila_helpers import (
    FAMILIES,
    FINAL_KEYS,
    STATE_KEYS,
    SavedTensorIntegrity,
    clone_inputs,
    compare_gradients,
    compare_outputs,
    connected_inputs,
    cotangents,
    gradients,
    make_inputs,
    objective,
    run_fused,
    run_packed_reference,
    run_reference,
)

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason="SwiLA Triton preservation gate requires an NVIDIA GPU")


# Complete tuples keep every family/precision pair deliberate, rather than
# multiplying dimensions that would compile the same path repeatedly.
CASES = [
    pytest.param("plain", torch.float32, 1, 5, 1, 5, 3, 3, False, id="plain-fp32-odd-zero"),
    pytest.param("gated", torch.float32, 1, 5, 1, 5, 3, 3, True, id="gated-fp32-odd-initial"),
    pytest.param("temporal", torch.float32, 1, 5, 1, 5, 3, 3, False, id="temporal-fp32-odd-default-priors"),
    pytest.param("temporal_gated", torch.float32, 1, 5, 1, 5, 3, 3, True, id="temporal-gated-fp32-odd-initial"),
    pytest.param("plain", torch.bfloat16, 2, 7, 2, 8, 4, 4, True, id="plain-bf16-batch-head-initial"),
    pytest.param("gated", torch.bfloat16, 2, 7, 2, 8, 4, 4, False, id="gated-bf16-batch-head-zero"),
    pytest.param("temporal", torch.bfloat16, 2, 7, 2, 8, 4, 4, True, id="temporal-bf16-batch-head-initial"),
    pytest.param("temporal_gated", torch.bfloat16, 2, 7, 2, 8, 4, 4, False, id="temporal-gated-bf16-default-priors"),
    pytest.param("plain", torch.float16, 1, 5, 1, 5, 3, 3, True, id="plain-fp16-initial"),
    pytest.param("gated", torch.float16, 1, 5, 1, 5, 3, 3, True, id="gated-fp16-initial"),
]


@pytest.mark.parametrize(("family", "dtype", "batch", "length", "heads", "keys", "values", "experts", "initial"), CASES)
def test_fused_recurrent(family, dtype, batch, length, heads, keys, values, experts, initial):
    data = make_inputs(family, dtype=dtype, device=device, batch=batch, length=length, heads=heads,
                       keys=keys, values=values, experts=experts, initial=initial)
    ref_inputs, actual_inputs = clone_inputs(data), clone_inputs(data)
    ref, actual = run_reference(ref_inputs, family), run_fused(actual_inputs, family)
    compare_outputs(ref, actual)
    weights = cotangents(ref)
    compare_gradients(gradients(objective(ref, weights), ref_inputs),
                      gradients(objective(actual, weights), actual_inputs))


@pytest.mark.parametrize("family", FAMILIES)
def test_fused_isolated_vjps(family):
    data = make_inputs(family, device=device)
    ref_inputs, actual_inputs = clone_inputs(data), clone_inputs(data)
    with SavedTensorIntegrity() as integrity:
        ref, actual = run_reference(ref_inputs, family), run_fused(actual_inputs, family)
    compare_outputs(ref, actual)
    weights = cotangents(ref)
    for name in ref:
        ref_grad = gradients(objective(ref, weights, [name]), ref_inputs, retain_graph=True)
        actual_grad = gradients(objective(actual, weights, [name]), actual_inputs, retain_graph=True)
        compare_gradients(ref_grad, actual_grad, connected=connected_inputs(ref_inputs, name), prefix=f"{name}/")
        integrity.assert_unchanged()
    shared_grad = gradients(objective(actual, weights), actual_inputs)
    integrity.assert_unchanged()
    fresh_inputs = clone_inputs(data)
    fresh_grad = gradients(objective(run_fused(fresh_inputs, family), weights), fresh_inputs)
    compare_gradients(shared_grad, fresh_grad, prefix="retained-versus-fresh/")


@pytest.mark.parametrize("family", FAMILIES)
def test_fused_packed_boundaries(family):
    lengths = (1, 4, 2)
    data = make_inputs(family, device=device, length=sum(lengths), sequences=len(lengths))
    ref_inputs, actual_inputs = clone_inputs(data), clone_inputs(data)
    cu_seqlens = torch.tensor([0, 1, 5, 7], dtype=torch.int32, device=device)
    ref = run_packed_reference(ref_inputs, family, lengths)
    actual = run_fused(actual_inputs, family, cu_seqlens=cu_seqlens)
    compare_outputs(ref, actual)
    weights = cotangents(ref)
    compare_gradients(gradients(objective(ref, weights), ref_inputs),
                      gradients(objective(actual, weights), actual_inputs, retain_graph=True))
    boundary_grad = gradients(actual["o"][:, :1].sum(), actual_inputs)
    for name, grad in boundary_grad.items():
        assert grad is not None and torch.isfinite(grad).all()
        outside = grad[1:] if name in STATE_KEYS else grad[:, 1:]
        assert torch.count_nonzero(outside) == 0, f"{name}: gradient crossed a packed boundary"


@pytest.mark.parametrize("family", FAMILIES)
def test_fused_unequal_packed_cache_continuation(family):
    lengths, prefixes = (3, 5), (1, 2)
    data = make_inputs(family, device=device, length=sum(lengths), sequences=2)
    full_inputs, split_inputs = clone_inputs(data), clone_inputs(data)
    full = run_fused(full_inputs, family, cu_seqlens=torch.tensor([0, 3, 8], device=device, dtype=torch.int64))

    def pick(parts):
        return {name: tensor if name in STATE_KEYS else torch.cat([tensor[:, a:b] for a, b in parts], dim=1)
                for name, tensor in split_inputs.items()}

    prefix = run_fused(pick(((0, 1), (3, 5))), family,
                       cu_seqlens=torch.tensor([0, 1, 3], device=device, dtype=torch.int64))
    suffix_inputs = pick(((1, 3), (5, 8)))
    suffix_inputs["initial_state"] = prefix["state"]
    if "temporal" in family:
        suffix_inputs["initial_r"], suffix_inputs["initial_rho"] = prefix["r_final"], prefix["rho_final"]
    suffix = run_fused(suffix_inputs, family, cu_seqlens=torch.tensor([0, 2, 5], device=device, dtype=torch.int64))
    split = {}
    for name in full:
        if name in FINAL_KEYS:
            split[name] = suffix[name]
        else:
            split[name] = torch.cat((prefix[name][:, :prefixes[0]], suffix[name][:, :lengths[0] - prefixes[0]],
                                     prefix[name][:, prefixes[0]:], suffix[name][:, lengths[0] - prefixes[0]:]), dim=1)
    compare_outputs(full, split)
    # Independent sequential execution prevents agreement between two wrong cache routes.
    oracle_inputs = clone_inputs(data)
    reference = run_packed_reference(oracle_inputs, family, lengths)
    compare_outputs(reference, full)
    weights = cotangents(reference)
    expected = gradients(objective(reference, weights), oracle_inputs)
    compare_gradients(expected, gradients(objective(full, weights), full_inputs), prefix="full/")
    compare_gradients(expected, gradients(objective(split, weights), split_inputs), prefix="split/")


@pytest.mark.parametrize(("family", "dtype"), [
    pytest.param("plain", torch.float32, id="plain-fp32"),
    pytest.param("temporal_gated", torch.bfloat16, id="temporal-gated-bf16"),
])
def test_fused_qk_normalization(family, dtype):
    data = make_inputs(family, device=device, dtype=dtype)
    data["q"] = .35 + 1.1 * data["q"]
    data["k"] = -.2 + .9 * data["k"]
    ref_inputs, actual_inputs = clone_inputs(data), clone_inputs(data)
    ref = run_reference(ref_inputs, family, normalize=True, scale=.7)
    actual = run_fused(actual_inputs, family, normalize=True, scale=.7)
    compare_outputs(ref, actual)
    weights = cotangents(ref)
    compare_gradients(gradients(objective(ref, weights), ref_inputs),
                      gradients(objective(actual, weights), actual_inputs))


@pytest.mark.parametrize(("family", "experts", "zero_mass"), [
    pytest.param("gated", 1, False, id="gated-single-expert"),
    pytest.param("temporal_gated", 8, True, id="temporal-gated-e8-zero-prior-mass"),
])
def test_fused_expert_boundaries(family, experts, zero_mass):
    data = make_inputs(family, device=device, experts=experts)
    if zero_mass:
        data["initial_r"][..., 0] = 0
        data["initial_rho"][..., 0] = 0
    ref_inputs, actual_inputs = clone_inputs(data), clone_inputs(data)
    ref, actual = run_reference(ref_inputs, family), run_fused(actual_inputs, family)
    compare_outputs(ref, actual)
    weights = cotangents(ref)
    compare_gradients(gradients(objective(ref, weights), ref_inputs),
                      gradients(objective(actual, weights), actual_inputs))
    if experts == 1:
        torch.testing.assert_close(actual["r"], torch.ones_like(actual["r"]), rtol=0, atol=0)


@pytest.mark.parametrize("family", FAMILIES)
def test_fused_rejects_invalid_packed_boundaries(family):
    inputs = make_inputs(family, device=device)
    for boundaries in ([1, 5], [0, 4], [0, 2, 2, 5], [0, 4, 3, 5]):
        with pytest.raises(ValueError):
            run_fused(inputs, family, cu_seqlens=torch.tensor(boundaries, device=device, dtype=torch.int32))


@pytest.mark.parametrize("family", ("temporal", "temporal_gated"))
def test_fused_requires_paired_temporal_priors(family):
    inputs = make_inputs(family, device=device)
    for name in ("initial_r", "initial_rho"):
        incomplete = {key: value for key, value in inputs.items() if key != name}
        with pytest.raises(ValueError, match="both"):
            run_fused(incomplete, family)
