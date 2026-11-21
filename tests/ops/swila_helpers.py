# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Strict SwiLA comparison policy and independent operator fixtures.

The policy, oracle, and saved-tensor guard originate in lindermanlab/
switching-linear-attention at 4b43bb5c. Global RMS, coordinate, and per-value
slice gates all apply; FLA's global comparison is an additional check.
"""

import math

import torch
import torch.nn.functional as F

from fla.utils import assert_close
from tests.ops.swila_oracle import swila_recurrence_oracle

POLICIES = {
    (torch.float32, False): (1e-5, 1e-4, 1e-5),
    (torch.float32, True): (5e-5, 2e-4, 5e-5),
    (torch.float16, False): (2e-3, 5e-3, 2e-3),
    (torch.float16, True): (4e-3, 1e-2, 4e-3),
    (torch.bfloat16, False): (5e-3, 1e-2, 5e-3),
    (torch.bfloat16, True): (5e-3, 2e-2, 8e-3),
}
FAMILIES = ("plain", "gated", "temporal", "temporal_gated")
STATE_KEYS = {"initial_state", "initial_r", "initial_rho"}
FINAL_KEYS = {"state", "r_final", "rho_final"}


def assert_swila_close(
    name, ref, actual, *, gradient=False, profile=None,
    gradient_ratio=None, require_relative=False, slice_dims=None,
):
    """Preserve all three published precision gates, including the absolute floor."""
    assert ref.shape == actual.shape, f"{name}: shape {ref.shape} != {actual.shape}"
    assert ref.dtype == actual.dtype, f"{name}: dtype {ref.dtype} != {actual.dtype}"
    assert ref.device == actual.device, f"{name}: devices differ"
    profile = actual.dtype if profile is None else profile
    ratio, coordinate_rtol, slice_rtol = POLICIES[(profile, gradient)]
    if gradient_ratio is not None:
        assert gradient_ratio <= ratio, "an override may tighten, never relax, the policy"
        ratio = gradient_ratio
    ref, actual = ref.detach().float(), actual.detach().float()
    assert torch.isfinite(ref).all() and torch.isfinite(actual).all(), f"{name}: nonfinite endpoint"
    if ref.numel() == 0:
        return
    difference = (ref - actual).abs()
    reference_rms = ref.flatten().square().mean().sqrt().item()
    raw_rms = difference.flatten().square().mean().sqrt().item()
    maximum = difference.max().item()
    relative = raw_rms / (reference_rms + 1e-8)
    assert all(math.isfinite(x) for x in (reference_rms, raw_rms, maximum, relative)), name
    assert_close(name, ref, actual, ratio=ratio, warning=False, err_atol=1e-6)
    assert maximum <= 1e-6 or relative < ratio, f"{name}: RMS {relative:.9g} >= {ratio}"
    if require_relative:
        assert relative <= ratio, f"{name}: mandatory relative RMS {relative:.9g} > {ratio}"
    local_scale = torch.maximum(ref.abs().double(), ref.new_tensor(reference_rms).double())
    scores = difference.double() / (1e-6 + coordinate_rtol * local_scale)
    index = scores.flatten().argmax().item()
    coordinate_error = difference.flatten()[index].item()
    coordinate_limit = 1e-6 + coordinate_rtol * max(ref.abs().flatten()[index].item(), reference_rms)
    assert coordinate_error / coordinate_limit <= 1.0, f"{name}: coordinate {index}: {coordinate_error} > {coordinate_limit}"
    slice_errors = difference.square().mean(dim=-1).sqrt() if ref.ndim else difference
    slice_refs = ref.square().mean(dim=-1).sqrt() if ref.ndim else ref.abs()
    slice_scale = torch.maximum(slice_refs.double(), ref.new_tensor(reference_rms).double())
    scores = slice_errors.double() / (1e-6 + slice_rtol * slice_scale)
    index = scores.flatten().argmax().item()
    slice_error = slice_errors.flatten()[index].item()
    slice_limit = 1e-6 + slice_rtol * max(slice_refs.flatten()[index].item(), reference_rms)
    assert slice_error / slice_limit <= 1.0, f"{name}: slice {index}: {slice_error} > {slice_limit}"
    if slice_dims is not None:
        assert slice_dims and all(-ref.ndim <= d < ref.ndim for d in slice_dims), "invalid slice dimensions"
        dims = tuple(sorted(d % ref.ndim for d in slice_dims))
        assert len(set(dims)) == len(dims), "slice dimensions must be unique"
        if dims != (ref.ndim - 1,):
            retained = tuple(d for d in range(ref.ndim) if d not in dims)
            shape = tuple(ref.shape[d] for d in retained) + (math.prod(ref.shape[d] for d in dims),)
            assert_swila_close(
                f"{name}/extra-slices", ref.permute(retained + dims).reshape(shape),
                actual.permute(retained + dims).reshape(shape), gradient=gradient, profile=profile,
                gradient_ratio=gradient_ratio, require_relative=require_relative,
            )


def assert_swila_grad(
    name, ref, actual, *, connected=True, profile=None,
    gradient_ratio=None, require_relative=False, slice_dims=None,
):
    if type(connected) is not bool:
        raise TypeError("connected must be a bool")
    for tensor in (ref, actual):
        if tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError("a gradient must be a tensor or None")
    if not connected:
        if ref is not None and actual is not None:
            assert_swila_close(
                name, ref, actual, gradient=True, profile=profile, gradient_ratio=gradient_ratio,
                require_relative=require_relative, slice_dims=slice_dims,
            )
        for label, tensor in (("reference", ref), ("fused", actual)):
            assert tensor is None or (torch.isfinite(tensor).all() and torch.count_nonzero(tensor) == 0), (
                f"{name}: disconnected {label} gradient must be None or exactly zero"
            )
        return
    assert ref is not None and actual is not None, f"{name}: connected gradient missing"
    assert_swila_close(
        name, ref, actual, gradient=True, profile=profile, gradient_ratio=gradient_ratio,
        require_relative=require_relative, slice_dims=slice_dims,
    )


class SavedTensorIntegrity:
    """Observe actual autograd storage; never substitute a cloned saved tensor."""

    def __init__(self):
        self.snapshots = []

    def watch(self, tensor):
        alias = tensor.detach()
        self.snapshots.append((alias, alias.clone(), alias._version, tuple(alias.shape), tuple(alias.stride())))
        return alias

    def __enter__(self):
        self.hooks = torch.autograd.graph.saved_tensors_hooks(self.watch, lambda tensor: tensor)
        self.hooks.__enter__()
        return self

    def __exit__(self, *args):
        return self.hooks.__exit__(*args)

    def assert_unchanged(self):
        by_device = {}
        for tensor, snapshot, version, shape, stride in self.snapshots:
            assert (tuple(tensor.shape), tuple(tensor.stride())) == (shape, stride)
            assert tensor.dtype == snapshot.dtype and tensor.device == snapshot.device
            assert tensor._version == version, "saved tensor version changed"
            current, saved = by_device.setdefault(tensor.device, ([], []))
            current.append(tensor.contiguous().reshape(-1).view(torch.uint8))
            saved.append(snapshot.contiguous().reshape(-1).view(torch.uint8))
        for current, saved in by_device.values():
            assert torch.equal(torch.cat(current), torch.cat(saved)), "saved tensor bytes changed"


def make_inputs(
    family, *, dtype=torch.float32, device="cpu", seed=42, batch=1, length=5,
    heads=1, keys=5, values=3, experts=3, initial=True, sequences=None,
):
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, storage=dtype):
        return torch.randn(*shape, generator=generator, device=device, dtype=storage)

    shape = (batch, length, heads, values, experts)
    inputs = {
        "q": F.normalize(randn(batch, length, heads, keys), dim=-1),
        "k": F.normalize(randn(batch, length, heads, keys), dim=-1),
        "v": 0.4 * randn(batch, length, heads, values),
        "beta": torch.sigmoid(randn(*shape)),
        "pi_q": 0.5 * randn(*shape),
        "pi_k": 0.5 * randn(*shape),
    }
    count = batch if sequences is None else sequences
    if initial:
        inputs["initial_state"] = 0.05 * randn(count, heads, values, experts, keys)
    if "gated" in family:
        inputs["log_decay"] = -(0.02 + 0.2 * torch.rand(
            batch, length, heads, keys, generator=generator, device=device,
            dtype=torch.float64 if dtype == torch.float64 else torch.float32,
        ))
    if "temporal" in family:
        inputs["gate_kv"] = randn(batch, length, heads, values)
        inputs["gate_q"] = randn(batch, length, heads, values)
        if initial:
            inputs["initial_r"] = F.softmax(randn(count, heads, values, experts), dim=-1)
            inputs["initial_rho"] = F.softmax(randn(count, heads, values, experts), dim=-1)
    return inputs


def clone_inputs(inputs, *, dtype=None, requires_grad=True):
    return {
        key: value.detach().clone().to(dtype=dtype or value.dtype).requires_grad_(requires_grad)
        for key, value in inputs.items()
    }


def run_oracle(inputs, family, *, scale=None):
    kwargs = dict(inputs)
    for key in ("gate_kv", "gate_q"):
        if key in kwargs:
            kwargs[key] = kwargs[key].unsqueeze(-1)
    return swila_recurrence_oracle(**kwargs, temporal="temporal" in family, scale=scale)


def run_reference(inputs, family, *, normalize=False, scale=None):
    from fla.ops.swila import swila_recurrence

    kwargs = dict(inputs)
    if normalize:
        for key in ("q", "k"):
            value = kwargs[key].float()
            kwargs[key] = (value * torch.rsqrt(value.square().sum(-1, keepdim=True) + 1e-6)).to(inputs[key].dtype)
    oracle_inputs = {key: value.float() for key, value in kwargs.items()}
    for key in ("gate_kv", "gate_q"):
        if key in kwargs:
            kwargs[key] = kwargs[key].unsqueeze(-1)
    o, state, responsibilities = swila_recurrence(
        **kwargs, scale=scale, temporal="temporal" in family, output_final_state=True,
    )
    if "temporal" not in family:
        return {"o": o, "state": state, "r": responsibilities, "rho": F.softmax(inputs["pi_q"], dim=-1)}
    # The eager public API does not return these two direct-operator auxiliaries.
    auxiliary = run_oracle(oracle_inputs, family, scale=scale)
    return {
        "o": o, "state": state["S"], "r_final": state["r_prev"], "rho_final": state["rho_prev"],
        **responsibilities, "u": auxiliary["u"], "pi_bar": auxiliary["pi_bar"],
    }


def run_fused(
    inputs, family, *, cu_seqlens=None, normalize=False, scale=None, output_final_state=True, return_rho_seq=True,
):
    from fla.ops.swila import (
        fused_recurrent_gated_swila,
        fused_recurrent_gated_swila_temporal,
        fused_recurrent_swila,
        fused_recurrent_swila_temporal,
    )

    functions = dict(zip(FAMILIES, (
        fused_recurrent_swila, fused_recurrent_gated_swila,
        fused_recurrent_swila_temporal, fused_recurrent_gated_swila_temporal,
    )))
    kwargs = dict(inputs)
    for source, dest in (("initial_r", "r_prev_init"), ("initial_rho", "rho_prev_init")):
        if source in kwargs:
            kwargs[dest] = kwargs.pop(source)
    if "temporal" not in family:
        kwargs["return_rho_seq"] = return_rho_seq
    outputs = functions[family](
        **kwargs, scale=scale, output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=normalize, cu_seqlens=cu_seqlens,
    )
    if "temporal" not in family:
        return dict(zip(("o", "state", "r", "rho"), outputs))
    assert len(outputs) == 9 and outputs[6] is None
    return dict(zip(("o", "state", "r_final", "rho_final", "r", "rho", "u", "pi_bar"), (*outputs[:6], *outputs[7:])))


def segment_inputs(inputs, start, end, sequence):
    return {
        name: tensor[sequence:sequence + 1] if name in STATE_KEYS else tensor[:, start:end]
        for name, tensor in inputs.items()
    }


def concatenate_outputs(outputs):
    return {name: torch.cat([part[name] for part in outputs], dim=0 if name in FINAL_KEYS else 1) for name in outputs[0]}


def run_packed_reference(inputs, family, lengths, *, scale=None):
    outputs, start = [], 0
    for sequence, length in enumerate(lengths):
        outputs.append(run_reference(segment_inputs(inputs, start, start + length, sequence), family, scale=scale))
        start += length
    return concatenate_outputs(outputs)


def reference_graph(data, family, *, lengths=None, scale=None):
    inputs = clone_inputs(data)
    # share the temporal branches' input casts so their VJPs merge before activation-dtype rounding.
    shared = {name: value.float() for name, value in inputs.items()} if "temporal" in family else inputs
    outputs = run_reference(shared, family, scale=scale) if lengths is None else run_packed_reference(
        shared, family, lengths, scale=scale,
    )
    outputs["o"] = outputs["o"].to(data["q"].dtype)
    return inputs, outputs


def cotangents(outputs, *, seed=42):
    generator = torch.Generator(device=next(iter(outputs.values())).device).manual_seed(seed)
    return {
        name: torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
        for name, value in outputs.items()
    }


def objective(outputs, weights, names=None):
    names = outputs if names is None else names
    return sum((outputs[name].float() * weights[name].float()).sum() for name in names)


def gradients(loss, inputs, *, retain_graph=False):
    return dict(zip(inputs, torch.autograd.grad(loss, tuple(inputs.values()), allow_unused=True, retain_graph=retain_graph)))


def connected_inputs(inputs, output):
    query = {"pi_q", "gate_q", "initial_rho"}
    write = {"k", "v", "beta", "pi_k", "initial_state", "log_decay", "gate_kv", "initial_r"}
    if output in ("rho", "rho_final"):
        return query & inputs.keys()
    if output in ("state", "r", "r_final", "u", "pi_bar"):
        return write & inputs.keys()
    return set(inputs)


def compare_outputs(ref, actual, *, prefix=""):
    assert ref.keys() == actual.keys()
    for name in ref:
        assert_swila_close(f"{prefix}{name}", ref[name], actual[name])


def compare_gradients(ref, actual, *, connected=None, prefix="", **kwargs):
    assert ref.keys() == actual.keys()
    connected = ref.keys() if connected is None else connected
    for name in ref:
        assert_swila_grad(f"{prefix}d{name}", ref[name], actual[name], connected=name in connected, **kwargs)


def make_historical_inputs(
    *,
    seed: int,
    length: int = 7,
    experts: int = 3,
    temporal: bool,
    gated: bool,
) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch, heads, keys, values = 1, 1, 3, 2
    sequence_shape = (batch, length, heads)
    q = F.normalize(
        torch.randn(*sequence_shape, keys, generator=generator, dtype=torch.float64),
        dim=-1,
    )
    k = F.normalize(
        torch.randn(*sequence_shape, keys, generator=generator, dtype=torch.float64),
        dim=-1,
    )
    v = 0.4 * torch.randn(
        *sequence_shape,
        values,
        generator=generator,
        dtype=torch.float64,
    )
    beta = torch.sigmoid(
        torch.randn(
            *sequence_shape,
            values,
            experts,
            generator=generator,
            dtype=torch.float64,
        )
    )
    pi_q = 0.5 * torch.randn(
        *sequence_shape,
        values,
        experts,
        generator=generator,
        dtype=torch.float64,
    )
    pi_k = 0.5 * torch.randn(
        *sequence_shape,
        values,
        experts,
        generator=generator,
        dtype=torch.float64,
    )
    inputs = {
        "q": q,
        "k": k,
        "v": v,
        "beta": beta,
        "pi_q": pi_q,
        "pi_k": pi_k,
        "scale": keys**-0.5,
        "initial_state": 0.05
        * torch.randn(
            batch,
            heads,
            values,
            experts,
            keys,
            generator=generator,
            dtype=torch.float64,
        ),
        "temporal": temporal,
        "initial_r": None,
        "initial_rho": None,
        "gate_kv": None,
        "gate_q": None,
        "log_decay": None,
    }
    if temporal:
        inputs["initial_r"] = F.softmax(
            torch.randn(
                batch,
                heads,
                values,
                experts,
                generator=generator,
                dtype=torch.float64,
            ),
            dim=-1,
        )
        inputs["initial_rho"] = F.softmax(
            torch.randn(
                batch,
                heads,
                values,
                experts,
                generator=generator,
                dtype=torch.float64,
            ),
            dim=-1,
        )
        inputs["gate_kv"] = torch.randn(
            batch,
            length,
            heads,
            values,
            1,
            generator=generator,
            dtype=torch.float64,
        )
        inputs["gate_q"] = torch.randn(
            batch,
            length,
            heads,
            values,
            1,
            generator=generator,
            dtype=torch.float64,
        )
    if gated:
        decay_shape = (*sequence_shape, keys)
        inputs["log_decay"] = -(
            0.02
            + 0.2
            * torch.rand(
                *decay_shape,
                generator=generator,
                dtype=torch.float64,
            )
        )
    return {
        name: value.squeeze(-1) if name in ("gate_kv", "gate_q") else value
        for name, value in inputs.items() if isinstance(value, torch.Tensor)
    }


def probabilities(shape, target, dtype, seed):
    generator = torch.Generator(device=target).manual_seed(seed)
    logits = torch.randn(shape, generator=generator, device=target, dtype=torch.float32)
    return logits.softmax(-1).to(dtype).detach().requires_grad_()


def mi_reference(r, rho, weight, boundaries=None):
    # Specify sequence weighting independently of the production packed reduction.
    def entropies(prob):
        if boundaries is None:
            sequences = list(prob.unbind(0))
        else:
            flat = prob.reshape(-1, *prob.shape[2:])
            sequences = [flat[start:end] for start, end in zip(boundaries[:-1], boundaries[1:]) if end > start]
        p_bar = torch.stack([sequence.mean(0) for sequence in sequences])
        marginal = -(p_bar * torch.log(p_bar + 1e-8)).sum(-1)
        conditional = torch.stack([
            -(sequence * torch.log(sequence + 1e-8)).sum(-1).mean(0)
            for sequence in sequences
        ])
        return p_bar, marginal, conditional

    p_r, marginal_r, conditional_r = entropies(r)
    p_rho, marginal_rho, conditional_rho = entropies(rho)
    if r.shape[-1] == 1:
        mi_r = mi_rho = normalized_r = normalized_rho = r.sum() * 0 + rho.sum() * 0
        loss = weight * .5 * (normalized_r + normalized_rho)
    else:
        normalizer = r.new_tensor(float(r.shape[-1])).log()
        mi_r = (marginal_r - conditional_r).mean()
        mi_rho = (marginal_rho - conditional_rho).mean()
        normalized_r, normalized_rho = mi_r / normalizer, mi_rho / normalizer
        loss = weight * .5 * (-normalized_r - normalized_rho)
    return loss, {
        "H_marginal_r": marginal_r.mean(), "H_marginal_rho": marginal_rho.mean(),
        "H_cond_r": conditional_r.mean(), "H_cond_rho": conditional_rho.mean(),
        "MI_r": mi_r, "MI_rho": mi_rho,
        "expert_util_r": p_r, "expert_util_rho": p_rho,
        "MI_r_normalized": normalized_r, "MI_rho_normalized": normalized_rho,
    }


def check_mi_result(actual, expected, *, rtol, atol):
    torch.testing.assert_close(actual[0], expected[0], rtol=rtol, atol=atol, check_dtype=False)
    assert set(actual[1]) == set(expected[1])
    for key, value in actual[1].items():
        assert not value.requires_grad and value.grad_fn is None
        torch.testing.assert_close(value, expected[1][key], rtol=rtol, atol=atol, check_dtype=False)


def dense_mi_gradient(prob, weight, upstream):
    work = prob.detach().to(torch.float64 if prob.dtype == torch.float64 else torch.float32)
    p_bar = work.mean(1)
    marginal = torch.log(p_bar + 1e-8) + p_bar / (p_bar + 1e-8)
    conditional = torch.log(work + 1e-8) + work / (work + 1e-8)
    scale = upstream * .5 * weight / (math.prod(prob.shape[:-1]) * math.log(prob.shape[-1]))
    return (scale * (marginal[:, None] - conditional)).to(prob.dtype)
