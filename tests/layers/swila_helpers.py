# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Explicit eager recurrence adapters; FLA convolutions, norms and caches stay real."""

import math
from contextlib import contextmanager
from unittest.mock import patch

import torch

import fla.ops.swila as operators
from fla.layers import SwitchingLinearAttention
from tests.ops.swila_helpers import (
    assert_swila_close,
    assert_swila_grad,
    concatenate_outputs,
    run_reference,
    segment_inputs,
)


def make_layer(family="plain", **overrides):
    options = dict(
        d_model=16, num_heads=2, num_experts=3, pi_proj_rank=4,
        b_proj_rank=4, alpha_proj_rank=4, use_short_conv=False,
        temporal="temporal" in family, use_forget_gate="gated" in family,
        layer_idx=0, return_output_only=False,
    )
    return SwitchingLinearAttention(**(options | overrides))


def make_gpu_layer(family="plain", **overrides):
    # Reuse the operator gate's Dk8/Dv4/E4/H2 compilation domain.
    return make_layer(family, **(dict(expand_v=.5, num_experts=4) | overrides))


def eager_operator(temporal=False, **kwargs):
    family = ("temporal" if temporal else "plain")
    if kwargs.get("log_decay") is not None:
        family = "temporal_gated" if temporal else "gated"
    inputs = {name: kwargs[name] for name in ("q", "k", "v", "beta", "pi_q", "pi_k")}
    for name in ("initial_state", "log_decay", "gate_kv", "gate_q"):
        if kwargs.get(name) is not None:
            inputs[name] = kwargs[name]
    for old, new in (("r_prev_init", "initial_r"), ("rho_prev_init", "initial_rho")):
        if kwargs.get(old) is not None:
            inputs[new] = kwargs[old]

    def evaluate(values):
        return run_reference(values, family, normalize=kwargs.get("use_qk_l2norm_in_kernel", False), scale=kwargs.get("scale"))

    boundaries = kwargs.get("cu_seqlens")
    if boundaries is None:
        result = evaluate(inputs)
    else:
        offsets = boundaries.tolist()
        result = concatenate_outputs([
            evaluate(segment_inputs(inputs, start, end, index))
            for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))
        ])
    state = result["state"] if kwargs.get("output_final_state", False) else None
    if not temporal:
        return result["o"], state, result["r"], result["rho"] if kwargs.get("return_rho_seq", True) else None
    r_final = result["r_final"] if kwargs.get("output_final_state", False) else None
    rho_final = result["rho_final"] if kwargs.get("output_final_state", False) else None
    return result["o"], state, r_final, rho_final, result["r"], result["rho"], None, result["u"], result["pi_bar"]


@contextmanager
def eager_recurrence():
    with patch.object(operators, "fused_recurrent_swila", lambda **kwargs: eager_operator(**kwargs)), patch.object(
        operators, "fused_recurrent_swila_temporal", lambda **kwargs: eager_operator(temporal=True, **kwargs),
    ):
        yield


def compare_tree(reference, actual, *, prefix="", profile=None):
    if reference is None:
        assert actual is None, prefix
    elif isinstance(reference, torch.Tensor):
        assert_swila_close(prefix, reference, actual, profile=profile)
    elif isinstance(reference, dict):
        assert reference.keys() == actual.keys(), prefix
        for key in reference:
            compare_tree(reference[key], actual[key], prefix=f"{prefix}/{key}", profile=profile)
    else:
        assert len(reference) == len(actual), prefix
        for index, (expected, observed) in enumerate(zip(reference, actual, strict=True)):
            compare_tree(expected, observed, prefix=f"{prefix}/{index}", profile=profile)


def layer_gradients(loss, layer, hidden, *, retain_graph=False):
    leaves = {"hidden": hidden, **dict(layer.named_parameters())}
    return dict(zip(leaves, torch.autograd.grad(loss, tuple(leaves.values()), allow_unused=True, retain_graph=retain_graph)))


def compare_layer_gradients(reference, actual, *, profile=None):
    assert reference.keys() == actual.keys()
    for name in reference:
        assert_swila_grad(name, reference[name], actual[name], connected=True, profile=profile)


def layer_objective(output, attentions, auxiliary, cotangent):
    loss = (output.float() * cotangent.float()).sum()
    tensors = attentions.values() if isinstance(attentions, dict) else (attentions,)
    for tensor in tensors:
        if tensor is not None:
            weights = torch.arange(tensor.numel(), device=tensor.device, dtype=torch.float32).sin().reshape(tensor.shape)
            loss = loss + .17 * (tensor.float() * weights).sum()
    return loss + (auxiliary if auxiliary is not None else 0)


def recurrent_cache(cache, index=0):
    state = cache[index]["recurrent_state"]
    return state if isinstance(state, tuple) else (state,)


def install_nonuniform_router_fixture(layer):
    """One fixed, finite, input-dependent trained-state fixture; consumes no RNG.

    The same recipe applies to all four compact layer families. The input/output
    sine patterns are prescribed analytically, independent of observed errors.
    Identity SwiGLU branches keep the construction easy to inspect. Temporal
    reset probability 1/2 avoids the small default reset Jacobian. No non-router
    parameter, learned memory, hidden input, threshold, or objective is changed.
    """
    import torch

    with torch.no_grad():
        for route, phase in (("q", .17), ("k", .83)):
            input_projection = getattr(layer, f"pi_{route}_input_proj")
            weight = input_projection.weight
            rows = torch.arange(1, weight.shape[0] + 1, device=weight.device, dtype=torch.float32)[:, None]
            columns = torch.arange(1, weight.shape[1] + 1, device=weight.device, dtype=torch.float32)[None, :]
            weight.copy_((math.sqrt(2 / weight.shape[1]) * torch.sin(.37 * rows * columns + phase)).to(weight.dtype))

            convolution = getattr(layer, f"pi_{route}_conv1d")
            if convolution is not None:
                convolution.weight.zero_()
                convolution.weight[..., -1] = 1.
                if convolution.weight.shape[-1] > 1:
                    convolution.weight[..., -2] = .25
                if convolution.bias is not None:
                    convolution.bias.zero_()

            projection = getattr(layer, f"pi_{route}_proj")
            for branch in (projection.gate, projection.feat):
                weight = branch.weight
                assert weight.shape[0] == weight.shape[1] == 4, weight.shape
                weight.copy_(2 * torch.eye(4, device=weight.device, dtype=weight.dtype))
            weight = projection.out.weight
            rows = torch.arange(1, weight.shape[0] + 1, device=weight.device, dtype=torch.float32)[:, None]
            columns = torch.arange(1, weight.shape[1] + 1, device=weight.device, dtype=torch.float32)[None, :]
            weight.copy_((1.25 / math.sqrt(weight.shape[1]) * torch.sin(.7 * rows * columns + phase)).to(weight.dtype))

        if layer.temporal:
            for projection in (layer.gate_q_proj, layer.gate_kv_proj):
                projection.weight.zero_()
                projection.bias.zero_()
