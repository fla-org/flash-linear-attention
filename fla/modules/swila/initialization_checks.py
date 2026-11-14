# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).


"""Optional, stateless checks of SwiLA's deterministic initialization values.

Call explicitly after initialization, or after loading a checkpoint containing
fresh-initialization values. Trained checkpoints need not satisfy these checks.
They validate current values and optimizer metadata, not reset history or the
random draw that produced unconstrained weights. This module has no layer-class,
model-framework, or kernel dependency; package initialization retains its own
import behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules.swila.projections import LowRankProjection, SwiGLUProjection


def local_initialization_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a detached local shard for ordinary tensors or DTensors."""
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        tensor = to_local()
    return tensor.detach()


def named_initialized_parameters(layer):
    """Yield parameters governed by SwiLA's specialized initialization rules."""
    modules: list[tuple[str, nn.Module | None]] = [
        ("b_proj", layer.b_proj),
        ("pi_q_input_proj", layer.pi_q_input_proj),
        ("pi_k_input_proj", layer.pi_k_input_proj),
        ("pi_q_proj", layer.pi_q_proj),
        ("pi_k_proj", layer.pi_k_proj),
        ("pi_q_conv1d", layer.pi_q_conv1d),
        ("pi_k_conv1d", layer.pi_k_conv1d),
    ]
    if layer.use_forget_gate:
        modules.append(("a_proj", layer.a_proj))
    if layer.temporal:
        modules.extend(
            [
                ("gate_kv_proj", layer.gate_kv_proj),
                ("gate_q_proj", layer.gate_q_proj),
            ]
        )
    if layer.inject_noise:
        modules.extend(
            [
                ("W_noise_k", layer.W_noise_k),
                ("W_noise_q", layer.W_noise_q),
            ]
        )
    if layer.temporal and layer.inject_gate_noise:
        modules.extend(
            [
                ("W_noise_gate_kv", layer.W_noise_gate_kv),
                ("W_noise_gate_q", layer.W_noise_gate_q),
            ]
        )

    seen: set[int] = set()
    for prefix, module in modules:
        if module is None:
            continue
        for name, parameter in module.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            yield f"{prefix}.{name}", parameter

    if layer.use_forget_gate:
        for name, parameter in (
            ("A_log", layer.A_log),
            ("dt_bias", layer.dt_bias),
        ):
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield name, parameter
    if layer.init_state is not None and id(layer.init_state) not in seen:
        yield "init_state", layer.init_state


def replicated_initialization_tensor(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Return a detached full tensor for global DTensor invariants."""
    if not callable(getattr(tensor, "to_local", None)):
        return tensor.detach()

    from torch.distributed.tensor import Replicate

    replicated = tensor.redistribute(
        device_mesh=tensor.device_mesh,
        placements=tuple(Replicate() for _ in tensor.placements),
    )
    return replicated.to_local().detach()


def terminal_projection(projection: nn.Module) -> nn.Linear:
    if isinstance(projection, LowRankProjection):
        return projection.up
    if isinstance(projection, SwiGLUProjection):
        return projection.out
    return projection


def assert_local_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    message: str,
) -> None:
    actual = local_initialization_tensor(actual)
    expected = local_initialization_tensor(expected)
    if actual.shape != expected.shape or not torch.equal(actual, expected):
        raise RuntimeError(message)


def assert_local_centered(
    centered: torch.Tensor,
    source: torch.Tensor,
    message: str,
) -> None:
    """Check a floating-point centering invariant on a local tensor shard."""
    centered = local_initialization_tensor(centered)
    source = local_initialization_tensor(source)
    if centered.numel() == 0:
        return
    if not torch.isfinite(centered).all().item():
        raise RuntimeError(message)
    dtype = source.dtype
    epsilon = torch.finfo(dtype).eps if dtype.is_floating_point else 0.0
    source_scale = source.float().abs().max().item() if source.numel() else 0.0
    tolerance = max(1e-7, 8.0 * epsilon * source_scale)
    if centered.float().abs().max().item() > tolerance:
        raise RuntimeError(message)


def check_layer_initialization(layer) -> None:
    parameter_names = {id(parameter): name for name, parameter in layer.named_parameters()}
    historical = getattr(layer, "architecture_profile", "current") == "historical_8dc05f93"

    def context(*parameters):
        return ", ".join(repr(parameter_names[id(parameter)]) for parameter in parameters)

    if not historical and layer.pi_align_qk_init:
        aligned_pairs = (
            ("pi input", layer.pi_q_input_proj, layer.pi_k_input_proj),
            ("pi terminal", layer.pi_q_proj, layer.pi_k_proj),
            ("pi convolution", layer.pi_q_conv1d, layer.pi_k_conv1d),
        )
        for label, query_module, key_module in aligned_pairs:
            if query_module is None:
                continue
            query_parameters = dict(query_module.named_parameters())
            key_parameters = dict(key_module.named_parameters())
            if query_parameters.keys() != key_parameters.keys():
                raise RuntimeError(f"{label} q/k parameter layouts differ")
            for name in query_parameters:
                assert_local_equal(
                    query_parameters[name],
                    key_parameters[name],
                    f"{context(query_parameters[name], key_parameters[name])}: "
                    f"{label} q/k parameters are not initially aligned",
                )

    # Historical routers are independent Kaiming draws, not centered/aligned.
    # Their random samples have no deterministic value invariant to assert;
    # the common finite and learned-state checks still apply.
    router_projections = () if historical else (layer.pi_q_proj, layer.pi_k_proj)
    for projection in router_projections:
        terminal = replicated_initialization_tensor(
            terminal_projection(projection).weight
        )
        if layer.pi_proj_rank > 0:
            pi_weight = terminal.view(
                layer.num_heads,
                layer.head_v_dim,
                layer.num_experts,
                terminal.shape[-1],
            )
            expert_axis = 2
        else:
            pi_weight = terminal.view(
                layer.head_v_dim,
                layer.num_experts,
                terminal.shape[-1],
            )
            expert_axis = 1
        assert_local_centered(
            pi_weight.mean(dim=expert_axis),
            pi_weight,
            f"{context(terminal_projection(projection).weight)}: "
            "pi terminal rows are not centered over experts",
        )
        if layer.num_experts > 1:
            expert_difference = (
                pi_weight.select(expert_axis, 0)
                - pi_weight.select(expert_axis, 1)
            )
            local_difference = local_initialization_tensor(
                expert_difference
            )
            if (
                local_difference.numel()
                and torch.count_nonzero(local_difference).item() == 0
            ):
                raise RuntimeError(
                    f"{context(terminal_projection(projection).weight)}: "
                    "pi terminal experts are initially identical"
                )

    if layer.use_forget_gate:
        A_log = local_initialization_tensor(layer.A_log)
        dt_bias = local_initialization_tensor(layer.dt_bias)
        if layer.A_log.dtype != torch.float32 or layer.dt_bias.dtype != torch.float32:
            raise RuntimeError("A_log and dt_bias must remain float32")
        if not getattr(layer.A_log, "_no_weight_decay", False):
            raise RuntimeError("A_log lost its no-weight-decay marker")
        if not getattr(layer.dt_bias, "_no_weight_decay", False):
            raise RuntimeError("dt_bias lost its no-weight-decay marker")
        if A_log.numel() and torch.count_nonzero(A_log).item() != 0:
            raise RuntimeError("A_log must initialize exactly to zero")

        dt = F.softplus(dt_bias.float())
        if dt.numel() and (
            dt.min().item() < 1e-3 - 1e-7
            or dt.max().item() > 1e-1 + 1e-7
        ):
            raise RuntimeError(
                "softplus(dt_bias) is outside [1e-3, 1e-1]"
            )

    if layer.inject_noise:
        expected_noise = layer.pi_noise_init_std
        for label, projection in (
            ("query", layer.W_noise_q),
            ("key", layer.W_noise_k),
        ):
            weight = local_initialization_tensor(projection.weight)
            bias = local_initialization_tensor(projection.bias)
            if weight.numel() and torch.count_nonzero(weight).item() != 0:
                raise RuntimeError(
                    f"{context(projection.weight)}: "
                    f"{label} pi-noise weights must initialize to zero"
                )
            actual_noise = F.softplus(bias.float())
            expected = torch.full_like(actual_noise, expected_noise)
            if not torch.allclose(actual_noise, expected, rtol=1e-6, atol=1e-7):
                raise RuntimeError(
                    f"{context(projection.bias)}: "
                    f"{label} pi-noise scale does not match "
                    "pi_noise_init_std"
                )

    if layer.temporal:
        for label, projection in (
            ("KV", layer.gate_kv_proj),
            ("query", layer.gate_q_proj),
        ):
            weight = local_initialization_tensor(projection.weight)
            bias = replicated_initialization_tensor(projection.bias)
            if weight.numel() and torch.count_nonzero(weight).item() != 0:
                raise RuntimeError(
                    f"{context(projection.weight)}: "
                    f"{label} temporal-gate weights must initialize to zero"
                )
            if getattr(projection.weight, "_no_weight_decay", False):
                raise RuntimeError(
                    f"{context(projection.weight)}: "
                    f"{label} temporal-gate weight must retain weight decay"
                )
            if not getattr(
                projection.bias,
                "_no_weight_decay",
                False,
            ):
                raise RuntimeError(
                    f"{context(projection.bias)}: "
                    f"{label} temporal-gate bias lost its "
                    "no-weight-decay marker"
                )
            expected = torch.full_like(bias, layer.gate_init_bias)
            if not torch.equal(bias, expected):
                raise RuntimeError(
                    f"{context(projection.bias)}: "
                    f"{label} temporal-gate bias does not match the "
                    "configured initialization"
                )

    if layer.temporal and layer.inject_gate_noise:
        for label, projection in (
            ("KV", layer.W_noise_gate_kv),
            ("query", layer.W_noise_gate_q),
        ):
            weight = local_initialization_tensor(projection.weight)
            bias = local_initialization_tensor(projection.bias)
            if weight.numel() and torch.count_nonzero(weight).item() != 0:
                raise RuntimeError(
                    f"{context(projection.weight)}: "
                    f"{label} temporal-gate-noise weights must "
                    "initialize to zero"
                )
            if getattr(projection.weight, "_no_weight_decay", False):
                raise RuntimeError(
                    f"{context(projection.weight)}: "
                    f"{label} temporal-gate-noise weight must retain "
                    "weight decay"
                )
            if not getattr(
                projection.bias,
                "_no_weight_decay",
                False,
            ):
                raise RuntimeError(
                    f"{context(projection.bias)}: "
                    f"{label} temporal-gate-noise bias lost its "
                    "no-weight-decay marker"
                )
            actual_noise = F.softplus(bias.float())
            expected_noise = torch.full_like(
                actual_noise,
                layer.gate_noise_init_std,
            )
            if not torch.allclose(
                actual_noise,
                expected_noise,
                rtol=1e-6,
                atol=1e-7,
            ):
                raise RuntimeError(
                    f"{context(projection.bias)}: "
                    f"{label} temporal-gate-noise scale does not match "
                    "gate_noise_init_std"
                )

    if layer.init_state is not None:
        initial_state = local_initialization_tensor(layer.init_state)
        if not torch.isfinite(initial_state).all().item():
            raise RuntimeError("learnable init_state contains nonfinite values")
        if torch.count_nonzero(initial_state).item() != 0:
            raise RuntimeError(
                "zero-initialized learnable init_state is not exactly zero"
            )


@torch.no_grad()
def check_swila_initialization(model_or_layer: nn.Module) -> None:
    """Check every SwiLA layer without changing parameters, metadata, or RNG.

    All parameters must be materialized and finite. Deterministic centering,
    alignment (for the current profile), gate, noise, decay, and learnable-state invariants use
    the same analytic tolerances as initialization's former value checks.
    Distributed values are checked locally or replicated for global invariants;
    call on every rank. No reset record is required; loaded values are checked
    directly rather than inferring their initialization history.

    Raises:
        TypeError: The argument is not a torch.nn.Module.
        ValueError: No SwiLA layer is present.
        RuntimeError: A named layer violates an initialization invariant.
    """
    if not isinstance(model_or_layer, nn.Module):
        raise TypeError("check_swila_initialization expects a torch.nn.Module")
    found_layer = False
    for layer_name, layer in model_or_layer.named_modules():
        if not getattr(layer, "requires_swila_reset_after_generic_init", False):
            continue
        found_layer = True
        try:
            parameters = dict(layer.named_parameters())
            if not parameters:
                raise RuntimeError("SwiLA exposes no parameters")
            for name, parameter in parameters.items():
                local = local_initialization_tensor(parameter)
                if parameter.is_meta or local.is_meta:
                    raise RuntimeError(
                        f"parameter {name!r} is meta; initialization values are unavailable"
                    )
                if not torch.isfinite(local).all().item():
                    raise RuntimeError(f"parameter {name!r} contains nonfinite values")
            check_layer_initialization(layer)
        except RuntimeError as error:
            label = layer_name or "<root>"
            raise RuntimeError(f"SwiLA layer {label!r}: {error}") from error
    if not found_layer:
        raise ValueError("no SwiLA layers found in model_or_layer")
