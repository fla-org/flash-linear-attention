# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).


"""SwiLA parameter initialization and checkpoint metadata, without verification."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from fla.modules.swila.projections import (
    LowRankProjection,
    SwiGLUProjection,
)


def capture_load_state_dict_prefix(module, state_dict, prefix: str, *args) -> None:
    """Remember the layer prefix for its matching post-load hook."""
    del state_dict, args
    module._active_load_state_dict_prefix = prefix


@torch.no_grad()
def restore_parameter_metadata_after_load(module, incompatible_keys) -> None:
    """Restore metadata and initialize only missing legacy gate-noise biases."""
    restore_parameter_metadata(module)
    prefix = getattr(module, "_active_load_state_dict_prefix", "")
    if hasattr(module, "_active_load_state_dict_prefix"):
        del module._active_load_state_dict_prefix
    if not module.temporal or not module.inject_gate_noise:
        return

    missing_projection_biases = (
        ("W_noise_gate_kv.bias", module.W_noise_gate_kv),
        ("W_noise_gate_q.bias", module.W_noise_gate_q),
    )
    handled_keys = []
    for suffix, projection in missing_projection_biases:
        key = f"{prefix}{suffix}"
        if key in incompatible_keys.missing_keys:
            # Older checkpoints lack these biases. Preserve loaded siblings.
            initialize_missing_gate_noise_bias(module, projection)
            handled_keys.append(key)
    if handled_keys:
        incompatible_keys.missing_keys[:] = [
            key for key in incompatible_keys.missing_keys if key not in handled_keys
        ]


def copy_module_state(source: nn.Module, target: nn.Module) -> None:
    """Copy values without tying storage or overwriting loaded target state."""
    source_parameters = dict(source.named_parameters())
    for name, parameter in target.named_parameters():
        if not getattr(parameter, "_is_hf_initialized", False):
            parameter.copy_(source_parameters[name])

    source_buffers = dict(source.named_buffers())
    for name, buffer in target.named_buffers():
        if not getattr(buffer, "_is_hf_initialized", False):
            buffer.copy_(source_buffers[name])


def replicated_parameter_for_initialization(
    parameter: torch.Tensor,
) -> tuple[torch.Tensor, object | None]:
    """Expose a full local tensor plus an optional DTensor write-back.

    FSDP2 represents parameters as DTensors sharded over their first
    dimension. Directly viewing a 2-D terminal weight as ``[H, ...]`` is
    invalid when ``H`` is not divisible by the mesh size. These initialized
    terminal matrices are small, so initialization temporarily replicates
    only that parameter, performs the layout-sensitive operation locally,
    and redistributes it to its original placements.
    """
    if not callable(getattr(parameter, "to_local", None)):
        return parameter, None

    from torch.distributed.tensor import Replicate

    original_placements = parameter.placements
    replicated_placements = tuple(
        Replicate() for _ in original_placements
    )
    replicated = parameter.redistribute(
        device_mesh=parameter.device_mesh,
        placements=replicated_placements,
    )

    def write_back() -> None:
        parameter.copy_(
            replicated.redistribute(
                device_mesh=parameter.device_mesh,
                placements=original_placements,
            )
        )

    return replicated.to_local(), write_back


def inverse_softplus_scalar(value: float) -> float:
    """Compute inverse softplus stably for every positive finite value."""
    return value + math.log(-math.expm1(-value))


@torch.no_grad()
def restore_parameter_metadata(layer) -> None:
    """Restore dtype and optimizer metadata omitted from state dicts."""
    if layer.use_forget_gate:
        for parameter in (layer.A_log, layer.dt_bias):
            if parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
            parameter._no_weight_decay = True

    if not layer.temporal:
        return
    layer.gate_kv_proj.bias._no_weight_decay = True
    layer.gate_q_proj.bias._no_weight_decay = True
    if layer.inject_gate_noise:
        layer.W_noise_gate_kv.bias._no_weight_decay = True
        layer.W_noise_gate_q.bias._no_weight_decay = True


@torch.no_grad()
def initialize_missing_gate_noise_bias(
    layer,
    projection: nn.Linear,
) -> None:
    """Initialize one newly added legacy-checkpoint noise bias."""
    if projection.bias.is_meta:
        return
    inverse_softplus = inverse_softplus_scalar(
        layer.gate_noise_init_std
    )
    bias, write_back = replicated_parameter_for_initialization(
        projection.bias
    )
    nn.init.constant_(bias, inverse_softplus)
    if write_back is not None:
        write_back()
    projection.bias._no_weight_decay = True


def kaiming_init(module: nn.Module):
    """Apply default Kaiming init without overwriting HF-loaded tensors."""
    if not getattr(module.weight, "_is_hf_initialized", False):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    if (
        module.bias is not None
        and not getattr(module.bias, "_is_hf_initialized", False)
    ):
        fan_in = nn.init._calculate_correct_fan(module.weight, 'fan_in')
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(module.bias, -bound, bound)


def kaiming_init_linear_modules(module: nn.Module):
    """Kaiming-initialize Linear/Conv1d leaves, chiefly optional pi convs."""
    for submodule in module.modules():
        if isinstance(submodule, (nn.Linear, nn.Conv1d)):
            kaiming_init(submodule)


def initialize_router_projection(layer, projection: nn.Module) -> None:
    """Initialize one pi projection and center its terminal expert rows.

        Linear/low-rank input stages preserve unit feature variance. The
        terminal scale targets ``pi_init_logit_std``; SwiGLU uses a 1.7 factor
        to compensate for ``silu(gate) * feature``. Terminal rows are reshaped
        according to the global or per-head router layout, centered over E,
        and variance-corrected by ``sqrt(E / (E - 1))``. Consequently logits
        sum to zero over experts while retaining small expert differences.

        Parameters marked as loaded by Transformers are left untouched.
        """
    terminal_scale = layer.pi_init_logit_std
    if isinstance(projection, LowRankProjection):
        if not getattr(projection.down.weight, "_is_hf_initialized", False):
            nn.init.normal_(
                projection.down.weight,
                mean=0.0,
                std=projection.down.in_features ** -0.5,
            )
        terminal = projection.up
    elif isinstance(projection, SwiGLUProjection):
        for input_projection in (projection.gate, projection.feat):
            if not getattr(
                input_projection.weight,
                "_is_hf_initialized",
                False,
            ):
                nn.init.normal_(
                    input_projection.weight,
                    mean=0.0,
                    std=input_projection.in_features ** -0.5,
                )
        terminal = projection.out
        # For standard-normal gate/feature activations,
        # std(silu(gate) * feature) is about 0.59.
        terminal_scale *= 1.7
    else:
        terminal = projection

    terminal_is_loaded = getattr(
        terminal.weight,
        "_is_hf_initialized",
        False,
    )
    terminal_weight = terminal.weight
    write_back = None
    if not terminal_is_loaded:
        nn.init.normal_(
            terminal.weight,
            mean=0.0,
            std=terminal_scale * terminal.in_features ** -0.5,
        )
        terminal_weight, write_back = (
            replicated_parameter_for_initialization(
                terminal.weight
            )
        )
    if (
        terminal.bias is not None
        and not getattr(terminal.bias, "_is_hf_initialized", False)
    ):
        nn.init.zeros_(terminal.bias)

    # Loaded checkpoint tensors carry this marker in recent Transformers
    # versions. Match guarded nn.init behavior and do not recenter them.
    if not terminal_is_loaded:
        if layer.pi_proj_rank > 0:
            weight = terminal_weight.view(
                layer.num_heads,
                layer.head_v_dim,
                layer.num_experts,
                terminal.in_features,
            )
            expert_axis = 2
        else:
            weight = terminal_weight.view(
                layer.head_v_dim,
                layer.num_experts,
                terminal.in_features,
            )
            expert_axis = 1
        weight.sub_(weight.mean(dim=expert_axis, keepdim=True))
        if layer.num_experts > 1:
            weight.mul_(math.sqrt(layer.num_experts / (layer.num_experts - 1)))
        if write_back is not None:
            write_back()


def initialize_router_path(
    layer,
    input_projection: nn.Linear,
    projection: nn.Module,
    convolution: nn.Module | None,
) -> None:
    """Initialize one router's input, optional convolution, and output path."""
    if not getattr(input_projection.weight, "_is_hf_initialized", False):
        nn.init.normal_(
            input_projection.weight,
            mean=0.0,
            std=input_projection.in_features ** -0.5,
        )
    if (
        input_projection.bias is not None
        and not getattr(input_projection.bias, "_is_hf_initialized", False)
    ):
        nn.init.zeros_(input_projection.bias)
    if convolution is not None:
        kaiming_init_linear_modules(convolution)
    initialize_router_projection(layer, projection)


@torch.no_grad()
def initialize_router_parameters(layer) -> None:
    """Initialize pi and optionally align separately owned q/k routers.

    With alignment enabled, q is initialized once and copied into k. This
    gives identical deterministic logits at initialization without tying
    parameters; subsequent optimization and sampled router noise can make
    the paths diverge.
    """
    initialize_router_path(
        layer,
        layer.pi_q_input_proj,
        layer.pi_q_proj,
        layer.pi_q_conv1d,
    )
    if layer.pi_align_qk_init:
        copy_module_state(layer.pi_q_input_proj, layer.pi_k_input_proj)
        copy_module_state(layer.pi_q_proj, layer.pi_k_proj)
        if layer.pi_q_conv1d is not None:
            copy_module_state(layer.pi_q_conv1d, layer.pi_k_conv1d)
    else:
        initialize_router_path(
            layer,
            layer.pi_k_input_proj,
            layer.pi_k_proj,
            layer.pi_k_conv1d,
        )


@torch.no_grad()
def initialize_router_noise_parameters(layer) -> None:
    """Make initial router-noise std input-independent and configurable.

        Zero weights plus ``inverse_softplus(pi_noise_init_std)`` bias make
        ``softplus(W_noise(x))`` equal the requested scale for every input.
        """
    if not layer.inject_noise or layer.W_noise_k is None:
        return

    inverse_softplus = math.log(math.expm1(layer.pi_noise_init_std))
    for projection in (layer.W_noise_q, layer.W_noise_k):
        if not getattr(projection.weight, "_is_hf_initialized", False):
            nn.init.zeros_(projection.weight)
        if not getattr(projection.bias, "_is_hf_initialized", False):
            nn.init.constant_(projection.bias, inverse_softplus)


@torch.no_grad()
def initialize_temporal_gate_parameters(layer) -> None:
    """Initialize Q/KV reset projections with zero weights and a scalar bias."""
    if not layer.temporal:
        return

    for projection in (layer.gate_kv_proj, layer.gate_q_proj):
        if not getattr(projection.weight, "_is_hf_initialized", False):
            nn.init.zeros_(projection.weight)
        if not getattr(projection.bias, "_is_hf_initialized", False):
            bias, write_back = replicated_parameter_for_initialization(
                projection.bias
            )
            nn.init.constant_(bias, layer.gate_init_bias)
            if write_back is not None:
                write_back()
        projection.bias._no_weight_decay = True


@torch.no_grad()
def initialize_temporal_gate_noise_parameters(layer) -> None:
    """Initialize optional temporal-gate noise in softplus space."""
    if not layer.inject_gate_noise or layer.W_noise_gate_kv is None:
        return

    inverse_softplus = inverse_softplus_scalar(
        layer.gate_noise_init_std
    )
    for projection in (
        layer.W_noise_gate_kv,
        layer.W_noise_gate_q,
    ):
        if not getattr(projection.weight, "_is_hf_initialized", False):
            nn.init.zeros_(projection.weight)
        if not getattr(projection.bias, "_is_hf_initialized", False):
            nn.init.constant_(projection.bias, inverse_softplus)
        projection.bias._no_weight_decay = True


def initialize_gate_projection(
    projection: nn.Module,
    *,
    output_scale: float,
) -> None:
    """Initialize a beta/alpha projection with a small terminal output.

        Input/down stages use ``N(0, 1/sqrt(fan_in))`` and the terminal stage
        uses ``N(0, output_scale/sqrt(fan_in))``. Each output channel is
        shared across experts by the recurrence layout.
        """
    if isinstance(projection, LowRankProjection):
        if not getattr(
            projection.down.weight,
            "_is_hf_initialized",
            False,
        ):
            nn.init.normal_(
                projection.down.weight,
                mean=0.0,
                std=projection.down.in_features ** -0.5,
            )
        output_projection = projection.up
        output_std = output_scale * projection.up.in_features ** -0.5
    elif isinstance(projection, SwiGLUProjection):
        for input_projection in (projection.gate, projection.feat):
            if not getattr(
                input_projection.weight,
                "_is_hf_initialized",
                False,
            ):
                nn.init.normal_(
                    input_projection.weight,
                    mean=0.0,
                    std=input_projection.in_features ** -0.5,
                )
        output_projection = projection.out
        output_std = output_scale * projection.out.in_features ** -0.5
    else:
        output_projection = projection
        output_std = output_scale * projection.in_features ** -0.5

    output_is_loaded = getattr(
        output_projection.weight,
        "_is_hf_initialized",
        False,
    )
    if not output_is_loaded:
        nn.init.normal_(
            output_projection.weight,
            mean=0.0,
            std=output_std,
        )
    if (
        output_projection.bias is not None
        and not getattr(
            output_projection.bias,
            "_is_hf_initialized",
            False,
        )
    ):
        nn.init.zeros_(output_projection.bias)


def initialize_decay_parameters(layer) -> None:
    """Initialize conservative per-head decay magnitudes and time steps.

        ``A_log=0`` gives ``exp(A_log)=1``. Time steps are sampled
        LogUniform(1e-3, 1e-1) and stored as inverse-softplus values.
        Both parameters remain float32 and carry ``_no_weight_decay=True``.
        """
    if not getattr(layer.A_log, "_is_hf_initialized", False):
        layer.A_log.zero_()

    if not getattr(layer.dt_bias, "_is_hf_initialized", False):
        # Keep sampling DTensor-native in FineWeb/FSDP.
        log_dt = torch.empty_like(
            layer.dt_bias,
            dtype=torch.float32,
        ).uniform_(math.log(1e-3), math.log(1e-1))
        dt = log_dt.exp()
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        layer.dt_bias.copy_(inv_dt)

    layer.A_log._no_weight_decay = True
    layer.dt_bias._no_weight_decay = True


def initialize_state_parameters(layer) -> None:
    """Zero a fresh learnable state without overwriting HF-loaded parameters."""
    if (
        layer.init_state is None
        or getattr(layer.init_state, "_is_hf_initialized", False)
    ):
        return

    nn.init.zeros_(layer.init_state)


@torch.no_grad()
def initialize_gate_parameters(layer) -> None:
    """Initialize near-neutral beta and conservative alpha pathways.

        Beta uses terminal scale 0.1 so its sigmoid starts near 0.5. Alpha uses
        terminal scale 0.05 so input dependence initially perturbs, rather than
        dominates, the log-uniform time-step baseline.
        """
    initialize_gate_projection(
        layer.b_proj,
        output_scale=0.1,
    )

    if layer.use_forget_gate:
        initialize_gate_projection(
            layer.a_proj,
            output_scale=0.05,
        )
        initialize_decay_parameters(layer)


@torch.no_grad()
def initialize_historical_swila_parameters(layer) -> None:
    """Replay the plain ``8dc05f93`` layer's specialized initialization.

    The order below is the executable ``reset_parameters`` at
    ``6fc1fa6921af``: both router inputs, both router convolutions, beta,
    then both router outputs. In particular, beta *was* Kaiming-initialized,
    despite the historical method's contrary docstring. There is no router
    centering, Q/K copying, or calibrated terminal scale in this profile.

    Generic Q/K/V/output parameters stay under their existing constructor/HF
    initialization. The old HF model explicitly repeated this reset after
    ``post_init`` in both model constructors; the caller owns that lifecycle.
    Learned memory is zeroed here as well so meta -> to_empty materialization
    cannot leave it uninitialized. Zeroing consumes no random numbers.

    Unlike the historical helper, loaded-parameter guards are retained: a
    checkpoint's trained tensors must never be replaced by a fresh draw.
    """
    for projection in (layer.pi_q_input_proj, layer.pi_k_input_proj):
        kaiming_init(projection)
    for convolution in (layer.pi_q_conv1d, layer.pi_k_conv1d):
        if convolution is not None:
            kaiming_init_linear_modules(convolution)

    for projection in (layer.b_proj, layer.pi_q_proj, layer.pi_k_proj):
        # The historical profile admits only positive-rank SwiGLU paths.
        for component in (projection.gate, projection.feat, projection.out):
            kaiming_init(component)
    initialize_state_parameters(layer)


def initialize_swila_parameters(layer):
    """Reapply SwiLA initialization after generic model initialization.

        Generic q/k/v/o weights are untouched. This restores centered and
        optionally aligned pi routers, near-0.5 beta, conservative alpha,
        temporal reset-gate biases, and configured router-noise scales.
        Transformer-loaded pi tensors are preserved by the helper-level guards.
        """
    if getattr(layer, "architecture_profile", "current") == "historical_8dc05f93":
        initialize_historical_swila_parameters(layer)
        return
    with torch.no_grad():
        initialize_router_parameters(layer)
        initialize_gate_parameters(layer)
        initialize_state_parameters(layer)
        initialize_temporal_gate_parameters(layer)

        # --- Noise pathways: initialized pi and temporal-gate scales ---
        initialize_router_noise_parameters(layer)
        initialize_temporal_gate_noise_parameters(layer)
