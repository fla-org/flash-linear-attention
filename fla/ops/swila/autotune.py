# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Pure-Python autotune policies shared by SwiLA Triton kernels."""

from __future__ import annotations

from typing import Any

import torch

# Temporal kernels keep a state tile [BV, BE, BK] and two transition tiles
# [BV, BE, BE] live.  The budget retains the production E=4, Dk=Dv=64,
# BV=64 candidate while excluding the substantially larger high-expert tiles;
# BV=64, BE=8, BK=64 produced an illegal access during H100 autotuning.
_MAX_TEMPORAL_TILE_WORKING_ELEMENTS = 18_432
_MIN_TEMPORAL_VALUE_TILE = 8

# The historical H100 D64/E4 sweeps consistently favored one-warp forward
# launches near BV=16.  The current Equation-41 kernels are not identical, so
# retain neighboring value tiles plus one stage/warp escape hatch instead of
# pinning one old winner.  This table is used only for the exact FineWeb SM90
# specialization; every other shape/device keeps the full generic sweep.
_FINEWEB_SM90_FORWARD_CONFIGS = frozenset(
    (block_v, num_warps, num_stages)
    for block_v in (8, 16, 32)
    for num_warps, num_stages in ((1, 2), (1, 3), (2, 2))
)


def _is_exact_nvidia_sm90_tensor(tensor: Any) -> bool:
    """Return whether ``tensor`` is on its own exact NVIDIA SM90 device."""
    return bool(
        tensor is not None
        and getattr(tensor, "is_cuda", False)
        and torch.version.hip is None
        and torch.cuda.get_device_capability(tensor.device) == (9, 0)
    )


def _is_fineweb_sm90_forward(
    arguments: dict[str, Any],
    *,
    temporal: bool,
) -> bool:
    """Recognize the dense BF16 D64/E4 key-decay FineWeb forward."""
    q = arguments.get("q")
    log_decay = arguments.get("log_decay")
    if (
        not _is_exact_nvidia_sm90_tensor(q)
        or getattr(q, "dtype", None) != torch.bfloat16
        or log_decay is None
        or getattr(log_decay, "dtype", None) != torch.float32
        or not getattr(log_decay, "is_cuda", False)
        or getattr(log_decay, "device", None) != getattr(q, "device", None)
        or (
            hasattr(log_decay, "is_contiguous")
            and not log_decay.is_contiguous()
        )
        or int(arguments.get("D_k", -1)) != 64
        or int(arguments.get("D_v", -1)) != 64
        or int(arguments.get("E", -1)) != 4
        or int(arguments.get("T", -1)) != 4096
        or bool(arguments.get("IS_VARLEN", False))
    ):
        return False

    if temporal:
        return bool(
            arguments.get("IDENTITY_TRANSITIONS", False)
            and arguments.get("USE_GATE", False)
            and arguments.get("USE_LOG_DECAY", False)
        )
    return True


def _fineweb_sm90_forward_subset(configs: list[Any]) -> list[Any]:
    selected = [
        config
        for config in configs
        if (
            int(config.kwargs["BV"]),
            int(config.num_warps),
            int(config.num_stages),
        )
        in _FINEWEB_SM90_FORWARD_CONFIGS
    ]
    if selected:
        return selected
    raise RuntimeError(
        "SM90 D64/E4 forward autotuning requires BV in {8,16,32} with "
        "(warps, stages) in {(1,2),(1,3),(2,2)}"
    )


def prune_gated_forward_configs(
    configs: list[Any],
    named_args: dict[str, Any],
    **kwargs: Any,
) -> list[Any]:
    """Use a small current-semantic table for dense SM90 D64/E4 gated SwiLA."""
    arguments = dict(named_args)
    arguments.update(kwargs)
    if not _is_fineweb_sm90_forward(arguments, temporal=False):
        return configs
    return _fineweb_sm90_forward_subset(configs)


def prune_gated_backward_configs(
    configs: list[Any],
    named_args: dict[str, Any],
    **kwargs: Any,
) -> list[Any]:
    """Exclude the diagnosed SM90 D64/E8 one-warp backward compiler tile."""
    if not configs:
        raise ValueError("gated backward autotuning requires at least one config")
    arguments = dict(named_args)
    arguments.update(kwargs)
    if (
        not _is_exact_nvidia_sm90_tensor(arguments.get("q"))
        or tuple(int(arguments.get(name, -1)) for name in ("D_k", "D_v", "E", "BK", "BE"))
        != (64, 64, 8, 64, 8)
    ):
        return configs

    # This 32,768-element FP32 state tile generated 58 MB of LLVM IR and
    # spent hours in LLVM SLP vectorization. Keep every other launch unchanged,
    # including neighboring shapes; this is not a general register budget.
    pruned = [config for config in configs
              if not (int(config.kwargs["BV"]) == 64 and int(config.num_warps) == 1)]
    if not pruned:
        raise RuntimeError("no gated backward config remains after SM90 D64/E8 compiler pruning")
    return pruned


def prune_temporal_forward_configs(
    configs: list[Any],
    named_args: dict[str, Any],
    **kwargs: Any,
) -> list[Any]:
    """Safety-prune temporal forward, then narrow the FineWeb SM90 case."""
    pruned = prune_temporal_value_configs(configs, named_args, **kwargs)
    arguments = dict(named_args)
    arguments.update(kwargs)
    if not _is_fineweb_sm90_forward(arguments, temporal=True):
        return pruned
    return _fineweb_sm90_forward_subset(pruned)


def prune_temporal_value_configs(
    configs: list[Any],
    named_args: dict[str, Any],
    **kwargs: Any,
) -> list[Any]:
    """Remove temporal value tiles that are invalid or too register-heavy.

    Triton's autotuner benchmarks every retained configuration, so one unsafe
    candidate can invalidate the CUDA context even when it would never be the
    fastest choice.  A value tile should not exceed the actual value dimension
    (apart from the minimum supported tile), and the approximate live working
    set must stay bounded.
    """
    if not configs:
        raise ValueError("temporal autotuning requires at least one config")

    arguments = dict(named_args)
    arguments.update(kwargs)
    try:
        d_v = int(arguments["D_v"])
        block_e = int(arguments["BE"])
        block_k = int(arguments["BK"])
    except KeyError as exc:
        raise KeyError(
            f"missing temporal autotune dimension {exc.args[0]!r}"
        ) from exc

    max_value_tile = max(_MIN_TEMPORAL_VALUE_TILE, d_v)

    def is_safe(config: Any) -> bool:
        block_v = int(config.kwargs["BV"])
        working_elements = block_v * block_e * (block_k + 2 * block_e)
        return (
            block_v <= max_value_tile
            and working_elements <= _MAX_TEMPORAL_TILE_WORKING_ELEMENTS
        )

    pruned = [config for config in configs if is_safe(config)]
    if pruned:
        return pruned

    raise RuntimeError(
        "no safe temporal Triton value tile for "
        f"D_v={d_v}, BE={block_e}, BK={block_k}; "
        "reduce the key dimension or number of experts"
    )


def prune_temporal_backward_configs(
    configs: list[Any],
    named_args: dict[str, Any],
    **kwargs: Any,
) -> list[Any]:
    """Prune temporal backward configs, fixing the BE=16 safe specialization.

    The public high-expert path splits Q-temporal reverse and forward replay
    out of the large reverse kernel and removes the fixed identity transition
    tiles.  Even with that smaller live set, keeping its launch deterministic
    avoids reintroducing the H100 compiler/register-pressure failure through a
    future autotune winner.  Lower-expert calls retain the ordinary safe sweep.
    """
    pruned = prune_temporal_value_configs(configs, named_args, **kwargs)
    arguments = dict(named_args)
    arguments.update(kwargs)
    if bool(arguments.get("PRODUCTION_FAST", False)):
        production = [
            config
            for config in pruned
            if int(config.kwargs["BV"]) in (16, 32)
            and int(config.num_warps) == 1
            and int(config.num_stages) == 2
        ]
        if production:
            return production
        raise RuntimeError(
            "production temporal backward requires BV in {16, 32}, "
            "num_warps=1, num_stages=2"
        )

    if not bool(arguments.get("RESOURCE_SAFE", False)):
        return pruned

    fixed = [
        config
        for config in pruned
        if int(config.kwargs["BV"]) == 8
        and int(config.num_warps) == 1
        and int(config.num_stages) == 2
    ]
    if fixed:
        return fixed
    raise RuntimeError(
        "resource-safe temporal backward requires BV=8, "
        "num_warps=1, num_stages=2"
    )
