# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Lazy recurrent SwiLA operators and the sequential Torch reference."""

from importlib import import_module

_EXPORTS = {
    "swila_recurrence": "fla.ops.swila.naive",
    "fused_recurrent_swila": "fla.ops.swila.fused_recurrent",
    "fused_recurrent_swila_temporal": "fla.ops.swila.fused_recurrent_temporal",
    "fused_recurrent_gated_swila": "fla.ops.swila.fused_recurrent_gated",
    "fused_recurrent_gated_swila_temporal": "fla.ops.swila.fused_recurrent_gated_temporal",
    "compute_swila_mi_loss": "fla.ops.swila.router_stats",
    "fused_swila_log_decay_gate": "fla.ops.swila.alpha_gate",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
