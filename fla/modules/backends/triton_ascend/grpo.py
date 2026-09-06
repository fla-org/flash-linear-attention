# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Deprecated import path for the grpo Triton-Ascend implementation."""

from importlib import import_module

_RENAMED = {
    'GrpoLossNPU': 'GrpoLoss',
    '_npu_vocab_block_size': '_vocab_block_size',
    'fused_grpo_loss_npu': 'fused_grpo_loss',
}

_PUBLIC_NAMES = (
    'GrpoLossNPU',
    'fused_grpo_loss_npu',
    'grpo_bwd_kernel',
    'grpo_fwd_kernel',
)


def __getattr__(name):
    if name == "__all__":
        return _PUBLIC_NAMES
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("fla.modules.backends.grpo.triton_ascend")
    return getattr(module, _RENAMED.get(name, name))
