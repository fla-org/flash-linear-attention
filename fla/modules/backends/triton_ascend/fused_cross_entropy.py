# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Deprecated import path for the fused_cross_entropy Triton-Ascend implementation."""

from importlib import import_module

_RENAMED = {
    'CrossEntropyLossFunctionNPU': 'CrossEntropyLossFunction',
    '_npu_block_size': '_block_size',
    'cross_entropy_loss_npu': 'cross_entropy_loss',
    'fused_cross_entropy_backward_npu': 'fused_cross_entropy_backward',
    'fused_cross_entropy_forward_npu': 'fused_cross_entropy_forward',
}

_PUBLIC_NAMES = (
    'CrossEntropyLossFunctionNPU',
    'cross_entropy_bwd_kernel',
    'cross_entropy_fwd_kernel',
    'cross_entropy_loss_npu',
    'fused_cross_entropy_backward_npu',
    'fused_cross_entropy_forward_npu',
)


def __getattr__(name):
    if name == "__all__":
        return _PUBLIC_NAMES
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("fla.modules.backends.fused_cross_entropy.triton_ascend")
    return getattr(module, _RENAMED.get(name, name))
