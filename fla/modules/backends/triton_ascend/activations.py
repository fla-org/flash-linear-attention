# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Deprecated import path for the activations Triton-Ascend implementation."""

from importlib import import_module

_RENAMED = {
    'PowGLULinearFunctionNPU': 'PowGLULinearFunction',
    'SwiGLULinearFunctionNPU': 'SwiGLULinearFunction',
    'logsigmoid_bwd_npu': 'logsigmoid_bwd',
    'logsigmoid_fwd_npu': 'logsigmoid_fwd',
    'powglu_fwd_npu': 'powglu_fwd',
    'powglu_fwdbwd_npu': 'powglu_fwdbwd',
    'powglu_linear_npu': 'powglu_linear',
    'sigmoid_bwd_npu': 'sigmoid_bwd',
    'sigmoid_fwd_npu': 'sigmoid_fwd',
    'swiglu_fwd_npu': 'swiglu_fwd',
    'swiglu_fwdbwd_npu': 'swiglu_fwdbwd',
    'swiglu_linear_npu': 'swiglu_linear',
    'swish_bwd_npu': 'swish_bwd',
    'swish_fwd_npu': 'swish_fwd',
}

_PUBLIC_NAMES = (
    'PowGLULinearFunctionNPU',
    'SwiGLULinearFunctionNPU',
    'logsigmoid_bwd_kernel',
    'logsigmoid_bwd_npu',
    'logsigmoid_fwd_kernel',
    'logsigmoid_fwd_npu',
    'powglu_fwd_kernel',
    'powglu_fwd_npu',
    'powglu_fwdbwd_kernel',
    'powglu_fwdbwd_npu',
    'powglu_linear_npu',
    'sigmoid_bwd_kernel',
    'sigmoid_bwd_npu',
    'sigmoid_fwd_kernel',
    'sigmoid_fwd_npu',
    'swiglu_fwd_kernel',
    'swiglu_fwd_npu',
    'swiglu_fwdbwd_kernel',
    'swiglu_fwdbwd_npu',
    'swiglu_linear_npu',
    'swish_bwd_kernel',
    'swish_bwd_npu',
    'swish_fwd_kernel',
    'swish_fwd_npu',
)


def __getattr__(name):
    if name == "__all__":
        return _PUBLIC_NAMES
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("fla.modules.backends.activations.triton_ascend")
    return getattr(module, _RENAMED.get(name, name))
