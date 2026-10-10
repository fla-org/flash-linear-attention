# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import TritonAscendBackend
from fla.modules import _deprecated_getattr
from fla.modules.activations.ops import (
    ACT2FN,
    elu_p1,
    fast_gelu_impl,
    logsigmoid,
    powglu,
    powglu_linear,
    sigmoid,
    sigmoidglu,
    sigmoidglu_linear,
    sqrelu,
    swiglu,
    swiglu_linear,
    swish,
)

__all__ = [
    'ACT2FN',
    'elu_p1',
    'fast_gelu_impl',
    'logsigmoid',
    'powglu',
    'powglu_linear',
    'sigmoid',
    'sigmoidglu',
    'sigmoidglu_linear',
    'sqrelu',
    'swiglu',
    'swiglu_linear',
    'swish',
]


if TritonAscendBackend.is_available():
    from fla.modules.activations import triton_ascend  # noqa: F401


__getattr__ = _deprecated_getattr(module_name=__name__, targets=('fla.modules.activations.ops',))
