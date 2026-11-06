# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import TritonAscendBackend
from fla.ops.norm.fused_norm_gate import layer_norm_gated, rms_norm_gated
from fla.ops.norm.l2norm import l2_norm, l2norm
from fla.ops.norm.layernorm import group_norm, layer_norm, rms_norm

__all__ = ['group_norm', 'l2_norm', 'l2norm', 'layer_norm', 'layer_norm_gated', 'rms_norm', 'rms_norm_gated']

if TritonAscendBackend.is_available():
    from fla.ops.norm import triton_ascend  # noqa: F401
