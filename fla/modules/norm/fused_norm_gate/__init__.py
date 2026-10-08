# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.modules.norm.fused_norm_gate.module import (
    FusedLayerNormGated,
    FusedLayerNormGatedLinear,
    FusedLayerNormSwishGate,
    FusedLayerNormSwishGateLinear,
    FusedRMSNormGated,
    FusedRMSNormGatedLinear,
    FusedRMSNormSwishGate,
    FusedRMSNormSwishGateLinear,
)
from fla.modules.norm.fused_norm_gate.ops import (
    LayerNormGatedFunction,
    LayerNormGatedLinearFunction,
    layer_norm_gated,
    layer_norm_gated_bwd,
    layer_norm_gated_bwd_kernel,
    layer_norm_gated_bwd_kernel_row,
    layer_norm_gated_fwd,
    layer_norm_gated_fwd_kernel,
    layer_norm_gated_fwd_kernel_row,
    layer_norm_swish_gate_linear,
    rms_norm_gated,
    rms_norm_swish_gate_linear,
)

__all__ = [
    'FusedLayerNormGated',
    'FusedLayerNormGatedLinear',
    'FusedLayerNormSwishGate',
    'FusedLayerNormSwishGateLinear',
    'FusedRMSNormGated',
    'FusedRMSNormGatedLinear',
    'FusedRMSNormSwishGate',
    'FusedRMSNormSwishGateLinear',
    'LayerNormGatedFunction',
    'LayerNormGatedLinearFunction',
    'layer_norm_gated',
    'layer_norm_gated_bwd',
    'layer_norm_gated_bwd_kernel',
    'layer_norm_gated_bwd_kernel_row',
    'layer_norm_gated_fwd',
    'layer_norm_gated_fwd_kernel',
    'layer_norm_gated_fwd_kernel_row',
    'layer_norm_swish_gate_linear',
    'rms_norm_gated',
    'rms_norm_swish_gate_linear',
]
