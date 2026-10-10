# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import TritonAscendBackend
from fla.modules.norm.fused_norm_gate import (
    FusedLayerNormGated,
    FusedLayerNormGatedLinear,
    FusedLayerNormSwishGate,
    FusedLayerNormSwishGateLinear,
    FusedRMSNormGated,
    FusedRMSNormGatedLinear,
    FusedRMSNormSwishGate,
    FusedRMSNormSwishGateLinear,
)
from fla.modules.norm.l2norm import L2Norm
from fla.modules.norm.layernorm import (
    GroupNorm,
    GroupNormLinear,
    LayerNorm,
    LayerNormLinear,
    NormParallel,
    RMSNorm,
    RMSNormLinear,
)
from fla.modules.norm.layernorm_gated import LayerNormGated, RMSNormGated
from fla.modules.norm.layernorm_quant import (
    activation_quant,
    bit_linear,
    layer_norm_linear_quant,
    rms_norm_linear_quant,
    weight_quant,
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
    'GroupNorm',
    'GroupNormLinear',
    'L2Norm',
    'LayerNorm',
    'LayerNormGated',
    'LayerNormLinear',
    'NormParallel',
    'RMSNorm',
    'RMSNormGated',
    'RMSNormLinear',
    'activation_quant',
    'bit_linear',
    'layer_norm_linear_quant',
    'rms_norm_linear_quant',
    'weight_quant',
]


if TritonAscendBackend.is_available():
    from fla.modules.norm.triton_ascend import fused_norm_gate, l2norm, layernorm  # noqa: F401
