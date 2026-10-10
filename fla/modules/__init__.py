# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import sys
import warnings
from importlib import import_module
from importlib.machinery import ModuleSpec
from types import ModuleType

from fla.modules.causal_conv1d import ShortConvolution
from fla.modules.fused_bitlinear import BitLinear, FusedBitLinear
from fla.modules.fused_cross_entropy import FusedCrossEntropyLoss
from fla.modules.fused_kl_div import FusedKLDivLoss
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.modules.long_conv import ImplicitLongConvolution, LongConvolution
from fla.modules.mlp import GatedMLP
from fla.modules.norm import (
    FusedLayerNormGated,
    FusedLayerNormSwishGate,
    FusedLayerNormSwishGateLinear,
    FusedRMSNormGated,
    FusedRMSNormSwishGate,
    FusedRMSNormSwishGateLinear,
    GroupNorm,
    GroupNormLinear,
    L2Norm,
    LayerNorm,
    LayerNormLinear,
    RMSNorm,
    RMSNormLinear,
)
from fla.modules.rotary import RotaryEmbedding
from fla.modules.token_shift import TokenShift

_MODULE_ALIASES = {
    'fused_norm_gate': ('norm.fused_norm_gate',),
    'l2norm': ('norm.l2norm',),
    'layernorm': ('norm.layernorm',),
    'layernorm_gated': ('norm.layernorm_gated',),
    'convolution': ('causal_conv1d', 'long_conv', 'causal_conv1d.cp', 'causal_conv1d.backends.cuda'),
    'conv': ('causal_conv1d', 'long_conv'),
    'conv.causal_conv1d': ('causal_conv1d.ops',),
    'conv.short_conv': ('causal_conv1d.ops',),
    'conv.long_conv': ('long_conv',),
    'conv.cp': ('causal_conv1d.cp',),
    'conv.cp.ops': ('causal_conv1d.cp',),
    'conv.cuda': ('causal_conv1d.backends.cuda',),
    'conv.cuda.ops': ('causal_conv1d.backends.cuda',),
    'conv.triton': ('causal_conv1d.ops',),
    'conv.triton.ops': ('causal_conv1d.ops',),
    'conv.triton.kernels': ('causal_conv1d.ops',),
}

for _old, _targets in _MODULE_ALIASES.items():
    _fullname = f'{__name__}.{_old}'
    if len(_targets) == 1:
        _module = import_module(f'{__name__}.{_targets[0]}')
    else:
        _module = ModuleType(_fullname)
        _module.__spec__ = ModuleSpec(_fullname, loader=None, is_package=True)
        _module.__path__ = []
        for _target in _targets:
            _source = import_module(f'{__name__}.{_target}')
            for _symbol in getattr(_source, '__all__', vars(_source)):
                if not _symbol.startswith('_'):
                    vars(_module).setdefault(_symbol, getattr(_source, _symbol))
        _module.__all__ = [name for name in vars(_module) if not name.startswith('_')]
    sys.modules[_fullname] = _module
    # conv.causal_conv1d is a function on the old package.
    if _old != 'conv.causal_conv1d':
        _parent, _, _child = _fullname.rpartition('.')
        setattr(sys.modules[_parent], _child, _module)

# defer GRPO's compile policy until it is explicitly imported
for _name in ('activations', 'rotary', 'fused_cross_entropy', 'fused_kl_div', 'fused_linear_cross_entropy'):
    _module = import_module(f'{__name__}.{_name}')
    for _symbol, _value in vars(import_module(f'{__name__}.{_name}.ops')).items():
        if not _symbol.startswith('_'):
            vars(_module).setdefault(_symbol, _value)

_SYMBOL_ALIASES = {
    'fused_bitlinear': {
        'layer_norm_fwd_quant': 'norm.layernorm_quant.layer_norm_quant_fwd',
        'layer_norm_bwd': 'norm.layernorm_quant.layer_norm_quant_bwd',
        'LayerNormLinearQuantFn': 'LayerNormLinearQuantFunction',
        'layer_norm_linear_quant_fn': 'layer_norm_linear_quant',
    },
    'fused_cross_entropy': {
        'fused_cross_entropy_forward': 'cross_entropy_fwd',
        'CrossEntropyLossFunction': 'FusedCrossEntropyFunction',
    },
    'fused_kl_div': {'fused_kl_div_forward': 'fused_kl_div_fwd', 'fused_kl_div_backward': 'fused_kl_div_bwd'},
    'fused_linear_cross_entropy': {
        'fused_linear_cross_entropy_forward': 'fused_linear_cross_entropy_fwd',
        'fused_linear_cross_entropy_backward': 'fused_linear_cross_entropy_bwd',
    },
}
for _name, _aliases in _SYMBOL_ALIASES.items():
    _module = sys.modules[f'{__name__}.{_name}']
    for _old, _new in _aliases.items():
        _target, _, _symbol = _new.rpartition('.')
        _source = import_module(f'{__name__}.{_target}') if _target else _module
        setattr(_module, _old, getattr(_source, _symbol))

warnings.warn(
    'Legacy fla.modules imports will be deprecated in FLA 0.6.1. '
    'Use fla.modules.norm, fla.modules.causal_conv1d, fla.modules.long_conv '
    'or the owning package.ops for moved symbols.',
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    'BitLinear',
    'FusedBitLinear',
    'FusedCrossEntropyLoss',
    'FusedKLDivLoss',
    'FusedLayerNormGated',
    'FusedLayerNormSwishGate',
    'FusedLayerNormSwishGateLinear',
    'FusedLinearCrossEntropyLoss',
    'FusedRMSNormGated',
    'FusedRMSNormSwishGate',
    'FusedRMSNormSwishGateLinear',
    'GatedMLP',
    'GroupNorm',
    'GroupNormLinear',
    'ImplicitLongConvolution',
    'L2Norm',
    'LayerNorm',
    'LayerNormLinear',
    'LongConvolution',
    'RMSNorm',
    'RMSNormLinear',
    'RotaryEmbedding',
    'ShortConvolution',
    'TokenShift',
]
