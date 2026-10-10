# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# ruff: noqa: E402
import sys
import warnings
from importlib import import_module
from importlib.machinery import ModuleSpec
from types import ModuleType


def _deprecated_getattr(module_name, targets, aliases=None):
    warned = False

    def resolve(name):
        nonlocal warned
        if name == '__all__':
            return sorted({
                key for target in targets
                for key in getattr(import_module(target), '__all__', vars(import_module(target)))
                if not key.startswith('_')
            })
        if not name.startswith('__'):
            new_name = (aliases or {}).get(name, name)
            for target in targets:
                source = import_module(target)
                if hasattr(source, new_name):
                    if not warned:
                        warnings.warn(
                            f'{module_name}.{name} will be deprecated in FLA 0.6.1; use {target}.{new_name} instead.',
                            FutureWarning,
                            stacklevel=2,
                        )
                        warned = True
                    value = getattr(source, new_name)
                    setattr(sys.modules[module_name], name, value)
                    return value
        raise AttributeError(f'module {module_name!r} has no attribute {name!r}')

    return resolve


_DEPRECATED_MODULES = {
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
    'fused_norm_gate': ('norm.fused_norm_gate',),
    'l2norm': ('norm.l2norm',),
    'layernorm': ('norm.layernorm',),
    'layernorm_gated': ('norm.layernorm_gated',),
}

for _name, _targets in _DEPRECATED_MODULES.items():
    _fullname = f'{__name__}.{_name}'
    _module = ModuleType(_fullname)
    _is_package = _name in {'conv', 'conv.cp', 'conv.cuda', 'conv.triton'}
    _module.__spec__ = ModuleSpec(_fullname, loader=None, is_package=_is_package)
    _module.__package__ = _fullname if _is_package else _fullname.rpartition('.')[0]
    if _is_package:
        _module.__path__ = []
    _module.__getattr__ = _deprecated_getattr(_fullname, tuple(f'{__name__}.{target}' for target in _targets))
    sys.modules[_fullname] = _module
    # conv.causal_conv1d is a function on the old package.
    if _name != 'conv.causal_conv1d':
        _parent, _, _child = _fullname.rpartition('.')
        setattr(sys.modules[_parent], _child, _module)


# legacy aliases must exist before these imports resolve their dependencies.
# autopep8: off
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

# autopep8: on

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
