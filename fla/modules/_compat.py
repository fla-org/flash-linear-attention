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

_MODULE_ALIASES = {
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

_PACKAGE_EXPORTS = {
    'convolution': (
        'CausalConv1dFunction', 'CausalConv1dFunctionCP', 'FastCausalConv1dFn', 'ImplicitLongConvolution',
        'LongConvolution', 'PositionalEmbedding', 'ShortConvolution', 'causal_conv1d', 'causal_conv1d_bwd',
        'causal_conv1d_cp', 'causal_conv1d_fwd', 'causal_conv1d_update', 'causal_conv1d_update_states',
        'fast_causal_conv1d_fn', 'fft_conv',
    ),
    'conv': (
        'ImplicitLongConvolution', 'LongConvolution', 'PositionalEmbedding', 'ShortConvolution', 'causal_conv1d', 'fft_conv',
    ),
    'conv.cp': ('CausalConv1dFunctionCP', 'causal_conv1d_cp'),
    'conv.cuda': ('FastCausalConv1dFn', 'causal_conv1d_cuda', 'fast_causal_conv1d_fn'),
    'conv.triton': (
        'CausalConv1dFunction', 'causal_conv1d_bwd', 'causal_conv1d_fwd', 'causal_conv1d_update',
        'causal_conv1d_update_states', 'compute_dh0_triton',
    ),
}


def _warn_deprecated(old: str, new: str):
    warnings.warn(f'{old} will be deprecated in FLA 0.6.1; use {new} instead.', FutureWarning, stacklevel=3)


def _deprecated_getattr(module_name: str, targets: tuple[str, ...], aliases: dict[str, str] | None = None):
    """Resolve former exports without wrapping functions or changing checkpoint class identities."""
    warned = False

    def __getattr__(name):
        nonlocal warned
        if name.startswith('__') and name != '__all__':
            raise AttributeError(f'module {module_name!r} has no attribute {name!r}')
        new_name = (aliases or {}).get(name, name)
        for target in targets:
            source = import_module(target)
            if name == '__all__':
                return getattr(source, '__all__', [key for key in vars(source) if not key.startswith('_')])
            try:
                value = getattr(source, new_name)
                break
            except AttributeError:
                continue
        else:
            raise AttributeError(f'module {module_name!r} has no attribute {name!r}') from None
        if not warned:
            _warn_deprecated(old=f'{module_name}.{name}', new=f'{target}.{new_name}')
            warned = True
        setattr(sys.modules[module_name], name, value)
        return value

    return __getattr__


def _install_legacy_modules():
    for name, targets in _MODULE_ALIASES.items():
        fullname = 'fla.modules.' + name
        if fullname in sys.modules:
            continue
        module = ModuleType(fullname)
        is_package = any(alias.startswith(name + '.') for alias in _MODULE_ALIASES)
        module.__spec__ = ModuleSpec(fullname, loader=None, is_package=is_package)
        module.__package__ = fullname if is_package else fullname.rpartition('.')[0]
        if is_package:
            module.__path__ = []
        if name in _PACKAGE_EXPORTS:
            module.__all__ = list(_PACKAGE_EXPORTS[name])
        module.__getattr__ = _deprecated_getattr(
            module_name=fullname,
            targets=tuple('fla.modules.' + target for target in targets),
        )
        sys.modules[fullname] = module
        parent, _, child = fullname.rpartition('.')
        # conv.causal_conv1d remains a function on the package, as it was in 0.5.2.
        if child not in _PACKAGE_EXPORTS.get(parent.removeprefix('fla.modules.'), ()):
            setattr(sys.modules[parent], child, module)
