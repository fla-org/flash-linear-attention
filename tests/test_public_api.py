# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib
import inspect
import pickle

import pytest
import torch

import fla
from fla import layers, models, modules
from fla.modules import l2norm


def test_top_level_exports_layers_and_non_config_models():
    expected_exports = [
        *layers.__all__,
        *[name for name in models.__all__ if not name.endswith("Config")],
    ]

    assert fla.__all__ == expected_exports

    for name in expected_exports:
        source = layers if hasattr(layers, name) else models
        assert getattr(fla, name) is getattr(source, name)

    config_exports = [name for name in models.__all__ if name.endswith("Config")]
    assert not set(config_exports).intersection(fla.__all__)
    assert not any(name in fla.__dict__ for name in config_exports)


@pytest.mark.parametrize(
    ('owner', 'name', 'defaults'),
    [
        (
            'causal_conv.causal_conv',
            'causal_conv1d_fwd',
            {'chunk_size': 64, 'layout_fallback': False, 'output_final_state': False},
        ),
        ('layernorm', 'layer_norm', {'eps': 1e-5, 'prenorm': False}),
        ('l2norm', 'l2norm', {'eps': 1e-6, 'output_dtype': None}),
        ('fused_norm_gate', 'layer_norm_gated', {'activation': 'swish', 'eps': 1e-6}),
    ],
    ids=[
        'conv',
        'layernorm',
        'l2norm',
        'norm-gate',
    ],
)
def test_public_call_defaults(owner, name, defaults):
    function = getattr(importlib.import_module(f'fla.modules.{owner}'), name)
    signature = inspect.signature(function)
    for parameter, expected in defaults.items():
        assert signature.parameters[parameter].default == expected


@pytest.mark.parametrize(
    ('legacy', 'current', 'names'),
    [
        ('convolution', 'conv', ('ShortConvolution', 'LongConvolution', 'ImplicitLongConvolution')),
        ('convolution', 'causal_conv', ('ShortConvolution', 'causal_conv1d')),
        ('convolution', 'long_conv', ('LongConvolution', 'ImplicitLongConvolution', 'PositionalEmbedding', 'fft_conv')),
    ],
    ids=['convolution', 'causal-conv', 'long-conv'],
)
def test_legacy_imports_preserve_symbol_identity(legacy, current, names):
    old_package = importlib.import_module(f'fla.modules.{legacy}')
    new_package = importlib.import_module(f'fla.modules.{current}')
    for name in names:
        assert getattr(old_package, name) is getattr(new_package, name)


def test_public_function_aliases():
    assert l2norm.l2_norm is l2norm.l2norm


@pytest.mark.parametrize(
    ('legacy', 'name', 'kwargs', 'state_keys'),
    [
        ('layernorm', 'LayerNorm', {'hidden_size': 4, 'bias': True}, ('weight', 'bias')),
        ('layernorm', 'RMSNorm', {'hidden_size': 4}, ('weight',)),
        ('layernorm_gated', 'LayerNormGated', {'hidden_size': 4}, ('weight', 'bias')),
        ('layernorm_gated', 'RMSNormGated', {'hidden_size': 4}, ('weight',)),
        ('fused_norm_gate', 'FusedLayerNormGated', {'hidden_size': 4, 'bias': True}, ('weight', 'bias')),
        ('fused_norm_gate', 'FusedRMSNormGated', {'hidden_size': 4}, ('weight',)),
        ('l2norm', 'L2Norm', {}, ()),
        ('convolution', 'ShortConvolution', {'hidden_size': 4, 'kernel_size': 3, 'bias': True}, ('weight', 'bias')),
        ('conv.long_conv', 'LongConvolution', {'hidden_size': 4, 'max_len': 8}, ('filter',)),
    ],
    ids=[
        'layernorm',
        'rmsnorm',
        'layernorm-gated',
        'rmsnorm-gated',
        'fused-layernorm-gated',
        'fused-rmsnorm-gated',
        'l2norm',
        'convolution',
        'long-convolution',
    ],
)
def test_legacy_module_pickle_and_state_dict(monkeypatch, legacy, name, kwargs, state_keys):
    torch.manual_seed(42)
    module_class = getattr(importlib.import_module(f'fla.modules.{legacy}'), name)
    module = module_class(**kwargs)
    state = module.state_dict()
    assert set(state) == set(state_keys)
    if hasattr(modules, name):
        assert getattr(modules, name) is module_class

    paths = [f'fla.modules.{legacy}']
    if legacy in {'fused_norm_gate', 'l2norm', 'layernorm', 'layernorm_gated'}:
        paths.append(f'fla.modules.norm.{legacy}')
    if name == 'ShortConvolution':
        paths.extend(['fla.modules.conv', 'fla.modules.conv.module', 'fla.modules.conv.short_conv'])
    for path in paths:
        with monkeypatch.context() as patch:
            patch.setattr(module_class, '__module__', path)
            checkpoint = pickle.dumps(module)
        restored = pickle.loads(checkpoint)

        assert type(restored) is module_class
        assert repr(restored) == repr(module)
        assert set(restored.state_dict()) == set(state_keys)
        for key in state:
            torch.testing.assert_close(restored.state_dict()[key], state[key])
    module_class(**kwargs).load_state_dict(state, strict=True)


@pytest.mark.parametrize('disabled', ['0', '1'], ids=['dispatch-enabled', 'dispatch-disabled'])
def test_normalization_imports_preserve_public_exports(run_python, disabled):
    run_python(
        """
        import importlib
        import warnings

        expected_symbols = {
            'layernorm': (
                'GroupNorm', 'GroupNormLinear', 'GroupNormRef', 'LayerNorm', 'LayerNormLinear',
                'NormParallel', 'RMSNorm', 'RMSNormLinear', 'group_norm', 'group_norm_linear',
                'group_norm_ref', 'layer_norm', 'layer_norm_bwd', 'layer_norm_fwd', 'layer_norm_linear',
                'layer_norm_ref', 'rms_norm', 'rms_norm_linear', 'rms_norm_ref',
            ),
            'l2norm': ('L2Norm', 'l2norm', 'l2_norm', 'l2norm_fwd', 'l2norm_bwd'),
            'fused_norm_gate': (
                'FusedLayerNormGated', 'FusedLayerNormGatedLinear', 'FusedLayerNormSwishGate',
                'FusedLayerNormSwishGateLinear', 'FusedRMSNormGated', 'FusedRMSNormGatedLinear',
                'FusedRMSNormSwishGate', 'FusedRMSNormSwishGateLinear', 'layer_norm_gated',
                'layer_norm_gated_bwd', 'layer_norm_gated_fwd', 'layer_norm_swish_gate_linear',
                'rms_norm_gated', 'rms_norm_swish_gate_linear',
            ),
            'layernorm_gated': (
                'LayerNormGated', 'RMSNormGated', 'layernorm_fn', 'rmsnorm_fn', 'rms_norm_ref',
                'layer_norm_fwd', 'layer_norm_bwd',
            ),
        }
        canonical_paths = tuple('fla.modules.norm.' + name for name in expected_symbols)

        def norm_warnings(records):
            return [
                warning for warning in records
                if issubclass(warning.category, FutureWarning)
                and any(path in str(warning.message) for path in canonical_paths)
            ]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', FutureWarning)
            from fla import layers, modules
            from fla.layers.gla import GatedLinearAttention
            from fla.layers.mamba2 import Mamba2
            from fla.modules import L2Norm, RMSNorm, RotaryEmbedding
            from fla.modules import norm, rotary
            for canonical_path in canonical_paths:
                importlib.import_module(canonical_path)

        assert layers.GatedLinearAttention is GatedLinearAttention
        assert layers.Mamba2 is Mamba2
        assert modules.L2Norm is L2Norm is norm.L2Norm
        assert modules.RMSNorm is RMSNorm is norm.RMSNorm
        assert modules.RotaryEmbedding is RotaryEmbedding is rotary.RotaryEmbedding
        assert not norm_warnings(caught), [str(w.message) for w in caught]

        for name, canonical_path in zip(expected_symbols, canonical_paths):
            canonical = importlib.import_module(canonical_path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', FutureWarning)
                legacy = importlib.import_module('fla.modules.' + name)
                assert importlib.import_module('fla.modules.' + name) is legacy
                if name == 'l2norm':
                    from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
                    assert l2norm_fwd is canonical.l2norm_fwd
                    assert l2norm_bwd is canonical.l2norm_bwd
            assert not norm_warnings(caught), (name, [str(w.message) for w in caught])
            assert set(legacy.__all__) == set(canonical.__all__) == set(expected_symbols[name])
            for symbol in expected_symbols[name]:
                assert getattr(legacy, symbol) is getattr(canonical, symbol), (name, symbol)

        from fla.modules.conv import causal_conv1d
        importlib.import_module('fla.modules.conv.causal_conv1d')
        from fla.modules.conv import causal_conv1d as after_legacy_import
        assert callable(causal_conv1d)
        assert after_legacy_import is causal_conv1d
        """,
        FLA_DISABLE_BACKEND_DISPATCH=disabled,
    )
