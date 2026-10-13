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
from fla.modules import activations, l2norm


def test_top_level_exports_layers_and_non_config_models():
    expected_exports = [*layers.__all__, *[name for name in models.__all__ if not name.endswith("Config")]]

    assert fla.__all__ == expected_exports

    for name in expected_exports:
        source = layers if hasattr(layers, name) else models
        assert getattr(fla, name) is getattr(source, name)

    config_exports = [name for name in models.__all__ if name.endswith("Config")]
    assert not set(config_exports).intersection(fla.__all__)
    assert not any(name in fla.__dict__ for name in config_exports)


@pytest.mark.parametrize('disabled', ['0', '1'], ids=['dispatch-enabled', 'dispatch-disabled'])
def test_public_imports_preserve_callables(run_python, disabled):
    run_python(
        """
        from fla.ops.kda import chunk_kda
        from fla.backends import dispatch
        from fla.modules import ShortConvolution

        assert callable(chunk_kda)
        assert callable(dispatch)
        assert callable(ShortConvolution)
        """,
        FLA_DISABLE_BACKEND_DISPATCH=disabled,
    )


@pytest.mark.parametrize(
    ('owner', 'name', 'defaults'),
    [
        ('activations', 'powglu', {'power': 3.0}),
        ('rotary', 'rotary_embedding', {'seqlen_offsets': 0, 'interleaved': False, 'inplace': False}),
        ('causal_conv1d', 'causal_conv1d_fwd', {'chunk_size': 64, 'layout_fallback': False, 'output_final_state': False}),
        ('causal_conv1d.backends.cuda', 'causal_conv1d_cuda', {'activation': None, 'output_final_state': False}),
        ('grpo', 'fused_grpo_loss', {'beta': 0.1, 'save_kl': False, 'inplace': False}),
        ('fused_cross_entropy', 'cross_entropy_loss', {'ignore_index': -100, 'process_group': None}),
        ('fused_linear_cross_entropy', 'fused_linear_cross_entropy_loss', {'num_chunks': 8, 'reduction': 'mean'}),
        ('fused_kl_div', 'fused_kl_div_loss', {'reduction': 'batchmean', 'accumulate_grad_in_fp32': True}),
        ('layernorm', 'layer_norm', {'eps': 1e-5, 'prenorm': False}),
        ('l2norm', 'l2norm', {'eps': 1e-6, 'output_dtype': None}),
        ('fused_norm_gate', 'layer_norm_gated', {'activation': 'swish', 'eps': 1e-6}),
    ],
    ids=[
        'activations', 'rotary', 'conv', 'conv-cuda', 'grpo', 'cross-entropy', 'linear-cross-entropy',
        'kl-div', 'layernorm', 'l2norm', 'norm-gate',
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
        ('convolution', 'causal_conv1d', ('ShortConvolution', 'causal_conv1d')),
        ('convolution', 'long_conv', ('LongConvolution', 'ImplicitLongConvolution', 'PositionalEmbedding', 'fft_conv')),
    ],
    ids=['causal-conv', 'long-conv'],
)
def test_legacy_imports_preserve_symbol_identity(legacy, current, names):
    old_package = importlib.import_module(f'fla.modules.{legacy}')
    new_package = importlib.import_module(f'fla.modules.{current}')
    for name in names:
        assert getattr(old_package, name) is getattr(new_package, name)


def test_public_function_aliases():
    for name in ('sigmoid', 'logsigmoid', 'swish', 'sqrelu'):
        assert activations.ACT2FN[name] is getattr(activations, name)
    assert activations.ACT2FN['silu'] is activations.swish
    assert activations.ACT2FN['gelu'] is activations.fast_gelu_impl
    assert l2norm.l2_norm is l2norm.l2norm
    bitlinear = importlib.import_module('fla.modules.fused_bitlinear')
    layernorm_quant = importlib.import_module('fla.modules.norm.layernorm_quant')
    assert bitlinear.layer_norm_fwd_quant is layernorm_quant.layer_norm_quant_fwd
    assert bitlinear.layer_norm_bwd is layernorm_quant.layer_norm_quant_bwd
    assert bitlinear.LayerNormLinearQuantFunction is layernorm_quant.LayerNormLinearQuantFunction
    assert bitlinear.LayerNormLinearQuantFn is bitlinear.LayerNormLinearQuantFunction
    assert bitlinear.layer_norm_linear_quant_fn is bitlinear.layer_norm_linear_quant


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
        ('convolution', 'LongConvolution', {'hidden_size': 4, 'max_len': 8}, ('filter',)),
        ('fused_bitlinear', 'BitLinear', {'in_features': 4, 'out_features': 4}, ('weight', 'norm.weight')),
        ('fused_bitlinear', 'FusedBitLinear', {'in_features': 4, 'out_features': 4}, ('weight', 'norm.weight')),
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
        'bitlinear',
        'fused-bitlinear',
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

        for name, canonical_path in zip(expected_symbols, canonical_paths):
            canonical = importlib.import_module(canonical_path)
            legacy = importlib.import_module('fla.modules.' + name)
            assert importlib.import_module('fla.modules.' + name) is legacy
            if name == 'l2norm':
                from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
                assert l2norm_fwd is canonical.l2norm_fwd
                assert l2norm_bwd is canonical.l2norm_bwd
            assert set(legacy.__all__) == set(canonical.__all__) == set(expected_symbols[name])
            for symbol in expected_symbols[name]:
                assert getattr(legacy, symbol) is getattr(canonical, symbol), (name, symbol)

        from fla.modules.causal_conv1d import causal_conv1d
        importlib.import_module('fla.modules.causal_conv1d.ops')
        from fla.modules.causal_conv1d import causal_conv1d as after_implementation_import
        from fla.modules.convolution import causal_conv1d as legacy
        assert callable(causal_conv1d)
        assert after_implementation_import is causal_conv1d
        assert causal_conv1d is legacy
        """,
        FLA_DISABLE_BACKEND_DISPATCH=disabled,
    )
