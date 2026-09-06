# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest
import torch

import fla.backends as backends
import fla.utils as utils

ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINTS = {
    'activations': (
        'sigmoid_fwd', 'sigmoid_bwd', 'logsigmoid_fwd', 'logsigmoid_bwd', 'swish_fwd', 'swish_bwd',
        'swiglu_fwd', 'swiglu_fwdbwd', 'swiglu_linear', 'powglu_fwd', 'powglu_fwdbwd', 'powglu_linear',
    ),
    'causal_conv1d': (
        'causal_conv1d_fwd', 'causal_conv1d_bwd', 'compute_dh0_triton', 'causal_conv1d_update_states', 'causal_conv1d_update',
    ),
    'fused_cross_entropy': ('cross_entropy_loss',),
    'fused_kl_div': ('fused_kl_div_forward', 'fused_kl_div_backward'),
    'fused_linear_cross_entropy': (
        'logsumexp_fwd', 'fused_linear_cross_entropy_forward', 'fused_linear_cross_entropy_backward',
    ),
    'fused_norm_gate': ('layer_norm_gated_fwd', 'layer_norm_gated_bwd'),
    'grpo': ('fused_grpo_loss',),
    'l2norm': ('l2norm_fwd', 'l2norm_bwd'),
    'layernorm': ('layer_norm_fwd', 'layer_norm_bwd'),
    'rotary': ('rotary_embedding_fwdbwd',),
}


def _run_python(source, **env):
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(source)],
        cwd=ROOT,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ('operation', 'name'),
    [pytest.param(operation, name, id=f'{operation}-{name}') for operation, names in ENTRY_POINTS.items() for name in names],
)
@pytest.mark.parametrize('explicit_defaults', [False, True], ids=['omitted-defaults', 'explicit-arguments'])
def test_module_backend_dispatch(monkeypatch, operation, name, explicit_defaults):
    """Every module entry point forwards its full call surface to its own backend."""
    monkeypatch.setattr(utils, 'IS_NPU', True)
    backends.BackendRegistry.ensure_initialized(f'modules.{operation}')
    source = 'conv.triton.ops' if operation == 'causal_conv1d' else operation
    entry_point = getattr(importlib.import_module(f'fla.modules.{source}'), name)
    implementation_name = 'compute_dh0' if name == 'compute_dh0_triton' else name
    implementation = ModuleType(f'fla.modules.backends.{operation}.triton_ascend')
    calls = []
    output = object()

    def invoke(**kwargs):
        calls.append(kwargs)
        return output

    setattr(implementation, implementation_name, invoke)
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)
    signature = inspect.signature(entry_point)
    arguments = {
        key: object()
        for key, parameter in signature.parameters.items()
        if explicit_defaults or parameter.default is inspect.Parameter.empty
    }
    if operation == 'causal_conv1d' and name in ('causal_conv1d_fwd', 'causal_conv1d_bwd'):
        arguments['x'] = torch.zeros(1, 2, 3)
    expected = signature.bind(**arguments)
    expected.apply_defaults()
    expected_arguments = dict(expected.arguments)
    if name == 'l2norm_bwd':
        expected_arguments.pop('eps')

    assert entry_point(**arguments) is output
    assert calls == [expected_arguments]


@pytest.mark.parametrize('name', ['causal_conv1d_fwd', 'causal_conv1d_bwd'])
@pytest.mark.parametrize('strided', [False, True], ids=['contiguous', 'qkv-view'])
def test_conv_layout_handling_belongs_to_ascend_backend(monkeypatch, name, strided):
    monkeypatch.setattr(utils, 'IS_NPU', True)
    backends.BackendRegistry.ensure_initialized('modules.causal_conv1d')
    target = ModuleType('fla.modules.backends.causal_conv1d.triton_ascend')
    calls = []

    def invoke(**kwargs):
        calls.append(kwargs['layout_fallback'])
        return None

    setattr(target, name, invoke)
    monkeypatch.setitem(sys.modules, target.__name__, target)
    x = torch.zeros(1, 4, 15, dtype=torch.bfloat16)
    if strided:
        x = x[..., :5]
    entry_point = getattr(importlib.import_module('fla.modules.conv.triton.ops'), name)
    if name == 'causal_conv1d_fwd':
        entry_point(x=x, weight=None, bias=None, residual=None)
    else:
        entry_point(x=x, dy=None, dht=None)
    assert calls == [strided]


def test_module_backend_fallback_and_independent_registries(monkeypatch):
    """A backend registered for one operation cannot intercept another operation."""
    monkeypatch.setattr(backends.BackendRegistry, '_registries', {})
    monkeypatch.setattr(backends.BackendRegistry, '_initialized', {'modules.first', 'modules.second'})
    monkeypatch.setattr(backends, '_DISPATCH_DISABLED', False)

    class TestBackend(backends.BaseBackend):
        def compute(self, x):
            return x + 10

        def compute_verifier(self, x):
            return x >= 0, 'negative input'

    first = backends.BackendRegistry('modules.first')
    second = backends.BackendRegistry('modules.second')
    first.register(TestBackend())

    def compute(x):
        return x - 1

    first_compute = backends.dispatch('modules.first')(compute)
    second_compute = backends.dispatch('modules.second')(compute)

    assert first_compute(2) == 12
    assert first_compute(-2) == -3
    assert second_compute(2) == 1
    assert not second._backends
    monkeypatch.setattr(TestBackend, 'is_available', classmethod(lambda cls: False))
    assert first_compute(2) == 1


@pytest.mark.parametrize('disabled', ['0', '1'], ids=['enabled', 'disabled'])
def test_public_imports_are_lazy_and_do_not_warn(disabled):
    _run_python(
        """
        import sys
        import warnings
        import torch

        original_compile = torch.compile
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', DeprecationWarning)
            from fla.modules import RMSNorm, RotaryEmbedding, ShortConvolution
            from fla.backends import BackendRegistry
            BackendRegistry.ensure_initialized('modules.layernorm')
        assert torch.compile is original_compile
        assert not any(issubclass(w.category, DeprecationWarning) for w in caught)
        assert 'fla.ops.backends' not in sys.modules
        assert 'fla.modules.backends.triton_ascend' not in sys.modules
        assert 'fla.modules.backends._legacy' not in sys.modules
        assert 'fla.modules.backends.layernorm.triton_ascend' not in sys.modules
        assert 'modules' not in BackendRegistry._registries
        assert set(BackendRegistry._registries['modules.layernorm']._backends) == {'triton_ascend'}
        """,
        FLA_DISABLE_BACKEND_DISPATCH=disabled,
    )


def test_disabled_dispatch_uses_default_without_backend_import():
    _run_python(
        """
        import sys
        from fla.backends import dispatch

        def compute(x):
            return x + 1

        assert dispatch('modules.layernorm')(compute) is compute
        assert compute(1) == 2
        assert 'fla.modules.backends.layernorm' not in sys.modules
        """,
        FLA_DISABLE_BACKEND_DISPATCH='1',
    )


def test_deprecated_dispatch_imports_share_the_current_registry():
    _run_python(
        """
        import importlib
        import warnings
        import fla.backends as current

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', DeprecationWarning)
            old = importlib.import_module('fla.ops.backends')
            from fla.modules.backends import dispatch, modules_registry
            from fla.modules.backends.triton_ascend import TritonAscendBackend
            decorator = dispatch('modules')
        assert old.BackendRegistry is current.BackendRegistry
        assert old.BaseBackend is current.BaseBackend
        assert old.dispatch is dispatch is current.dispatch
        assert current.BackendRegistry._registries['modules'] is modules_registry
        assert isinstance(modules_registry._backends['triton_ascend'], TritonAscendBackend)
        messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any('fla.ops.backends' in message for message in messages)
        assert any('modules_registry' in message for message in messages)
        assert any("dispatch('modules')" in message for message in messages)
        assert all('next release after 0.6.0' in message for message in messages)

        class CustomBackend(current.BaseBackend):
            backend_type = 'legacy_test'
            priority = -1

            def l2norm_fwd(self, x, eps=1e-6, output_dtype=None):
                return x

        modules_registry.register(CustomBackend())
        from fla.modules.l2norm import l2norm_fwd
        token = object()
        assert l2norm_fwd(token) is token
        """,
    )


@pytest.mark.parametrize('operation', ENTRY_POINTS)
def test_deprecated_kernel_symbols_resolve_to_current_implementations(monkeypatch, operation):
    target = ModuleType(f'fla.modules.backends.{operation}.triton_ascend')

    def unused_symbol(name):
        if name.startswith('__'):
            raise AttributeError(name)
        return object()

    target.__getattr__ = unused_symbol
    expected = {}
    for name in ENTRY_POINTS[operation]:
        current_name = 'compute_dh0' if name == 'compute_dh0_triton' else name
        value = object()
        setattr(target, current_name, value)
        expected[current_name + '_npu'] = value
    monkeypatch.setitem(sys.modules, target.__name__, target)
    with pytest.warns(DeprecationWarning, match='next release after 0.6.0'):
        legacy_package = importlib.import_module('fla.modules.backends.triton_ascend')
        importlib.reload(legacy_package)
    legacy = importlib.import_module(f'fla.modules.backends.triton_ascend.{operation}')
    for name, value in expected.items():
        assert getattr(legacy, name) is value


def test_npu_grpo_compile_setting_is_local():
    _run_python(
        """
        import importlib
        import torch
        import fla.utils as utils

        original_compile = torch.compile
        utils.IS_NPU = True
        importlib.reload(importlib.import_module('fla.modules.backends'))
        grpo = importlib.reload(importlib.import_module('fla.modules.grpo'))
        assert torch.compile is original_compile
        assert not hasattr(grpo.grpo_loss_with_old_logps, '_torchdynamo_orig_callable')

        logps = torch.zeros(2, 1, requires_grad=True)
        loss = grpo.grpo_loss_with_old_logps(
            logps=logps,
            ref_logps=torch.zeros_like(logps),
            old_logps=torch.zeros_like(logps),
            pad_mask=torch.ones_like(logps, dtype=torch.bool),
            logits_to_keep=1,
            rewards=torch.tensor([1.0, -1.0]),
            beta=0.0,
        )
        loss.backward()
        assert logps.grad[0, 0] < 0
        assert logps.grad[1, 0] > 0
        assert torch.compile is original_compile
        """,
    )
