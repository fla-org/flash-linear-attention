# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib
import inspect
import sys
from types import SimpleNamespace

import pytest
import torch

from fla import backends, utils
from fla.backends import (
    BackendRegistry,
    BaseBackend,
    TritonAscendBackend,
    dispatch,
    register,
)


@pytest.fixture(autouse=True)
def clear_test_registries():
    yield
    for entry in tuple(backends._function_registries):
        if getattr(entry, '__module__', None) == __name__:
            del backends._function_registries[entry]


@pytest.fixture
def enable_dispatch(monkeypatch):
    monkeypatch.setattr(backends, '_DISPATCH_DISABLED', False)


@pytest.fixture(params=['registry', 'registered'])
def backend_dispatch(request):
    def decorate(*candidates):
        if request.param == 'registered':
            def decorate_entry(func):
                entry = dispatch(func)
                for backend in candidates:
                    register(entry)(type(backend))
                return entry
            return decorate_entry
        registry = BackendRegistry('test')
        for backend in candidates:
            registry.register(backend)
        return registry.dispatch

    return decorate


@pytest.fixture
def lazy_backend(monkeypatch, tmp_path, enable_dispatch):
    class Backend(BaseBackend):
        backend_type = 'test_lazy'
        env_var = 'FLA_TEST_LAZY'

    @dispatch
    def compute(x, scale=2):
        return x * 3

    module = 'fla_test_registered_backend'
    path = tmp_path / f'{module}.py'
    path.write_text(
        'from fla.backends import register\n'
        'from fla_test_entry import Backend, compute\n'
        '@register(compute, backend=Backend, verifier=lambda x, scale=2: (x.numel() > 1, None))\n'
        'def implementation(x, scale=2):\n'
        '    import fla_test_optional_dependency\n'
        '    return x * scale\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, 'fla_test_entry', SimpleNamespace(compute=compute, Backend=Backend))
    monkeypatch.setitem(sys.modules, 'fla_test_optional_dependency', SimpleNamespace())

    importlib.import_module(module)
    monkeypatch.setenv('FLA_TEST_LAZY', '1')
    yield compute, Backend, path
    sys.modules.pop(module, None)


@pytest.mark.parametrize("default_enable", [False, True], ids=["default-off", "default-on"])
@pytest.mark.parametrize(
    ("global_enable", "local_enable", "expected"),
    [
        pytest.param(None, None, None, id="global-unset-local-unset"),
        pytest.param(None, "0", False, id="global-unset-local-off"),
        pytest.param(None, "1", True, id="global-unset-local-on"),
        pytest.param("0", None, None, id="global-off-local-unset"),
        pytest.param("0", "0", False, id="global-off-local-off"),
        pytest.param("0", "1", True, id="global-off-local-on"),
        pytest.param("1", None, True, id="global-on-local-unset"),
        pytest.param("1", "0", True, id="global-on-local-off"),
        pytest.param("1", "1", True, id="global-on-local-on"),
    ],
)
def test_backend_enabled(monkeypatch, global_enable, local_enable, expected, default_enable):
    monkeypatch.setattr(BaseBackend, "backend_type", "gluon")
    monkeypatch.setattr(BaseBackend, "env_var", "FLA_TEST_GLUON")
    monkeypatch.setattr(BaseBackend, "default_enable", default_enable)
    for name, value in (("FLA_GLUON", global_enable), ("FLA_TEST_GLUON", local_enable)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert BaseBackend().is_enabled() is (default_enable if expected is None else expected)


@pytest.mark.parametrize("default_enable", [False, True], ids=["default-off", "default-on"])
@pytest.mark.parametrize("global_enable", [None, "0", "1"], ids=["global-unset", "global-off", "global-on"])
def test_backend_without_switch(monkeypatch, global_enable, default_enable):
    monkeypatch.setattr(BaseBackend, "backend_type", "gluon")
    monkeypatch.setattr(BaseBackend, "env_var", None)
    monkeypatch.setattr(BaseBackend, "default_enable", default_enable)
    if global_enable is None:
        monkeypatch.delenv("FLA_GLUON", raising=False)
    else:
        monkeypatch.setenv("FLA_GLUON", global_enable)

    assert BaseBackend().is_enabled() is True


@pytest.mark.parametrize('global_enable', [None, '0', '1'], ids=['default', 'disabled', 'enabled'])
@pytest.mark.parametrize('conv_enable', ['0', '1'], ids=['conv-disabled', 'conv-enabled'])
def test_gluon_enabled(monkeypatch, global_enable, conv_enable):
    from fla.backends import GluonBackend

    if global_enable is None:
        monkeypatch.delenv('FLA_GLUON', raising=False)
    else:
        monkeypatch.setenv('FLA_GLUON', global_enable)
    monkeypatch.setenv('FLA_CONV_GLUON', conv_enable)
    conv_backend = GluonBackend(env_var='FLA_CONV_GLUON')
    assert conv_backend.is_enabled() is (global_enable == '1' or conv_enable == '1')
    assert GluonBackend().is_enabled() is (global_enable == '1')


def test_registered_backend_switches(monkeypatch, enable_dispatch):
    from fla.backends import GluonBackend

    monkeypatch.setattr(GluonBackend, 'is_available', classmethod(lambda cls: True))

    @dispatch
    def local_entry():
        return 'default'

    @dispatch
    def global_entry():
        return 'default'

    @register(local_entry, backend=GluonBackend, env_var='FLA_TEST_GLUON')
    @register(global_entry, backend=GluonBackend)
    def implementation():
        return 'gluon'

    monkeypatch.setenv('FLA_GLUON', '0')
    monkeypatch.setenv('FLA_TEST_GLUON', '1')
    assert local_entry() == 'gluon'
    assert global_entry() == 'default'

    monkeypatch.setenv('FLA_GLUON', '1')
    monkeypatch.setenv('FLA_TEST_GLUON', '0')
    assert local_entry() == 'gluon'
    assert global_entry() == 'gluon'

    monkeypatch.setenv('FLA_GLUON', '0')
    assert local_entry() == 'default'
    assert global_entry() == 'default'


def test_ascend_available(monkeypatch):
    monkeypatch.setattr(utils, 'IS_NPU', True)
    assert TritonAscendBackend.is_available() is True
    monkeypatch.setattr(utils, 'IS_NPU', False)
    assert TritonAscendBackend.is_available() is False


def test_registry_priority(enable_dispatch):
    class Backend(BaseBackend):
        def __init__(self, name, priority, value):
            super().__init__()
            self.backend_type = name
            self.priority = priority
            self.value = value

        def compute(self):
            return self.value

    first = BackendRegistry('same-name')
    second = BackendRegistry('same-name')
    first.register(Backend('first', 5, 1))
    first.register(Backend('second', 5, 2))

    def compute():
        return 0

    wrapped = first.dispatch(compute)
    assert wrapped() == 1
    assert second.dispatch(compute)() == 0
    first.register(Backend('priority', 0, 3))
    assert wrapped() == 3


@pytest.mark.parametrize('disabled', [False, True], ids=['dispatch-enabled', 'dispatch-disabled'])
def test_register_operation(monkeypatch, disabled):
    monkeypatch.setattr(backends, '_DISPATCH_DISABLED', disabled)
    monkeypatch.setattr(backends, '_operation_registries', {})

    class Backend(BaseBackend):
        pass

    assert register('test')(Backend) is Backend
    registry = backends._get_operation_registry('test')
    assert type(registry.get_active()) is Backend

    @register('test')
    class Replacement(Backend):
        pass

    assert type(registry.get_active()) is Replacement
    assert len(registry._backends) == 1


def test_dispatch_owner(monkeypatch):
    monkeypatch.setattr(backends, '_operation_registries', {})
    registry = backends._get_operation_registry('attn')
    imports = []
    monkeypatch.setattr(backends.importlib, 'import_module', imports.append)

    assert dispatch('attn').__self__ is registry
    assert imports == ['fla.ops.attn.backends']


@pytest.mark.parametrize('operation', ['unknown_dispatch_test', 'modules.unknown_dispatch_test', 'modules'])
def test_dispatch_unknown_owner(monkeypatch, operation):
    monkeypatch.setattr(backends, '_operation_registries', {})
    backends._get_operation_registry(operation)
    with pytest.raises(ModuleNotFoundError):
        dispatch(operation)


@pytest.mark.parametrize('dependency', ['missing_optional_dependency', 'fla.ops.attn.backends.required'])
def test_dispatch_import_error(monkeypatch, dependency):
    monkeypatch.setattr(backends, '_operation_registries', {})
    backends._get_operation_registry('attn')
    error = ModuleNotFoundError(f'Missing dependency: {dependency}', name=dependency)

    def import_owner(name):
        raise error

    monkeypatch.setattr(backends.importlib, 'import_module', import_owner)
    with pytest.raises(ModuleNotFoundError) as caught:
        backends._load_operation_registry('attn')
    assert caught.value is error


@pytest.mark.parametrize('route', ['accepted', 'rejected', 'unavailable', 'disabled', 'missing', 'verifier-error'])
def test_dispatch(monkeypatch, enable_dispatch, backend_dispatch, route):
    class Backend(BaseBackend):
        env_var = 'FLA_TEST_BACKEND'

        def compute(self, x):
            return x * 2

        def compute_verifier(self, x):
            if route == 'verifier-error':
                raise ValueError('unsupported call')
            return route != 'rejected', None

    if route == 'unavailable':
        monkeypatch.setattr(Backend, 'is_available', classmethod(lambda cls: False))
    elif route == 'disabled':
        monkeypatch.setenv('FLA_TEST_BACKEND', '0')
    elif route == 'missing':
        monkeypatch.delattr(Backend, 'compute')

    @backend_dispatch(Backend())
    def compute(x):
        return x * 3

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    result = compute(x)
    factor = 2 if route == 'accepted' else 3
    torch.testing.assert_close(result, x * factor)
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, factor))


def test_dispatch_runtime(monkeypatch, enable_dispatch, backend_dispatch):
    class Primary(BaseBackend):
        backend_type = 'test_primary'
        env_var = 'FLA_TEST_PRIMARY'
        priority = 0

        def compute(self, x):
            return x * 2

        def compute_verifier(self, x):
            return x >= 0, None

    class Secondary(BaseBackend):
        backend_type = 'test_secondary'
        env_var = 'FLA_TEST_SECONDARY'

        def compute(self, x):
            return x * 4

    monkeypatch.setenv('FLA_TEST_PRIMARY', '1')
    monkeypatch.setenv('FLA_TEST_SECONDARY', '1')

    @backend_dispatch(Secondary(), Primary())
    def compute(x):
        return x * 3

    assert compute(1) == 2
    assert compute(-1) == -4
    monkeypatch.setenv('FLA_TEST_PRIMARY', '0')
    assert compute(1) == 4
    monkeypatch.setenv('FLA_TEST_SECONDARY', '0')
    assert compute(1) == 3
    monkeypatch.setenv('FLA_TEST_PRIMARY', '1')
    assert compute(1) == 2


def test_dispatch_error(enable_dispatch, backend_dispatch):
    error = RuntimeError('backend execution failed')

    class Backend(BaseBackend):
        def compute(self, x):
            raise error

    @backend_dispatch(Backend())
    def compute(x):
        return x

    with pytest.raises(RuntimeError) as caught:
        compute(1)
    assert caught.value is error


def test_register(monkeypatch, enable_dispatch):
    class Primary(BaseBackend):
        backend_type = 'primary'
        priority = 0

    @torch.compiler.disable
    @dispatch
    def compute(x):
        return x * 3 + 1

    @register(compute)
    class Secondary(BaseBackend):
        def compute_verifier(self, x):
            return x != 0, None

        def compute(self, x):
            return x * 4

    @register(compute, backend=Primary, verifier=lambda x: (x > 0, None))
    def first(x):
        return x * 2

    assert compute(1) == 2
    assert compute(-1) == -4
    assert compute(0) == 1
    assert first(1) == 2
    assert Secondary().compute(-1) == -4


def test_register_arguments(lazy_backend):
    compute, _, _ = lazy_backend
    x = torch.ones(2, requires_grad=True)
    for args, kwargs, scale in [((x,), {}, 2), ((x, 5), {}, 5), ((), {'x': x, 'scale': 7}, 7)]:
        result = compute(*args, **kwargs)
        torch.testing.assert_close(result, x * scale)
        torch.testing.assert_close(torch.autograd.grad(result.sum(), x)[0], torch.full_like(x, scale))


def test_load_runtime(monkeypatch, lazy_backend):
    compute, Backend, path = lazy_backend
    x = torch.ones(2)
    monkeypatch.setattr(Backend, 'is_available', classmethod(lambda cls: False))
    torch.testing.assert_close(compute(x), x * 3)
    assert path.stem in sys.modules

    monkeypatch.setattr(Backend, 'is_available', classmethod(lambda cls: True))
    monkeypatch.setenv('FLA_TEST_LAZY', '0')
    torch.testing.assert_close(compute(x), x * 3)
    assert path.stem in sys.modules

    monkeypatch.setenv('FLA_TEST_LAZY', '1')
    torch.testing.assert_close(compute(x), x * 2)
    assert path.stem in sys.modules
    monkeypatch.setenv('FLA_TEST_LAZY', '0')
    torch.testing.assert_close(compute(x), x * 3)


def test_load_dependency_error(monkeypatch, lazy_backend):
    compute, _, path = lazy_backend
    monkeypatch.delitem(sys.modules, 'fla_test_optional_dependency')
    x = torch.ones(2)
    torch.testing.assert_close(compute(x[:1]), x[:1] * 3)
    assert path.stem in sys.modules
    with pytest.raises(ModuleNotFoundError, match='fla_test_optional_dependency'):
        compute(x)

    monkeypatch.setitem(sys.modules, 'fla_test_optional_dependency', SimpleNamespace())
    torch.testing.assert_close(compute(x), x * 2)


@pytest.mark.parametrize('disabled', ['0', '1'], ids=['dispatch-enabled', 'dispatch-disabled'])
def test_dispatch_disabled(run_python, disabled):
    run_python(
        """
        import os
        import sys
        import warnings

        from fla import backends
        from fla.backends import GluonBackend, TritonAscendBackend, dispatch, register

        enabled = os.environ['FLA_DISABLE_BACKEND_DISPATCH'] != '1'

        def compute(x):
            return x + 1

        for operation in [
            'attn', 'attnres', 'common', 'gated_delta_rule', 'gdn2', 'generalized_delta_rule.dplr',
            'gla', 'kda', 'rwkv6', 'utils',
        ]:
            with warnings.catch_warnings():
                warnings.simplefilter('error', DeprecationWarning)
                decorate = dispatch(operation)
            registry = backends._operation_registries[operation]
            assert registry.operation_name == operation
            assert registry._backends
            wrapped = decorate(compute)
            assert (wrapped is compute) == (not enabled)
            assert wrapped(1) == 2

        wrapped = dispatch(compute)
        assert (wrapped is compute) == (not enabled)
        assert wrapped(1) == 2

        @register(wrapped)
        def replacement(x):
            return x + 2

        assert replacement(1) == 3
        assert wrapped(1) == (3 if enabled else 2)
        assert 'fla.ops.kda.backends.flash_kda' in sys.modules
        for owner in ['activations', 'rotary', 'grpo', 'fused_cross_entropy', 'fused_linear_cross_entropy', 'fused_kl_div']:
            assert f'modules.{owner}' not in backends._operation_registries
        assert 'flash_kda' not in backends._operation_registries['kda']._backends
        assert 'tilelang' not in sys.modules
        assert 'flash_kda' not in sys.modules
        assert 'fla.ops.kda.backends.triton_ascend.chunk_intra' not in sys.modules
        ascend_available = TritonAscendBackend.is_available()
        assert ('fla.ops.common.backends.triton_ascend.chunk_o' in sys.modules) == ascend_available
        assert any(name.startswith('fla.ops.simple_gla.backends.triton_ascend.') for name in sys.modules) == ascend_available
        assert any(
            name.startswith('fla.modules.') and 'triton_ascend' in name.split('.')
            for name in sys.modules
        ) == ascend_available
        assert ('fla.modules.causal_conv1d.backends.gluon' in sys.modules) == GluonBackend.is_available()
        """,
        FLA_DISABLE_BACKEND_DISPATCH=disabled,
    )


def test_register_import_order(run_python):
    run_python(
        """
        import importlib
        import inspect
        import sys

        from triton.language import math

        from fla import backends

        # host-side registration checks do not execute Ascend intrinsics.
        if not hasattr(math, 'tanh'):
            math.tanh = None
        backends.TritonAscendBackend.is_available = classmethod(lambda cls: True)
        for owner, names in {
            'activations': [
                'sigmoid_fwd', 'sigmoid_bwd', 'logsigmoid_fwd', 'logsigmoid_bwd', 'swish_fwd', 'swish_bwd',
                'swiglu_fwd', 'swiglu_fwdbwd', 'swiglu_linear', 'powglu_fwd', 'powglu_fwdbwd', 'powglu_linear',
            ],
            'rotary': ['rotary_embedding_fwdbwd'],
            'grpo': ['fused_grpo_loss'],
            'fused_cross_entropy': ['cross_entropy_loss'],
            'fused_linear_cross_entropy': [
                'logsumexp_fwd', 'fused_linear_cross_entropy_fwd', 'fused_linear_cross_entropy_bwd',
            ],
            'fused_kl_div': ['fused_kl_div_fwd', 'fused_kl_div_bwd'],
            'causal_conv1d': [
                'causal_conv1d_fwd', 'causal_conv1d_bwd', 'compute_dh0_triton',
                'causal_conv1d_update', 'causal_conv1d_update_states',
            ],
            'norm.layernorm': ['layer_norm_fwd', 'layer_norm_bwd'],
            'norm.l2norm': ['l2norm_fwd', 'l2norm_bwd'],
            'norm.fused_norm_gate': ['layer_norm_gated_fwd', 'layer_norm_gated_bwd'],
        }.items():
            module = f'fla.modules.{owner}' if owner.startswith('norm.') else f'fla.modules.{owner}.ops'
            if owner == 'causal_conv1d':
                backend_module = 'fla.modules.causal_conv1d.backends.triton_ascend'
            elif owner.startswith('norm.'):
                backend_module = f'fla.modules.norm.triton_ascend.{owner.split(".")[1]}'
            else:
                backend_module = f'fla.modules.{owner}.triton_ascend'
            importlib.reload(importlib.import_module(f'fla.modules.{owner.split(".")[0]}'))
            entries = importlib.import_module(module)
            implementation = sys.modules[backend_module]
            for name in names:
                entry = getattr(entries, name)
                registry = backends._function_registries[inspect.unwrap(entry)]
                target = registry._backends['triton_ascend'].get_implementation(name)
                implementation_name = 'compute_dh0_npu' if name == 'compute_dh0_triton' else name + '_npu'
                assert target is getattr(implementation, implementation_name)
                expected = inspect.signature(entry).parameters
                actual = inspect.signature(target).parameters
                assert expected.keys() == actual.keys()
                for parameter in expected:
                    assert expected[parameter].kind == actual[parameter].kind
                    assert expected[parameter].default == actual[parameter].default
        """,
        FLA_DISABLE_BACKEND_DISPATCH='0',
    )


@pytest.mark.parametrize(
    ('owner', 'name', 'tensor_args', 'options'),
    [
        ('activations', 'logsigmoid_fwd', (), {'temperature': 0.5, 'output_contiguous': True}),
        ('activations', 'logsigmoid_bwd', ('dy',), {'temperature': 0.5, 'output_contiguous': True}),
        ('rotary', 'rotary_embedding_fwdbwd', ('cos', 'sin'), {'interleaved': True, 'conjugate': False}),
        ('rotary', 'rotary_embedding_fwdbwd', ('cos', 'sin'), {'interleaved': True, 'conjugate': True}),
        ('causal_conv1d', 'causal_conv1d_fwd', ('weight', 'bias', 'residual'), {'chunk_size': 32, 'output_final_state': True}),
        ('causal_conv1d', 'causal_conv1d_bwd', ('dy', 'dht'), {'chunk_size': 32, 'layout_fallback': True}),
        ('causal_conv1d', 'causal_conv1d_update', ('cache', 'weight', 'bias', 'residual'), {'activation': 'silu'}),
        ('causal_conv1d', 'causal_conv1d_update_states', ('initial_state',), {'state_len': 4}),
        ('grpo', 'fused_grpo_loss', ('ref_logp', 'input_ids', 'advantages'), {'beta': 0.2, 'save_kl': True}),
        ('fused_cross_entropy', 'cross_entropy_loss', ('target',), {'ignore_index': -1, 'label_smoothing': 0.1}),
        ('fused_linear_cross_entropy', 'fused_linear_cross_entropy_fwd', ('target', 'weight'), {'reduction': 'sum'}),
        ('fused_linear_cross_entropy', 'fused_linear_cross_entropy_bwd', ('dx', 'dw', 'db'), {}),
        ('fused_kl_div', 'fused_kl_div_fwd', ('target_x', 'weight', 'target_weight'), {'use_dw': False}),
        ('fused_kl_div', 'fused_kl_div_bwd', ('dx', 'dw'), {}),
        ('norm.layernorm', 'layer_norm_fwd', ('weight', 'bias'), {'is_rms_norm': True, 'num_groups': 2}),
        ('norm.layernorm', 'layer_norm_bwd', ('x', 'weight', 'bias'), {'recompute_output': True, 'num_groups': 2}),
        ('norm.l2norm', 'l2norm_fwd', (), {'eps': 1e-4, 'output_dtype': torch.float32}),
        ('norm.l2norm', 'l2norm_bwd', ('rstd', 'dy'), {'eps': 1e-4}),
        ('norm.fused_norm_gate', 'layer_norm_gated_fwd', ('g', 'weight', 'bias'), {'activation': 'sigmoid'}),
        ('norm.fused_norm_gate', 'layer_norm_gated_bwd', ('x', 'g', 'weight', 'bias'), {'activation': 'sigmoid'}),
    ],
    ids=[
        'activations-forward', 'activations-backward', 'rotary-forward', 'rotary-backward',
        'conv-forward', 'conv-backward', 'conv-update', 'conv-update-states', 'grpo', 'cross-entropy',
        'linear-cross-entropy-forward', 'linear-cross-entropy-backward', 'kl-div-forward', 'kl-div-backward',
        'layernorm-forward', 'layernorm-backward', 'l2norm-forward', 'l2norm-backward',
        'norm-gate-forward', 'norm-gate-backward',
    ],
)
@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason='dispatch was disabled before the entry points were imported')
def test_dispatch_modules(monkeypatch, owner, name, tensor_args, options):
    module = f'fla.modules.{owner}' if owner.startswith('norm.') else f'fla.modules.{owner}.ops'
    entry = getattr(importlib.import_module(module), name)
    calls = []
    expected = object()

    def implementation(value, **kwargs):
        assert torch.is_grad_enabled()
        calls.append((value, kwargs))
        return expected

    registry = backends._function_registries[inspect.unwrap(entry)]
    monkeypatch.setattr(registry, '_backends', {})
    candidate = BaseBackend()
    candidate.implementation = implementation
    registry.register(candidate)

    x = torch.tensor([-2.0, 0.5, 3.0], requires_grad=True)
    kwargs = {argument: torch.ones_like(x) for argument in tensor_args}
    kwargs.update(options)
    inspect.signature(entry).bind(x, **kwargs)
    result = entry(x, **kwargs)

    assert len(calls) == 1
    assert calls[0][0] is x
    assert calls[0][1].keys() == kwargs.keys()
    for argument, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            assert calls[0][1][argument] is value
        else:
            assert calls[0][1][argument] == value
    assert result is expected


@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason='Backend dispatch was disabled before import')
@pytest.mark.parametrize('direction', ['fwd', 'bwd'], ids=['forward', 'backward'])
@pytest.mark.parametrize('env_var', ['FLA_GLUON', 'FLA_CONV_GLUON'], ids=['global', 'local'])
def test_dispatch_gluon(monkeypatch, direction, env_var):
    pytest.importorskip('triton.experimental.gluon')
    from fla.backends import GluonBackend, TritonAscendBackend
    from fla.modules.causal_conv1d import ops

    func_name = f'causal_conv1d_{direction}'
    monkeypatch.setenv('FLA_GLUON', '0')
    monkeypatch.setenv('FLA_CONV_GLUON', '0')
    monkeypatch.setenv(env_var, '1')
    monkeypatch.setattr(GluonBackend, 'is_available', classmethod(lambda cls: True))
    monkeypatch.setattr(TritonAscendBackend, 'is_available', classmethod(lambda cls: False))
    entry = getattr(ops, func_name)
    importlib.reload(importlib.import_module('fla.modules.causal_conv1d'))
    candidate = backends._function_registries[inspect.unwrap(entry)]._backends['gluon']
    monkeypatch.setattr(candidate, 'verifier', lambda **kwargs: (True, None))
    result = object()

    def implementation(x, **kwargs):
        assert torch.is_grad_enabled()
        assert x.requires_grad
        return result

    monkeypatch.setattr(candidate, 'implementation', implementation)
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    kwargs = {'weight': x, 'bias': None, 'residual': None} if direction == 'fwd' else {'dy': x, 'dht': None}
    inspect.signature(entry).bind(x=x, **kwargs)
    assert entry(x=x, **kwargs) is result
