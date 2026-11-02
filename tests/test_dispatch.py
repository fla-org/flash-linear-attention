# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import inspect

import pytest
import torch

from fla import backends, utils
from fla.backends import BackendRegistry, BaseBackend, TritonAscendBackend, dispatch, register


@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason='Backend dispatch was disabled before import')
@pytest.mark.parametrize(
    'graph_option,grad_enabled',
    [(None, False), (False, True), (True, False), (True, True)],
    ids=['default-inference', 'eager-training', 'graph-inference', 'graph-training'],
)
def test_gdn_graph_not_consumed_by_flash_qla(monkeypatch, graph_option, grad_enabled):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.backends import flash_qla

    registry = backends._function_registries[inspect.unwrap(chunk_gated_delta_rule)]
    candidate = flash_qla.FlashQLABackend()
    monkeypatch.setattr(registry, '_backends', {'flash_qla': candidate})
    monkeypatch.setattr(candidate, 'is_available', lambda: True)
    monkeypatch.setattr(candidate, 'is_enabled', lambda: True)
    monkeypatch.setattr(flash_qla, 'IS_NVIDIA_HOPPER', True)
    monkeypatch.setattr(flash_qla, 'IS_NVIDIA_SM120', False)
    calls = []
    expected = object()

    def implementation(*args, **kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(candidate, 'chunk_gated_delta_rule', implementation)
    q = torch.zeros(1, 65, 1, 128, dtype=torch.bfloat16, requires_grad=grad_enabled)
    g = torch.zeros(1, 65, 1)
    beta = torch.ones_like(g, dtype=q.dtype)
    kwargs = {} if graph_option is None else {'use_graph': graph_option}
    with torch.set_grad_enabled(grad_enabled):
        if graph_option:
            with pytest.raises(ValueError, match='only on NPU'):
                chunk_gated_delta_rule(q, q, q, g, beta, **kwargs)
            assert not calls
        else:
            assert chunk_gated_delta_rule(q, q, q, g, beta, **kwargs) is expected
            assert len(calls) == 1


@pytest.mark.parametrize(
    'use_graph,requires_grad',
    [(False, False), (False, True), (True, True)],
    ids=['eager-inference', 'eager-training', 'graph-training'],
)
def test_gdn_graph_routes_through_autograd(monkeypatch, use_graph, requires_grad):
    from fla.ops.gated_delta_rule import chunk, graph

    validation = []
    monkeypatch.setattr(graph, '_validate_graph_inputs', lambda *args: validation.append(args))
    calls = []
    expected = (object(), object())

    def apply(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(chunk.ChunkGatedDeltaRuleFunction, 'apply', apply)
    q = torch.zeros(1, 65, 1, 64, dtype=torch.bfloat16, requires_grad=requires_grad)
    g = torch.zeros(1, 65, 1)
    beta = torch.ones_like(g, dtype=q.dtype)
    cu = torch.tensor([0, 65])
    # inspect the default entry's routing without launching a kernel or requiring an NPU.
    entry = inspect.unwrap(chunk.chunk_gated_delta_rule)
    result = entry(q, q, q, g, beta, cu_seqlens=cu, use_graph=use_graph)
    assert all(actual is reference for actual, reference in zip(result, expected))
    assert len(calls) == 1
    assert calls[0][0] is q
    assert calls[0][-1] is use_graph
    assert len(validation) == int(use_graph)


def test_solve_tril_ascend_eager_forwarding(run_python):
    run_python(
        """
        import importlib
        from unittest.mock import patch

        from triton.language import math

        from fla import backends

        if not hasattr(math, 'tanh'):
            math.tanh = None
        backends.TritonAscendBackend.is_available = classmethod(lambda cls: True)
        importlib.reload(importlib.import_module('fla.ops.utils'))
        solve = importlib.import_module('fla.ops.utils.solve_tril')
        ascend = importlib.import_module('fla.ops.utils.triton_ascend.solve_tril')
        A, cu, indices, expected = object(), object(), object(), object()
        with patch.object(ascend, '_solve_tril_npu', return_value=expected) as implementation:
            assert solve.solve_tril(A) is expected
            implementation.assert_called_once_with(
                A=A, cu_seqlens=None, chunk_indices=None, output_dtype=ascend.torch.float, use_graph=False,
            )
            implementation.reset_mock()
            assert solve.solve_tril(A, cu, indices, None) is expected
            implementation.assert_called_once_with(
                A=A, cu_seqlens=cu, chunk_indices=indices, output_dtype=None, use_graph=False,
            )
        """,
        FLA_DISABLE_BACKEND_DISPATCH='0',
    )


@pytest.fixture
def enable_dispatch(monkeypatch):
    monkeypatch.setattr(backends, '_DISPATCH_DISABLED', False)
    monkeypatch.setattr(backends, '_function_registries', {})


@pytest.fixture
def backend_dispatch():
    def decorate(*candidates):
        registry = BackendRegistry('test')
        for backend in candidates:
            registry.register(backend)
        return registry.dispatch

    return decorate


@pytest.mark.parametrize(
    ("global_enable", "local_enable", "default_enable", "expected"),
    [
        (None, None, False, False),
        (None, None, True, True),
        ('0', '1', False, True),
        ('1', '0', False, True),
        ('0', '0', True, False),
    ],
)
def test_backend_enabled(monkeypatch, global_enable, local_enable, default_enable, expected):
    monkeypatch.setattr(BaseBackend, "backend_type", "gluon")
    monkeypatch.setattr(BaseBackend, "env_var", "FLA_TEST_GLUON")
    monkeypatch.setattr(BaseBackend, "default_enable", default_enable)
    for name, value in (("FLA_GLUON", global_enable), ("FLA_TEST_GLUON", local_enable)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert BaseBackend().is_enabled() is expected


def test_backend_without_switch(monkeypatch):
    monkeypatch.setattr(BaseBackend, "backend_type", "gluon")
    monkeypatch.setattr(BaseBackend, "env_var", None)
    monkeypatch.setattr(BaseBackend, "default_enable", False)
    monkeypatch.setenv("FLA_GLUON", '0')

    assert BaseBackend().is_enabled() is True


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


def test_gluon_requires_triton_3_5(monkeypatch):
    from fla.backends import GluonBackend

    monkeypatch.setattr(utils, 'IS_NVIDIA', True)
    monkeypatch.setattr(backends, 'TRITON_ABOVE_3_5_0', False)
    assert GluonBackend.is_available() is False
    monkeypatch.setattr(backends, 'TRITON_ABOVE_3_5_0', True)
    assert GluonBackend.is_available() is (utils.find_spec_cached('triton.experimental.gluon') is not None)


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
    monkeypatch.setattr(Primary, 'is_available', classmethod(lambda cls: False))
    assert compute(1) == 3
    monkeypatch.setattr(Primary, 'is_available', classmethod(lambda cls: True))
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


def test_register(enable_dispatch):
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


def test_register_arguments(enable_dispatch):
    @dispatch
    def compute(x, scale=2):
        return x * 3

    @register(compute, backend=BaseBackend)
    def implementation(x, scale=2):
        return x * scale
    x = torch.ones(2, requires_grad=True)
    for args, kwargs, scale in [((x,), {}, 2), ((x, 5), {}, 5), ((), {'x': x, 'scale': 7}, 7)]:
        result = compute(*args, **kwargs)
        torch.testing.assert_close(result, x * scale)
        torch.testing.assert_close(torch.autograd.grad(result.sum(), x)[0], torch.full_like(x, scale))


@pytest.mark.parametrize('disabled', ['0', '1'], ids=['dispatch-enabled', 'dispatch-disabled'])
def test_dispatch_disabled(run_python, disabled):
    run_python(
        """
        import os

        from fla import backends
        from fla.backends import BackendRegistry, BaseBackend, dispatch, register

        enabled = os.environ['FLA_DISABLE_BACKEND_DISPATCH'] != '1'
        registry = BackendRegistry('test')
        backends._load_operation_registry = lambda operation: registry
        backends._operation_registries['test'] = registry

        @register('test')
        class Backend(BaseBackend):
            def compute(self, x):
                return x + 2

        def compute(x):
            return x + 1

        wrapped = dispatch('test')(compute)
        assert (wrapped is compute) == (not enabled)
        assert wrapped(1) == (3 if enabled else 2)

        wrapped = dispatch(compute)
        assert (wrapped is compute) == (not enabled)

        @register(wrapped)
        def replacement(x):
            return x + 2

        assert replacement(1) == 3
        assert wrapped(1) == (3 if enabled else 2)
        """,
        FLA_DISABLE_BACKEND_DISPATCH=disabled,
    )


@pytest.mark.parametrize('default_enable', [False, True])
def test_register_default_enable_override(monkeypatch, enable_dispatch, default_enable):
    class Backend(BaseBackend):
        backend_type = 'test_policy'
        env_var = 'FLA_TEST_POLICY'

    Backend.default_enable = not default_enable
    monkeypatch.delenv('FLA_TEST_POLICY', raising=False)

    @dispatch
    def overridden():
        return 'default'

    @dispatch
    def inherited():
        return 'default'

    @register(overridden, backend=Backend, default_enable=default_enable)
    @register(inherited, backend=Backend)
    def implementation():
        return 'backend'

    assert overridden() == ('backend' if default_enable else 'default')
    assert inherited() == ('default' if default_enable else 'backend')
    assert Backend.default_enable is not default_enable


def test_register_priority_override(enable_dispatch):
    class Primary(BaseBackend):
        backend_type = 'primary'

    class Secondary(BaseBackend):
        backend_type = 'secondary'
        priority = 3

    @dispatch
    def overridden():
        return 'default'

    @dispatch
    def inherited():
        return 'default'

    @register(overridden, backend=Primary, priority=0)
    @register(inherited, backend=Primary)
    def primary():
        return 'primary'

    @register(overridden, backend=Secondary)
    @register(inherited, backend=Secondary)
    def secondary():
        return 'secondary'

    assert overridden() == 'primary'
    assert inherited() == 'secondary'
    assert Primary.priority == 5
