# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib.metadata
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fla import backends as registry_module
from fla.backends import BackendRegistry, BaseBackend, dispatch, register_backend
from fla.ops.attn.backends import tilelang as attn_tilelang_backend
from fla.ops.common.backends import tilelang as common_tilelang_backend
from fla.ops.generalized_delta_rule.dplr.backends import tilelang as dplr_tilelang_backend
from fla.ops.kda.backends import tilelang as kda_tilelang_backend
from fla.ops.rwkv6.backends import tilelang as rwkv6_tilelang_backend
from fla.utils import env

_REAL_PATH_EXISTS = Path.exists


@pytest.fixture(autouse=True)
def clear_nvcc_probe_cache():
    env.has_usable_nvcc.cache_clear()
    yield
    env.has_usable_nvcc.cache_clear()


@pytest.fixture
def enable_dispatch(monkeypatch):
    monkeypatch.setattr(registry_module, '_DISPATCH_DISABLED', False)


def _configure_no_nvcc(monkeypatch):
    """Hide every nvcc source probed by has_usable_nvcc (CI runners have a real toolkit)."""
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(env.shutil, "which", lambda name: None)

    def no_such_dist(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "files", no_such_dist)

    def fake_exists(self):
        if str(self).startswith("/usr/local/cuda"):
            return False
        return _REAL_PATH_EXISTS(self)

    monkeypatch.setattr(env.Path, "exists", fake_exists)


def _backend_cls(backend_module):
    if backend_module is attn_tilelang_backend:
        return backend_module.AttnTileLangBackend
    if backend_module is common_tilelang_backend:
        return backend_module.CommonTileLangBackend
    if backend_module is rwkv6_tilelang_backend:
        return backend_module.RWKV6TileLangBackend
    if backend_module is kda_tilelang_backend:
        return backend_module.KDATileLangBackend
    if backend_module is dplr_tilelang_backend:
        return backend_module.DPLRTileLangBackend
    raise ValueError(f"unrecognized TileLang backend module: {backend_module}")


@pytest.mark.parametrize("default_enable", [False, True], ids=["default-off", "default-on"])
@pytest.mark.parametrize(("global_enable", "local_enable", "expected"), [
    pytest.param(None, None, None, id="global-unset-local-unset"),
    pytest.param(None, "0", False, id="global-unset-local-off"),
    pytest.param(None, "1", True, id="global-unset-local-on"),
    pytest.param("0", None, None, id="global-off-local-unset"),
    pytest.param("0", "0", False, id="global-off-local-off"),
    pytest.param("0", "1", True, id="global-off-local-on"),
    pytest.param("1", None, True, id="global-on-local-unset"),
    pytest.param("1", "0", True, id="global-on-local-off"),
    pytest.param("1", "1", True, id="global-on-local-on"),
])
def test_base_backend_is_enabled(monkeypatch, global_enable, local_enable, expected, default_enable):
    monkeypatch.setattr(BaseBackend, "backend_type", "gluon")
    monkeypatch.setattr(BaseBackend, "env_var", "FLA_TEST_GLUON")
    monkeypatch.setattr(BaseBackend, "default_enable", default_enable)
    for name, value in (("FLA_GLUON", global_enable), ("FLA_TEST_GLUON", local_enable)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert BaseBackend.is_enabled() is (default_enable if expected is None else expected)


@pytest.mark.parametrize("default_enable", [False, True], ids=["default-off", "default-on"])
@pytest.mark.parametrize("global_enable", [None, "0", "1"], ids=["global-unset", "global-off", "global-on"])
def test_base_backend_is_enabled_without_env_var(monkeypatch, global_enable, default_enable):
    monkeypatch.setattr(BaseBackend, "backend_type", "gluon")
    monkeypatch.setattr(BaseBackend, "env_var", None)
    monkeypatch.setattr(BaseBackend, "default_enable", default_enable)
    if global_enable is None:
        monkeypatch.delenv("FLA_GLUON", raising=False)
    else:
        monkeypatch.setenv("FLA_GLUON", global_enable)

    assert BaseBackend.is_enabled() is True


@pytest.mark.parametrize('route', ['accepted', 'rejected', 'unavailable', 'disabled', 'missing', 'verifier-error'])
def test_dispatch_selection_preserves_outputs_and_gradients(monkeypatch, enable_dispatch, route):
    class Backend(BaseBackend):
        env_var = 'FLA_TEST_BACKEND'

        def compute(self, x):
            return x * 2

        def compute_verifier(self, x):
            if route == 'verifier-error':
                raise ValueError('unsupported call')
            return route != 'rejected', None

    registry = BackendRegistry('test')
    registry.register(Backend())
    if route == 'unavailable':
        monkeypatch.setattr(Backend, 'is_available', classmethod(lambda cls: False))
    elif route == 'disabled':
        monkeypatch.setenv('FLA_TEST_BACKEND', '0')
    elif route == 'missing':
        monkeypatch.delattr(Backend, 'compute')

    @registry.dispatch
    def compute(x):
        return x * 3

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    result = compute(x)
    factor = 2 if route == 'accepted' else 3
    torch.testing.assert_close(result, x * factor)
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, factor))


def test_registry_ownership_and_priority(enable_dispatch):
    class Backend(BaseBackend):
        def __init__(self, name, priority, value):
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


def test_class_registration_preserves_identity_and_owner_state(monkeypatch, enable_dispatch):
    monkeypatch.setattr(registry_module, '_registries', {})

    class Backend(BaseBackend):
        env_var = 'FLA_TEST_BACKEND'

        def __init__(self):
            self.value = 1

        def compute(self):
            return self.value

    assert register_backend('first', 'second')(Backend) is Backend
    first = registry_module._registry_for('first')
    second = registry_module._registry_for('second')
    first_backend, second_backend = first.get_active(), second.get_active()
    assert type(first_backend) is type(second_backend) is Backend
    assert first_backend is not second_backend
    first_backend.value = 3
    assert second_backend.value == 1

    def compute():
        return 0

    first_call, second_call = first.dispatch(compute), second.dispatch(compute)
    assert first_call() == 3
    assert second_call() == 1

    @register_backend('first')
    class Replacement(Backend):
        def __init__(self):
            self.value = 2

    assert type(first.get_active()) is Replacement
    assert first_call() == 2
    assert second.get_active() is second_backend
    assert second_call() == 1
    monkeypatch.setenv('FLA_TEST_BACKEND', '0')
    assert first_call() == second_call() == 0
    assert first.get_active() is second.get_active() is None


@pytest.mark.filterwarnings('error::DeprecationWarning')
def test_bound_operation_key(enable_dispatch):
    registry = BackendRegistry('test')

    @registry.dispatch('test')
    def compute(x):
        return x + 1

    assert compute(1) == 2


@pytest.mark.parametrize(('operation', 'error'), [
    ('unknown_dispatch_test', ModuleNotFoundError),
    ('modules.unknown_dispatch_test', ModuleNotFoundError),
])
def test_dispatch_rejects_unknown_operation(monkeypatch, operation, error):
    monkeypatch.setattr(registry_module, '_registries', {})
    registry_module._registry_for(operation)
    with pytest.raises(error):
        dispatch(operation)


def test_dispatch_loads_owner_after_shared_backend_registration(monkeypatch, enable_dispatch):
    monkeypatch.setattr(registry_module, '_registries', {})

    @register_backend('attn')
    class SharedBackend(BaseBackend):
        backend_type = 'shared'

        def compute(self):
            return 'shared'

    registry = registry_module._registry_for('attn')
    imports = []

    def import_owner(name):
        imports.append(name)

        @register_backend('attn')
        class OwnerBackend(BaseBackend):
            backend_type = 'owner'
            priority = 0

            def compute(self):
                return 'owner'

        return SimpleNamespace()

    with monkeypatch.context() as patch:
        patch.setattr(registry_module.importlib, 'import_module', import_owner)
        decorate = dispatch('attn')

    assert imports == ['fla.ops.attn.backends']
    assert registry_module._registry_for('attn') is registry
    assert set(registry._backends) == {'shared', 'owner'}

    @decorate
    def compute():
        return 'default'

    assert compute() == 'owner'


@pytest.mark.parametrize('dependency', ['missing_optional_dependency', 'fla.ops.attn.backends.required'])
@pytest.mark.parametrize('allow_unknown', [False, True], ids=['central', 'legacy'])
def test_resolver_preserves_backend_dependency_errors(monkeypatch, dependency, allow_unknown):
    monkeypatch.setattr(registry_module, '_registries', {})
    registry_module._registry_for('attn')
    error = ModuleNotFoundError(f'Missing dependency: {dependency}', name=dependency)

    def import_owner(name):
        raise error

    monkeypatch.setattr(registry_module.importlib, 'import_module', import_owner)
    with pytest.raises(ModuleNotFoundError) as caught:
        registry_module._resolve_registry('attn', allow_unknown=allow_unknown)
    assert caught.value is error


@pytest.mark.skipif(registry_module._DISPATCH_DISABLED, reason='Backend dispatch was disabled before import')
@pytest.mark.parametrize('direction', ['fwd', 'bwd'], ids=['forward', 'backward'])
def test_aggregate_module_dispatch_preserves_arguments_and_result(monkeypatch, direction):
    from fla.modules import activations
    from fla.modules.backends import modules_registry
    from fla.modules.backends.triton_ascend import TritonAscendBackend

    assert dispatch('modules').__self__ is modules_registry
    name = f'logsigmoid_{direction}'
    monkeypatch.setattr(TritonAscendBackend, 'is_available', classmethod(lambda cls: True))
    monkeypatch.setattr(TritonAscendBackend, 'verify', lambda self, *args, **kwargs: (True, None))
    result = object()
    calls = []

    def implementation(self, x, **kwargs):
        assert torch.is_grad_enabled()
        calls.append((x, kwargs))
        return result

    monkeypatch.setattr(TritonAscendBackend, name, implementation)
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    kwargs = {'temperature': 0.5, 'output_contiguous': True}
    if direction == 'bwd':
        kwargs['dy'] = torch.ones_like(x)
    assert getattr(activations, name)(x, **kwargs) is result
    assert len(calls) == 1
    assert calls[0][0] is x
    assert calls[0][1].keys() == kwargs.keys()
    for argument, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            assert calls[0][1][argument] is value
        else:
            assert calls[0][1][argument] == value


@pytest.mark.parametrize(
    ('module_name', 'func_name'),
    [
        ('chunk_intra', 'chunk_gdn2_fwd_intra'),
        ('chunk_bwd', 'chunk_gdn2_bwd_wy_dqkg_fused'),
        ('fused_recurrent', 'fused_recurrent_gdn2_fwd'),
    ],
    ids=['forward', 'backward', 'fused-recurrent-forward'],
)
def test_gdn2_dispatch_uses_local_registry(monkeypatch, module_name, func_name):
    from fla.ops.gdn2.backends.triton_ascend import TritonAscendGDN2Backend

    entry = getattr(importlib.import_module(f'fla.ops.gdn2.{module_name}'), func_name)
    if registry_module._DISPATCH_DISABLED or not hasattr(entry, '__wrapped__'):
        pytest.skip('Backend dispatch was disabled before import')
    monkeypatch.setattr(TritonAscendGDN2Backend, 'is_available', classmethod(lambda cls: True))
    monkeypatch.setattr(TritonAscendGDN2Backend, 'is_enabled', classmethod(lambda cls: True))
    monkeypatch.setattr(TritonAscendGDN2Backend, f'{func_name}_verifier', lambda self, q: (True, None))
    result = object()

    def implementation(self, q):
        assert torch.is_grad_enabled()
        assert q.requires_grad
        return result

    monkeypatch.setattr(TritonAscendGDN2Backend, func_name, implementation)
    assert entry(q=torch.tensor([1.0, 2.0], requires_grad=True)) is result


@pytest.mark.parametrize('first_import', ['fla.backends', 'fla.ops.backends'])
def test_legacy_dispatch_uses_shared_registry(run_python, first_import):
    run_python(
        """
        import importlib
        import os
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', DeprecationWarning)
            importlib.import_module(os.environ['FIRST_IMPORT'])
            import fla.ops.backends as legacy
        matching = [
            warning for warning in caught
            if warning.category is DeprecationWarning
            and str(warning.message).startswith('fla.ops.backends is deprecated')
        ]
        assert len(matching) == 1, [str(w.message) for w in caught]
        assert 'next release after 0.6.0' in str(matching[0].message)

        from fla.backends import BaseBackend, dispatch
        from fla import backends as registry_module
        from fla.ops.gdn2.backends import gdn2_registry
        from fla.ops.kda.backends import kda_registry
        assert legacy.BaseBackend is BaseBackend
        assert legacy.BackendRegistry._registries is registry_module._registries
        assert legacy.BackendRegistry('gdn2') is gdn2_registry
        assert legacy.BackendRegistry('kda') is kda_registry
        legacy.BackendRegistry.ensure_initialized('kda')
        assert legacy.BackendRegistry._registries['kda'] is kda_registry

        class Backend(BaseBackend):
            backend_type = 'import_test'
            priority = -1

            def compute(self, x, use_backend):
                return x * 2

            def compute_verifier(self, x, use_backend):
                return use_backend, None

        legacy.BackendRegistry('kda').register(Backend())

        def compute(x, use_backend):
            return x * 3

        current = dispatch('kda')(compute)
        bound = kda_registry.dispatch(compute)
        old = legacy.dispatch('kda')(compute)
        for use_backend, factor in [(True, 2), (False, 3)]:
            for implementation in [current, bound, old]:
                assert implementation(2, use_backend=use_backend) == 2 * factor

        custom = legacy.BackendRegistry('custom_import_test')
        custom.register(Backend())
        assert legacy.dispatch('custom_import_test')(compute)(2, use_backend=True) == 4
        try:
            dispatch('custom_import_test')
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError('central dispatch must reject an operation without an owner package')

        from fla.modules.backends import modules_registry
        assert legacy.BackendRegistry('modules') is modules_registry
        assert dispatch('modules').__self__ is modules_registry
        assert legacy.dispatch('modules').__self__ is modules_registry
        """,
        FIRST_IMPORT=first_import,
        FLA_DISABLE_BACKEND_DISPATCH='0',
    )


@pytest.mark.parametrize('disabled', ['0', '1'], ids=['dispatch-enabled', 'dispatch-disabled'])
def test_dispatch_policy_and_optional_dependencies(run_python, disabled):
    run_python(
        """
        import os
        import sys
        import warnings

        from fla.backends import dispatch
        from fla import backends as registry_module
        enabled = os.environ['FLA_DISABLE_BACKEND_DISPATCH'] != '1'

        def compute(x):
            return x + 1

        for operation in [
            'attn',
            'attnres',
            'common',
            'gated_delta_rule',
            'gdn2',
            'generalized_delta_rule.dplr',
            'gla',
            'kda',
            'rwkv6',
            'utils',
            'modules',
        ]:
            with warnings.catch_warnings():
                warnings.simplefilter('error', DeprecationWarning)
                decorate = dispatch(operation)
            registry = registry_module._registries[operation]
            assert registry.operation_name == operation
            assert registry._backends
            wrapped = decorate(compute)
            assert (wrapped is compute) == (not enabled)
            assert wrapped(1) == 2

        from fla.ops.attn.backends.tilelang import AttnTileLangBackend
        from fla.ops.common.backends.tilelang import CommonTileLangBackend, TileLangBackend
        common = registry_module._registries['common']._backends['tilelang']
        attention = registry_module._registries['attn']._backends['tilelang']
        assert type(common) is CommonTileLangBackend is TileLangBackend
        assert type(attention) is AttnTileLangBackend
        assert hasattr(common, 'chunk_bwd_dqkwg')
        assert not hasattr(attention, 'chunk_bwd_dqkwg')
        for name in ['parallel_attn_fwd', 'parallel_attn_bwd']:
            assert hasattr(attention, name)
            assert not hasattr(common, name)
        assert 'tilelang' not in sys.modules
        assert 'flash_kda' not in sys.modules
        assert 'fla.ops.kda.backends.triton_ascend.chunk_intra' not in sys.modules
        assert not any(
            name.startswith('fla.modules.') and '.backends.triton_ascend.ops' in name
            for name in sys.modules
        )
        assert not any(name.startswith('fla.modules.backends.triton_ascend.') for name in sys.modules)
        assert 'fla.modules.backends.gluon.causal_conv1d' not in sys.modules
        assert 'fla.modules.conv.backends.gluon.ops' not in sys.modules
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            from fla.ops.backends import dispatch
        legacy = dispatch('kda')(compute)
        assert (legacy is compute) == (not enabled)
        assert legacy(1) == 2
        """,
        FLA_DISABLE_BACKEND_DISPATCH=disabled,
    )


def test_nvcc_from_cuda_home_env(monkeypatch, tmp_path):
    _configure_no_nvcc(monkeypatch)
    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))

    assert env.has_usable_nvcc() is True


def test_nvcc_from_path(monkeypatch):
    _configure_no_nvcc(monkeypatch)
    monkeypatch.setattr(env.shutil, "which", lambda name: "/usr/local/cuda/bin/nvcc")

    assert env.has_usable_nvcc() is True


def test_nvcc_from_pip_wheel(monkeypatch):
    _configure_no_nvcc(monkeypatch)
    monkeypatch.setattr(
        importlib.metadata,
        "files",
        lambda dist: [SimpleNamespace(name="ptxas"), SimpleNamespace(name="nvcc")],
    )

    assert env.has_usable_nvcc() is True


def test_nvcc_pip_wheel_without_nvcc_binary(monkeypatch):
    # nvidia-cuda-nvcc-cu12 ships only ptxas; it must not count as a usable compiler
    _configure_no_nvcc(monkeypatch)
    monkeypatch.setattr(importlib.metadata, "files", lambda dist: [SimpleNamespace(name="ptxas")])

    assert env.has_usable_nvcc() is False


def test_no_nvcc_logs_fallback_once(monkeypatch, caplog):
    _configure_no_nvcc(monkeypatch)

    with caplog.at_level(logging.INFO, logger=env.__name__):
        assert env.has_usable_nvcc() is False
        assert env.has_usable_nvcc() is False

    fallback_messages = [record.message for record in caplog.records if "falling back to Triton" in record.message]
    assert len(fallback_messages) == 1
    assert "FLA_TILELANG=0" in fallback_messages[0]


@pytest.mark.parametrize("backend_module", [
    attn_tilelang_backend, common_tilelang_backend, kda_tilelang_backend, rwkv6_tilelang_backend, dplr_tilelang_backend,
])
def test_tilelang_backend_gated_by_nvcc_probe(monkeypatch, backend_module):
    monkeypatch.setattr(backend_module, "_TILELANG_AVAILABLE", True)
    monkeypatch.setattr(backend_module, "has_usable_nvcc", lambda: False)
    assert _backend_cls(backend_module).is_available() is False

    monkeypatch.setattr(backend_module, "has_usable_nvcc", lambda: True)
    assert _backend_cls(backend_module).is_available() is True


@pytest.mark.parametrize("backend_module", [
    attn_tilelang_backend, common_tilelang_backend, kda_tilelang_backend, rwkv6_tilelang_backend, dplr_tilelang_backend,
])
def test_tilelang_backend_unavailable_without_tilelang(monkeypatch, backend_module):
    monkeypatch.setattr(backend_module, "_TILELANG_AVAILABLE", False)
    monkeypatch.setattr(backend_module, "has_usable_nvcc", lambda: True)
    assert _backend_cls(backend_module).is_available() is False


@pytest.mark.parametrize('backend_module', [attn_tilelang_backend, common_tilelang_backend], ids=['attn', 'common'])
@pytest.mark.parametrize('setting', [None, '0', '1'], ids=['default', 'disabled', 'enabled'])
def test_tilelang_backend_default_and_override(monkeypatch, backend_module, setting):
    if setting is None:
        monkeypatch.delenv('FLA_TILELANG', raising=False)
    else:
        monkeypatch.setenv('FLA_TILELANG', setting)
    expected = backend_module.IS_NVIDIA_HOPPER and backend_module.TRITON_ABOVE_3_4_0
    if setting is not None:
        expected = setting != '0'
    assert _backend_cls(backend_module).is_enabled() is expected


def test_rwkv6_tilelang_backend_requires_opt_in(monkeypatch):
    monkeypatch.delenv("FLA_TILELANG", raising=False)
    assert rwkv6_tilelang_backend.RWKV6TileLangBackend.is_enabled() is False

    monkeypatch.setenv("FLA_TILELANG", "1")
    assert rwkv6_tilelang_backend.RWKV6TileLangBackend.is_enabled() is True


def test_rwkv6_tilelang_backend_verifier_accepts_supported_shape():
    q = SimpleNamespace(dtype=torch.bfloat16, is_cuda=True, shape=(1, 64, 2, 64), ndim=4)
    k = SimpleNamespace(dtype=torch.bfloat16, shape=q.shape)
    gi = SimpleNamespace(dtype=torch.float32, shape=q.shape)
    ge = SimpleNamespace(dtype=torch.float32, shape=q.shape)
    u = SimpleNamespace(dtype=torch.bfloat16, shape=(2, 64))

    accepted, reason = rwkv6_tilelang_backend.RWKV6TileLangBackend().chunk_rwkv6_fwd_intra_verifier(
        q=q,
        k=k,
        gi=gi,
        ge=ge,
        u=u,
        scale=1.0,
    )

    assert accepted is True
    assert reason is None


def test_rwkv6_tilelang_backend_verifier_rejects_unsupported_dimension():
    q = SimpleNamespace(dtype=torch.bfloat16, is_cuda=True, shape=(1, 64, 2, 128), ndim=4)
    k = SimpleNamespace(dtype=torch.bfloat16, shape=q.shape)
    gi = SimpleNamespace(dtype=torch.float32, shape=q.shape)
    ge = SimpleNamespace(dtype=torch.float32, shape=q.shape)
    u = SimpleNamespace(dtype=torch.bfloat16, shape=(2, 128))

    accepted, reason = rwkv6_tilelang_backend.RWKV6TileLangBackend().chunk_rwkv6_fwd_intra_verifier(
        q=q,
        k=k,
        gi=gi,
        ge=ge,
        u=u,
        scale=1.0,
    )

    assert accepted is False
    assert reason == "TileLang RWKV6 intra backend currently supports the D=64 benchmark bucket only, got K=128"
