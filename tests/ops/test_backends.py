# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib
import importlib.metadata
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch

from fla import backends, utils
from fla.ops.common.backends import tilelang as common_tilelang_backend
from fla.ops.generalized_delta_rule.dplr.backends import tilelang as dplr_tilelang_backend
from fla.ops.kda.backends import tilelang as kda_tilelang_backend
from fla.ops.simple_gla.backends.triton_ascend.utils import simple_gla_verifier
from fla.utils import env

_REAL_PATH_EXISTS = Path.exists
_TILELANG_BACKENDS = [
    pytest.param(common_tilelang_backend.CommonTileLangBackend, id="common"),
    pytest.param(kda_tilelang_backend.KDATileLangBackend, id='kda'),
    pytest.param(dplr_tilelang_backend.DPLRTileLangBackend, id='dplr'),
]


@pytest.fixture(autouse=True)
def clear_nvcc_probe_cache():
    env.has_usable_nvcc.cache_clear()
    yield
    env.has_usable_nvcc.cache_clear()


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
    if backends._DISPATCH_DISABLED:
        pytest.skip('Backend dispatch was disabled before import')
    assert hasattr(entry, '__wrapped__')
    monkeypatch.setattr(TritonAscendGDN2Backend, 'is_available', classmethod(lambda cls: True))
    monkeypatch.setattr(TritonAscendGDN2Backend, 'is_enabled', lambda self: True)
    monkeypatch.setattr(TritonAscendGDN2Backend, f'{func_name}_verifier', lambda self, q: (True, None))
    result = object()

    def implementation(self, q):
        assert torch.is_grad_enabled()
        assert q.requires_grad
        return result

    monkeypatch.setattr(TritonAscendGDN2Backend, func_name, implementation)
    assert entry(q=torch.tensor([1.0, 2.0], requires_grad=True)) is result


@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason='Backend dispatch was disabled before import')
@pytest.mark.parametrize('route', ['accepted', 'disabled', 'unavailable', 'training', 'gate-not-fused'])
def test_flash_kda_registered_dispatch(monkeypatch, route):
    from fla.ops.kda.backends.flash_kda import FlashKDABackend
    from fla.ops.kda.chunk import ChunkKDAFunction, chunk_kda

    monkeypatch.setenv('FLA_FLASH_KDA', '0' if route == 'disabled' else '1')
    monkeypatch.setattr(FlashKDABackend, 'is_available', classmethod(lambda cls: route != 'unavailable'))
    calls = []

    def flash_forward(self, *args, **kwargs):
        calls.append('flash_kda')
        return args[2], None

    def triton_forward(*args):
        calls.append('triton')
        return args[2], None

    monkeypatch.setattr(FlashKDABackend, 'chunk_kda', flash_forward)
    monkeypatch.setattr(ChunkKDAFunction, 'apply', triton_forward)
    q = torch.ones(1, 2, 1, 128, dtype=torch.bfloat16)
    beta = torch.ones(1, 2, 1, dtype=torch.bfloat16)
    with torch.set_grad_enabled(route == 'training'):
        result = chunk_kda(
            q,
            q,
            q,
            q,
            beta,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=route != 'gate-not-fused',
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=-5,
            state_v_first=True,
            A_log=torch.ones(1),
            dt_bias=torch.ones(128),
        )
    assert result[0] is q
    assert result[1] is None
    assert calls == ['flash_kda' if route == 'accepted' else 'triton']


@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason="Backend dispatch was disabled before import")
def test_attention_and_common_backends_have_separate_implementations():
    import inspect

    from fla.ops.attn.parallel import parallel_attn_bwd, parallel_attn_fwd

    if not backends.TileLangBackend.is_available():
        pytest.skip('TileLang is unavailable')
    common = backends._load_operation_registry('common')._backends['tilelang']
    assert type(common) is common_tilelang_backend.CommonTileLangBackend
    for entry in [parallel_attn_fwd, parallel_attn_bwd]:
        candidate = backends._function_registries[inspect.unwrap(entry)]._backends['tilelang']
        assert type(candidate) is backends.TileLangBackend
        assert candidate.implementation.__name__ == f'{entry.__name__}_tilelang'
        assert not hasattr(common, entry.__name__)


@pytest.fixture
def rwkv6_tilelang_backend():
    if not backends.TileLangBackend.is_available():
        pytest.skip('TileLang is unavailable')
    return importlib.import_module('fla.ops.rwkv6.backends.tilelang')


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


@pytest.mark.parametrize('backend_class', _TILELANG_BACKENDS)
def test_tilelang_backend_gated_by_nvcc_probe(monkeypatch, backend_class):
    monkeypatch.setattr(backends, "find_spec_cached", lambda name: object())
    monkeypatch.setattr(backends, "has_usable_nvcc", lambda: False)
    assert backend_class.is_available() is False

    monkeypatch.setattr(backends, "has_usable_nvcc", lambda: True)
    assert backend_class.is_available() is True


@pytest.mark.parametrize('backend_class', _TILELANG_BACKENDS)
def test_tilelang_backend_unavailable_without_tilelang(monkeypatch, backend_class):
    monkeypatch.setattr(backends, "find_spec_cached", lambda name: None)
    monkeypatch.setattr(backends, "has_usable_nvcc", lambda: True)
    assert backend_class.is_available() is False


@pytest.mark.parametrize(
    ('backend_class', 'default_enabled'),
    [
        pytest.param(
            common_tilelang_backend.CommonTileLangBackend,
            utils.IS_NVIDIA_HOPPER and utils.TRITON_ABOVE_3_4_0,
            id='common',
        ),
        pytest.param(kda_tilelang_backend.KDATileLangBackend, True, id='kda'),
        pytest.param(dplr_tilelang_backend.DPLRTileLangBackend, True, id='dplr'),
    ],
)
@pytest.mark.parametrize('setting', [None, '0', '1'], ids=['default', 'disabled', 'enabled'])
def test_tilelang_backend_default_and_override(monkeypatch, backend_class, default_enabled, setting):
    if setting is None:
        monkeypatch.delenv('FLA_TILELANG', raising=False)
    else:
        monkeypatch.setenv('FLA_TILELANG', setting)
    expected = default_enabled if setting is None else setting != '0'
    assert backend_class().is_enabled() is expected


def test_rwkv6_tilelang_backend_verifier_accepts_supported_shape(rwkv6_tilelang_backend):
    q = SimpleNamespace(dtype=torch.bfloat16, is_cuda=True, shape=(1, 64, 2, 64), ndim=4)
    k = SimpleNamespace(dtype=torch.bfloat16, shape=q.shape)
    gi = SimpleNamespace(dtype=torch.float32, shape=q.shape)
    ge = SimpleNamespace(dtype=torch.float32, shape=q.shape)
    u = SimpleNamespace(dtype=torch.bfloat16, shape=(2, 64))

    accepted, reason = rwkv6_tilelang_backend.chunk_rwkv6_fwd_intra_verifier(
        q=q,
        k=k,
        gi=gi,
        ge=ge,
        u=u,
        scale=1.0,
    )

    assert accepted is True
    assert reason is None


def test_rwkv6_tilelang_backend_verifier_rejects_unsupported_dimension(rwkv6_tilelang_backend):
    q = SimpleNamespace(dtype=torch.bfloat16, is_cuda=True, shape=(1, 64, 2, 128), ndim=4)
    k = SimpleNamespace(dtype=torch.bfloat16, shape=q.shape)
    gi = SimpleNamespace(dtype=torch.float32, shape=q.shape)
    ge = SimpleNamespace(dtype=torch.float32, shape=q.shape)
    u = SimpleNamespace(dtype=torch.bfloat16, shape=(2, 128))

    accepted, reason = rwkv6_tilelang_backend.chunk_rwkv6_fwd_intra_verifier(
        q=q,
        k=k,
        gi=gi,
        ge=ge,
        u=u,
        scale=1.0,
    )

    assert accepted is False
    assert reason == "TileLang RWKV6 intra backend currently supports the D=64 benchmark bucket only, got K=128"


@pytest.fixture
def kda_bwd_inputs():
    x = SimpleNamespace(shape=(1, 64, 2, 64))
    return dict(q=x, k=x, v=x, v_new=x, g=x, beta=x, A=x, h=x, do=x, dh=x, dv=x, chunk_offsets=None)


@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason='Backend dispatch was disabled before import')
def test_kda_tilelang_backend_dispatch(monkeypatch, kda_bwd_inputs):
    from fla.ops.kda.backends.triton_ascend import TritonAscendKDABackend
    from fla.ops.kda.chunk_bwd import chunk_kda_bwd_wy_dqkg_fused

    monkeypatch.setattr(TritonAscendKDABackend, 'is_available', classmethod(lambda cls: False))
    backend = kda_tilelang_backend.KDATileLangBackend
    monkeypatch.setattr(backend, 'is_available', classmethod(lambda cls: True))
    monkeypatch.setenv('FLA_TILELANG', '1')
    result = object()
    implementation = create_autospec(backend.chunk_kda_bwd_wy_dqkg_fused, return_value=result)
    monkeypatch.setattr(backend, 'chunk_kda_bwd_wy_dqkg_fused', implementation)

    assert chunk_kda_bwd_wy_dqkg_fused(**kda_bwd_inputs, use_graph=False) is result
    implementation.assert_called_once()


@pytest.mark.parametrize(
    ('use_graph', 'accepted', 'reason'),
    [
        pytest.param(False, True, None, id='eager'),
        pytest.param(
            True,
            False,
            'TileLang backend does not support graph capture (use_graph=True); fall back to Triton',
            id='graph',
        ),
    ],
)
def test_kda_tilelang_backend_verifier(kda_bwd_inputs, use_graph, accepted, reason):
    backend = kda_tilelang_backend.KDATileLangBackend()
    assert backend.chunk_kda_bwd_wy_dqkg_fused_verifier(**kda_bwd_inputs, use_graph=use_graph) == (accepted, reason)


@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason='Backend dispatch was disabled before import')
@pytest.mark.parametrize('extra', [{}, {'chunk_offsets': None}], ids=['chunk_offsets_omitted', 'chunk_offsets_none'])
def test_intracard_backend_dispatch(monkeypatch, extra):
    """KDA's eager forward passes chunk_offsets=None; the opt-in intra-card backend must still be selected."""
    from fla.ops.common.backends.intracard import IntraCardCPBackend
    from fla.ops.common.backends.triton_ascend import TritonAscendCommonBackend
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h

    monkeypatch.setattr(TritonAscendCommonBackend, 'is_available', classmethod(lambda cls: False))
    monkeypatch.setenv('FLA_INTRACARD_CP', '1')
    result = object()
    implementation = create_autospec(IntraCardCPBackend.chunk_gated_delta_rule_fwd_h, return_value=result)
    monkeypatch.setattr(IntraCardCPBackend, 'chunk_gated_delta_rule_fwd_h', implementation)

    x = SimpleNamespace(shape=(1, 64, 2, 64))
    with torch.inference_mode():
        assert chunk_gated_delta_rule_fwd_h(k=x, w=x, u=x, cu_seqlens=torch.tensor([0, 64]), **extra) is result
    implementation.assert_called_once()


def test_intracard_backend_verifier_rejects_graph_mode():
    """Graph-mode callers pass fixed-shape chunk_offsets; intra-card CP cannot be captured."""
    from fla.ops.common.backends.intracard import IntraCardCPBackend

    x = SimpleNamespace(shape=(1, 64, 2, 64))
    with torch.inference_mode():
        verdict = IntraCardCPBackend().chunk_gated_delta_rule_fwd_h_verifier(
            k=x, w=x, u=x, cu_seqlens=torch.tensor([0, 64]), chunk_offsets=torch.tensor([0, 1]),
        )
    assert verdict == (False, 'chunk_offsets is given (CUDA Graph mode)')


@pytest.mark.parametrize(
    ('is_npu', 'device', 'accepted', 'reason'),
    [
        (False, 'cpu', False, 'not running on NPU'),
        (True, 'cpu', False, 'input device is not NPU'),
        (True, None, False, 'input device is not NPU'),
        (True, 'npu', True, None),
    ],
    ids=['off-npu', 'cpu-input', 'missing-input', 'npu-input'],
)
def test_simple_gla_ascend_verifier(monkeypatch, is_npu, device, accepted, reason):
    monkeypatch.setattr(utils, 'IS_NPU', is_npu)
    q = SimpleNamespace(device=SimpleNamespace(type=device)) if device is not None else None
    assert simple_gla_verifier(q=q, k=q, v=q) == (accepted, reason)


def test_retention_ascend_registration_import_order(run_python):
    run_python(
        """
        import importlib
        import inspect
        import sys

        from fla import backends
        from fla.ops.simple_gla.backends.triton_ascend.utils import simple_gla_verifier

        backends.TritonAscendBackend.is_available = classmethod(lambda cls: True)
        for operation, name in [
            ('common.chunk_o', 'chunk_bwd_dv'),
            ('simple_gla.fused_chunk', 'fused_chunk_simple_gla'),
            ('simple_gla.parallel', 'parallel_simple_gla'),
            ('simple_gla.fused_recurrent', 'fused_recurrent_simple_gla'),
        ]:
            module = 'fla.ops.' + operation
            owner, file = operation.split('.')
            backend_module = f'fla.ops.{owner}.backends.triton_ascend.{file}'
            importlib.reload(importlib.import_module(f'fla.ops.{owner}'))
            entry = getattr(importlib.import_module(module), name)
            implementation = sys.modules[backend_module]
            registry = backends._function_registries[inspect.unwrap(entry)]
            assert len(registry._backends) == 1
            candidate = registry._backends['triton_ascend']
            assert candidate.implementation is getattr(implementation, name + '_npu')
            assert candidate.verifier is (simple_gla_verifier if owner == 'simple_gla' else None)
            expected = inspect.signature(entry).parameters
            actual = inspect.signature(candidate.implementation).parameters
            assert expected.keys() == actual.keys()
            for parameter in expected:
                assert expected[parameter].kind == actual[parameter].kind
                assert expected[parameter].default == actual[parameter].default
        assert 'simple_gla' not in backends._operation_registries
        """,
        FLA_DISABLE_BACKEND_DISPATCH='0',
    )


@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason='Backend dispatch was disabled before import')
@pytest.mark.parametrize(
    ('module_name', 'entry_name', 'backend_type'),
    [
        ('fla.ops.attn.parallel', 'parallel_attn_fwd', 'tilelang'),
        ('fla.ops.attn.parallel', 'parallel_attn_bwd', 'tilelang'),
        ('fla.ops.attnres.fused', 'fused_attnres', 'triton_ascend'),
        ('fla.ops.attnres.fused', 'fused_attnres', 'gluon'),
        ('fla.ops.rwkv6.chunk', 'chunk_rwkv6_fwd_intra', 'tilelang'),
    ],
)
def test_registered_op_dispatch(monkeypatch, module_name, entry_name, backend_type):
    import inspect

    entry = getattr(importlib.import_module(module_name), entry_name)
    registry = backends._function_registries[inspect.unwrap(entry)]
    backend_class = {
        'triton_ascend': backends.TritonAscendBackend,
        'tilelang': backends.TileLangBackend,
        'gluon': backends.GluonBackend,
    }[backend_type]
    available = backend_class.is_available()
    assert (backend_type in registry._backends) == available
    if not available:
        return
    candidate = registry._backends[backend_type]
    assert type(candidate) is backend_class
    suffix = 'npu' if backend_type == 'triton_ascend' else backend_type
    assert candidate.implementation.__name__ == f'{entry_name}_{suffix}'
    if backend_type in ('tilelang', 'gluon'):
        parameters = inspect.signature(entry).parameters.values()
        arguments = {p.name: object() for p in parameters if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
        inspect.signature(candidate.implementation).bind(**arguments)
        inspect.signature(candidate.implementation).bind(*arguments.values())
    for other in registry._backends.values():
        monkeypatch.setattr(other, 'is_enabled', lambda: False)
    monkeypatch.setattr(candidate, 'is_enabled', lambda: True)
    monkeypatch.setattr(candidate, 'verifier', lambda *args, **kwargs: (True, None))
    result = object()
    monkeypatch.setattr(candidate, 'implementation', lambda *args, **kwargs: result)
    assert entry() is result


@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason='Backend dispatch was disabled before import')
@pytest.mark.parametrize(
    ('module_name', 'entry_name', 'default_enabled'),
    [
        ('fla.ops.attn.parallel', 'parallel_attn_fwd', utils.IS_NVIDIA_HOPPER and utils.TRITON_ABOVE_3_4_0),
        ('fla.ops.attn.parallel', 'parallel_attn_bwd', utils.IS_NVIDIA_HOPPER and utils.TRITON_ABOVE_3_4_0),
        ('fla.ops.rwkv6.chunk', 'chunk_rwkv6_fwd_intra', False),
    ],
)
@pytest.mark.parametrize('setting', [None, '0', '1'])
def test_registered_tilelang_default_and_override(monkeypatch, module_name, entry_name, default_enabled, setting):
    import inspect

    if not backends.TileLangBackend.is_available():
        pytest.skip('TileLang is unavailable')
    entry = getattr(importlib.import_module(module_name), entry_name)
    candidate = backends._function_registries[inspect.unwrap(entry)]._backends['tilelang']
    if setting is None:
        monkeypatch.delenv('FLA_TILELANG', raising=False)
    else:
        monkeypatch.setenv('FLA_TILELANG', setting)
    assert candidate.default_enable is default_enabled
    assert candidate.is_enabled() is (default_enabled if setting is None else setting != '0')
