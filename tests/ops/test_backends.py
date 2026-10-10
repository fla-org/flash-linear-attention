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

from fla import backends, utils
from fla.ops.attn.backends import tilelang as attn_tilelang_backend
from fla.ops.common.backends import tilelang as common_tilelang_backend
from fla.ops.generalized_delta_rule.dplr.backends import tilelang as dplr_tilelang_backend
from fla.ops.kda.backends import tilelang as kda_tilelang_backend
from fla.ops.rwkv6.backends import tilelang as rwkv6_tilelang_backend
from fla.ops.simple_gla.backends.triton_ascend.utils import simple_gla_verifier
from fla.utils import env

_REAL_PATH_EXISTS = Path.exists
_TILELANG_BACKENDS = [
    pytest.param(attn_tilelang_backend.AttnTileLangBackend, id='attn'),
    pytest.param(common_tilelang_backend.CommonTileLangBackend, id='common'),
    pytest.param(kda_tilelang_backend.KDATileLangBackend, id='kda'),
    pytest.param(rwkv6_tilelang_backend.RWKV6TileLangBackend, id='rwkv6'),
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


def test_attention_and_common_backends_have_separate_implementations():
    common = backends._load_operation_registry('common')._backends['tilelang']
    attention = backends._load_operation_registry('attn')._backends['tilelang']
    assert type(common) is common_tilelang_backend.CommonTileLangBackend
    assert type(attention) is attn_tilelang_backend.AttnTileLangBackend
    assert hasattr(common, 'chunk_bwd_dqkwg')
    assert not hasattr(attention, 'chunk_bwd_dqkwg')
    for name in ('parallel_attn_fwd', 'parallel_attn_bwd'):
        assert hasattr(attention, name)
        assert not hasattr(common, name)


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
            attn_tilelang_backend.AttnTileLangBackend,
            utils.IS_NVIDIA_HOPPER and utils.TRITON_ABOVE_3_4_0,
            id='attn',
        ),
        pytest.param(
            common_tilelang_backend.CommonTileLangBackend,
            utils.IS_NVIDIA_HOPPER and utils.TRITON_ABOVE_3_4_0,
            id='common',
        ),
        pytest.param(kda_tilelang_backend.KDATileLangBackend, True, id='kda'),
        pytest.param(rwkv6_tilelang_backend.RWKV6TileLangBackend, False, id='rwkv6'),
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
