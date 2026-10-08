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

from fla.ops.attn.backends.gluon import AttnGluonBackend
from fla.ops.backends import BaseBackend
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


def _backend_cls(backend_module):
    if backend_module is common_tilelang_backend:
        return backend_module.TileLangBackend
    if backend_module is rwkv6_tilelang_backend:
        return backend_module.RWKV6TileLangBackend
    if backend_module is kda_tilelang_backend:
        return backend_module.KDATileLangBackend
    if backend_module is dplr_tilelang_backend:
        return backend_module.DPLRTileLangBackend
    raise ValueError(f"unrecognized TileLang backend module: {backend_module}")


@pytest.mark.parametrize("backend_module", [common_tilelang_backend, kda_tilelang_backend, rwkv6_tilelang_backend, dplr_tilelang_backend])
def test_tilelang_backend_gated_by_nvcc_probe(monkeypatch, backend_module):
    monkeypatch.setattr(backend_module, "_TILELANG_AVAILABLE", True)
    monkeypatch.setattr(backend_module, "has_usable_nvcc", lambda: False)
    assert _backend_cls(backend_module).is_available() is False

    monkeypatch.setattr(backend_module, "has_usable_nvcc", lambda: True)
    assert _backend_cls(backend_module).is_available() is True


@pytest.mark.parametrize("backend_module", [common_tilelang_backend, kda_tilelang_backend, rwkv6_tilelang_backend, dplr_tilelang_backend])
def test_tilelang_backend_unavailable_without_tilelang(monkeypatch, backend_module):
    monkeypatch.setattr(backend_module, "_TILELANG_AVAILABLE", False)
    monkeypatch.setattr(backend_module, "has_usable_nvcc", lambda: True)
    assert _backend_cls(backend_module).is_available() is False


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


def test_attn_gluon_backend_requires_opt_in(monkeypatch):
    monkeypatch.delenv('FLA_GLUON', raising=False)
    monkeypatch.delenv('FLA_ATTN_GLUON', raising=False)
    assert not AttnGluonBackend.is_enabled()
    monkeypatch.setenv('FLA_ATTN_GLUON', '1')
    assert AttnGluonBackend.is_enabled()
    monkeypatch.setenv('FLA_ATTN_GLUON', '0')
    monkeypatch.setenv('FLA_GLUON', '1')
    assert AttnGluonBackend.is_enabled()


@pytest.mark.parametrize('method', ['parallel_attn_fwd', 'parallel_attn_bwd', 'attn_decoding_fwd'])
@pytest.mark.parametrize(
    ('device_type', 'capability', 'dtype', 'K', 'V', 'reason'),
    [
        pytest.param('cpu', 10, torch.float16, 64, 64, 'compute capability', id='cpu'),
        pytest.param('cuda', 8, torch.float16, 64, 64, 'compute capability', id='unsupported-device'),
        pytest.param('cuda', 10, torch.float32, 64, 64, 'matching fp16 or bf16', id='fp32'),
        pytest.param('cuda', 10, torch.float64, 64, 64, 'matching fp16 or bf16', id='fp64'),
        pytest.param('cuda', 10, torch.float16, 0, 64, 'dimensions', id='empty-key'),
        pytest.param('cuda', 10, torch.float16, 64, 0, 'dimensions', id='empty-value'),
        pytest.param('cuda', 10, torch.float16, 513, 64, 'dimensions', id='wide-key'),
        pytest.param('cuda', 10, torch.float16, 64, 513, 'dimensions', id='wide-value'),
        pytest.param('cuda', 9, torch.float16, 256, 512, None, id='fp16'),
        pytest.param('cuda', 10, torch.bfloat16, 512, 256, None, id='bf16'),
    ],
)
def test_attn_gluon_backend_verifier(monkeypatch, method, device_type, capability, dtype, K, V, reason):
    monkeypatch.setattr('fla.ops.attn.backends.gluon.get_device_capability', lambda *args: (capability, 0))
    q = SimpleNamespace(device=SimpleNamespace(type=device_type, index=0), dtype=dtype, shape=(1, 1, 1, K))
    v = SimpleNamespace(dtype=dtype, shape=(1, 1, 1, V))
    kwargs = dict(q=q, k=q, v=v, g_cumsum=None, sink_bias=None, scale=0.125)
    if method == 'parallel_attn_bwd':
        kwargs.update(o=None, lse=None, do=None)
    elif method == 'attn_decoding_fwd':
        kwargs['cu_seqlens'] = None
    accepted, actual_reason = getattr(AttnGluonBackend(), method + '_verifier')(**kwargs)
    assert accepted is (reason is None)
    if reason is None:
        assert actual_reason is None
    else:
        assert reason in actual_reason
