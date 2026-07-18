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

import fla.ops.common.backends.tilelang as common_tilelang_backend
import fla.ops.kda.backends.tilelang as kda_tilelang_backend
import fla.ops.rwkv6.backends.tilelang as rwkv6_tilelang_backend
from fla.utils import _compat

_REAL_PATH_EXISTS = Path.exists


@pytest.fixture(autouse=True)
def clear_nvcc_probe_cache():
    _compat.has_usable_nvcc.cache_clear()
    yield
    _compat.has_usable_nvcc.cache_clear()


def _configure_no_nvcc(monkeypatch):
    """Hide every nvcc source probed by has_usable_nvcc (CI runners have a real toolkit)."""
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(_compat.shutil, "which", lambda name: None)

    def no_such_dist(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "files", no_such_dist)

    def fake_exists(self):
        if str(self).startswith("/usr/local/cuda"):
            return False
        return _REAL_PATH_EXISTS(self)

    monkeypatch.setattr(_compat.Path, "exists", fake_exists)


def test_nvcc_from_cuda_home_env(monkeypatch, tmp_path):
    _configure_no_nvcc(monkeypatch)
    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))

    assert _compat.has_usable_nvcc() is True


def test_nvcc_from_path(monkeypatch):
    _configure_no_nvcc(monkeypatch)
    monkeypatch.setattr(_compat.shutil, "which", lambda name: "/usr/local/cuda/bin/nvcc")

    assert _compat.has_usable_nvcc() is True


def test_nvcc_from_pip_wheel(monkeypatch):
    _configure_no_nvcc(monkeypatch)
    monkeypatch.setattr(
        importlib.metadata,
        "files",
        lambda dist: [SimpleNamespace(name="ptxas"), SimpleNamespace(name="nvcc")],
    )

    assert _compat.has_usable_nvcc() is True


def test_nvcc_pip_wheel_without_nvcc_binary(monkeypatch):
    # nvidia-cuda-nvcc-cu12 ships only ptxas; it must not count as a usable compiler
    _configure_no_nvcc(monkeypatch)
    monkeypatch.setattr(importlib.metadata, "files", lambda dist: [SimpleNamespace(name="ptxas")])

    assert _compat.has_usable_nvcc() is False


def test_no_nvcc_logs_fallback_once(monkeypatch, caplog):
    _configure_no_nvcc(monkeypatch)

    with caplog.at_level(logging.INFO, logger=_compat.__name__):
        assert _compat.has_usable_nvcc() is False
        assert _compat.has_usable_nvcc() is False

    fallback_messages = [record.message for record in caplog.records if "falling back to Triton" in record.message]
    assert len(fallback_messages) == 1
    assert "FLA_TILELANG=0" in fallback_messages[0]


def _backend_cls(backend_module):
    if backend_module is common_tilelang_backend:
        return backend_module.TileLangBackend
    if backend_module is rwkv6_tilelang_backend:
        return backend_module.RWKV6TileLangBackend
    return backend_module.KDATileLangBackend


@pytest.mark.parametrize("backend_module", [common_tilelang_backend, kda_tilelang_backend, rwkv6_tilelang_backend])
def test_tilelang_backend_gated_by_nvcc_probe(monkeypatch, backend_module):
    monkeypatch.setattr(backend_module, "_TILELANG_AVAILABLE", True)
    monkeypatch.setattr(backend_module, "has_usable_nvcc", lambda: False)
    assert _backend_cls(backend_module).is_available() is False

    monkeypatch.setattr(backend_module, "has_usable_nvcc", lambda: True)
    assert _backend_cls(backend_module).is_available() is True


@pytest.mark.parametrize("backend_module", [common_tilelang_backend, kda_tilelang_backend, rwkv6_tilelang_backend])
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
