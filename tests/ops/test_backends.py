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

import fla.ops.common.backends.tilelang as common_tilelang_backend
import fla.ops.kda.backends.tilelang as kda_tilelang_backend
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
    return backend_module.KDATileLangBackend


@pytest.mark.parametrize("backend_module", [common_tilelang_backend, kda_tilelang_backend])
def test_tilelang_backend_gated_by_nvcc_probe(monkeypatch, backend_module):
    monkeypatch.setattr(backend_module, "_TILELANG_AVAILABLE", True)
    monkeypatch.setattr(backend_module, "has_usable_nvcc", lambda: False)
    assert _backend_cls(backend_module).is_available() is False

    monkeypatch.setattr(backend_module, "has_usable_nvcc", lambda: True)
    assert _backend_cls(backend_module).is_available() is True


@pytest.mark.parametrize("backend_module", [common_tilelang_backend, kda_tilelang_backend])
def test_tilelang_backend_unavailable_without_tilelang(monkeypatch, backend_module):
    monkeypatch.setattr(backend_module, "_TILELANG_AVAILABLE", False)
    monkeypatch.setattr(backend_module, "has_usable_nvcc", lambda: True)
    assert _backend_cls(backend_module).is_available() is False
