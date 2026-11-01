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

from fla import backends
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


@pytest.mark.parametrize(
    ('package_available', 'nvcc_available', 'expected'),
    [(True, False, False), (True, True, True), (False, True, False)],
)
def test_tilelang_available(monkeypatch, package_available, nvcc_available, expected):
    monkeypatch.setattr(backends, 'find_spec_cached', lambda name: object() if package_available else None)
    monkeypatch.setattr(backends, 'has_usable_nvcc', lambda: nvcc_available)
    assert backends.TileLangBackend.is_available() is expected


@pytest.mark.parametrize(('setting', 'expected'), [(None, True), ('0', False), ('1', True)])
def test_tilelang_enabled(monkeypatch, setting, expected):
    if setting is None:
        monkeypatch.delenv('FLA_TILELANG', raising=False)
    else:
        monkeypatch.setenv('FLA_TILELANG', setting)
    assert backends.TileLangBackend().is_enabled() is expected
