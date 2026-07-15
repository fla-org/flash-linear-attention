# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import logging

import pytest

import fla.ops.common.backends.tilelang as tilelang_backend


@pytest.fixture(autouse=True)
def clear_nvcc_probe_cache():
    tilelang_backend._has_usable_nvcc.cache_clear()
    yield
    tilelang_backend._has_usable_nvcc.cache_clear()


def _configure_tilelang_without_nvcc(monkeypatch):
    monkeypatch.setattr(tilelang_backend, "_TILELANG_AVAILABLE", True)
    monkeypatch.setattr(tilelang_backend.shutil, "which", lambda name: None)
    monkeypatch.setattr(tilelang_backend.cpp_extension, "CUDA_HOME", None)
    monkeypatch.setattr(tilelang_backend, "find_spec_cached", lambda name: None)


def test_tilelang_available_with_path_nvcc(monkeypatch):
    _configure_tilelang_without_nvcc(monkeypatch)
    monkeypatch.setattr(tilelang_backend.shutil, "which", lambda name: "/usr/local/cuda/bin/nvcc")

    assert tilelang_backend.TileLangBackend.is_available() is True


def test_tilelang_available_with_cuda_home_nvcc(monkeypatch, tmp_path):
    _configure_tilelang_without_nvcc(monkeypatch)
    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    monkeypatch.setattr(tilelang_backend.cpp_extension, "CUDA_HOME", str(nvcc.parents[1]))

    assert tilelang_backend.TileLangBackend.is_available() is True


def test_tilelang_available_with_pip_nvcc(monkeypatch):
    _configure_tilelang_without_nvcc(monkeypatch)
    monkeypatch.setattr(
        tilelang_backend,
        "find_spec_cached",
        lambda name: object() if name == "nvidia.cuda_nvcc" else None,
    )

    assert tilelang_backend.TileLangBackend.is_available() is True


def test_tilelang_unavailable_without_nvcc(monkeypatch, caplog):
    _configure_tilelang_without_nvcc(monkeypatch)

    def find_missing_spec(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(tilelang_backend, "find_spec_cached", find_missing_spec)

    with caplog.at_level(logging.INFO, logger=tilelang_backend.__name__):
        assert tilelang_backend.TileLangBackend.is_available() is False
        assert tilelang_backend.TileLangBackend.is_available() is False

    fallback_messages = [record.message for record in caplog.records if "falling back to Triton" in record.message]
    assert len(fallback_messages) == 1
    assert "FLA_TILELANG=0" in fallback_messages[0]
