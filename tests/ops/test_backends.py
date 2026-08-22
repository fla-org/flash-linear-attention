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
import fla.ops.generalized_delta_rule.dplr.backends.tilelang as dplr_tilelang_backend
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


def _kda_tilelang_verifier_args(
    H=2,
    HV=4,
    K=64,
    V=64,
    dtype=torch.bfloat16,
    chunk_size=64,
    state_v_first=False,
    requires_grad=False,
    g_dtype=torch.float32,
    beta_dtype=torch.float32,
    g_is_cuda=True,
    beta_is_cuda=True,
    g_contiguous=True,
    beta_contiguous=True,
):
    B, T = 1, 64
    q = SimpleNamespace(shape=(B, T, H, K), dtype=dtype, is_cuda=True, requires_grad=requires_grad)
    k = SimpleNamespace(shape=q.shape, dtype=dtype, is_cuda=True, requires_grad=requires_grad)
    v = SimpleNamespace(shape=(B, T, HV, V), dtype=dtype, is_cuda=True, requires_grad=requires_grad)
    state_shape = (V, K) if state_v_first else (K, V)
    h = SimpleNamespace(shape=(B, T // chunk_size, HV, *state_shape), dtype=dtype, is_cuda=True, requires_grad=requires_grad)
    g = SimpleNamespace(
        shape=(B, T, HV, K),
        dtype=g_dtype,
        is_cuda=g_is_cuda,
        requires_grad=requires_grad,
        is_contiguous=lambda: g_contiguous,
    )
    beta = SimpleNamespace(
        shape=(B, T, HV),
        dtype=beta_dtype,
        is_cuda=beta_is_cuda,
        requires_grad=requires_grad,
        is_contiguous=lambda: beta_contiguous,
    )
    return dict(
        q=q,
        k=k,
        v=v,
        v_new=v,
        g=g,
        beta=beta,
        A=SimpleNamespace(shape=(B, T, HV, chunk_size), dtype=dtype, is_cuda=True, requires_grad=requires_grad),
        h=h,
        do=v,
        dh=h,
        dv=v,
        chunk_size=chunk_size,
        state_v_first=state_v_first,
    )


def _kda_tilelang_public_verifier_args(
    H=2,
    HV=4,
    K=128,
    V=128,
    dtype=torch.bfloat16,
    chunk_size=32,
    q_contiguous=True,
    h0_contiguous=True,
):
    B, T = 1, 64
    q = SimpleNamespace(shape=(B, T, H, K), dtype=dtype, is_cuda=True, is_contiguous=lambda: q_contiguous)
    k = SimpleNamespace(shape=q.shape, dtype=dtype, is_cuda=True, is_contiguous=lambda: True)
    v = SimpleNamespace(shape=(B, T, HV, V), dtype=dtype, is_cuda=True, is_contiguous=lambda: True)
    g = SimpleNamespace(shape=(B, T, HV, K), dtype=torch.float32, is_cuda=True, is_contiguous=lambda: True)
    beta = SimpleNamespace(shape=(B, T, HV), dtype=dtype, is_cuda=True, is_contiguous=lambda: True)
    h0 = SimpleNamespace(shape=(B, HV, K, V), dtype=torch.float32, is_cuda=True, is_contiguous=lambda: h0_contiguous)
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )


@pytest.mark.parametrize(
    ("H", "HV"),
    [
        pytest.param(4, 4, id="mha"),
        pytest.param(2, 4, id="gva"),
    ],
)
def test_kda_tilelang_backend_public_chunk_kda_verifier_accepts_measured_bucket(H, HV):
    accepted, reason = kda_tilelang_backend.KDATileLangBackend().chunk_kda_verifier(
        **_kda_tilelang_public_verifier_args(H=H, HV=HV)
    )

    assert accepted is True
    assert reason is None


@pytest.mark.parametrize(
    ("overrides", "kwargs", "expected_reason"),
    [
        pytest.param({"dtype": torch.float16}, {}, "requires q dtype torch.bfloat16", id="dtype"),
        pytest.param({"K": 64, "V": 64}, {}, "K=V=128", id="dimension"),
        pytest.param({"H": 2, "HV": 3}, {}, "divisible by H=2", id="non-divisible-gva"),
        pytest.param({"chunk_size": 64}, {}, "supports chunk_size=32", id="chunk-size"),
        pytest.param({"q_contiguous": False}, {}, "requires q to be contiguous", id="q-layout"),
        pytest.param({"h0_contiguous": False}, {}, "requires initial_state to be contiguous", id="state-layout"),
        pytest.param({}, {"safe_gate": True}, "safe_gate=False only", id="safe-gate"),
        pytest.param({}, {"state_v_first": True}, "state_v_first=False only", id="state-v-first"),
        pytest.param({}, {"cu_seqlens": SimpleNamespace(is_cuda=True)}, "dense fixed-length", id="varlen"),
    ],
)
def test_kda_tilelang_backend_public_chunk_kda_verifier_rejects_unmeasured_cases(overrides, kwargs, expected_reason):
    accepted, reason = kda_tilelang_backend.KDATileLangBackend().chunk_kda_verifier(
        **_kda_tilelang_public_verifier_args(**overrides),
        **kwargs,
    )

    assert accepted is False
    assert expected_reason in reason


def test_kda_tilelang_backend_verifier_accepts_gva():
    accepted, reason = kda_tilelang_backend.KDATileLangBackend().chunk_kda_bwd_wy_dqkg_fused_verifier(
        **_kda_tilelang_verifier_args(H=2, HV=4)
    )

    assert accepted is True
    assert reason is None


@pytest.mark.parametrize(
    ("K", "V", "state_v_first"),
    [
        pytest.param(128, 128, False, id="d128-state-k-first"),
        pytest.param(128, 128, True, id="d128-state-v-first"),
    ],
)
def test_kda_tilelang_backend_verifier_accepts_d128_state_layouts(K, V, state_v_first):
    accepted, reason = kda_tilelang_backend.KDATileLangBackend().chunk_kda_bwd_wy_dqkg_fused_verifier(
        **_kda_tilelang_verifier_args(K=K, V=V, state_v_first=state_v_first)
    )

    assert accepted is True
    assert reason is None


def test_kda_tilelang_backend_verifier_rejects_reordered_state_tiles():
    args = _kda_tilelang_verifier_args()
    state_shape = args["h"].shape
    reordered = SimpleNamespace(
        shape=(state_shape[2], state_shape[0], state_shape[1], *state_shape[3:]),
        dtype=args["h"].dtype,
        is_cuda=True,
        requires_grad=False,
    )
    args["h"] = reordered
    args["dh"] = reordered

    accepted, reason = kda_tilelang_backend.KDATileLangBackend().chunk_kda_bwd_wy_dqkg_fused_verifier(**args)

    assert accepted is False
    assert "requires h/dh shape" in reason


def test_kda_tilelang_backend_verifier_rejects_non_divisible_gva():
    accepted, reason = kda_tilelang_backend.KDATileLangBackend().chunk_kda_bwd_wy_dqkg_fused_verifier(
        **_kda_tilelang_verifier_args(H=2, HV=3)
    )

    assert accepted is False
    assert "HV % H must be 0 for GVA" in reason


def test_kda_tilelang_backend_verifier_rejects_untiled_dimension():
    accepted, reason = kda_tilelang_backend.KDATileLangBackend().chunk_kda_bwd_wy_dqkg_fused_verifier(
        **_kda_tilelang_verifier_args(K=60)
    )

    assert accepted is False
    assert "to be divisible by its BK tile" in reason


def test_kda_tilelang_backend_verifier_rejects_varlen():
    accepted, reason = kda_tilelang_backend.KDATileLangBackend().chunk_kda_bwd_wy_dqkg_fused_verifier(
        **_kda_tilelang_verifier_args(),
        cu_seqlens=SimpleNamespace(is_cuda=True),
    )

    assert accepted is False
    assert "dense fixed-length sequences only" in reason


def test_kda_tilelang_backend_verifier_rejects_autograd_inputs():
    accepted, reason = kda_tilelang_backend.KDATileLangBackend().chunk_kda_bwd_wy_dqkg_fused_verifier(
        **_kda_tilelang_verifier_args(requires_grad=True)
    )

    assert accepted is False
    assert "manual fused-backward calls" in reason


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        pytest.param({"g_is_cuda": False}, "requires CUDA tensors", id="g-cpu"),
        pytest.param({"beta_is_cuda": False}, "requires CUDA tensors", id="beta-cpu"),
        pytest.param({"g_dtype": torch.float16}, "requires g dtype torch.float32", id="g-dtype"),
        pytest.param({"beta_dtype": torch.int32}, "beta dtype torch.int32", id="beta-dtype"),
        pytest.param({"g_contiguous": False}, "requires g to be contiguous", id="g-layout"),
        pytest.param({"beta_contiguous": False}, "requires beta to be contiguous", id="beta-layout"),
    ],
)
def test_kda_tilelang_backend_verifier_rejects_g_beta_unsupported_cases(overrides, expected_reason):
    accepted, reason = kda_tilelang_backend.KDATileLangBackend().chunk_kda_bwd_wy_dqkg_fused_verifier(
        **_kda_tilelang_verifier_args(**overrides)
    )

    assert accepted is False
    assert expected_reason in reason


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
