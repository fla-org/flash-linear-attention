# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib.metadata
import inspect
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

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


@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason='Backend dispatch was disabled before import')
@pytest.mark.parametrize(
    ('module_name', 'names', 'backend_types'),
    [
        ('atk.chunk_atk_bwd', ['chunk_atk_bwd'], ['triton_ascend']),
        ('atk.chunk_atk_fwd', ['chunk_atk_fwd', 'recompute_atk_fwd'], ['triton_ascend']),
        ('attn.parallel', ['parallel_attn_bwd', 'parallel_attn_fwd'], ['tilelang']),
        ('attnres.fused', ['fused_attnres'], ['gluon', 'triton_ascend']),
        ('common.chunk_delta_h', ['chunk_gated_delta_rule_bwd_dhu'], ['triton_ascend']),
        ('common.chunk_delta_h', ['chunk_gated_delta_rule_fwd_h'], ['intracard_cp', 'triton_ascend']),
        ('common.chunk_h', ['chunk_bwd_dh', 'chunk_fwd_h'], ['triton_ascend']),
        ('common.chunk_o', ['chunk_bwd_dqkwg'], ['tilelang', 'triton_ascend']),
        ('common.chunk_o', ['chunk_bwd_dv_local', 'chunk_fwd_o'], ['triton_ascend']),
        ('common.chunk_scaled_dot_kkt', ['chunk_scaled_dot_kkt_fwd'], ['triton_ascend']),
        ('common.gate', ['fused_beta_sigmoid_bwd', 'fused_beta_sigmoid_fwd'], ['triton_ascend']),
        ('gated_delta_rule.chunk', ['chunk_gated_delta_rule'], ['flash_qla']),
        ('gated_delta_rule.chunk_fwd', ['chunk_gated_delta_rule_fwd_intra'], ['triton_ascend']),
        ('gated_delta_rule.gate', ['gdn_gate_bwd', 'gdn_gate_chunk_cumsum', 'gdn_gate_fwd'], ['triton_ascend']),
        ('gated_delta_rule.wy_fast', ['prepare_wy_repr_bwd', 'recompute_w_u_fwd'], ['triton_ascend']),
        ('gdn2.chunk_bwd', ['chunk_gdn2_bwd_wy_dqkg_fused'], ['triton_ascend']),
        ('gdn2.chunk_intra', ['chunk_gdn2_fwd_intra'], ['triton_ascend']),
        ('gdn2.fused_recurrent', ['fused_recurrent_gdn2_fwd'], ['triton_ascend']),
        ('generalized_delta_rule.dplr.chunk', ['chunk_dplr_delta_rule'], ['tilelang']),
        ('gla.chunk', ['chunk_gla_bwd_dA', 'chunk_gla_bwd_dqk_intra', 'chunk_gla_bwd_dqkg',
         'chunk_gla_bwd_dv', 'chunk_gla_fwd_intra_gk', 'chunk_gla_fwd_o_gk'], ['triton_ascend']),
        ('kda.chunk_bwd', ['chunk_kda_bwd_dAv'], ['triton_ascend']),
        ('kda.chunk_bwd', ['chunk_kda_bwd_wy_dqkg_fused'], ['tilelang', 'triton_ascend']),
        ('kda.chunk_intra', ['chunk_kda_bwd_intra', 'chunk_kda_fwd_intra'], ['triton_ascend']),
        ('kda.chunk_intra_token_parallel', ['chunk_kda_fwd_intra_token_parallel'], ['triton_ascend']),
        ('kda.fused_recurrent', ['fused_recurrent_kda_fwd'], ['triton_ascend']),
        ('kda.gate', ['fused_kda_gate', 'kda_gate_bwd', 'kda_gate_chunk_cumsum', 'kda_gate_fwd'], ['triton_ascend']),
        ('kda.wy_fast', ['recompute_w_u_fwd'], ['triton_ascend']),
        ('precond_gated_delta_rule.chunk_precond_kkt_fwd', ['chunk_precond_kkt_fwd'], ['triton_ascend']),
        ('precond_gated_delta_rule.chunk_precond_wy_bwd', ['prepare_precond_wy_repr_bwd'], ['triton_ascend']),
        ('precond_gated_delta_rule.fused_recurrent', ['fused_recurrent_precond_gated_delta_rule_fwd'], ['triton_ascend']),
        ('rwkv6.chunk', ['chunk_rwkv6_fwd_intra'], ['tilelang']),
        ('wall_attn.parallel', ['parallel_wall_attn_bwd', 'parallel_wall_attn_fwd'], []),
    ],
)
def test_ops_registered_dispatch(monkeypatch, module_name, names, backend_types):
    module = importlib.import_module(f'fla.ops.{module_name}')
    expected = set(backend_types)
    if not backends.GluonBackend.is_available():
        expected.discard('gluon')
    for name in names:
        entry = getattr(module, name)
        registry = backends._function_registries[inspect.unwrap(entry)]
        assert registry._backends.keys() == expected
        for backend_type, candidate in registry._backends.items():
            assert candidate.get_implementation(name) is not None
            with monkeypatch.context() as patch:
                patch.setattr(registry, '_backends', {backend_type: candidate})
                patch.setattr(type(candidate), 'is_available', classmethod(lambda cls: True))
                patch.setattr(type(candidate), 'is_enabled', lambda self: True)
                patch.setattr(candidate, 'verifier', lambda x: (True, None))
                patch.setattr(candidate, 'implementation', lambda x: x.square())
                x = torch.tensor([1., 2.], requires_grad=True)
                y = entry(x)
                torch.testing.assert_close(y, x.square())
                torch.testing.assert_close(torch.autograd.grad(y.sum(), x)[0], 2 * x)


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


@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason='Backend dispatch was disabled before import')
def test_attention_and_common_backends_have_separate_implementations():
    from fla.ops.attn.parallel import parallel_attn_fwd
    from fla.ops.common.chunk_o import chunk_bwd_dqkwg

    common = backends._function_registries[inspect.unwrap(chunk_bwd_dqkwg)]._backends['tilelang']
    attention = backends._function_registries[inspect.unwrap(parallel_attn_fwd)]._backends['tilelang']
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
