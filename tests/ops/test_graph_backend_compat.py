# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from fla.ops.gated_delta_rule.backends.triton_ascend import TritonAscendGDNBackend

ASCEND_ENTRIES = [
    ('gated_delta_rule', 'gate', 'gdn_gate_chunk_cumsum'),
    ('gated_delta_rule', 'gate', 'gdn_gate_bwd'),
    ('kda', 'gate', 'kda_gate_bwd'),
    ('utils', 'solve_tril', 'solve_tril'),
    ('gated_delta_rule', 'chunk_fwd', 'chunk_gated_delta_rule_fwd_intra'),
    ('gated_delta_rule', 'wy_fast', 'recompute_w_u_fwd'),
    ('gated_delta_rule', 'wy_fast', 'prepare_wy_repr_bwd'),
    ('common', 'chunk_delta_h', 'chunk_gated_delta_rule_fwd_h'),
    ('common', 'chunk_delta_h', 'chunk_gated_delta_rule_bwd_dhu'),
    ('common', 'chunk_o', 'chunk_bwd_dv_local'),
    ('common', 'chunk_o', 'chunk_fwd_o'),
    ('common', 'chunk_o', 'chunk_bwd_dqkwg'),
]


@pytest.mark.parametrize('operation,module,name', ASCEND_ENTRIES)
def test_ascend_eager_graph_keyword_compatibility(operation, module, name):
    public = getattr(importlib.import_module(f'fla.ops.{operation}.{module}'), name)
    backend = getattr(importlib.import_module(f'fla.ops.{operation}.backends.triton_ascend.{module}'), f'{name}_npu')
    kwargs = {
        key: parameter.default if parameter.default is not inspect.Parameter.empty else None
        for key, parameter in inspect.signature(public).parameters.items()
    }
    inspect.signature(backend).bind(**kwargs)
    assert kwargs['use_graph'] is False
    kwargs['use_graph'] = True
    with pytest.raises(NotImplementedError, match='Ascend'):
        inspect.unwrap(backend)(**kwargs)


def test_ascend_wy_verifier_accepts_explicit_eager(monkeypatch):
    monkeypatch.setattr('fla.utils.IS_NPU', True)
    tensor = SimpleNamespace(device=SimpleNamespace(type='npu'), dtype=torch.bfloat16)
    backend = TritonAscendGDNBackend()
    assert backend.recompute_w_u_fwd_verifier(tensor, tensor, tensor, tensor, use_graph=False) == (True, None)


def test_ascend_wy_forwards_graph_keyword(monkeypatch):
    module = importlib.import_module('fla.ops.gated_delta_rule.backends.triton_ascend.wy_fast')
    calls = []
    result = (object(), object())

    def recompute(*args, **kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(module, 'recompute_w_u_fwd_npu', recompute)
    backend = TritonAscendGDNBackend()
    for use_graph in (False, True):
        assert backend.recompute_w_u_fwd(None, None, None, None, use_graph=use_graph) is result
    assert calls == [{'use_graph': False}, {'use_graph': True}]


def test_ascend_output_eager_without_chunk_indices(monkeypatch):
    module = importlib.import_module('fla.ops.common.backends.triton_ascend.chunk_o')
    calls = []

    class LaunchRecorder:
        def __getitem__(self, grid):
            return lambda **kwargs: calls.append(kwargs)

    monkeypatch.setattr(module, 'get_npu_properties', lambda: {'num_aicore': 1})
    monkeypatch.setattr(module, 'chunk_fwd_kernel_o_npu', LaunchRecorder())
    q = torch.zeros(1, 128, 1, 32)
    cu = torch.tensor([0, 32, 128])
    offsets = torch.tensor([0, 1, 3])
    monkeypatch.setattr(module, 'prepare_chunk_offsets', lambda *args: offsets)
    output = inspect.unwrap(module.chunk_fwd_o_npu)(q, q, q, q, cu_seqlens=cu, use_graph=False)
    assert output.shape == q.shape
    assert calls[0]['total_chunks'] == 3
    assert calls[0]['chunk_offsets'] is offsets

    supplied_offsets = offsets.clone()
    inspect.unwrap(module.chunk_fwd_o_npu)(q, q, q, q, cu_seqlens=cu, chunk_offsets=supplied_offsets)
    assert calls[1]['chunk_offsets'] is supplied_offsets


@pytest.mark.parametrize('unsupported', ['ascend', 'platform', 'batch', 'cp'])
@pytest.mark.parametrize('use_graph', [False, True])
def test_gdn_auto_unsupported_routes_to_eager(monkeypatch, unsupported, use_graph):
    module = importlib.import_module('fla.ops.gated_delta_rule.chunk')
    monkeypatch.setattr(module, 'IS_NPU', unsupported == 'ascend')
    monkeypatch.setattr(module, 'IS_NVIDIA', unsupported != 'platform')
    q = torch.zeros(2 if unsupported == 'batch' else 1, 128, 1, 32)
    g = torch.zeros(*q.shape[:3])
    calls = []
    signature = inspect.signature(module.ChunkGatedDeltaRuleFunction.forward)

    def apply(*args):
        calls.append(signature.bind(None, *args).arguments)
        return q, None

    monkeypatch.setattr(module.ChunkGatedDeltaRuleFunction, 'apply', apply)
    cp = SimpleNamespace(cu_seqlens=torch.tensor([0, 128]), cu_seqlens_cpu=None) if unsupported == 'cp' else None
    entry = inspect.unwrap(module.chunk_gated_delta_rule)
    kwargs = dict(cp_context=cp, chunk_indices=torch.zeros(2, 2, dtype=torch.long),
                  chunk_offsets=torch.tensor([0, 2]))
    entry(q, q, q, g, g, graph_mode='auto', use_graph=use_graph, **kwargs)
    assert calls[-1]['use_graph'] is False
    assert calls[-1]['chunk_indices'] is None
    assert calls[-1]['chunk_offsets'] is None
    with pytest.raises((NotImplementedError, RuntimeError, ValueError)):
        entry(q, q, q, g, g, graph_mode='force_graph', **kwargs)
    assert len(calls) == 1


@pytest.mark.parametrize('mode', ['eager', 'auto'])
def test_kda_eager_route_discards_graph_metadata(monkeypatch, mode):
    module = importlib.import_module('fla.ops.kda.chunk')
    monkeypatch.setattr(module, 'IS_NVIDIA', True)
    monkeypatch.setattr(module, 'IS_NPU', False)
    monkeypatch.setattr(module, 'is_graph_capable_device', lambda device: False)
    q = torch.zeros(1, 128, 1, 32)
    beta = torch.zeros(*q.shape[:3])
    calls = []
    signature = inspect.signature(module.ChunkKDAFunction.forward)

    def apply(*args):
        calls.append(signature.bind(None, *args).arguments)
        return q, None

    monkeypatch.setattr(module.ChunkKDAFunction, 'apply', apply)
    inspect.unwrap(module.chunk_kda)(
        q, q, q, q, beta,
        cu_seqlens=torch.tensor([0, 128]),
        chunk_indices=torch.tensor([[0, 0], [0, 1]]),
        chunk_offsets=torch.tensor([0, 2]),
        use_graph=mode == 'auto',
        graph_mode=mode,
    )
    assert len(calls) == 1
    assert calls[0]['use_graph'] is False
    assert calls[0]['chunk_indices'] is None
    assert calls[0]['chunk_offsets'] is None
    assert calls[0]['graph_nt_max'] is None
    assert calls[0]['max_num_seqs'] is None
    with pytest.raises(ValueError, match='conflicts'):
        inspect.unwrap(module.chunk_kda)(q, q, q, q, beta, graph_mode='eager', use_graph=True)
