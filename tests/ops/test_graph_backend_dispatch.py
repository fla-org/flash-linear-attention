# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib
import inspect

import pytest
import torch

from fla.ops.backends import BackendRegistry
from fla.ops.gated_delta_rule.backends.flash_qla import FlashQLABackend
from fla.ops.kda.backends.flash_kda import FlashKDABackend


@pytest.fixture(params=['gated_delta_rule', 'kda'])
def dispatch_case(request, monkeypatch):
    operation = request.param
    module = importlib.import_module(f'fla.ops.{operation}.chunk')
    is_gdn = operation == 'gated_delta_rule'
    name = 'chunk_gated_delta_rule' if is_gdn else 'chunk_kda'
    backend = FlashQLABackend() if is_gdn else FlashKDABackend()
    BackendRegistry.ensure_initialized(operation)
    registry = BackendRegistry._registries[operation]
    monkeypatch.setattr(registry, '_backends', {backend.backend_type: backend})
    monkeypatch.setattr(backend, 'is_available', lambda: True)
    monkeypatch.setattr(backend, 'is_enabled', lambda: True)
    monkeypatch.setattr(module, 'IS_NVIDIA', True)
    monkeypatch.setattr(module, 'IS_NPU', False)
    monkeypatch.setattr(module, 'is_graph_capable_device', lambda device: True)
    if is_gdn:
        monkeypatch.setattr('fla.ops.gated_delta_rule.backends.flash_qla.IS_NVIDIA_HOPPER', True)
        monkeypatch.setattr('fla.ops.gated_delta_rule.backends.flash_qla.IS_NVIDIA_SM120', False)
    calls = []

    def external(*args, **kwargs):
        assert kwargs.get('chunk_indices') is None
        assert kwargs.get('chunk_offsets') is None
        calls.append('external')
        return kwargs['v'] * 2, None

    monkeypatch.setattr(backend, name, external)
    function = module.ChunkGatedDeltaRuleFunction if is_gdn else module.ChunkKDAFunction
    signature = inspect.signature(function.forward)

    def native(*args):
        arguments = signature.bind(None, *args).arguments
        calls.append('graph' if arguments['use_graph'] else 'native_eager')
        return arguments['v'] * 3, None

    monkeypatch.setattr(function, 'apply', native)
    q = torch.zeros(1, 128, 1, 128, dtype=torch.bfloat16)
    kwargs = dict(q=q, k=q, v=torch.ones_like(q), g=q if not is_gdn else q[..., 0], beta=q[..., 0])
    if not is_gdn:
        kwargs.update(use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                      state_v_first=True, safe_gate=True, lower_bound=-5, A_log=torch.zeros(1))
    return getattr(module, name), kwargs, calls


@pytest.mark.parametrize('case', ['dense', 'packed_low', 'packed_high', 'over_capacity', 'force_graph'])
@torch.no_grad()
def test_auto_preserves_eager_backend_dispatch(dispatch_case, case):
    entry, kwargs, calls = dispatch_case
    graph_kwargs = dict(graph_mode='auto')
    if case in ('packed_low', 'packed_high', 'force_graph'):
        kwargs['cu_seqlens'] = torch.tensor([0, 1, 1 if case == 'packed_low' else 128])
        graph_kwargs['chunk_indices'] = torch.tensor(
            [[0, 0], [-1, 0], [-1, 0]] if case == 'packed_low' else [[0, 0], [1, 0], [1, 1]],
        )
        graph_kwargs['chunk_offsets'] = torch.tensor([0, 1, 1 if case == 'packed_low' else 3])
    if case == 'over_capacity':
        graph_kwargs['graph_t_max'] = 64
    if case == 'force_graph':
        graph_kwargs['graph_mode'] = 'force_graph'
    output, _ = entry(**kwargs, **graph_kwargs)
    expected = 'external' if case in ('packed_low', 'over_capacity') else 'graph'
    assert calls == [expected]
    if expected == 'external':
        for mode in ('eager', None):
            reference, _ = entry(**kwargs, graph_mode=mode)
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
        assert calls == ['external', 'external', 'external']


@torch.no_grad()
def test_auto_backend_rejection_keeps_native_fallback(dispatch_case):
    entry, kwargs, calls = dispatch_case
    kwargs['q'] = kwargs['q'][..., :64]
    kwargs['k'] = kwargs['k'][..., :64]
    if kwargs['g'].ndim == 4:
        kwargs['g'] = kwargs['g'][..., :64]
    output, _ = entry(**kwargs, graph_mode='auto', graph_t_max=64)
    assert calls == ['native_eager']
    torch.testing.assert_close(output, kwargs['v'] * 3, rtol=0, atol=0)


def test_auto_backend_retains_autograd(dispatch_case):
    entry, kwargs, calls = dispatch_case
    kwargs['v'].requires_grad_(True)
    output, _ = entry(**kwargs, graph_mode='auto', graph_t_max=64)
    output.float().sum().backward()
    is_gdn = entry.__name__ == 'chunk_gated_delta_rule'
    assert calls == ['external' if is_gdn else 'native_eager']
    torch.testing.assert_close(kwargs['v'].grad, torch.full_like(kwargs['v'], 2 if is_gdn else 3), rtol=0, atol=0)
