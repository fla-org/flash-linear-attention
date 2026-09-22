# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib

import pytest
import torch

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.kda import KimiDeltaAttention
from fla.utils import IS_NVIDIA, assert_close, device

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='CUDA Graph layer tests require NVIDIA')
LAYERS = [GatedDeltaNet, KimiDeltaAttention]


def _make_layer(layer_cls):
    return layer_cls(
        hidden_size=128,
        head_dim=32,
        num_heads=2,
        expand_v=1,
        mode='chunk',
        use_short_conv=True,
        conv_size=4,
        conv_bias=True,
    ).to(device=device, dtype=torch.bfloat16).train()


@pytest.mark.parametrize('layer_cls', LAYERS)
@pytest.mark.parametrize('graph_mode', ['force_graph', 'auto'])
def test_dense_layer_graph_replay_forward_backward(layer_cls, graph_mode, monkeypatch):
    torch.manual_seed(42)
    layer = _make_layer(layer_cls)
    reference = _make_layer(layer_cls)
    reference.load_state_dict(layer.state_dict())
    x = torch.randn(1, 128, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    do = torch.randn_like(x)
    kwargs = dict(graph_mode=graph_mode, chunk_size=32)
    module = importlib.import_module(layer_cls.__module__)
    name = 'chunk_gated_delta_rule' if layer_cls is GatedDeltaNet else 'chunk_kda'
    original = getattr(module, name)
    routes = []

    def record_route(*args, **kwargs):
        routes.append(kwargs['use_graph'])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, record_route)

    for _ in range(2):
        layer(x, **kwargs)[0].backward(do)
        layer.zero_grad(set_to_none=True)
        x.grad = None
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = layer(x, **kwargs)[0]
        output.backward(do)
    assert routes and all(routes)
    pointers = (output.data_ptr(), x.grad.data_ptr(), *(p.grad.data_ptr() for p in layer.parameters()))

    tolerances = (0.02, 0.03, 0.04) if layer_cls is GatedDeltaNet else (0.03, 0.04, 0.05)
    for _ in range(2):
        with torch.no_grad():
            x.copy_(torch.randn_like(x))
            do.copy_(torch.randn_like(do))
            x.grad.zero_()
            for parameter in layer.parameters():
                parameter.grad.zero_()
        graph.replay()
        torch.cuda.synchronize()

        reference.zero_grad(set_to_none=True)
        reference_x = x.detach().clone().requires_grad_(True)
        expected = reference(reference_x, graph_mode='eager', chunk_size=32)[0]
        expected.backward(do)
        assert_close('dense output', expected, output, tolerances[0])
        assert_close('dense input gradient', reference_x.grad, x.grad, tolerances[1])
        assert torch.isfinite(output).all()
        assert torch.isfinite(x.grad).all()
        for (name, parameter), expected_parameter in zip(layer.named_parameters(), reference.parameters(), strict=True):
            assert_close(name, expected_parameter.grad, parameter.grad, tolerances[2])
            assert torch.isfinite(parameter.grad).all()
        assert pointers == (output.data_ptr(), x.grad.data_ptr(), *(p.grad.data_ptr() for p in layer.parameters()))


@pytest.mark.parametrize('layer_cls', LAYERS)
@pytest.mark.parametrize('layout', ['batch', 'mask'])
def test_dense_layer_graph_unsupported_layout(layer_cls, layout):
    torch.manual_seed(42)
    layer = _make_layer(layer_cls)
    x = torch.randn(2 if layout == 'batch' else 1, 128, 128, device=device, dtype=torch.bfloat16)
    kwargs = {}
    if layout == 'mask':
        mask = torch.ones(1, 128, device=device, dtype=torch.long)
        mask[:, :32] = 0
        kwargs['attention_mask'] = mask
    reason = 'batch size 1' if layout == 'batch' else 'prepacked inputs'
    with pytest.raises(ValueError, match=reason):
        layer(x, graph_mode='force_graph', **kwargs)
    eager = layer(x, graph_mode='eager', **kwargs)[0]
    auto = layer(x, graph_mode='auto', **kwargs)[0]
    torch.testing.assert_close(auto, eager, rtol=0, atol=0)
    assert torch.isfinite(auto).all()


@pytest.mark.parametrize('layer_cls', LAYERS)
def test_layer_graph_ascend_routing(layer_cls, monkeypatch):
    layer = _make_layer(layer_cls)
    module = importlib.import_module(layer_cls.__module__)
    monkeypatch.setattr(module, 'IS_NPU', True)
    x = torch.randn(1, 128, 128, device=device, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match='Ascend'):
        layer(x, graph_mode='force_graph')
    auto = layer(x, graph_mode='auto')[0]
    eager = layer(x, graph_mode='eager')[0]
    torch.testing.assert_close(auto, eager, rtol=0, atol=0)
