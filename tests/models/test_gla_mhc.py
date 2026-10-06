# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import copy

import pytest
import torch
import torch.nn.functional as F

from fla.models.gla.modeling_gla import GLAModel
from fla.modules.residuals.mhc import ManifoldHyperConnection, sinkhorn

from .test_gla_external_norm import cpu_ops, make_config  # noqa: F401


def reference_routing(routing, streams):
    flat = streams.flatten(-2)
    flat = F.rms_norm(flat, (flat.shape[-1],), eps=routing.eps)
    logits = F.linear(flat, routing.weight)
    n = routing.num_streams
    pre = torch.sigmoid(routing.scale[0] * logits[..., :n] + routing.pre_bias)
    post = 2 * torch.sigmoid(routing.scale[1] * logits[..., n:2 * n] + routing.post_bias)
    mixing = (routing.scale[2] * logits[..., 2 * n:].unflatten(-1, (n, n)) + routing.res_bias).exp()
    for _ in range(routing.num_iters):
        mixing = mixing / mixing.sum(dim=-2, keepdim=True)
        mixing = mixing / mixing.sum(dim=-1, keepdim=True)
    return pre, post, mixing


def reference_model(model, ids):
    embeddings = model.embeddings(ids)
    states = [embeddings]
    streams = embeddings.unsqueeze(-2).repeat(1, 1, model.layers[0].residual_attn.num_streams, 1)
    routing = model.layers[0].residual_attn.input_routing
    norm = model.layers[0].residual_attn.input_norm
    for layer_idx, layer in enumerate(model.layers):
        for is_attn, branch, residual in [(True, layer.attn, layer.residual_attn), (False, layer.mlp, layer.residual_mlp)]:
            pre, post, mixing = reference_routing(routing, streams)
            readout = (pre.unsqueeze(-1) * streams).sum(dim=-2)
            if is_attn and layer_idx > 0:
                states.append(readout)
            incoming = norm(readout)
            output = branch(incoming)
            output = output[0] if is_attn else output
            # all three mappings use the same state from before the sublayer.
            streams = mixing @ streams + post.unsqueeze(-1) * output.unsqueeze(-2)
            routing, norm = residual.routing, residual.norm
    output = norm(streams.mean(dim=-2))
    return output, (*states, output)


@pytest.mark.parametrize('num_streams', [2, 4])
def test_mhc_matches_independent_recurrence(num_streams):
    torch.manual_seed(42)
    actual_model = GLAModel(make_config(mode='mhc', residual_kwargs={'num_streams': num_streams})).double()
    with torch.no_grad():
        for name, parameter in actual_model.named_parameters():
            if name.endswith(('scale', 'pre_bias', 'post_bias', 'res_bias')):
                parameter.uniform_(-0.4, 0.4)
    reference = copy.deepcopy(actual_model)
    ids = torch.arange(6).reshape(2, 3)
    result = actual_model(ids, output_hidden_states=True)
    actual = result.last_hidden_state
    expected, expected_states = reference_model(reference, ids)
    assert len(result.hidden_states) == len(expected_states) == actual_model.config.num_hidden_layers + 1
    assert result.hidden_states[-1] is actual
    for state, target in zip(result.hidden_states, expected_states):
        torch.testing.assert_close(state, target, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for (name, parameter), (_, target) in zip(actual_model.named_parameters(), reference.named_parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, target.grad, atol=1e-9, rtol=1e-9, msg=name)


@pytest.mark.parametrize('n', [2, 4, 8])
def test_sinkhorn_constraints_and_gradients(n):
    torch.manual_seed(42)
    logits = torch.randn(2, 3, n, n, dtype=torch.float64, requires_grad=True) * 0.3
    actual = sinkhorn(logits)
    expected = logits.exp()
    for _ in range(20):
        expected = expected / expected.sum(dim=-2, keepdim=True)
        expected = expected / expected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert (actual >= 0).all()
    for axis in (-1, -2):
        torch.testing.assert_close(actual.sum(axis), torch.ones_like(actual.sum(axis)), atol=1e-10, rtol=0)
    weights = torch.randn_like(actual)
    actual_grad, = torch.autograd.grad((actual * weights).sum(), logits)
    expected_grad, = torch.autograd.grad((expected * weights).sum(), logits)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)


def test_mhc_gradcheck_and_history_lifetime():
    torch.manual_seed(42)
    residual = ManifoldHyperConnection(3, num_streams=2, num_sublayers=1, fuse_norm=False).double()
    x = torch.randn(1, 1, 3, dtype=torch.float64, requires_grad=True)

    def forward(value):
        incoming, history = residual.initialize(value)
        original = tuple(history)
        original_readout = history.hidden_state
        assert residual.get_hidden_state(history) is original_readout
        expected_readout = (history.pre.unsqueeze(-1) * history.streams).sum(dim=-2)
        torch.testing.assert_close(original_readout, expected_readout, atol=0, rtol=0)
        result, updated = residual(incoming.sin(), history)
        assert updated is not history and history.hidden_state is original_readout
        assert all(a is b for a, b in zip(history, original))
        assert updated.streams.shape == (1, 1, 2, 3)
        assert residual.get_hidden_state(updated) is updated.hidden_state
        torch.testing.assert_close(updated.hidden_state, updated.streams.mean(dim=-2), atol=0, rtol=0)
        return result, residual.get_hidden_state(updated)

    assert torch.autograd.gradcheck(forward, (x,), atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_mhc_autocast_and_extreme_logits(dtype):
    torch.manual_seed(42)
    residual = ManifoldHyperConnection(8, num_streams=4, num_sublayers=1, fuse_norm=False).to(dtype)
    x = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        incoming, history = residual.initialize(x)
        output, updated = residual(incoming.sin(), history)
    assert history.mixing.dtype == history.post.dtype == torch.float32
    assert output.dtype == updated.streams.dtype == updated.hidden_state.dtype == dtype
    output.float().square().sum().backward()
    assert torch.isfinite(x.grad).all()
    logits = (torch.randn(2, 4, 4) * 1000).requires_grad_()
    result = sinkhorn(logits)
    assert torch.isfinite(result).all()
    (result * torch.randn_like(result)).sum().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('kwargs', [
    {'num_streams': 1}, {'num_streams': True}, {'num_iters': 0}, {'num_iters': 1.5},
    {'init_scale': 0.0}, {'init_scale': float('nan')}, {'num_sublayers': 0},
])
def test_invalid_mhc_options(kwargs):
    with pytest.raises(ValueError):
        ManifoldHyperConnection(8, **kwargs)
