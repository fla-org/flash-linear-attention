# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.models.gla.modeling_gla import GLAModel
from fla.modules.residuals import AttentionResidual, ManifoldHyperConnection, StandardResidual
from fla.ops.attnres import naive_attnres

from .test_gla_external_norm import cpu_ops, make_config  # noqa: F401


def reference_forward(model, ids, block_size):
    x = model.embeddings(ids)
    completed, partial = [x], None
    previous = None
    states = []
    for i, layer in enumerate(model.layers):
        states.append(x)
        for j, (branch, residual) in enumerate([(layer.attn, layer.residual_attn), (layer.mlp, layer.residual_mlp)]):
            index = 2 * i + j
            if index == 0:
                incoming = residual.input_norm(x)
            elif block_size is None:
                incoming = previous.norm(x)
            else:
                if index % block_size == 0:
                    completed, partial = completed + [partial], None
                sources = completed + ([] if partial is None else [partial])
                incoming = naive_attnres(
                    previous.query.weight, sources, previous.key_norm.weight, previous.norm.weight, previous.norm.eps,
                )
            output = branch(incoming)
            output = output[0] if j == 0 else output
            if block_size is None:
                x = x + output
            else:
                partial = output if partial is None else partial + output
                x = partial
            previous = residual
    if block_size is None:
        x = previous.norm(x)
    else:
        x = naive_attnres(
            previous.query.weight, completed + [partial],
            previous.key_norm.weight, previous.norm.weight, previous.norm.eps,
        )
    return x, (*states, x)


@pytest.mark.parametrize('block_size', [None, 1, 2, 4, 6, 12])
@pytest.mark.parametrize('fuse_norm', [False, True])
def test_gla_matches_prenorm_reference(block_size, fuse_norm):
    torch.manual_seed(42)
    actual_model = GLAModel(make_config(block_size, fuse_norm))
    with torch.no_grad():
        for module in actual_model.modules():
            if isinstance(module, nn.RMSNorm):
                module.weight.uniform_(0.5, 1.5)
            if getattr(module, '_is_attnres_proj', False):
                module.weight.normal_(std=0.1)
    reference = copy.deepcopy(actual_model)
    ids = torch.arange(10).reshape(2, 5)
    actual = actual_model(ids, output_hidden_states=True)
    expected, expected_states = reference_forward(reference, ids, block_size)
    torch.testing.assert_close(actual.last_hidden_state, expected, atol=0, rtol=0)
    assert len(actual.hidden_states) == len(expected_states) == actual_model.config.num_hidden_layers + 1
    assert actual.hidden_states[-1] is actual.last_hidden_state
    for result, target in zip(actual.hidden_states, expected_states):
        torch.testing.assert_close(result, target, atol=0, rtol=0)
    grad = torch.randn_like(expected)
    actual.last_hidden_state.backward(grad)
    expected.backward(grad)
    for (name, parameter), (_, target) in zip(actual_model.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(parameter.grad, target.grad, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize('fuse_norm', [False, True])
def test_standard_residual_state_and_fusion(fuse_norm):
    torch.manual_seed(42)
    residual = StandardResidual(32, fuse_norm=fuse_norm, norm_eps=1e-5)
    x, delta = torch.randn(2, 3, 32), torch.randn(2, 3, 32)
    prepared, history = residual.initialize(x)
    assert history is x and residual.get_hidden_state(history) is x
    torch.testing.assert_close(prepared, residual.input_norm(x), atol=0, rtol=0)
    calls = []
    handle = residual.norm.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append((args, kwargs)), with_kwargs=True,
    )
    output, updated = residual(delta, history)
    handle.remove()
    assert len(calls) == 1 and len(calls[0][0]) == 1
    if fuse_norm:
        assert calls[0][1]['residual'] is history and calls[0][1]['prenorm'] is True
    else:
        assert not calls[0][1]
    assert history is x and residual.get_hidden_state(history) is x
    assert updated is not history and residual.get_hidden_state(updated) is updated
    torch.testing.assert_close(updated, x + delta, atol=0, rtol=0)
    torch.testing.assert_close(output, residual.norm(updated), atol=0, rtol=0)
    assert residual.norm.eps == residual.input_norm.eps == 1e-5


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sub_layer_idx', [0, 1])
def test_standard_residual_fuses_before_low_precision_rounding(dtype, sub_layer_idx):
    class AccumulateFP32Norm(nn.RMSNorm):
        def forward(self, x, residual=None, prenorm=False):
            value = x.float() if residual is None else x.float() + residual.float()
            output = F.rms_norm(value, self.normalized_shape, self.weight.float(), self.eps).to(x.dtype)
            return (output, value.to(x.dtype)) if prenorm else output

    torch.manual_seed(42)
    residual = StandardResidual(32, sub_layer_idx=sub_layer_idx, fuse_norm=True)
    residual.norm = AccumulateFP32Norm(32, eps=1e-6).to(dtype)
    state, branch = torch.randn(2, 3, 32).to(dtype), torch.randn(2, 3, 32).to(dtype)
    history = state
    output, updated = residual(branch, history)
    expected_history = state + branch
    assert residual.get_hidden_state(updated) is updated
    torch.testing.assert_close(updated, expected_history, atol=0, rtol=0)
    expected_output = F.rms_norm(
        state.float() + branch.float(), (32,), residual.norm.weight.float(), residual.norm.eps,
    ).to(dtype)
    torch.testing.assert_close(output, expected_output, atol=0, rtol=0)
    assert not torch.equal(output, residual.norm(expected_history))


@pytest.mark.parametrize('block_size', [1, 2, 4, 12])
def test_attnres_history_immutability(block_size):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 32)
    first = AttentionResidual(32, block_size=block_size, fuse_norm=False)
    _, history = first.initialize(x)
    assert len(history) == 1 and history[0] is first.get_hidden_state(history) is x
    outputs = []
    for i in range(6):
        residual = AttentionResidual(32, sub_layer_idx=i, block_size=block_size, fuse_norm=False)
        output = torch.randn_like(x)
        outputs.append(output)
        incoming, snapshot = history, tuple(history)
        _, history = residual(output, history)
        assert residual.get_hidden_state(history) is history[-1]
        assert history is not incoming
        assert len(incoming) == len(snapshot) and all(a is b for a, b in zip(incoming, snapshot))
        assert residual.get_hidden_state(incoming) is snapshot[-1]
        assert history[0] is x
        expected = [x]
        for start in range(0, len(outputs), block_size):
            partial = outputs[start]
            for value in outputs[start + 1:start + block_size]:
                partial = partial + value
            expected.append(partial)
        assert len(history) == len(expected)
        for actual, target in zip(history, expected):
            torch.testing.assert_close(actual, target, atol=0, rtol=0)


@pytest.mark.parametrize('block_size', [1, 4])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_attnres_aligns_initial_history_to_branch_dtype(block_size, dtype):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 32, requires_grad=True)
    history = [x]
    initial_history = history
    expected = [x.to(dtype)]
    for i in range(6):
        residual = AttentionResidual(32, sub_layer_idx=i, block_size=block_size, fuse_norm=False)
        branch = torch.randn_like(x, dtype=dtype, requires_grad=True)
        output, history = residual(branch, history)
        if i % block_size == 0:
            expected = [*expected, branch]
        else:
            expected = [*expected[:-1], expected[-1] + branch]
        assert initial_history[0] is x and x.dtype == torch.float32
        assert all(state.dtype == dtype for state in history)
        if i % block_size == 0:
            assert history[-1] is branch
        for actual, target in zip(history, expected):
            torch.testing.assert_close(actual, target, atol=0, rtol=0)
        output.square().sum().backward(retain_graph=True)
        assert branch.grad is not None and torch.isfinite(branch.grad).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize('residual_cls', [StandardResidual, AttentionResidual, ManifoldHyperConnection])
def test_history_initialization_only_at_first_sublayer(residual_cls):
    residual = residual_cls(32, sub_layer_idx=1, fuse_norm=False)
    with pytest.raises(ValueError, match='first sublayer'):
        residual.initialize(torch.randn(1, 2, 32))


@pytest.mark.parametrize('mode', ['standard', 'attnres', 'mhc'])
def test_gla_hybrid_and_repeated_forward(mode):
    torch.manual_seed(42)
    config = make_config(4 if mode == 'attnres' else None, mode=mode, attn={'layers': [1], 'num_heads': 2})
    model = GLAModel(config)
    ids = torch.arange(10).reshape(2, 5)
    first = model(ids, return_dict=False)[0]
    second = model(ids).last_hidden_state
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert first.shape == (2, 5, 32)
    assert not hasattr(model, 'attn_norms') and not hasattr(model, 'norm')


@pytest.mark.parametrize('mode', ['standard', 'attnres', 'mhc'])
@pytest.mark.parametrize('num_layers', [1, 3])
def test_gla_reported_hidden_states(mode, num_layers):
    torch.manual_seed(42)
    config = make_config(mode=mode)
    config.num_hidden_layers = num_layers
    model = GLAModel(config)
    ids = torch.arange(6).reshape(2, 3)
    plain = model(ids, output_hidden_states=False)
    result = model(ids, output_hidden_states=True)
    as_tuple = model(ids, output_hidden_states=True, return_dict=False)
    assert plain.hidden_states is None
    assert len(result.hidden_states) == num_layers + 1
    assert result.hidden_states[-1] is result.last_hidden_state
    torch.testing.assert_close(result.hidden_states[0], model.embeddings(ids), atol=0, rtol=0)
    torch.testing.assert_close(result.last_hidden_state, plain.last_hidden_state, atol=0, rtol=0)
    torch.testing.assert_close(as_tuple[0], result.last_hidden_state, atol=0, rtol=0)
    assert len(as_tuple[1]) == len(result.hidden_states)
    for state, target in zip(as_tuple[1], result.hidden_states):
        assert state.shape == (2, 3, config.hidden_size)
        torch.testing.assert_close(state, target, atol=0, rtol=0)


@pytest.mark.parametrize(('kwargs', 'error'), [
    ({'mode': 'unknown'}, ValueError),
    ({'mode': 'mhc', 'block_size': 4}, TypeError),
    ({'block_size': 0}, ValueError),
    ({'block_size': True}, ValueError),
    ({'block_size': 1.5}, ValueError),
])
def test_invalid_residual_config(kwargs, error):
    with pytest.raises(error):
        GLAModel(make_config(**kwargs))
