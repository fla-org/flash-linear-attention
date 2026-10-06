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
from safetensors.torch import save_file

from fla.models.gla import modeling_gla
from fla.models.gla.configuration_gla import GLAConfig
from fla.modules import mlp
from fla.modules.residuals import attnres, mhc, standard
from fla.ops.attnres import naive_attnres


class TokenMixer(nn.Module):
    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, past_key_values=None, **kwargs):
        return self.proj(hidden_states).tanh(), None, past_key_values


class TorchRMSNorm(nn.RMSNorm):
    def forward(self, x, residual=None, prenorm=False):
        x = x if residual is None else x + residual
        output = super().forward(x)
        return (output, x) if prenorm else output


@pytest.fixture(autouse=True)
def cpu_ops(monkeypatch):
    # isolate residual scheduling, ownership and gradients from the GPU kernels.
    monkeypatch.setattr(modeling_gla, 'GatedLinearAttention', TokenMixer)
    monkeypatch.setattr(modeling_gla, 'Attention', TokenMixer)
    for module in (standard, attnres, mhc):
        monkeypatch.setattr(module, 'RMSNorm', TorchRMSNorm)
    monkeypatch.setattr(attnres, 'fused_attnres', naive_attnres)
    monkeypatch.setattr(mlp, 'swiglu', lambda gate, value: F.silu(gate) * value)


def make_config(block_size=None, fuse_norm=False, mode=None, **kwargs):
    residual_kwargs = dict(kwargs.pop('residual_kwargs', None) or {})
    if block_size is not None:
        residual_kwargs['block_size'] = block_size
    if mode is None:
        mode = 'attnres' if block_size is not None else 'standard'
    return GLAConfig(
        hidden_size=32, num_hidden_layers=3, num_heads=2, intermediate_size=48, vocab_size=64,
        fuse_norm=fuse_norm, fuse_swiglu=False, fuse_cross_entropy=False,
        residual_mode=mode, residual_kwargs=residual_kwargs, use_cache=False, **kwargs,
    )


def legacy_state(model, external=False):
    backbone = model if isinstance(model, modeling_gla.GLAModel) else model.model
    prefix = '' if backbone is model else 'model.'
    state = {key: value.clone() for key, value in model.state_dict().items()}
    for old, new in modeling_gla._legacy_residual_keys(len(backbone.layers)).items():
        if old.startswith('attn_norms.'):
            continue
        if prefix + new in state:
            state[prefix + old] = state.pop(prefix + new)
    if external:
        for i in range(len(backbone.layers)):
            state[f'{prefix}attn_norms.{i}.weight'] = state.pop(f'{prefix}layers.{i}.attn_norm.weight')
    return state


@pytest.mark.parametrize(('mode', 'block_size'), [('standard', None), ('attnres', 1), ('attnres', 4), ('mhc', None)])
@pytest.mark.parametrize('fuse_norm', [False, True])
@pytest.mark.parametrize('output_hidden_states', [False, True])
def test_residual_checkpointing(mode, block_size, fuse_norm, output_hidden_states):
    torch.manual_seed(42)
    model = modeling_gla.GLAModel(make_config(block_size, fuse_norm, mode))
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.RMSNorm):
                module.weight.uniform_(0.5, 1.5)
            if getattr(module, '_is_attnres_proj', False):
                module.weight.normal_(std=0.1)
    checkpointed = copy.deepcopy(model)
    checkpointed.gradient_checkpointing_enable()
    ids = torch.arange(10).reshape(2, 5)
    result = model(ids, output_hidden_states=output_hidden_states)
    checkpointed_result = checkpointed(ids, output_hidden_states=output_hidden_states)
    actual = result.last_hidden_state
    expected = checkpointed_result.last_hidden_state
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    grad = torch.randn_like(actual)
    actual_loss, expected_loss = (actual * grad).sum(), (expected * grad).sum()
    if output_hidden_states:
        assert len(result.hidden_states) == len(checkpointed_result.hidden_states) == model.config.num_hidden_layers + 1
        for state, target in zip(result.hidden_states, checkpointed_result.hidden_states):
            torch.testing.assert_close(state, target, atol=0, rtol=0)
            actual_loss = actual_loss + state.square().mean()
            expected_loss = expected_loss + target.square().mean()
    actual_loss.backward()
    expected_loss.backward()
    for (name, parameter), (_, other) in zip(model.named_parameters(), checkpointed.named_parameters()):
        torch.testing.assert_close(parameter.grad, other.grad, atol=0, rtol=0, msg=name)
        if 'input_query' not in name and 'input_key_norm' not in name:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize('mode', ['standard', 'attnres', 'mhc'])
def test_reentrant_checkpointing_rejected(mode):
    model = modeling_gla.GLAModel(make_config(4 if mode == 'attnres' else None, mode=mode))
    with pytest.raises(ValueError, match='use_reentrant=False'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})


@pytest.mark.parametrize('model_class', [modeling_gla.GLAModel, modeling_gla.GLAForCausalLM])
@pytest.mark.parametrize('layout', ['official', 'external', 'current'])
@pytest.mark.parametrize('block_size', [None, 4])
def test_residual_checkpoint_load(model_class, layout, block_size, tmp_path):
    torch.manual_seed(42)
    config = make_config(block_size)
    model = model_class(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.2)
    state = model.state_dict() if layout == 'current' else legacy_state(model, external=layout == 'external')
    check_checkpoint_load(model, state, tmp_path)


def check_checkpoint_load(model, state, tmp_path):
    model_class, config = type(model), model.config
    restored = model_class(config)
    restored.load_state_dict(state, strict=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    config.save_pretrained(tmp_path)
    save_file(state, tmp_path / 'model.safetensors', metadata={'format': 'pt'})
    loaded, info = model_class.from_pretrained(tmp_path, output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys']
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize('mode', ['standard', 'attnres', 'mhc'])
def test_save_roundtrip_and_causal_lm(mode, tmp_path):
    torch.manual_seed(42)
    config = make_config(4 if mode == 'attnres' else None, mode=mode)
    model = modeling_gla.GLAModel(config)
    ids = torch.arange(10).reshape(2, 5)
    expected = model(ids).last_hidden_state
    model.save_pretrained(tmp_path)
    restored = modeling_gla.GLAModel.from_pretrained(tmp_path)
    torch.testing.assert_close(expected, restored(ids).last_hidden_state, atol=0, rtol=0)
    causal_lm = modeling_gla.GLAForCausalLM(config)
    output = causal_lm(ids, labels=ids)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert torch.isfinite(causal_lm.model.embeddings.weight.grad).all()


@pytest.mark.parametrize('block_size', [None, 1, 4])
@pytest.mark.parametrize('tie_word_embeddings', [False, True])
def test_causal_lm_save_pretrained(block_size, tie_word_embeddings, tmp_path):
    torch.manual_seed(42)
    model = modeling_gla.GLAForCausalLM(make_config(block_size, tie_word_embeddings=tie_word_embeddings)).eval()
    ids = torch.arange(10).reshape(2, 5)
    expected = model(ids).logits
    model.save_pretrained(tmp_path)
    restored, info = modeling_gla.GLAForCausalLM.from_pretrained(tmp_path, output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys']
    assert (restored.lm_head.weight is restored.model.embeddings.weight) == tie_word_embeddings
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0, msg=name)
    torch.testing.assert_close(restored(ids).logits, expected, atol=0, rtol=0)
