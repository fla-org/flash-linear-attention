# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import re

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from fla import models
from fla.models.gla.modeling_gla import GLAForCausalLM, GLAModel
from fla.modules import residuals
from fla.modules.residuals import StandardResidual, register_residual
from fla.utils import find_spec_cached

CONFIG_CLASSES = [getattr(models, name) for name in models.__all__ if name.endswith('Config')]


@pytest.mark.parametrize('config_class', CONFIG_CLASSES, ids=lambda cls: cls.model_type)
@pytest.mark.parametrize('tie_word_embeddings', [False, True])
def test_model_accessors(config_class, tie_word_embeddings, make_model):
    torch.manual_seed(42)
    model = make_model(config_class, tie_word_embeddings=tie_word_embeddings)
    decoder = model.model if hasattr(model, 'model') else model.backbone
    assert model.get_decoder() is decoder
    assert model.get_input_embeddings() is decoder.get_input_embeddings()
    embeddings = nn.Embedding(64, 64)
    model.set_input_embeddings(embeddings)
    assert decoder.embeddings is embeddings
    embeddings = nn.Embedding(64, 64)
    decoder.set_input_embeddings(embeddings)
    assert model.get_input_embeddings() is embeddings
    output_embeddings = nn.Linear(64, 64, bias=False)
    model.set_output_embeddings(new_embeddings=None)
    assert model.get_output_embeddings() is None
    model.set_output_embeddings(new_embeddings=output_embeddings)
    assert model.get_output_embeddings() is output_embeddings
    model.tie_weights()
    assert (model.get_input_embeddings().weight is model.get_output_embeddings().weight) == tie_word_embeddings
    original_weights = model.get_input_embeddings().weight.detach().clone()
    model.resize_token_embeddings(72, mean_resizing=False)
    assert model.get_input_embeddings().weight.shape == (72, 64)
    assert model.get_output_embeddings().weight.shape == (72, 64)
    torch.testing.assert_close(model.get_input_embeddings().weight[:64], original_weights, rtol=0, atol=0)
    assert (model.get_input_embeddings().weight is model.get_output_embeddings().weight) == tie_word_embeddings
    if hasattr(model, 'model'):
        replacement = type(decoder)(model.config)
        model.set_decoder(decoder=replacement)
        assert model.model is replacement
        assert model.get_decoder() is replacement
        assert model.get_input_embeddings() is replacement.embeddings


@pytest.mark.parametrize('config_class', CONFIG_CLASSES, ids=lambda cls: cls.model_type)
def test_checkpoint(config_class, make_model, tmp_path):
    torch.manual_seed(42)
    model = make_model(config_class, tie_word_embeddings=True, downstream_config={'enabled': True})
    assert model.can_generate()
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight

    model.save_pretrained(tmp_path)
    restored_config = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(restored_config, config_class)
    assert restored_config.downstream_config == {'enabled': True}
    restored = AutoModelForCausalLM.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(restored, type(model))
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    assert restored.state_dict().keys() == model.state_dict().keys()
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, model.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize('config_class', [
    models.ABCConfig,
    models.DeltaNetConfig,
    models.GatedDeltaProductConfig,
    models.GLAConfig,
    models.GSAConfig,
    models.GSA2Config,
    models.HGRNConfig,
    models.HGRN2Config,
    models.LightNetConfig,
    models.LinearAttentionConfig,
    models.MesaNetConfig,
    models.MLAConfig,
    models.MoBAConfig,
    models.NSAConfig,
    models.RavenConfig,
    models.RetNetConfig,
], ids=lambda cls: cls.model_type)
@pytest.mark.parametrize('block_size', [None, 2])
def test_initialization(config_class, block_size, make_model):
    if config_class is models.MLAConfig and find_spec_cached('flash_attn') is None:
        pytest.skip('MLA initialization requires flash-attn.')
    torch.manual_seed(42)
    kwargs = {'expand_ratio': 16} if config_class is models.HGRN2Config else {}
    model = make_model(
        config_class,
        num_heads=4,
        head_dim=16,
        num_hidden_layers=2,
        pad_token_id=0,
        fuse_swiglu=False,
        attnres_block_size=block_size,
        **kwargs,
    )
    assert model.get_input_embeddings().weight[0].count_nonzero() == model.config.hidden_size
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if getattr(module, '_is_attnres_proj', False):
                assert module.weight.count_nonzero() == 0
            else:
                assert module.weight.count_nonzero() > 0
            if module.bias is not None:
                assert module.bias.count_nonzero() == 0
        elif isinstance(module, nn.RMSNorm):
            torch.testing.assert_close(module.weight, torch.ones_like(module.weight), rtol=0, atol=0)
    layer = model.model.layers[0]
    for module, weight in [(layer.attn, layer.attn.o_proj.weight), (layer.mlp, layer.mlp.down_proj.weight)]:
        for strategy in ['rescale', 'zero']:
            torch.manual_seed(42)
            model._init_weights(module, prenorm_residual_strategy=strategy)
            expected = weight.detach().clone()
            torch.manual_seed(42)
            model._init_weights(module, prenorm_residual_strategy=strategy)
            torch.testing.assert_close(weight, expected, rtol=0, atol=0)
            assert bool(expected.count_nonzero()) == (strategy == 'rescale')
        with pytest.raises(ValueError, match='Invalid prenorm_residual_strategy'):
            model._init_weights(module, prenorm_residual_strategy='invalid')


@pytest.mark.parametrize(('model_class', 'mode', 'legacy', 'fuse_norm', 'save_original_format'), [
    pytest.param(GLAModel, 'standard_defer', True, True, True, id='legacy-deferred'),
    pytest.param(GLAForCausalLM, 'standard', True, False, False, id='legacy-standard'),
    pytest.param(GLAForCausalLM, 'attnres', True, True, True, id='legacy-attnres'),
    pytest.param(GLAModel, 'mhc', False, True, False, id='mhc'),
    pytest.param(GLAForCausalLM, None, False, False, True, id='default'),
])
def test_gla_checkpoint(model_class, mode, legacy, fuse_norm, save_original_format, tmp_path):
    torch.manual_seed(42)
    config = models.GLAConfig(
        hidden_size=64,
        num_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=64,
        fuse_norm=fuse_norm,
        residual_mode=mode,
    )
    model = model_class(config)
    state = {name: torch.randn_like(value) for name, value in model.state_dict().items()}
    saved = dict(state)
    if legacy:
        saved = {}
        for name, value in state.items():
            name = re.sub(r'residual_(attn|mlp)\.norm\.', r'\1_norm.', name)
            name = re.sub(r'residual_(attn|mlp)\.query\.', r'\1_res_proj.', name)
            name = re.sub(r'residual_(attn|mlp)\.key_norm\.', r'\1_res_norm.', name)
            name = name.replace('residual_output.norm.', 'norm.')
            name = name.replace('residual_output.query.', 'res_proj.')
            name = name.replace('residual_output.key_norm.', 'res_norm.')
            saved[name] = value
    model.load_state_dict(saved, strict=True)
    config.save_pretrained(tmp_path)
    save_file(saved, tmp_path / 'model.safetensors', metadata={'format': 'pt'})
    restored, info = model_class.from_pretrained(tmp_path, output_loading_info=True)
    assert restored.config.residual_mode == model.config.residual_mode
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys']
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        torch.testing.assert_close(model.state_dict()[name], state[name], rtol=0, atol=0)

    resaved_path = tmp_path / 'resaved'
    restored.save_pretrained(resaved_path, save_original_format=save_original_format)
    resaved_state = load_file(resaved_path / 'model.safetensors')
    assert resaved_state.keys() == (saved.keys() if save_original_format else state.keys())
    reloaded, info = model_class.from_pretrained(resaved_path, output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys']
    reloaded.load_state_dict(reloaded.state_dict(), strict=True)
    for name, value in reloaded.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


@pytest.mark.parametrize('missing', [None, 'scale', 'projection.weight', 'norm.weight'])
def test_gla_custom_residual(missing, make_model, monkeypatch, tmp_path):
    class ProjectedResidual(StandardResidual):
        supports_reentrant_checkpointing = False

        def __init__(self, hidden_size, scale=0.25, **kwargs):
            super().__init__(hidden_size=hidden_size, **kwargs)
            self.initial_scale = scale
            self.scale = nn.Parameter(torch.empty(()))
            self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
            self.reset_parameters()

        def reset_parameters(self):
            super().reset_parameters()
            nn.init.constant_(self.scale, self.initial_scale)
            nn.init.zeros_(self.projection.weight)

        def post(self, state: torch.Tensor, branch_output: torch.Tensor, context: None = None) -> torch.Tensor:
            branch_output = self.scale * branch_output + self.projection(branch_output)
            return super().post(state, branch_output, context)

    torch.manual_seed(42)
    monkeypatch.setattr(residuals, '_RESIDUAL_CLASSES', residuals._RESIDUAL_CLASSES.copy())
    register_residual('test.projected', ProjectedResidual)
    with pytest.raises(ValueError, match='already registered'):
        register_residual('test.projected', ProjectedResidual)
    model = make_model(models.GLAConfig, num_hidden_layers=2, residual_mode='test.projected', residual_kwargs={'scale': 0.5})
    with pytest.raises(ValueError, match='use_reentrant=False'):
        model.gradient_checkpointing_enable({'use_reentrant': True})
    assert not model.is_gradient_checkpointing
    missing_key = f'model.layers.0.residual_attn.{missing}' if missing else None
    defaults = {name: value.clone() for name, value in model.state_dict().items()}
    model.load_state_dict(defaults, strict=True)
    residual = model.model.layers[0].residual_attn
    assert residual.scale.item() == 0.5 and residual.projection.weight.count_nonzero() == 0
    x, branch = torch.randn(2, 7, 64), torch.randn(2, 7, 64)
    residual_state = residual.initialize(x)
    _, context = residual.pre(residual_state)
    residual_state = residual.post(residual_state, branch, context)
    torch.testing.assert_close(residual.get_hidden_state(residual_state), x + 0.5 * branch)
    state = {name: torch.randn_like(value) for name, value in defaults.items() if name != missing_key}
    model.config.save_pretrained(tmp_path)
    save_file(state, tmp_path / 'model.safetensors', metadata={'format': 'pt'})
    restored, info = GLAForCausalLM.from_pretrained(tmp_path, output_loading_info=True)
    assert set(info['missing_keys']) == ({missing_key} if missing_key else set())
    assert not info['unexpected_keys'] and not info['mismatched_keys']
    restored.init_weights()
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, defaults[name] if name == missing_key else state[name], rtol=0, atol=0)
    monkeypatch.delitem(residuals._RESIDUAL_CLASSES, 'test.projected')
    with pytest.raises(ValueError, match='register_residual'):
        GLAForCausalLM.from_pretrained(tmp_path)


def test_gla_key_mapping(make_model, tmp_path):
    torch.manual_seed(42)
    model = make_model(models.GLAConfig, num_hidden_layers=1)
    state = model.state_dict()
    state['model.layers.0.attn_norm.weight'] = state.pop('model.layers.0.attn.q_proj.weight')
    model.config.save_pretrained(tmp_path)
    save_file(state, tmp_path / 'model.safetensors', metadata={'format': 'pt'})
    restored, info = GLAForCausalLM.from_pretrained(
        tmp_path,
        key_mapping={r'layers\.0\.attn_norm\.weight': 'layers.0.attn.q_proj.weight'},
        output_loading_info=True,
    )
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys']
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
