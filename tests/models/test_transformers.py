# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from fla import models
from fla.utils import device_name, find_spec_cached

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


@pytest.mark.parametrize('config_class', [models.GLAConfig, models.MambaConfig, models.TransformerConfig])
@pytest.mark.parametrize('use_cache', [False, True])
def test_generate_inputs_embeds(config_class, use_cache, make_model):
    torch.manual_seed(42)
    if config_class is models.MambaConfig and device_name == 'cpu':
        pytest.skip('Mamba normalization requires an accelerator.')
    model = make_model(config_class, hidden_size=32, vocab_size=32, pad_token_id=0, eos_token_id=None)
    model = model.to(device_name).eval()
    input_ids = torch.tensor([[1, 4, 3], [1, 5, 2]], device=device_name)
    kwargs = {'max_new_tokens': 3, 'do_sample': False, 'use_cache': use_cache}
    with torch.no_grad():
        tokens = model.generate(input_ids=input_ids, **kwargs)
        embeddings = model.generate(inputs_embeds=model.get_input_embeddings()(input_ids), **kwargs)
    assert torch.equal(tokens[:, input_ids.shape[1]:], embeddings)


@pytest.mark.parametrize('config_class', [
    models.ABCConfig,
    models.DeltaNetConfig,
    models.GatedDeltaProductConfig,
    models.GLAConfig,
    models.GSAConfig,
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
