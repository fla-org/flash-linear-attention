# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from fla import models
from fla.utils import device_name

CONFIG_CLASSES = [getattr(models, name) for name in models.__all__ if name.endswith('Config')]


@pytest.mark.parametrize('config_class', CONFIG_CLASSES, ids=lambda cls: cls.model_type)
def test_auto_model_checkpoint_round_trip(config_class, tmp_path):
    torch.manual_seed(42)
    if config_class is models.RWKV7Config and not torch._C.has_lapack:
        pytest.skip('RWKV7 initialization requires LAPACK.')
    config = config_class(
        hidden_size=64,
        num_heads=2,
        num_hidden_layers=0,
        num_self_decoder_layers=0,
        intermediate_size=128,
        vocab_size=64,
        tie_word_embeddings=True,
        fuse_norm=False,
        fuse_cross_entropy=False,
        downstream_config={'enabled': True},
    )
    model = AutoModelForCausalLM.from_config(config)
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
def test_generate_from_embeddings_matches_tokens(config_class, use_cache):
    torch.manual_seed(42)
    if config_class is models.MambaConfig and device_name == 'cpu':
        pytest.skip('Mamba normalization requires an accelerator.')
    config = config_class(
        hidden_size=32,
        num_hidden_layers=0,
        vocab_size=32,
        fuse_norm=False,
        fuse_cross_entropy=False,
        pad_token_id=0,
        eos_token_id=None,
    )
    model = AutoModelForCausalLM.from_config(config).to(device_name).eval()
    input_ids = torch.tensor([[1, 4, 3], [1, 5, 2]], device=device_name)
    kwargs = {'max_new_tokens': 3, 'do_sample': False, 'use_cache': use_cache}
    with torch.no_grad():
        tokens = model.generate(input_ids=input_ids, **kwargs)
        embeddings = model.generate(inputs_embeds=model.get_input_embeddings()(input_ids), **kwargs)
    assert torch.equal(tokens[:, input_ids.shape[1]:], embeddings)
