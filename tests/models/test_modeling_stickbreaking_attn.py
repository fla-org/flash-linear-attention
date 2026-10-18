# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models import StickBreakingAttentionConfig, StickBreakingAttentionForCausalLM, StickBreakingAttentionModel
from fla.utils import IS_AMD, IS_INTEL, IS_NVIDIA, assert_close, device

TOL = {torch.float16: 0.005, torch.bfloat16: 0.02}


def _config(**kwargs):
    return StickBreakingAttentionConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=4,
        num_kv_heads=2,
        vocab_size=64,
        pad_token_id=0,
        **kwargs,
    )


def test_registration_and_checkpoint(tmp_path):
    config = _config(attend_current=True, tie_word_embeddings=True)
    assert isinstance(AutoConfig.for_model(config.model_type), StickBreakingAttentionConfig)
    assert isinstance(AutoModel.from_config(config), StickBreakingAttentionModel)
    model = AutoModelForCausalLM.from_config(config)
    assert isinstance(model, StickBreakingAttentionForCausalLM)
    assert model.lm_head.weight is model.model.embeddings.weight
    assert model.generation_config.use_cache is False
    model.save_pretrained(tmp_path)
    restored = AutoModelForCausalLM.from_pretrained(tmp_path)
    assert restored.config.attend_current is True
    assert restored.config.use_cache is False
    assert restored.lm_head.weight is restored.model.embeddings.weight
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name


@pytest.mark.parametrize(
    'kwargs',
    [{'use_cache': True}, {'past_key_values': ()}, {'output_attentions': True}],
    ids=['cache', 'past', 'attention-matrix'],
)
def test_rejects_unsupported_outputs(kwargs):
    model = AutoModelForCausalLM.from_config(_config())
    with pytest.raises(NotImplementedError):
        model(input_ids=torch.ones(1, 3, dtype=torch.long), **kwargs)


def test_rejects_cache_configuration():
    with pytest.raises(ValueError, match='does not support KV-cache'):
        _config(use_cache=True)


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_INTEL), reason="Requires a supported GPU")
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16], ids=['fp16', 'bf16'])
@pytest.mark.parametrize('attend_current', [False, True])
@pytest.mark.parametrize('fusion', ['none', 'standard', 'linear-loss'])
def test_forward_backward(dtype, attend_current, fusion):
    torch.manual_seed(42)
    config = _config(
        attend_current=attend_current,
        fuse_norm=fusion != 'none',
        fuse_swiglu=fusion != 'none',
        fuse_cross_entropy=fusion == 'standard',
        fuse_linear_cross_entropy=fusion == 'linear-loss',
    )
    model = AutoModelForCausalLM.from_config(config).to(device=device, dtype=dtype)
    input_ids = torch.randint(3, config.vocab_size, (2, 47), device=device)
    result = model(input_ids=input_ids, labels=input_ids, output_hidden_states=True)
    assert torch.isfinite(result.loss)
    assert result.hidden_states[-1].shape == (2, 47, config.hidden_size)
    assert result.past_key_values is None
    result.loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    for layer in model.model.layers:
        assert torch.count_nonzero(layer.attn.q_proj.weight.grad) > 0
        assert torch.count_nonzero(layer.attn.k_proj.weight.grad) > 0
        assert torch.count_nonzero(layer.attn.v_proj.weight.grad) > 0


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_INTEL), reason="Requires a supported GPU")
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16], ids=['fp16', 'bf16'])
@pytest.mark.parametrize('attend_current', [False, True])
def test_packed_loss_and_padding(dtype, attend_current):
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(_config(attend_current=attend_current)).to(device=device, dtype=dtype)
    first = torch.randint(3, 64, (1, 17), device=device)
    second = torch.randint(3, 64, (1, 29), device=device)
    first_out = model(input_ids=first, labels=first)
    second_out = model(input_ids=second, labels=second)
    ids = torch.cat((first, second), dim=1)
    labels = ids.clone()
    cu_seqlens = torch.tensor([0, 17, 46], dtype=torch.int32, device=device)
    packed = model(input_ids=ids, labels=labels, cu_seqlens=cu_seqlens)
    assert torch.equal(labels, ids)
    assert_close(
        prefix='packed logits',
        ref=torch.cat((first_out.logits, second_out.logits), dim=1),
        tri=packed.logits,
        ratio=TOL[dtype],
    )
    expected_loss = (first_out.loss * 16 + second_out.loss * 28) / 44
    assert_close(prefix='packed loss', ref=expected_loss, tri=packed.loss, ratio=TOL[dtype])
    packed.loss.backward()
    assert torch.isfinite(model.model.layers[0].attn.q_proj.weight.grad).all()

    padded = torch.zeros(2, 29, dtype=torch.long, device=device)
    padded[0, 12:], padded[1] = first[0], second[0]
    mask = padded != 0
    result = model(input_ids=padded, attention_mask=mask)
    assert_close(prefix='padding', ref=first_out.logits, tri=result.logits[:1, 12:], ratio=TOL[dtype])


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_INTEL), reason="Requires a supported GPU")
def test_gradient_checkpointing():
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(_config()).to(device=device, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    input_ids = torch.randint(3, 64, (2, 31), device=device)
    model(input_ids=input_ids, labels=input_ids).loss.backward()
    for layer in model.model.layers:
        assert layer.attn.q_proj.weight.grad is not None
        assert torch.isfinite(layer.attn.q_proj.weight.grad).all()


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_INTEL), reason="Requires a supported GPU")
@pytest.mark.parametrize('attend_current', [False, True])
def test_generation_matches_full_prefix(attend_current):
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(_config(attend_current=attend_current)).to(device=device, dtype=torch.bfloat16)
    model.eval()
    ids = torch.randint(3, 64, (2, 9), device=device)
    mask = torch.ones_like(ids)
    ids[0, :3], mask[0, :3] = 0, 0
    with torch.no_grad():
        generated = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=4,
            do_sample=False,
            use_cache=False,
            eos_token_id=None,
        )
        expected, expected_mask = ids, mask
        for _ in range(4):
            logits = model(input_ids=expected, attention_mask=expected_mask, use_cache=False).logits
            expected = torch.cat((expected, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
            expected_mask = torch.cat((expected_mask, torch.ones_like(expected_mask[:, :1])), dim=1)
    assert torch.equal(generated, expected)
