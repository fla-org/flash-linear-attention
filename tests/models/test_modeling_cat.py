# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Contributed by: Jatin Prakash (bicycleman15)
# Controllably Efficient Language Models (https://arxiv.org/abs/2511.05313)

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from fla.models.cat import CATConfig, CATForCausalLM
from fla.models.cat import modeling_cat
from fla.utils import device


def create_cat_model(
    num_hidden_layers: int = 1,
    num_heads: int = 2,
    head_dim: int = 16,
    max_chunk_size: int = 8,
    max_position_embeddings: int = 64,
    pad_to_multiple_of: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    use_default_fused: bool = False,
) -> tuple[CATForCausalLM, CATConfig]:
    hidden_size = num_heads * head_dim
    fuse_kwargs = {} if use_default_fused else {
        'fuse_norm': False,
        'fuse_swiglu': False,
        'fuse_cross_entropy': False,
    }
    config = CATConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_heads=num_heads,
        max_chunk_size=max_chunk_size,
        dim_fx=hidden_size,
        compressor_hidden_size=hidden_size // 2,
        compressor_num_layers=1,
        compressor_num_heads=1,
        max_position_embeddings=max_position_embeddings,
        pad_to_multiple_of=pad_to_multiple_of,
        vocab_size=128,
        eos_token_id=None,
        use_cache=False,
        **fuse_kwargs,
    )
    model = CATForCausalLM(config)
    model.to(dtype).to(device)
    return model, config


def test_cat_forward_backward():
    model, config = create_cat_model()
    input_ids = torch.randint(0, config.vocab_size, (2, 40), device=device)

    with torch.no_grad():
        output_without_hidden_states = model(input_ids, output_hidden_states=False)
    output = model(input_ids, output_hidden_states=True)

    assert output_without_hidden_states.hidden_states is None
    assert output.logits.shape == (2, 40, config.vocab_size)
    assert len(output.hidden_states) == config.num_hidden_layers + 1
    assert all(state.shape == (2, 40, config.hidden_size) for state in output.hidden_states)

    output.logits.float().sum().backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_cat_with_labels():
    model, config = create_cat_model()
    input_ids = torch.randint(0, config.vocab_size, (2, 31), device=device)
    labels = input_ids.clone()

    output = model(input_ids, labels=labels)

    assert output.loss is not None
    assert output.loss.ndim == 0
    output.loss.backward()


def test_cat_default_fused_forward_backward():
    model, config = create_cat_model(use_default_fused=True)
    input_ids = torch.randint(0, config.vocab_size, (2, 31), device=device)

    assert config.fuse_norm
    assert config.fuse_swiglu
    assert config.fuse_cross_entropy

    output = model(input_ids, labels=input_ids)

    assert output.loss is not None
    assert output.logits.shape == (2, 31, config.vocab_size)
    output.loss.backward()


def test_cat_dynamo_config_is_scoped():
    config = {'cache_size_limit': 1}
    if hasattr(torch._dynamo.config, 'recompile_limit'):
        config['recompile_limit'] = 1

    with torch._dynamo.config.patch(config):
        with modeling_cat._cat_dynamo_config_patch():
            assert torch._dynamo.config.cache_size_limit == modeling_cat._CAT_DYNAMO_CACHE_LIMIT
            if hasattr(torch._dynamo.config, 'recompile_limit'):
                assert torch._dynamo.config.recompile_limit == modeling_cat._CAT_DYNAMO_CACHE_LIMIT

        assert torch._dynamo.config.cache_size_limit == 1
        if hasattr(torch._dynamo.config, 'recompile_limit'):
            assert torch._dynamo.config.recompile_limit == 1


@pytest.mark.parametrize('chunk_size', [4, 8])
def test_cat_adaptive_chunk_size(chunk_size: int):
    model, config = create_cat_model(max_chunk_size=8)
    input_ids = torch.randint(0, config.vocab_size, (2, 27), device=device)

    with torch.no_grad():
        output = model(input_ids, chunk_size=chunk_size)

    assert output.logits.shape == (2, 27, config.vocab_size)


def test_cat_training_rotates_power_of_two_chunk_sizes():
    model, _ = create_cat_model(max_chunk_size=8)
    model.train()

    assert [model.model._resolve_chunk_size(None) for _ in range(5)] == [2, 4, 8, 2, 4]


def test_cat_explicit_chunk_size_does_not_advance_training_schedule():
    model, _ = create_cat_model(max_chunk_size=8)
    model.train()

    assert model.model._resolve_chunk_size(8) == 8
    assert int(model.model._train_chunk_size_step.item()) == 0
    assert model.model._resolve_chunk_size(None) == 2


def test_cat_eval_defaults_to_max_chunk_size():
    model, _ = create_cat_model(max_chunk_size=8)
    model.eval()

    assert model.model._resolve_chunk_size(None) == 8
    assert int(model.model._train_chunk_size_step.item()) == 0


def test_cat_bucket_padding_and_trim():
    model, config = create_cat_model(max_position_embeddings=2048, pad_to_multiple_of=512)

    assert model.model._get_padded_seq_len(17, chunk_size=8) == 512
    assert model.model._get_padded_seq_len(513, chunk_size=8) == 1024

    input_ids = torch.randint(0, config.vocab_size, (1, 17), device=device)
    with torch.no_grad():
        output = model(input_ids)

    assert output.logits.shape == (1, 17, config.vocab_size)


@pytest.mark.parametrize('original_seq_len', [10, 12])
def test_cat_rearrange_hidden_states_matches_loop(original_seq_len: int):
    model, _ = create_cat_model()
    num_chunks, chunk_size = 3, 4
    block_len = chunk_size + 2
    internal_seq_len = num_chunks * block_len + 2
    hidden_states = torch.arange(internal_seq_len, device=device).view(1, internal_seq_len, 1)

    actual = model.model._rearrange_hidden_states(
        hidden_states,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        original_seq_len=original_seq_len,
    )

    expected_parts = []
    for chunk_idx in range(num_chunks):
        block_start = chunk_idx * block_len
        pred_start = block_start + (2 if chunk_idx == 0 else 1)
        expected_parts.append(hidden_states[:, pred_start:block_start + block_len - 1])
    expected_parts.append(hidden_states[:, -1:])
    expected = torch.cat(expected_parts, dim=1)[:, :original_seq_len]

    assert torch.equal(actual, expected)


def test_cat_padding_invariance():
    torch.manual_seed(42)
    model, config = create_cat_model(max_position_embeddings=64, pad_to_multiple_of=8)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 17), device=device)

    with torch.no_grad():
        output_small_bucket = model(input_ids, output_hidden_states=True)
        config.pad_to_multiple_of = 32
        output_large_bucket = model(input_ids, output_hidden_states=True)

    torch.testing.assert_close(output_small_bucket.logits, output_large_bucket.logits, rtol=1e-2, atol=1e-2)
    for small_bucket_state, large_bucket_state in zip(
        output_small_bucket.hidden_states,
        output_large_bucket.hidden_states,
        strict=True,
    ):
        torch.testing.assert_close(small_bucket_state, large_bucket_state, rtol=1e-2, atol=1e-2)


def test_cat_accepts_all_ones_attention_mask():
    model, config = create_cat_model()
    input_ids = torch.randint(0, config.vocab_size, (2, 25), device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)

    assert output.logits.shape == (2, 25, config.vocab_size)


def test_cat_accepts_flame_position_ids_and_empty_cu_seqlens():
    model, config = create_cat_model()
    input_ids = torch.randint(0, config.vocab_size, (2, 25), device=device)
    position_ids = torch.arange(input_ids.shape[-1], device=device).repeat(input_ids.shape[0], 1)

    with torch.no_grad():
        output = model(input_ids, position_ids=position_ids, cu_seqlens=None)

    assert output.logits.shape == (2, 25, config.vocab_size)


def test_cat_rejects_varlen_cu_seqlens():
    model, config = create_cat_model()
    input_ids = torch.randint(0, config.vocab_size, (1, 25), device=device)
    cu_seqlens = torch.tensor([0, 10, 25], device=device, dtype=torch.int32)

    with pytest.raises(ValueError, match="variable-length `cu_seqlens`"):
        model(input_ids, cu_seqlens=cu_seqlens)


def test_cat_rejects_padding_attention_mask():
    model, config = create_cat_model()
    input_ids = torch.randint(0, config.vocab_size, (2, 25), device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[0, 0] = False

    with pytest.raises(ValueError, match="padded batches are not supported"):
        model(input_ids, attention_mask=attention_mask)


def test_cat_chunk_size_validation():
    model, config = create_cat_model(max_chunk_size=8)
    input_ids = torch.randint(0, config.vocab_size, (1, 24), device=device)

    with pytest.raises(ValueError, match="cannot exceed"):
        model(input_ids, chunk_size=16)


def test_cat_auto_model_registration():
    config = CATConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_heads=2,
        max_chunk_size=8,
        compressor_hidden_size=16,
        compressor_num_layers=1,
        compressor_num_heads=1,
        max_position_embeddings=64,
        pad_to_multiple_of=32,
        vocab_size=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )

    assert AutoConfig.for_model('cat').model_type == 'cat'
    assert isinstance(AutoModelForCausalLM.from_config(config), CATForCausalLM)


def test_cat_save_load_transformers_5_tied_weight_metadata(tmp_path):
    model, config = create_cat_model(dtype=torch.float32)
    input_ids = torch.randint(0, config.vocab_size, (1, 16), device=device)

    if hasattr(model, 'get_expanded_tied_weights_keys'):
        assert model._tied_weights_keys == {"lm_head.weight": "model.embeddings.weight"}
    else:
        assert model._tied_weights_keys == ["lm_head.weight"]
    model.save_pretrained(tmp_path)
    reloaded = CATForCausalLM.from_pretrained(tmp_path).to(device)
    reloaded.eval()

    with torch.no_grad():
        output = reloaded(input_ids)

    assert output.logits.shape == (1, 16, config.vocab_size)


def test_cat_full_prefix_generation():
    model, config = create_cat_model(max_chunk_size=4, max_position_embeddings=32, pad_to_multiple_of=16)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 8), device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=2,
            do_sample=False,
            use_cache=False,
            chunk_size=4,
            pad_token_id=0,
        )

    assert output_ids.shape == (1, 10)
    assert torch.equal(output_ids[:, :8], input_ids)
