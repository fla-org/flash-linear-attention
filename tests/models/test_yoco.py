# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import pytest
import torch

import fla.layers.yoco_attn as yoco_attn_mod
from fla.layers.attn import Attention
from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.yoco_attn import YOCOCrossAttention, YOCOGatedRetention, YOCORotaryEmbedding, YOCOSharedKVBuilder
from fla.models import YOCOConfig, YOCOForCausalLM
from fla.utils import device

from .test_modeling_base import run_test_generation


def _linear_attn(
    *,
    linear_attn_type: str = 'gated_deltanet',
    mode: str = 'chunk',
    num_heads: int,
    num_v_heads: int | None = None,
    head_dim: int = 64,
    expand_v: float = 2.0,
    window_size: int | None = None,
):
    return {
        'type': linear_attn_type,
        'mode': mode,
        'num_heads': num_heads,
        'num_v_heads': num_heads if num_v_heads is None else num_v_heads,
        'head_dim': head_dim,
        'expand_v': expand_v,
        'window_size': window_size,
    }


def _attn(
    *,
    num_heads: int,
    num_kv_heads: int | None = None,
    layers: list[int] | None = None,
    qkv_bias: bool = False,
    window_size: int | None = None,
    rope_theta: float = 10000.0,
):
    return {
        'num_heads': num_heads,
        'num_kv_heads': num_heads if num_kv_heads is None else num_kv_heads,
        'qkv_bias': qkv_bias,
        'window_size': window_size,
        'rope_theta': rope_theta,
    }


def test_gated_retention_self_decoder_option():
    torch.manual_seed(42)
    config = YOCOConfig(
        num_hidden_layers=4,
        num_self_decoder_layers=2,
        hidden_size=128,
        self_decoder_attn=_linear_attn(
            linear_attn_type='gated_retention',
            mode='chunk',
            num_heads=2,
            num_v_heads=2,
            expand_v=1.0,
        ),
        cross_decoder_attn=_attn(num_heads=2, num_kv_heads=2),
        intermediate_size=256,
        vocab_size=128,
        max_position_embeddings=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = YOCOForCausalLM(config)
    assert isinstance(model.model.self_layers[0].attn, YOCOGatedRetention)
    assert isinstance(model.model.self_layers[1].attn, YOCOGatedRetention)
    assert model.model.self_layers[0].attn.rotary.interleaved
    assert model.model.self_layers[1].attn.rotary.interleaved
    assert isinstance(model.model.shared_kv_builder, YOCOSharedKVBuilder)
    assert isinstance(model.model.cross_layers[0].attn, YOCOCrossAttention)
    assert isinstance(model.model.cross_layers[1].attn, YOCOCrossAttention)


def test_gated_deltanet_self_decoder_default():
    torch.manual_seed(42)
    config = YOCOConfig(
        num_hidden_layers=4,
        num_self_decoder_layers=2,
        hidden_size=128,
        self_decoder_attn=_linear_attn(
            num_heads=4,
            num_v_heads=4,
            head_dim=32,
        ),
        cross_decoder_attn=_attn(num_heads=4, num_kv_heads=4),
        intermediate_size=256,
        vocab_size=128,
        max_position_embeddings=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = YOCOForCausalLM(config)
    assert isinstance(model.model.self_layers[0].attn, GatedDeltaNet)
    assert isinstance(model.model.self_layers[1].attn, GatedDeltaNet)
    assert isinstance(model.model.shared_kv_builder, YOCOSharedKVBuilder)
    assert isinstance(model.model.cross_layers[0].attn, YOCOCrossAttention)
    assert isinstance(model.model.cross_layers[1].attn, YOCOCrossAttention)


def test_swa_self_decoder_option():
    torch.manual_seed(42)
    config = YOCOConfig(
        num_hidden_layers=4,
        num_self_decoder_layers=2,
        hidden_size=128,
        self_decoder_attn=_linear_attn(
            linear_attn_type='swa',
            num_heads=2,
            num_v_heads=2,
            window_size=64,
        ),
        cross_decoder_attn=_attn(num_heads=2, num_kv_heads=2),
        intermediate_size=256,
        vocab_size=128,
        max_position_embeddings=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = YOCOForCausalLM(config)
    assert isinstance(model.model.self_layers[0].attn, Attention)
    assert isinstance(model.model.self_layers[1].attn, Attention)
    assert model.model.self_layers[0].attn.window_size == 64
    assert model.model.self_layers[1].attn.window_size == 64
    assert isinstance(model.model.shared_kv_builder, YOCOSharedKVBuilder)
    assert isinstance(model.model.cross_layers[0].attn, YOCOCrossAttention)
    assert isinstance(model.model.cross_layers[1].attn, YOCOCrossAttention)


def test_yoco_specific_modules_use_official_init():
    torch.manual_seed(42)

    config = YOCOConfig(
        num_hidden_layers=4,
        num_self_decoder_layers=2,
        hidden_size=512,
        self_decoder_attn=_linear_attn(
            num_heads=8,
            num_v_heads=8,
            head_dim=64,
        ),
        cross_decoder_attn=_attn(num_heads=8, num_kv_heads=8),
        intermediate_size=1024,
        vocab_size=4096,
        max_position_embeddings=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = YOCOForCausalLM(config)

    shared_kv_builder = model.model.shared_kv_builder
    cross_attn = model.model.cross_layers[0].attn
    mlp = model.model.self_layers[0].mlp

    hidden_bound = 1 / math.sqrt(config.hidden_size)
    intermediate_bound = 1 / math.sqrt(config.intermediate_size)

    assert shared_kv_builder.k_proj._is_hf_initialized
    assert shared_kv_builder.v_proj._is_hf_initialized
    assert cross_attn.q_proj._is_hf_initialized
    assert cross_attn.o_proj._is_hf_initialized
    assert mlp.gate_proj._is_hf_initialized
    assert mlp.up_proj._is_hf_initialized
    assert mlp.down_proj._is_hf_initialized

    assert shared_kv_builder.k_proj.weight.abs().max().item() <= hidden_bound + 1e-6
    assert shared_kv_builder.v_proj.weight.abs().max().item() <= hidden_bound + 1e-6
    assert cross_attn.q_proj.weight.abs().max().item() <= hidden_bound + 1e-6
    assert cross_attn.o_proj.weight.abs().max().item() <= hidden_bound + 1e-6
    assert mlp.gate_proj.weight.abs().max().item() <= hidden_bound + 1e-6
    assert mlp.up_proj.weight.abs().max().item() <= hidden_bound + 1e-6
    assert mlp.down_proj.weight.abs().max().item() <= intermediate_bound + 1e-6

    expected_embed_std = config.hidden_size ** -0.5
    embed_std = model.model.embeddings.weight.std().item()
    assert abs(embed_std - expected_embed_std) / expected_embed_std < 0.1

    gated_retention_model = YOCOForCausalLM(YOCOConfig(
        num_hidden_layers=4,
        num_self_decoder_layers=2,
        hidden_size=512,
        self_decoder_attn=_linear_attn(
            linear_attn_type='gated_retention',
            mode='chunk',
            num_heads=8,
            num_v_heads=8,
            expand_v=1.0,
        ),
        cross_decoder_attn=_attn(num_heads=8, num_kv_heads=8),
        intermediate_size=1024,
        vocab_size=4096,
        max_position_embeddings=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    ))
    gated_retention = gated_retention_model.model.self_layers[0].attn

    qkvg_bound = (2 ** -2.5) * math.sqrt(6.0 / (2 * gated_retention_model.config.hidden_size))
    gk_bound = (2 ** -2.5) * math.sqrt(
        6.0 / (gated_retention_model.config.hidden_size + gated_retention_model.config.self_decoder_attn['num_heads'])
    )
    out_bound = (2 ** -1) * math.sqrt(6.0 / (2 * gated_retention_model.config.hidden_size))

    assert isinstance(gated_retention, YOCOGatedRetention)
    assert gated_retention.q_proj._is_hf_initialized
    assert gated_retention.k_proj._is_hf_initialized
    assert gated_retention.v_proj._is_hf_initialized
    assert gated_retention.g_proj._is_hf_initialized
    assert gated_retention.gk_proj._is_hf_initialized
    assert gated_retention.o_proj._is_hf_initialized

    assert gated_retention.q_proj.weight.abs().max().item() <= qkvg_bound + 1e-6
    assert gated_retention.k_proj.weight.abs().max().item() <= qkvg_bound + 1e-6
    assert gated_retention.v_proj.weight.abs().max().item() <= qkvg_bound + 1e-6
    assert gated_retention.g_proj.weight.abs().max().item() <= qkvg_bound + 1e-6
    assert gated_retention.gk_proj.weight.abs().max().item() <= gk_bound + 1e-6
    assert gated_retention.o_proj.weight.abs().max().item() <= out_bound + 1e-6


def test_cross_decoder_rotary_matches_official_interleaved_usage():
    config = YOCOConfig(
        num_hidden_layers=4,
        num_self_decoder_layers=2,
        hidden_size=128,
        self_decoder_attn=_linear_attn(
            num_heads=2,
            num_v_heads=2,
            head_dim=64,
        ),
        cross_decoder_attn=_attn(num_heads=2, num_kv_heads=2),
        intermediate_size=256,
        vocab_size=128,
        max_position_embeddings=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = YOCOForCausalLM(config)

    assert model.model.shared_kv_builder.rotary.interleaved
    assert model.model.cross_layers[0].attn.rotary.interleaved


def test_yoco_rotary_supports_official_inv_freq_formula():
    rotary = YOCORotaryEmbedding(dim=128, base=10000.0, interleaved=True, rope_inv_freq='yoco')

    expected = 1.0 / (10000.0 ** torch.linspace(0, 1, 64, dtype=torch.float32))

    assert torch.allclose(rotary.inv_freq, expected, atol=0, rtol=0)


def test_yoco_config_propagates_rope_inv_freq():
    config = YOCOConfig(
        num_hidden_layers=4,
        num_self_decoder_layers=2,
        hidden_size=128,
        self_decoder_attn={
            **_linear_attn(
                linear_attn_type='gated_retention',
                mode='chunk',
                num_heads=2,
                num_v_heads=2,
                expand_v=1.0,
                window_size=None,
            ),
            'rope_inv_freq': 'yoco',
        },
        cross_decoder_attn={
            **_attn(num_heads=2, num_kv_heads=2),
            'rope_inv_freq': 'yoco',
        },
        intermediate_size=256,
        vocab_size=128,
        max_position_embeddings=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = YOCOForCausalLM(config)

    assert model.model.self_layers[0].attn.rotary.rope_inv_freq == 'yoco'
    assert model.model.shared_kv_builder.rotary.rope_inv_freq == 'yoco'
    assert model.model.cross_layers[0].attn.rotary.rope_inv_freq == 'yoco'


def test_yoco_config_preserves_extra_cross_decoder_attn_fields_and_propagates_qk_norm():
    config = YOCOConfig(
        num_hidden_layers=4,
        num_self_decoder_layers=2,
        hidden_size=128,
        self_decoder_attn=_linear_attn(
            linear_attn_type='gated_retention',
            mode='chunk',
            num_heads=2,
            num_v_heads=2,
            expand_v=1.0,
            window_size=None,
        ),
        cross_decoder_attn={
            **_attn(num_heads=2, num_kv_heads=2),
            'qk_norm': True,
            'custom_flag': 'kept',
        },
        intermediate_size=256,
        vocab_size=128,
        max_position_embeddings=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = YOCOForCausalLM(config)

    assert config.cross_decoder_attn['qk_norm'] is True
    assert config.cross_decoder_attn['custom_flag'] == 'kept'
    assert model.model.cross_layers[0].attn.qk_norm is True


def test_yoco_config_rejects_invalid_rope_inv_freq():
    with pytest.raises(ValueError, match="must be one of"):
        YOCOConfig(
            self_decoder_attn={**_linear_attn(num_heads=2, num_v_heads=2), 'rope_inv_freq': 'bad'},
            cross_decoder_attn=_attn(num_heads=2, num_kv_heads=2),
        )


def test_yoco_embeddings_are_scaled_like_official():
    torch.manual_seed(42)
    config = YOCOConfig(
        num_hidden_layers=0,
        num_self_decoder_layers=0,
        hidden_size=64,
        self_decoder_attn=_linear_attn(
            num_heads=2,
            num_v_heads=2,
        ),
        cross_decoder_attn=_attn(num_heads=2, num_kv_heads=2),
        intermediate_size=128,
        vocab_size=32,
        max_position_embeddings=32,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = YOCOForCausalLM(config)

    input_ids = torch.tensor([[1, 2, 3]])
    expected = model.model.embed_scale * model.model.embeddings(input_ids)
    captured = {}

    def capture_norm_input(module, inputs):
        del module
        captured['hidden_states'] = inputs[0].detach().clone()

    hook = model.model.norm.register_forward_pre_hook(capture_norm_input)
    try:
        model.model(input_ids=input_ids, use_cache=False, return_dict=True)
    finally:
        hook.remove()

    assert 'hidden_states' in captured
    assert torch.allclose(captured['hidden_states'], expected, atol=0, rtol=0)


def test_cross_attention_decode_without_attention_mask_uses_one_step(monkeypatch):
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 4
    num_heads = 2
    head_dim = 64
    hidden_size = num_heads * head_dim
    cross_attn = YOCOCrossAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_heads,
    )

    hidden_states = torch.randn(batch_size, 1, hidden_size)
    shared_k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    shared_v = torch.randn(batch_size, seq_len, num_heads, head_dim)

    monkeypatch.setattr(cross_attn.rotary, 'forward_states', lambda states, **kwargs: states)

    calls = {'decoding': 0}

    def fake_attn_decoding_one_step(q, k, v, cu_seqlens=None, **kwargs):
        del kwargs
        calls['decoding'] += 1
        assert q.shape == (1, batch_size, num_heads, head_dim)
        assert k.shape == (1, batch_size * seq_len, num_heads, head_dim)
        assert v.shape == (1, batch_size * seq_len, num_heads, head_dim)
        expected_cu_seqlens = torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32, device=q.device)
        assert torch.equal(cu_seqlens, expected_cu_seqlens)
        return torch.zeros(1, batch_size, num_heads, head_dim, dtype=q.dtype, device=q.device)

    def fail_parallel_attn(*args, **kwargs):
        raise AssertionError("parallel_attn should not be used for decode without attention_mask")

    monkeypatch.setattr(yoco_attn_mod, 'attn_decoding_one_step', fake_attn_decoding_one_step)
    monkeypatch.setattr(yoco_attn_mod, 'parallel_attn', fail_parallel_attn)

    outputs, attentions = cross_attn(hidden_states, shared_k, shared_v)

    assert calls['decoding'] == 1
    assert outputs.shape == (batch_size, 1, hidden_size)
    assert attentions is None


def test_cross_attention_varlen_decode_uses_per_sequence_rope_offsets(monkeypatch):
    torch.manual_seed(42)
    num_sequences = 2
    seq_lens = torch.tensor([2, 3], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    num_heads = 2
    head_dim = 64
    hidden_size = num_heads * head_dim
    cross_attn = YOCOCrossAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_heads,
    )

    hidden_states = torch.randn(1, num_sequences, hidden_size)
    shared_k = torch.randn(1, int(cu_seqlens[-1]), num_heads, head_dim)
    shared_v = torch.randn(1, int(cu_seqlens[-1]), num_heads, head_dim)

    captured = {}

    def capture_rotary(states, *, seqlen_offset, max_seqlen, cu_seqlens, **kwargs):
        del kwargs
        captured['states_shape'] = states.shape
        captured['seqlen_offset'] = seqlen_offset.clone()
        captured['max_seqlen'] = max_seqlen
        captured['cu_seqlens'] = cu_seqlens.clone()
        return states

    def fake_attn_decoding_one_step(q, k, v, cu_seqlens=None, **kwargs):
        del kwargs
        assert q.shape == (1, num_sequences, num_heads, head_dim)
        assert k.shape == (1, int(cu_seqlens[-1]), num_heads, head_dim)
        assert v.shape == (1, int(cu_seqlens[-1]), num_heads, head_dim)
        assert torch.equal(cu_seqlens, torch.tensor([0, 2, 5], dtype=torch.int32, device=q.device))
        return torch.zeros(1, num_sequences, num_heads, head_dim, dtype=q.dtype, device=q.device)

    monkeypatch.setattr(cross_attn.rotary, 'forward_states', capture_rotary)
    monkeypatch.setattr(yoco_attn_mod, 'attn_decoding_one_step', fake_attn_decoding_one_step)

    outputs, attentions = cross_attn(
        hidden_states,
        shared_k,
        shared_v,
        cu_seqlens=cu_seqlens,
    )

    expected_q_cu_seqlens = torch.arange(0, num_sequences + 1, dtype=torch.int32)
    assert captured['states_shape'] == (1, num_sequences, num_heads, head_dim)
    assert torch.equal(captured['seqlen_offset'], seq_lens - 1)
    assert captured['max_seqlen'] == int(cu_seqlens[-1])
    assert torch.equal(captured['cu_seqlens'], expected_q_cu_seqlens)
    assert outputs.shape == (1, num_sequences, hidden_size)
    assert attentions is None


def test_left_padded_prefill_matches_unpadded_logits_with_cross_decoder():
    if device == "cpu":
        pytest.skip("left-padded prefill test requires CUDA")

    torch.manual_seed(42)
    config = YOCOConfig(
        num_hidden_layers=4,
        num_self_decoder_layers=2,
        hidden_size=128,
        self_decoder_attn=_linear_attn(
            num_heads=2,
            num_v_heads=2,
            head_dim=64,
        ),
        cross_decoder_attn=_attn(num_heads=2, num_kv_heads=2),
        intermediate_size=256,
        vocab_size=128,
        max_position_embeddings=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = YOCOForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    model.eval()

    unpadded_input_ids = torch.tensor([[1, 2, 3]], device=device)
    unpadded_attention_mask = torch.ones_like(unpadded_input_ids, dtype=torch.bool)
    left_padded_input_ids = torch.tensor([[0, 0, 1, 2, 3]], device=device)
    left_padded_attention_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool, device=device)

    outputs_unpadded = model(
        input_ids=unpadded_input_ids,
        attention_mask=unpadded_attention_mask,
        use_cache=True,
        return_dict=True,
    )
    outputs_left_padded = model(
        input_ids=left_padded_input_ids,
        attention_mask=left_padded_attention_mask,
        use_cache=True,
        return_dict=True,
    )

    torch.testing.assert_close(
        outputs_left_padded.logits[:, -1:],
        outputs_unpadded.logits[:, -1:],
        atol=2e-2,
        rtol=2e-2,
    )


def test_skip_cross_decoder_matches_full_cross_decoder_logits_and_builds_cache():
    if device == "cpu":
        pytest.skip("skip cross decoder test requires CUDA")

    torch.manual_seed(42)
    config = YOCOConfig(
        num_hidden_layers=4,
        num_self_decoder_layers=2,
        hidden_size=128,
        self_decoder_attn=_linear_attn(
            num_heads=4,
            num_v_heads=4,
            head_dim=32,
        ),
        cross_decoder_attn=_attn(num_heads=4, num_kv_heads=4),
        intermediate_size=256,
        vocab_size=128,
        max_position_embeddings=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = YOCOForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    model.eval()
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(2, 7), device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[:, :2] = False

    outputs_full = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        skip_cross_decoder=False,
        return_dict=True,
    )
    outputs_skip = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        skip_cross_decoder=True,
        return_dict=True,
    )

    assert outputs_skip.logits.shape == (2, 1, config.vocab_size)
    torch.testing.assert_close(outputs_skip.logits, outputs_full.logits[:, -1:], atol=2e-2, rtol=2e-2)
    assert outputs_skip.past_key_values is not None
    assert outputs_skip.past_key_values.get_seq_length(config.num_self_decoder_layers) == input_ids.shape[1]


def test_skip_cross_decoder_prefill_matches_full_decode_loop():
    if device == "cpu":
        pytest.skip("skip cross decoder decode loop test requires CUDA")

    torch.manual_seed(42)
    config = YOCOConfig(
        num_hidden_layers=4,
        num_self_decoder_layers=2,
        hidden_size=128,
        self_decoder_attn=_linear_attn(
            num_heads=2,
            num_v_heads=2,
            head_dim=64,
        ),
        cross_decoder_attn=_attn(num_heads=2, num_kv_heads=2),
        intermediate_size=256,
        vocab_size=128,
        max_position_embeddings=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = YOCOForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    model.eval()

    input_ids = torch.randint(low=0, high=config.vocab_size, size=(2, 10), device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[:, :2] = False
    prefill_ids = input_ids[:, :6]
    decode_ids = input_ids[:, 6:]
    prefill_attention_mask = attention_mask[:, :6]

    prefill_full = model(
        input_ids=prefill_ids,
        attention_mask=prefill_attention_mask,
        use_cache=True,
        skip_cross_decoder=False,
        return_dict=True,
    )
    prefill_skip = model(
        input_ids=prefill_ids,
        attention_mask=prefill_attention_mask,
        use_cache=True,
        skip_cross_decoder=True,
        return_dict=True,
    )

    torch.testing.assert_close(prefill_skip.logits, prefill_full.logits[:, -1:], atol=2e-2, rtol=2e-2)

    full_past_key_values = prefill_full.past_key_values
    skip_past_key_values = prefill_skip.past_key_values
    full_decode_logits = []
    skip_decode_logits = []

    for step in range(decode_ids.shape[1]):
        decode_attention_mask = attention_mask[:, :prefill_ids.shape[1] + step + 1]
        decode_input_ids = decode_ids[:, step:step+1]

        decode_full = model(
            input_ids=decode_input_ids,
            attention_mask=decode_attention_mask,
            past_key_values=full_past_key_values,
            use_cache=True,
            skip_cross_decoder=False,
            return_dict=True,
        )
        decode_skip = model(
            input_ids=decode_input_ids,
            attention_mask=decode_attention_mask,
            past_key_values=skip_past_key_values,
            use_cache=True,
            skip_cross_decoder=False,
            return_dict=True,
        )

        full_decode_logits.append(decode_full.logits)
        skip_decode_logits.append(decode_skip.logits)
        full_past_key_values = decode_full.past_key_values
        skip_past_key_values = decode_skip.past_key_values

    torch.testing.assert_close(
        torch.cat(skip_decode_logits, dim=1),
        torch.cat(full_decode_logits, dim=1),
        atol=2e-2,
        rtol=2e-2,
    )

    assert skip_past_key_values is not None
    assert full_past_key_values is not None
    assert skip_past_key_values.get_seq_length(config.num_self_decoder_layers) == full_past_key_values.get_seq_length(
        config.num_self_decoder_layers
    )


def test_skip_cross_decoder_rejects_labels():
    config = YOCOConfig(
        num_hidden_layers=4,
        num_self_decoder_layers=2,
        hidden_size=128,
        self_decoder_attn=_linear_attn(
            num_heads=2,
            num_v_heads=2,
            head_dim=64,
        ),
        cross_decoder_attn=_attn(num_heads=2, num_kv_heads=2),
        intermediate_size=256,
        vocab_size=128,
        max_position_embeddings=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = YOCOForCausalLM(config)
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(1, 6))
    labels = torch.randint(low=0, high=config.vocab_size, size=(1, 6))

    with pytest.raises(ValueError, match="skip_cross_decoder=True.*labels"):
        model(
            input_ids=input_ids,
            labels=labels,
            skip_cross_decoder=True,
            return_dict=True,
        )


def test_skip_cross_decoder_rejects_output_hidden_states():
    config = YOCOConfig(
        num_hidden_layers=4,
        num_self_decoder_layers=2,
        hidden_size=128,
        self_decoder_attn=_linear_attn(
            num_heads=2,
            num_v_heads=2,
            head_dim=64,
        ),
        cross_decoder_attn=_attn(num_heads=2, num_kv_heads=2),
        intermediate_size=256,
        vocab_size=128,
        max_position_embeddings=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = YOCOForCausalLM(config)
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(1, 6))

    with pytest.raises(ValueError, match="skip_cross_decoder=True.*output_hidden_states=True"):
        model(
            input_ids=input_ids,
            output_hidden_states=True,
            skip_cross_decoder=True,
            return_dict=True,
        )


def test_training_defaults_disable_cache():
    if device == "cpu":
        pytest.skip("YOCO training cache test requires CUDA")

    torch.manual_seed(42)
    config = YOCOConfig(
        num_hidden_layers=2,
        num_self_decoder_layers=1,
        hidden_size=128,
        self_decoder_attn=_linear_attn(
            num_heads=2,
            num_v_heads=2,
            head_dim=64,
        ),
        cross_decoder_attn=_attn(num_heads=2, num_kv_heads=2),
        intermediate_size=256,
        vocab_size=128,
        max_position_embeddings=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = YOCOForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    model.train()

    input_ids = torch.randint(low=0, high=config.vocab_size, size=(1, 96), device=device)
    outputs = model(input_ids=input_ids, use_cache=None, return_dict=True)

    assert outputs.past_key_values is None


def test_gradient_checkpointing_backward_smoke():
    if device == "cpu":
        pytest.skip("gradient checkpointing smoke test requires CUDA")

    torch.manual_seed(42)
    config = YOCOConfig(
        num_hidden_layers=4,
        num_self_decoder_layers=2,
        hidden_size=128,
        self_decoder_attn=_linear_attn(
            num_heads=4,
            num_v_heads=4,
            head_dim=32,
        ),
        cross_decoder_attn=_attn(num_heads=4, num_kv_heads=4),
        intermediate_size=256,
        vocab_size=128,
        max_position_embeddings=128,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = YOCOForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.train()

    input_ids = torch.randint(low=0, high=config.vocab_size, size=(1, 96), device=device)
    labels = torch.randint(low=0, high=config.vocab_size, size=(1, 96), device=device)
    outputs = model(input_ids=input_ids, labels=labels, use_cache=True, return_dict=True)

    outputs.loss.backward()


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (4, 4, 2000, 8, 64, torch.float16),
        ]
    ]
)
def test_generation(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    config = YOCOConfig(
        num_hidden_layers=L,
        num_self_decoder_layers=L // 2,
        hidden_size=H * D,
        self_decoder_attn=_linear_attn(
            num_heads=H,
            num_v_heads=H,
            head_dim=D,
        ),
        cross_decoder_attn=_attn(num_heads=H, num_kv_heads=H),
        intermediate_size=4 * H * D,
        vocab_size=128,
        max_position_embeddings=T,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = YOCOForCausalLM(config)
    run_test_generation(L, B, T, H, D, YOCOConfig, dtype, model=model, config=config, tol=3e-3)
