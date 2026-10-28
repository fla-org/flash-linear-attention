# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from types import SimpleNamespace

import pytest
import torch
from transformers.generation import GenerationMixin

from fla import models
from fla.models import (
    BitNetConfig,
    DeltaFormerConfig,
    ForgettingTransformerConfig,
    GLAConfig,
    PaTHAttentionConfig,
)
from fla.models.cache_utils import Cache
from fla.models.generation_utils import FLAGenerationMixin, FLAUnsupportedCacheGenerationMixin
from fla.utils import assert_close, device, device_name, find_spec_cached

from .utils import create_model_and_config


@pytest.fixture
def preparation_model():
    model = FLAGenerationMixin()
    model.config = SimpleNamespace(is_encoder_decoder=False)
    model.device = torch.device('cpu')
    model.is_remote_code = lambda: False
    return model


class _Cache:
    def __init__(self, seq_length: int):
        self.seq_length = seq_length

    def get_seq_length(self) -> int:
        return self.seq_length

    def __len__(self) -> int:
        return 1


class _GenerationBackend(FLAGenerationMixin):
    def generate(self, *args, **kwargs):
        return args, kwargs


class _PastKeyValuesErrorBackend(FLAGenerationMixin):
    exception = AttributeError("cache does not expose `past_key_values`")

    def generate(self, *args, **kwargs):
        raise self.exception


class _OtherAttributeErrorBackend(FLAGenerationMixin):
    exception = AttributeError("unrelated failure")

    def generate(self, *args, **kwargs):
        raise self.exception


class _SuccessfulModel(FLAUnsupportedCacheGenerationMixin, _GenerationBackend):
    pass


class _UnsupportedCacheModel(FLAUnsupportedCacheGenerationMixin, _PastKeyValuesErrorBackend):
    pass


class _OtherAttributeErrorModel(FLAUnsupportedCacheGenerationMixin, _OtherAttributeErrorBackend):
    pass


def test_generate():
    args, kwargs = _SuccessfulModel().generate("input", max_new_tokens=1)

    assert args == ("input",)
    assert kwargs == {"max_new_tokens": 1}


def test_generate_cache_error():
    with pytest.raises(AttributeError, match="not supported for _UnsupportedCacheModel") as raised:
        _UnsupportedCacheModel().generate("input")

    assert raised.value.__context__ is _PastKeyValuesErrorBackend.exception


def test_generate_attribute_error():
    with pytest.raises(AttributeError) as raised:
        _OtherAttributeErrorModel().generate("input")

    assert raised.value is _OtherAttributeErrorBackend.exception


@pytest.mark.parametrize('with_cache_position', [False, True], ids=['no-position', 'with-position'])
@pytest.mark.parametrize(
    ('past_length', 'input_length', 'with_embeds', 'next_length', 'first_iteration', 'expected_key', 'expected_length'),
    [
        pytest.param(4, 6, True, 2, True, 'inputs_embeds', 2, id='cached-embeds'),
        pytest.param(4, 0, True, 2, True, 'inputs_embeds', 2, id='cached-embeds-empty-ids'),
        pytest.param(4, None, True, 2, True, 'inputs_embeds', 2, id='cached-embeds-no-ids'),
        pytest.param(4, 6, False, 2, True, 'input_ids', 2, id='cached-ids'),
        pytest.param(4, 2, False, None, True, 'input_ids', 2, id='already-sliced-prompt'),
        pytest.param(4, 6, True, None, True, 'inputs_embeds', 6, id='unsliced-embeds'),
        pytest.param(4, 7, True, 1, False, 'input_ids', 1, id='decode-with-stale-embeds'),
        pytest.param(0, 7, True, 1, None, 'input_ids', 1, id='decode-with-empty-cache'),
        pytest.param(None, 7, True, None, False, 'input_ids', 7, id='decode-without-cache'),
        pytest.param(None, 6, True, None, True, 'inputs_embeds', 6, id='prefill-embeds'),
    ],
)
def test_prepare_inputs_for_generation(
    preparation_model, with_cache_position, past_length, input_length, with_embeds, next_length,
    first_iteration, expected_key, expected_length,
):
    input_ids = torch.arange(2 * input_length).reshape(2, input_length) if input_length is not None else None
    inputs_embeds = torch.arange(96, dtype=torch.float32).reshape(2, 6, 8) if with_embeds else None
    cache = _Cache(past_length) if past_length is not None else None
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    cache_position = torch.arange(past_length or 0, (past_length or 0) + expected_length) if with_cache_position else None
    arguments = dict(
        input_ids=input_ids,
        past_key_values=cache,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        use_cache=cache is not None,
        logits_to_keep=1,
        cache_position=cache_position,
        next_sequence_length=next_length,
        is_first_iteration=first_iteration,
    )
    result = preparation_model.prepare_inputs_for_generation(**arguments)
    expected = inputs_embeds if expected_key == 'inputs_embeds' else input_ids
    torch.testing.assert_close(result[expected_key], expected[:, -expected_length:])
    assert result.get('input_ids' if expected_key == 'inputs_embeds' else 'inputs_embeds') is None
    assert result.get('past_key_values') is cache
    assert result['attention_mask'] is attention_mask
    assert result['use_cache'] == (cache is not None)
    assert result['logits_to_keep'] == 1
    if expected_key == 'input_ids':
        assert result['input_ids'].is_contiguous()
    reference = GenerationMixin.prepare_inputs_for_generation(
        SimpleNamespace(
            config=SimpleNamespace(is_encoder_decoder=False),
            device=torch.device('cpu'),
            is_remote_code=lambda: False,
        ),
        **arguments,
    )
    torch.testing.assert_close(result[expected_key], reference[expected_key])


def test_prepare_inputs_for_generation_decode_without_cache(preparation_model):
    input_ids = torch.arange(14).reshape(2, 7)
    result = preparation_model.prepare_inputs_for_generation(
        input_ids=input_ids,
        inputs_embeds=torch.arange(96, dtype=torch.float32).reshape(2, 6, 8),
        use_cache=False,
    )
    torch.testing.assert_close(result['input_ids'], input_ids)
    assert result.get('inputs_embeds') is None


@pytest.mark.parametrize('first_iteration', [False, True])
def test_prepare_inputs_cached(first_iteration, make_model):
    torch.manual_seed(42)
    model = make_model(GLAConfig, hidden_size=32, vocab_size=32)
    cache = Cache()
    cache.update(recurrent_state=torch.zeros(2, 1, 4, 4), offset=4)
    input_ids = torch.arange(14).reshape(2, 7)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    attention_mask = torch.ones_like(input_ids)
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    prepared = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        past_key_values=cache,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        next_sequence_length=3,
        is_first_iteration=first_iteration,
        cu_seqlens=cu_seqlens,
    )
    if first_iteration:
        assert prepared['input_ids'] is None
        torch.testing.assert_close(prepared['inputs_embeds'], inputs_embeds[:, -3:])
    else:
        torch.testing.assert_close(prepared['input_ids'], input_ids[:, -3:])
        assert prepared.get('inputs_embeds') is None
    assert prepared['attention_mask'] is attention_mask
    assert prepared['past_key_values'] is cache
    assert prepared['cu_seqlens'] is cu_seqlens


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


@pytest.mark.skipif(device_name == 'cpu', reason='Fused loss and model kernels require an accelerator.')
@pytest.mark.skipif(find_spec_cached('flash_attn') is None, reason='Fused-loss integration requires flash-attn.')
@pytest.mark.parametrize(
    "config_class",
    [
        pytest.param(config_class, id=config_class.__name__)
        for config_class in [
            BitNetConfig,
            DeltaFormerConfig,
            ForgettingTransformerConfig,
            PaTHAttentionConfig,
        ]
    ],
)
def test_generate_fused_loss(config_class):
    torch.manual_seed(42)
    L, B, T, H, D = 2, 2, 64, 4, 64
    model, config = create_model_and_config(config_class, L, H, D, dtype=torch.bfloat16)
    model.eval()

    input_ids = torch.randint(low=0, high=config.vocab_size, size=(B, T), device=device)
    labels = input_ids.clone()

    with torch.no_grad():
        expected = model(input_ids, labels=labels, use_cache=False).loss

        model.config.fuse_linear_cross_entropy = True
        outputs = model(input_ids, labels=labels, use_cache=False)
        assert outputs.logits is None
        assert_close("loss", expected, outputs.loss, 0.02)

        # the generate path passes no labels; logits must still be materialized
        outputs = model(input_ids, use_cache=False)
        assert outputs.logits is not None
        output_ids = model.generate(input_ids[:, :T // 2], max_new_tokens=2, use_cache=config_class is not DeltaFormerConfig)
    assert output_ids.shape[1] == T // 2 + 2
