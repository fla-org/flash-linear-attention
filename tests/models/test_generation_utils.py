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

from fla.models import generation_utils
from fla.models.generation_utils import FLAGenerationMixin, FLAUnsupportedCacheGenerationMixin


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


def test_generate_forwards_supported_generation_calls():
    args, kwargs = _SuccessfulModel().generate("input", max_new_tokens=1)

    assert args == ("input",)
    assert kwargs == {"max_new_tokens": 1}


def test_generate_translates_unsupported_cache_strategy_errors():
    with pytest.raises(AttributeError, match="not supported for _UnsupportedCacheModel") as raised:
        _UnsupportedCacheModel().generate("input")

    assert raised.value.__context__ is _PastKeyValuesErrorBackend.exception


def test_generate_preserves_unrelated_attribute_errors():
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
    with_cache_position, past_length, input_length, with_embeds, next_length, first_iteration, expected_key, expected_length,
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
    result = FLAGenerationMixin().prepare_inputs_for_generation(**arguments)
    expected = inputs_embeds if expected_key == 'inputs_embeds' else input_ids
    torch.testing.assert_close(result[expected_key], expected[:, -expected_length:])
    assert result['input_ids' if expected_key == 'inputs_embeds' else 'inputs_embeds'] is None
    assert result['past_key_values'] is cache
    assert result['attention_mask'] is attention_mask
    assert result['use_cache'] == (cache is not None)
    assert result['logits_to_keep'] == 1
    if expected_key == 'input_ids':
        assert result['input_ids'].is_contiguous()
    if generation_utils.supports_transformers_next_sequence_length():
        reference = GenerationMixin.prepare_inputs_for_generation(
            SimpleNamespace(
                config=SimpleNamespace(is_encoder_decoder=False),
                device=torch.device('cpu'),
                is_remote_code=lambda: False,
            ),
            **arguments,
        )
        torch.testing.assert_close(result[expected_key], reference[expected_key])


@pytest.mark.parametrize('cache_positions', [False, True], ids=['legacy', 'cache-positions'])
@pytest.mark.parametrize('past_length', [None, 0, 4], ids=['no-cache', 'empty-cache', 'populated-cache'])
@pytest.mark.parametrize('with_embeds', [False, True], ids=['ids', 'embeds'])
def test_prepare_inputs_for_generation_legacy(monkeypatch, cache_positions, past_length, with_embeds):
    monkeypatch.setattr(generation_utils, 'supports_transformers_next_sequence_length', lambda: False)
    monkeypatch.setattr(generation_utils, 'check_transformers_version', lambda _: cache_positions)
    input_ids = torch.arange(12).reshape(2, 6)
    inputs_embeds = torch.arange(96, dtype=torch.float32).reshape(2, 6, 8) if with_embeds else None
    cache = _Cache(past_length) if past_length is not None else None
    positions = torch.tensor([5]) if past_length else torch.arange(6)
    result = FLAGenerationMixin().prepare_inputs_for_generation(
        input_ids=input_ids,
        past_key_values=cache,
        inputs_embeds=inputs_embeds,
        cache_position=positions if cache_positions else None,
    )
    if with_embeds and not past_length:
        assert result['input_ids'] is None
        torch.testing.assert_close(result['inputs_embeds'], inputs_embeds)
    else:
        assert result['inputs_embeds'] is None
        torch.testing.assert_close(result['input_ids'], input_ids[:, -1:] if past_length else input_ids)
    assert result.get('cache_position') is (positions if cache_positions else None)


@pytest.mark.skipif(
    not generation_utils.supports_transformers_next_sequence_length(),
    reason='Requires the upstream sequence-length generation protocol.',
)
def test_prepare_inputs_for_generation_decode_without_cache():
    input_ids = torch.arange(14).reshape(2, 7)
    result = FLAGenerationMixin().prepare_inputs_for_generation(
        input_ids=input_ids,
        inputs_embeds=torch.arange(96, dtype=torch.float32).reshape(2, 6, 8),
        use_cache=False,
    )
    torch.testing.assert_close(result['input_ids'], input_ids)
    assert result['inputs_embeds'] is None


@pytest.mark.skipif(
    generation_utils.supports_transformers_next_sequence_length() or not generation_utils.check_transformers_version('4.56'),
    reason='Requires the upstream cache-position generation protocol.',
)
@pytest.mark.parametrize(('input_length', 'embed_length'), [(0, 6), (6, 2)], ids=['full-embeds', 'new-embeds'])
def test_prepare_inputs_for_generation_legacy_cached_embeds(input_length, embed_length):
    inputs_embeds = torch.arange(2 * embed_length * 8, dtype=torch.float32).reshape(2, embed_length, 8)
    result = FLAGenerationMixin().prepare_inputs_for_generation(
        input_ids=torch.arange(2 * input_length).reshape(2, input_length),
        past_key_values=_Cache(4),
        inputs_embeds=inputs_embeds,
        cache_position=torch.arange(4, 6),
    )
    assert result['input_ids'] is None
    torch.testing.assert_close(result['inputs_embeds'], inputs_embeds[:, -2:])
