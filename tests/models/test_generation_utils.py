# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import GLAConfig, GLAForCausalLM
from fla.models.cache_utils import Cache
from fla.models.utils import FLAGenerationMixin, FLAUnsupportedCacheGenerationMixin


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


@pytest.mark.parametrize('first_iteration', [False, True])
def test_cached_generation_selects_new_prompt_inputs(first_iteration):
    model = GLAForCausalLM(GLAConfig(hidden_size=32, num_hidden_layers=0, vocab_size=32, fuse_norm=False))
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
