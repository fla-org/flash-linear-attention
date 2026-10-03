# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models.utils import FLAGenerationMixin, FLAUnsupportedCacheGenerationMixin


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


def test_prepare_inputs_for_generation_slices_cached_embeds():
    model = object.__new__(FLAGenerationMixin)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    inputs_embeds = torch.arange(48, dtype=torch.float32).reshape(1, 6, 8)

    model_inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        past_key_values=_Cache(4),
        inputs_embeds=inputs_embeds,
        next_sequence_length=2,
        is_first_iteration=True,
        cache_position=torch.tensor([4, 5]),
    )

    assert model_inputs["input_ids"] is None
    torch.testing.assert_close(model_inputs["inputs_embeds"], inputs_embeds[:, -2:])


def test_prepare_inputs_for_generation_slices_cached_input_ids():
    model = object.__new__(FLAGenerationMixin)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    model_inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        past_key_values=_Cache(4),
        next_sequence_length=2,
        is_first_iteration=True,
    )

    torch.testing.assert_close(model_inputs["input_ids"], input_ids[:, -2:])
    assert model_inputs["inputs_embeds"] is None


def test_prepare_inputs_for_generation_uses_embeds_without_cache(monkeypatch):
    import fla.models.utils as utils

    monkeypatch.setattr(utils, "_IS_TRANSFORMERS_4_56_PLUS", False)
    model = object.__new__(FLAGenerationMixin)
    input_ids = torch.tensor([[1, 2, 3]])
    inputs_embeds = torch.randn(1, 3, 8)

    model_inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        is_first_iteration=True,
    )

    assert model_inputs["input_ids"] is None
    torch.testing.assert_close(model_inputs["inputs_embeds"], inputs_embeds)
