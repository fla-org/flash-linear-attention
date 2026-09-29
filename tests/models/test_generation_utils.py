# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import utils as models_utils
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


class _PartiallyFilledCache:
    """A cache that has already consumed the first 4 prompt tokens of a 1-layer model."""

    def get_seq_length(self) -> int:
        return 4

    def __len__(self) -> int:
        return 1


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


def test_prepare_inputs_for_generation_keeps_inputs_embeds_when_continuing_from_cache():
    model_inputs = _GenerationBackend().prepare_inputs_for_generation(
        input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]),
        past_key_values=_PartiallyFilledCache(),
        inputs_embeds=torch.arange(48, dtype=torch.float32).reshape(1, 6, 8),
        next_sequence_length=2,
        is_first_iteration=True,
        cache_position=torch.tensor([4, 5]),
    )

    assert model_inputs["input_ids"] is None
    assert torch.equal(model_inputs["inputs_embeds"], torch.arange(32, 48, dtype=torch.float32).reshape(1, 2, 8))


def test_prepare_inputs_for_generation_keeps_inputs_embeds_on_first_legacy_iteration(monkeypatch):
    monkeypatch.setattr(models_utils, "_IS_TRANSFORMERS_4_56_PLUS", False)

    model_inputs = _GenerationBackend().prepare_inputs_for_generation(
        input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]),
        past_key_values=_PartiallyFilledCache(),
        inputs_embeds=torch.arange(48, dtype=torch.float32).reshape(1, 6, 8),
        is_first_iteration=True,
    )

    assert "input_ids" not in model_inputs
    assert torch.equal(model_inputs["inputs_embeds"], torch.arange(48, dtype=torch.float32).reshape(1, 6, 8))


def test_prepare_inputs_for_generation_keeps_legacy_last_token_fallback(monkeypatch):
    monkeypatch.setattr(models_utils, "_IS_TRANSFORMERS_4_56_PLUS", False)

    model_inputs = _GenerationBackend().prepare_inputs_for_generation(
        input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]),
        past_key_values=_PartiallyFilledCache(),
        inputs_embeds=torch.arange(48, dtype=torch.float32).reshape(1, 6, 8),
    )

    assert torch.equal(model_inputs["input_ids"], torch.tensor([[6]]))
    assert model_inputs.get("inputs_embeds") is None
