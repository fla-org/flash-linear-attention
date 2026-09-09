# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest

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
