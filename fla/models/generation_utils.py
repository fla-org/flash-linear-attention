# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch
from transformers.cache_utils import Cache as HFCacheBase
from transformers.generation import GenerationMixin
from transformers.utils.deprecation import deprecate_kwarg

from fla.utils import check_transformers_version, supports_transformers_next_sequence_length


class FLAGenerationMixin(GenerationMixin):
    """Prepare recurrent-model inputs for supported Transformers generation protocols."""

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: HFCacheBase | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = True,
        logits_to_keep: int | None = None,
        cache_position: torch.LongTensor | None = None,
        next_sequence_length: int | None = None,
        is_first_iteration: bool | None = None,
        **kwargs,
    ):
        uses_sequence_length = (
            supports_transformers_next_sequence_length() or next_sequence_length is not None or is_first_iteration is not None
        )
        if uses_sequence_length:
            use_inputs_embeds = inputs_embeds is not None and is_first_iteration is True
            if next_sequence_length is not None:
                if use_inputs_embeds:
                    inputs_embeds = inputs_embeds[:, -next_sequence_length:]
                elif input_ids is not None:
                    input_ids = input_ids[:, -next_sequence_length:]
        else:
            has_past = past_key_values is not None and (
                past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else len(past_key_values)
            ) > 0
            if past_key_values is not None:
                if check_transformers_version('4.56') and cache_position is not None and input_ids is not None:
                    if hasattr(self, '_cache_dependant_input_preparation'):
                        inputs_embeds, input_ids = self._cache_dependant_input_preparation(
                            input_ids=input_ids,
                            inputs_embeds=inputs_embeds,
                            cache_position=cache_position,
                        )
                    elif input_ids.shape[1] != cache_position.shape[0]:
                        input_ids = input_ids[:, cache_position]
                elif has_past and input_ids is not None:
                    input_ids = input_ids[:, -1:]
            use_inputs_embeds = inputs_embeds is not None and (
                (cache_position is None or len(cache_position) == inputs_embeds.shape[1])
                if check_transformers_version('4.56') else not has_past
            )

        # contiguous token strides avoid recompilation during decoding: huggingface/transformers#29114.
        model_inputs = {
            'input_ids': None if use_inputs_embeds or input_ids is None else input_ids.contiguous(),
            'inputs_embeds': inputs_embeds if use_inputs_embeds else None,
        }
        if check_transformers_version('4.56'):
            model_inputs['cache_position'] = cache_position

        if logits_to_keep is not None:
            model_inputs['logits_to_keep'] = logits_to_keep

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
        })
        return model_inputs


class FLAUnsupportedCacheGenerationMixin(FLAGenerationMixin):
    """Generation mixin for models whose caches do not support every decoding strategy."""

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if "past_key_values" in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies",
                )
            raise exception
