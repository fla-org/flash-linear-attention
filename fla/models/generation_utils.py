# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch
import transformers
from packaging import version
from transformers.cache_utils import Cache as HFCacheBase
from transformers.generation import GenerationMixin
from transformers.utils.deprecation import deprecate_kwarg

_IS_TRANSFORMERS_4_56_PLUS = version.parse(transformers.__version__) >= version.parse("4.56.0")


class FLAGenerationMixin(GenerationMixin):
    """
    Flash Linear Attention Generation Mixin that provides version-compatible generation methods.
    This mixin handles transformers library version differences, particularly for prepare_inputs_for_generation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        **kwargs,
    ):
        # Use pre-computed version comparison for performance
        if _IS_TRANSFORMERS_4_56_PLUS:
            # For transformers 4.56.0+, use cache_position-based logic
            model_inputs = {}

            # Handle cache-dependent input preparation
            if past_key_values is not None:
                model_inputs["past_key_values"] = past_key_values

                # Use the new cache-dependent input preparation method if available
                if hasattr(self, '_cache_dependant_input_preparation') and cache_position is not None:
                    inputs_embeds, input_ids = self._cache_dependant_input_preparation(
                        input_ids, inputs_embeds, cache_position,
                    )
                elif cache_position is not None:
                    # Fallback: manually slice using cache_position
                    if input_ids is not None and input_ids.shape[1] != cache_position.shape[0]:
                        input_ids = input_ids[:, cache_position]
                elif (past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else len(past_key_values)) > 0:
                    # Ultimate fallback to old behavior
                    input_ids = input_ids[:, -1:]

            # Handle input format (similar to base class logic)
            if inputs_embeds is not None and (cache_position is None or len(cache_position) == inputs_embeds.shape[1]):
                model_inputs['inputs_embeds'] = inputs_embeds
                model_inputs['input_ids'] = None
            else:
                model_inputs['input_ids'] = input_ids.contiguous() if input_ids is not None else None
                model_inputs['inputs_embeds'] = None

            model_inputs['cache_position'] = cache_position

        else:
            # For older transformers versions, use the original logic
            model_inputs = {}
            # only last token for `inputs_ids` if the `past_key_values` is not empty.
            if past_key_values is not None and (past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else len(past_key_values)) > 0:
                input_ids = input_ids[:, -1:]
            # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
            if inputs_embeds is not None and hasattr(past_key_values, '__len__') and len(past_key_values) == 0:
                model_inputs = {'inputs_embeds': inputs_embeds}
            else:
                # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
                # recompiles graphs as the stride of the inputs is a guard.
                # Ref: https://github.com/huggingface/transformers/pull/29114
                # TODO: use `next_tokens` directly instead.
                model_inputs = {'input_ids': input_ids.contiguous()}

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
