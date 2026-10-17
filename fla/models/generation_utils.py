# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers.generation import GenerationMixin
from transformers.utils.deprecation import deprecate_kwarg

if TYPE_CHECKING:
    from fla.models.cache_utils import Cache


class FLAGenerationMixin(GenerationMixin):
    """Use Transformers generation with FLA's recurrent and hybrid cache."""

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # FLA layers consume recurrent state dictionaries, including in hybrid models.
        return False

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = True,
        logits_to_keep: int | torch.Tensor | None = None,
        cache_position: torch.LongTensor | None = None,
        next_sequence_length: int | None = None,
        is_first_iteration: bool = False,
        **kwargs,
    ):
        if cache_position is not None and next_sequence_length is None:
            next_sequence_length = len(cache_position)
        if logits_to_keep is not None:
            kwargs["logits_to_keep"] = logits_to_keep
        return super().prepare_inputs_for_generation(
            input_ids=input_ids,
            next_sequence_length=next_sequence_length,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration,
            use_cache=use_cache,
            **kwargs,
        )


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
