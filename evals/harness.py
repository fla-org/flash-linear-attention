# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

import fla  # noqa


@register_model('fla')
class FlashLinearAttentionLMWrapper(HFLM):
    def __init__(self, **kwargs) -> FlashLinearAttentionLMWrapper:

        # TODO: provide options for doing inference with different kernels

        super().__init__(**kwargs)


if __name__ == "__main__":
    cli_evaluate()
