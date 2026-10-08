# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.stickbreaking_attn.configuration_stickbreaking_attn import StickBreakingAttentionConfig
from fla.models.stickbreaking_attn.modeling_stickbreaking_attn import (
    StickBreakingAttentionForCausalLM,
    StickBreakingAttentionModel,
)

AutoConfig.register(StickBreakingAttentionConfig.model_type, StickBreakingAttentionConfig, exist_ok=True)
AutoModel.register(StickBreakingAttentionConfig, StickBreakingAttentionModel, exist_ok=True)
AutoModelForCausalLM.register(StickBreakingAttentionConfig, StickBreakingAttentionForCausalLM, exist_ok=True)

__all__ = ['StickBreakingAttentionConfig', 'StickBreakingAttentionForCausalLM', 'StickBreakingAttentionModel']
