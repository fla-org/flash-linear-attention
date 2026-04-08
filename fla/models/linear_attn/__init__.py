# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.linear_attn.configuration_linear_attn import LinearAttentionConfig
from fla.models.linear_attn.modeling_linear_attn import LinearAttentionForCausalLM, LinearAttentionModel

AutoConfig.register(LinearAttentionConfig.model_type, LinearAttentionConfig, exist_ok=True)
AutoModel.register(LinearAttentionConfig, LinearAttentionModel, exist_ok=True)
AutoModelForCausalLM.register(LinearAttentionConfig, LinearAttentionForCausalLM, exist_ok=True)

__all__ = ['LinearAttentionConfig', 'LinearAttentionForCausalLM', 'LinearAttentionModel']
