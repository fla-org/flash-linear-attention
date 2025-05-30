# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.linear_attn.configuration_linear_attn import LinearAttentionConfig
from fla.models.linear_attn.modeling_linear_attn import LinearAttentionForCausalLM, LinearAttentionModel

AutoConfig.register(LinearAttentionConfig.model_type, LinearAttentionConfig, exist_ok=True)
AutoModel.register(LinearAttentionConfig, LinearAttentionModel, exist_ok=True)
AutoModelForCausalLM.register(LinearAttentionConfig, LinearAttentionForCausalLM, exist_ok=True)

__all__ = ['LinearAttentionConfig', 'LinearAttentionForCausalLM', 'LinearAttentionModel']
