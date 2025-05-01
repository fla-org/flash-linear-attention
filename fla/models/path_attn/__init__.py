# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.path_attn.configuration_path_attention import PaTHAttentionConfig
from fla.models.path_attn.modeling_path_attention import PaTHAttentionForCausalLM, PaTHAttentionModel

AutoConfig.register(PaTHAttentionConfig.model_type, PaTHAttentionConfig)
AutoModel.register(PaTHAttentionConfig, PaTHAttentionModel)
AutoModelForCausalLM.register(PaTHAttentionConfig, PaTHAttentionForCausalLM)


__all__ = ['PaTHAttentionConfig', 'PaTHAttentionForCausalLM', 'PaTHAttentionModel']
