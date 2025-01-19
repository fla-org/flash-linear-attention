# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from fla.models.linear_attn.configuration_linear_attn import \
    LinearAttentionConfig, LinearAttentionVisionConfig
from fla.models.linear_attn.modeling_linear_attn import (
    LinearAttentionForCausalLM, LinearAttentionModel, LinearAttentionVisionModel, LinearAttentionForImageClassification, LinearAttentionForMaskedImageModeling)

AutoConfig.register(LinearAttentionConfig.model_type, LinearAttentionConfig)
AutoConfig.register(LinearAttentionVisionConfig.model_type, LinearAttentionVisionConfig)
AutoModel.register(LinearAttentionConfig, LinearAttentionModel)
AutoModelForCausalLM.register(LinearAttentionConfig, LinearAttentionForCausalLM)
AutoModelForImageClassification.register(LinearAttentionVisionConfig, LinearAttentionForImageClassification)
AutoModelForMaskedImageModeling.register(LinearAttentionVisionConfig, LinearAttentionForMaskedImageModeling)
AutoModel.register(LinearAttentionVisionConfig, LinearAttentionVisionModel)

__all__ = ['LinearAttentionConfig', 'LinearAttentionForCausalLM', 'LinearAttentionModel', 'LinearAttentionVisionModel', 'LinearAttentionForImageClassification', 'LinearAttentionForMaskedImageModeling', 'LinearAttentionVisionConfig']
