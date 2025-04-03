# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.forgetting_transformer.configuration_forgetting_transformer import ForgettingTransformerConfig
from fla.models.forgetting_transformer.modeling_forgetting_transformer import (
    ForgettingTransformerForCausalLM,
    ForgettingTransformerModel
)

AutoConfig.register(ForgettingTransformerConfig.model_type, ForgettingTransformerConfig)
AutoModel.register(ForgettingTransformerConfig, ForgettingTransformerModel)
AutoModelForCausalLM.register(ForgettingTransformerConfig, ForgettingTransformerForCausalLM)


__all__ = ['ForgettingTransformerConfig', 'ForgettingTransformerForCausalLM', 'ForgettingTransformerModel']
