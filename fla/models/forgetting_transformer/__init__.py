# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.forgetting_transformer.configuration_forgetting_transformer import ForgettingTransformerConfig
from fla.models.forgetting_transformer.modeling_forgetting_transformer import (
    ForgettingTransformerForCausalLM,
    ForgettingTransformerModel
)

AutoConfig.register(ForgettingTransformerConfig.model_type, ForgettingTransformerConfig, exist_ok=True)
AutoModel.register(ForgettingTransformerConfig, ForgettingTransformerModel, exist_ok=True)
AutoModelForCausalLM.register(ForgettingTransformerConfig, ForgettingTransformerForCausalLM, exist_ok=True)


__all__ = ['ForgettingTransformerConfig', 'ForgettingTransformerForCausalLM', 'ForgettingTransformerModel']
