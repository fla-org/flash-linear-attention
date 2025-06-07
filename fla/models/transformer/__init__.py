# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.transformer.configuration_transformer import TransformerConfig
from fla.models.transformer.modeling_transformer import TransformerForCausalLM, TransformerModel

AutoConfig.register(TransformerConfig.model_type, TransformerConfig, exist_ok=True)
AutoModel.register(TransformerConfig, TransformerModel, exist_ok=True)
AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)


__all__ = ['TransformerConfig', 'TransformerForCausalLM', 'TransformerModel']
