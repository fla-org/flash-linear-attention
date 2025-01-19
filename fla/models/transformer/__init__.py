# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from fla.models.transformer.configuration_transformer import TransformerConfig, TransformerVisionConfig
from fla.models.transformer.modeling_transformer import (
    TransformerForCausalLM, TransformerModel, TransformerVisionModel, TransformerForImageClassification, TransformerForMaskedImageModeling)

AutoConfig.register(TransformerConfig.model_type, TransformerConfig)
AutoConfig.register(TransformerVisionConfig.model_type, TransformerVisionConfig)
AutoModel.register(TransformerConfig, TransformerModel)
AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM)
AutoModelForImageClassification.register(TransformerVisionConfig, TransformerForImageClassification)
AutoModelForMaskedImageModeling.register(TransformerVisionConfig, TransformerForMaskedImageModeling)
AutoModel.register(TransformerVisionConfig, TransformerVisionModel)


__all__ = ['TransformerConfig', 'TransformerForCausalLM', 'TransformerModel', 'TransformerVisionModel', 'TransformerForImageClassification', 'TransformerForMaskedImageModeling', 'TransformerVisionConfig']
