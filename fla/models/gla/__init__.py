# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from fla.models.gla.configuration_gla import GLAConfig, GLAVisionConfig
from fla.models.gla.modeling_gla import GLAForCausalLM, GLAModel, GLAVisionModel, GLAForImageClassification, GLAForMaskedImageModeling

AutoConfig.register(GLAConfig.model_type, GLAConfig)
AutoConfig.register(GLAVisionConfig.model_type, GLAVisionConfig)
AutoModel.register(GLAConfig, GLAModel)
AutoModelForCausalLM.register(GLAConfig, GLAForCausalLM)
AutoModelForImageClassification.register(GLAVisionConfig, GLAForImageClassification)
AutoModelForMaskedImageModeling.register(GLAVisionConfig, GLAForMaskedImageModeling)
AutoModel.register(GLAVisionConfig, GLAVisionModel)


__all__ = ['GLAConfig', 'GLAForCausalLM', 'GLAModel', 'GLAVisionModel', 'GLAForImageClassification', 'GLAForMaskedImageModeling', 'GLAVisionConfig']
