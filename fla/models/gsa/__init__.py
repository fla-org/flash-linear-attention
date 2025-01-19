# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from fla.models.gsa.configuration_gsa import GSAConfig, GSAVisionConfig
from fla.models.gsa.modeling_gsa import GSAForCausalLM, GSAModel, GSAVisionModel, GSAForImageClassification, GSAForMaskedImageModeling

AutoConfig.register(GSAConfig.model_type, GSAConfig)
AutoConfig.register(GSAVisionConfig.model_type, GSAVisionConfig)
AutoModel.register(GSAConfig, GSAModel)
AutoModelForCausalLM.register(GSAConfig, GSAForCausalLM)
AutoModelForImageClassification.register(GSAVisionConfig, GSAForImageClassification)
AutoModelForMaskedImageModeling.register(GSAVisionConfig, GSAForMaskedImageModeling)
AutoModel.register(GSAVisionConfig, GSAVisionModel)

__all__ = ['GSAConfig', 'GSAForCausalLM', 'GSAModel', 'GSAVisionModel', 'GSAForImageClassification', 'GSAForMaskedImageModeling', 'GSAVisionConfig']
