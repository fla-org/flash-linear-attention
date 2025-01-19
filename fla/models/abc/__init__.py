# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from fla.models.abc.configuration_abc import ABCConfig, ABCVisionConfig
from fla.models.abc.modeling_abc import ABCForCausalLM, ABCModel, ABCVisionModel, ABCForImageClassification, ABCForMaskedImageModeling

AutoConfig.register(ABCConfig.model_type, ABCConfig)
AutoConfig.register(ABCVisionConfig.model_type, ABCVisionConfig)
AutoModel.register(ABCConfig, ABCModel)
AutoModelForCausalLM.register(ABCConfig, ABCForCausalLM)
AutoModelForImageClassification.register(ABCVisionConfig, ABCForImageClassification)
AutoModelForMaskedImageModeling.register(ABCVisionConfig, ABCForMaskedImageModeling)
AutoModel.register(ABCVisionConfig, ABCVisionModel)


__all__ = ['ABCConfig', 'ABCForCausalLM', 'ABCModel', 'ABCVisionModel', 'ABCForImageClassification', 'ABCForMaskedImageModeling', 'ABCVisionConfig']
