# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from fla.models.retnet.configuration_retnet import RetNetConfig, RetNetVisionConfig
from fla.models.retnet.modeling_retnet import RetNetForCausalLM, RetNetModel, RetNetVisionModel, RetNetForImageClassification, RetNetForMaskedImageModeling

AutoConfig.register(RetNetConfig.model_type, RetNetConfig)
AutoConfig.register(RetNetVisionConfig.model_type, RetNetVisionConfig)
AutoModel.register(RetNetConfig, RetNetModel)
AutoModelForCausalLM.register(RetNetConfig, RetNetForCausalLM)
AutoModel.register(RetNetVisionConfig, RetNetVisionModel)
AutoModelForImageClassification.register(RetNetVisionConfig, RetNetForImageClassification)
AutoModelForMaskedImageModeling.register(RetNetVisionConfig, RetNetForMaskedImageModeling)


__all__ = ['RetNetConfig', 'RetNetForCausalLM', 'RetNetModel', 'RetNetVisionModel', 'RetNetForImageClassification', 'RetNetForMaskedImageModeling', 'RetNetVisionConfig']
