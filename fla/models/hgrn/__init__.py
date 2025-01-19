# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from fla.models.hgrn.configuration_hgrn import HGRNConfig, HGRNVisionConfig
from fla.models.hgrn.modeling_hgrn import HGRNForCausalLM, HGRNModel, HGRNVisionModel, HGRNForImageClassification, HGRNForMaskedImageModeling

AutoConfig.register(HGRNConfig.model_type, HGRNConfig)
AutoConfig.register(HGRNVisionConfig.model_type, HGRNVisionConfig)
AutoModel.register(HGRNConfig, HGRNModel)
AutoModelForCausalLM.register(HGRNConfig, HGRNForCausalLM)
AutoModelForImageClassification.register(HGRNVisionConfig, HGRNForImageClassification)
AutoModelForMaskedImageModeling.register(HGRNVisionConfig, HGRNForMaskedImageModeling)
AutoModel.register(HGRNVisionConfig, HGRNVisionModel)


__all__ = ['HGRNConfig', 'HGRNForCausalLM', 'HGRNModel', 'HGRNVisionModel', 'HGRNForImageClassification', 'HGRNForMaskedImageModeling', 'HGRNVisionConfig']
