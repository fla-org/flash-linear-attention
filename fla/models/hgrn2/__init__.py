# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from fla.models.hgrn2.configuration_hgrn2 import HGRN2Config, HGRN2VisionConfig
from fla.models.hgrn2.modeling_hgrn2 import HGRN2ForCausalLM, HGRN2Model, HGRN2VisionModel, HGRN2ForImageClassification, HGRN2ForMaskedImageModeling

AutoConfig.register(HGRN2Config.model_type, HGRN2Config)
AutoConfig.register(HGRN2VisionConfig.model_type, HGRN2VisionConfig)
AutoModel.register(HGRN2Config, HGRN2Model)
AutoModelForCausalLM.register(HGRN2Config, HGRN2ForCausalLM)
AutoModelForImageClassification.register(HGRN2VisionConfig, HGRN2ForImageClassification)
AutoModelForMaskedImageModeling.register(HGRN2VisionConfig, HGRN2ForMaskedImageModeling)
AutoModel.register(HGRN2VisionConfig, HGRN2VisionModel)


__all__ = ['HGRN2Config', 'HGRN2ForCausalLM', 'HGRN2Model', 'HGRN2VisionModel', 'HGRN2ForImageClassification', 'HGRN2ForMaskedImageModeling', 'HGRN2VisionConfig']
