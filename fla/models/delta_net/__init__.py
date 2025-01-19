# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from fla.models.delta_net.configuration_delta_net import DeltaNetConfig, DeltaNetVisionConfig
from fla.models.delta_net.modeling_delta_net import (DeltaNetForCausalLM,
                                                     DeltaNetModel,
                                                     DeltaNetVisionModel,
                                                     DeltaNetForImageClassification,
                                                     DeltaNetForMaskedImageModeling)

AutoConfig.register(DeltaNetConfig.model_type, DeltaNetConfig)
AutoConfig.register(DeltaNetVisionConfig.model_type, DeltaNetVisionConfig)
AutoModel.register(DeltaNetConfig, DeltaNetModel)
AutoModelForCausalLM.register(DeltaNetConfig, DeltaNetForCausalLM)
AutoModel.register(DeltaNetVisionConfig, DeltaNetVisionModel)
AutoModelForImageClassification.register(DeltaNetVisionConfig, DeltaNetForImageClassification)
AutoModelForMaskedImageModeling.register(DeltaNetVisionConfig, DeltaNetForMaskedImageModeling)

__all__ = ['DeltaNetConfig', 'DeltaNetForCausalLM', 'DeltaNetModel', 'DeltaNetVisionModel', 'DeltaNetForImageClassification', 'DeltaNetForMaskedImageModeling', 'DeltaNetVisionConfig']
