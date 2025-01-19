# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from fla.models.gated_deltanet.configuration_gated_deltanet import \
    GatedDeltaNetConfig, GatedDeltaNetVisionConfig
from fla.models.gated_deltanet.modeling_gated_deltanet import (
    GatedDeltaNetForCausalLM, GatedDeltaNetModel, GatedDeltaNetVisionModel, GatedDeltaNetForImageClassification, GatedDeltaNetForMaskedImageModeling)

AutoConfig.register(GatedDeltaNetConfig.model_type, GatedDeltaNetConfig)
AutoConfig.register(GatedDeltaNetVisionConfig.model_type, GatedDeltaNetVisionConfig)
AutoModel.register(GatedDeltaNetConfig, GatedDeltaNetModel)
AutoModelForCausalLM.register(GatedDeltaNetConfig, GatedDeltaNetForCausalLM)
AutoModelForImageClassification.register(GatedDeltaNetVisionConfig, GatedDeltaNetForImageClassification)
AutoModelForMaskedImageModeling.register(GatedDeltaNetVisionConfig, GatedDeltaNetForMaskedImageModeling)
AutoModel.register(GatedDeltaNetVisionConfig, GatedDeltaNetVisionModel)

__all__ = ['GatedDeltaNetConfig', 'GatedDeltaNetForCausalLM', 'GatedDeltaNetModel', 'GatedDeltaNetVisionModel', 'GatedDeltaNetForImageClassification', 'GatedDeltaNetForMaskedImageModeling', 'GatedDeltaNetVisionConfig']
