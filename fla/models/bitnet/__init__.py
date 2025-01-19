# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from fla.models.bitnet.configuration_bitnet import BitNetConfig, BitNetVisionConfig
from fla.models.bitnet.modeling_bitnet import BitNetForCausalLM, BitNetModel, BitNetVisionModel, BitNetForImageClassification, BitNetForMaskedImageModeling

AutoConfig.register(BitNetConfig.model_type, BitNetConfig)
AutoConfig.register(BitNetVisionConfig.model_type, BitNetVisionConfig)
AutoModel.register(BitNetConfig, BitNetModel)
AutoModelForCausalLM.register(BitNetConfig, BitNetForCausalLM)
AutoModelForImageClassification.register(BitNetVisionConfig, BitNetForImageClassification)
AutoModelForMaskedImageModeling.register(BitNetVisionConfig, BitNetForMaskedImageModeling)
AutoModel.register(BitNetVisionConfig, BitNetVisionModel)


__all__ = ['BitNetConfig', 'BitNetForCausalLM', 'BitNetModel', 'BitNetVisionConfig', 'BitNetForImageClassification', 'BitNetForMaskedImageModeling', 'BitNetVisionModel']
