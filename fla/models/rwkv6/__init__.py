# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification, AutoModelForMaskedImageModeling

from fla.models.rwkv6.configuration_rwkv6 import RWKV6Config, RWKV6VisionConfig
from fla.models.rwkv6.modeling_rwkv6 import RWKV6ForCausalLM, RWKV6Model, RWKV6VisionModel, RWKV6ForImageClassification, RWKV6ForMaskedImageModeling

AutoConfig.register(RWKV6Config.model_type, RWKV6Config)
AutoConfig.register(RWKV6VisionConfig.model_type, RWKV6VisionConfig)
AutoModel.register(RWKV6Config, RWKV6Model)
AutoModelForCausalLM.register(RWKV6Config, RWKV6ForCausalLM)
AutoModel.register(RWKV6VisionConfig, RWKV6VisionModel)
AutoModelForImageClassification.register(RWKV6VisionConfig, RWKV6ForImageClassification)
AutoModelForMaskedImageModeling.register(RWKV6VisionConfig, RWKV6ForMaskedImageModeling)


__all__ = ['RWKV6Config', 'RWKV6ForCausalLM', 'RWKV6Model', 'RWKV6VisionModel', 'RWKV6ForImageClassification', 'RWKV6ForMaskedImageModeling', 'RWKV6VisionConfig']
