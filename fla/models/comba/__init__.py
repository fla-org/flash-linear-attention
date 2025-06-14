# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.comba.configuration_comba import CombaConfig
from fla.models.comba.modeling_comba import CombaForCausalLM, CombaModel

AutoConfig.register(CombaConfig.model_type, CombaConfig)
AutoModel.register(CombaConfig, CombaModel)
AutoModelForCausalLM.register(CombaConfig, CombaForCausalLM)

__all__ = ['CombaConfig', 'CombaForCausalLM', 'CombaModel']
