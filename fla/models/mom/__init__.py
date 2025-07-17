# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mom.configuration_mom import MomConfig
from fla.models.mom.modeling_mom import MomForCausalLM, MomModel

AutoConfig.register(MomConfig.model_type, MomConfig)
AutoModel.register(MomConfig, MomModel)
AutoModelForCausalLM.register(MomConfig, MomForCausalLM)

__all__ = ['MomConfig', 'MomForCausalLM', 'MomModel']
