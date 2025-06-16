# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.comba.configuration_comba import CombaConfig
from fla.models.comba.modeling_comba import CombaForCausalLM, CombaModel

AutoConfig.register(CombaConfig.model_type, CombaConfig, exist_ok=True)
AutoModel.register(CombaConfig, CombaModel, exist_ok=True)
AutoModelForCausalLM.register(CombaConfig, CombaForCausalLM, exist_ok=True)

__all__ = ['CombaConfig', 'CombaForCausalLM', 'CombaModel']
