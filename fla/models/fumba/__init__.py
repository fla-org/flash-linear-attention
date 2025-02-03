# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.fumba.configuration_fumba import FumbaConfig
from fla.models.fumba.modeling_fumba import (FumbaBlock, FumbaForCausalLM,
                                             FumbaModel)

AutoConfig.register(FumbaConfig.model_type, FumbaConfig, True)
AutoModel.register(FumbaConfig, FumbaModel, True)
AutoModelForCausalLM.register(FumbaConfig, FumbaForCausalLM, True)


__all__ = ['FumbaConfig', 'FumbaForCausalLM', 'FumbaModel', 'FumbaBlock']
