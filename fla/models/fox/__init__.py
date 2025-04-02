# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.fox.configuration_fox import FoXConfig
from fla.models.fox.modeling_fox import FoXForCausalLM, FoXModel

AutoConfig.register(FoXConfig.model_type, FoXConfig)
AutoModel.register(FoXConfig, FoXModel)
AutoModelForCausalLM.register(FoXConfig, FoXForCausalLM)


__all__ = ['FoXConfig', 'FoXForCausalLM', 'FoXModel']
