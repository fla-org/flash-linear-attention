# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.rodimus.configuration_rodimus import RodimusConfig
from fla.models.rodimus.modeling_rodimus import RodimusForCausalLM, RodimusModel

AutoConfig.register(RodimusConfig.model_type, RodimusConfig, exist_ok=True)
AutoModel.register(RodimusConfig, RodimusModel, exist_ok=True)
AutoModelForCausalLM.register(RodimusConfig, RodimusForCausalLM, exist_ok=True)


__all__ = ['RodimusConfig', 'RodimusForCausalLM', 'RodimusModel']
