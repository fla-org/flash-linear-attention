# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.abc.configuration_abc import ABCConfig
from fla.models.abc.modeling_abc import ABCForCausalLM, ABCModel

AutoConfig.register(ABCConfig.model_type, ABCConfig, exist_ok=True)
AutoModel.register(ABCConfig, ABCModel, exist_ok=True)
AutoModelForCausalLM.register(ABCConfig, ABCForCausalLM, exist_ok=True)


__all__ = ['ABCConfig', 'ABCForCausalLM', 'ABCModel']
