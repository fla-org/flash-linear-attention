# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.delta_net.configuration_delta_net import DeltaNetConfig
from fla.models.delta_net.modeling_delta_net import DeltaNetForCausalLM, DeltaNetModel

AutoConfig.register(DeltaNetConfig.model_type, DeltaNetConfig, exist_ok=True)
AutoModel.register(DeltaNetConfig, DeltaNetModel, exist_ok=True)
AutoModelForCausalLM.register(DeltaNetConfig, DeltaNetForCausalLM, exist_ok=True)

__all__ = ['DeltaNetConfig', 'DeltaNetForCausalLM', 'DeltaNetModel']
