# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetForCausalLM, GatedDeltaNetModel

AutoConfig.register(GatedDeltaNetConfig.model_type, GatedDeltaNetConfig)
AutoModel.register(GatedDeltaNetConfig, GatedDeltaNetModel)
AutoModelForCausalLM.register(GatedDeltaNetConfig, GatedDeltaNetForCausalLM)

__all__ = ['GatedDeltaNetConfig', 'GatedDeltaNetForCausalLM', 'GatedDeltaNetModel']
