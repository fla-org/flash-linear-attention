# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mom_gated_deltanet.configuration_mom_gated_deltanet import \
    MomGatedDeltaNetConfig
from fla.models.mom_gated_deltanet.modeling_mom_gated_deltanet import (
    MomGatedDeltaNetForCausalLM, MomGatedDeltaNetModel)

AutoConfig.register(MomGatedDeltaNetConfig.model_type, MomGatedDeltaNetConfig)
AutoModel.register(MomGatedDeltaNetConfig, MomGatedDeltaNetModel)
AutoModelForCausalLM.register(MomGatedDeltaNetConfig, MomGatedDeltaNetForCausalLM)

__all__ = ['MomGatedDeltaNetConfig', 'MomGatedDeltaNetForCausalLM', 'MomGatedDeltaNetModel']
