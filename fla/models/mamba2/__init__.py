# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mamba2.configuration_mamba2 import Mamba2Config
from fla.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM, Mamba2Model

AutoConfig.register(Mamba2Config.model_type, Mamba2Config, exist_ok=True)
AutoModel.register(Mamba2Config, Mamba2Model, exist_ok=True)
AutoModelForCausalLM.register(Mamba2Config, Mamba2ForCausalLM, exist_ok=True)


__all__ = ['Mamba2Config', 'Mamba2ForCausalLM', 'Mamba2Model']
