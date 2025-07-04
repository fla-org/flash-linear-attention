# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.gdn2.configuration_gdn2 import GDN2Config
from fla.models.gdn2.modeling_gdn2 import GDN2ForCausalLM, GDN2Model

AutoConfig.register(GDN2Config.model_type, GDN2Config, exist_ok=True)
AutoModel.register(GDN2Config, GDN2Model, exist_ok=True)
AutoModelForCausalLM.register(GDN2Config, GDN2ForCausalLM, exist_ok=True)

__all__ = ['GDN2Config', 'GDN2ForCausalLM', 'GDN2Model']
