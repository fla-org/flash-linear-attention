# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM, RWKV7Model

AutoConfig.register(RWKV7Config.model_type, RWKV7Config, exist_ok=True)
AutoModel.register(RWKV7Config, RWKV7Model, exist_ok=True)
AutoModelForCausalLM.register(RWKV7Config, RWKV7ForCausalLM, exist_ok=True)


__all__ = ['RWKV7Config', 'RWKV7ForCausalLM', 'RWKV7Model']
