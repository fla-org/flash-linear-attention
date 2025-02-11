# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM, RWKV7Model

AutoConfig.register(RWKV7Config.model_type, RWKV7Config, True)
AutoModel.register(RWKV7Config, RWKV7Model, True)
AutoModelForCausalLM.register(RWKV7Config, RWKV7ForCausalLM, True)


__all__ = ['RWKV7Config', 'RWKV7ForCausalLM', 'RWKV7Model']
