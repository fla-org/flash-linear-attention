# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.rwkv6.configuration_rwkv6 import RWKV6Config
from fla.models.rwkv6.modeling_rwkv6 import RWKV6ForCausalLM, RWKV6Model

AutoConfig.register(RWKV6Config.model_type, RWKV6Config, exist_ok=True)
AutoModel.register(RWKV6Config, RWKV6Model, exist_ok=True)
AutoModelForCausalLM.register(RWKV6Config, RWKV6ForCausalLM, exist_ok=True)


__all__ = ['RWKV6Config', 'RWKV6ForCausalLM', 'RWKV6Model']
