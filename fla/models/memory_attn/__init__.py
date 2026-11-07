# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.memory_attn.configuration_memory_attn import MemoryAttnConfig
from fla.models.memory_attn.modeling_memory_attn import MemoryAttnForCausalLM, MemoryAttnModel

AutoConfig.register(MemoryAttnConfig.model_type, MemoryAttnConfig, exist_ok=True)
AutoModel.register(MemoryAttnConfig, MemoryAttnModel, exist_ok=True)
AutoModelForCausalLM.register(MemoryAttnConfig, MemoryAttnForCausalLM, exist_ok=True)


__all__ = ['MemoryAttnConfig', 'MemoryAttnForCausalLM', 'MemoryAttnModel']
