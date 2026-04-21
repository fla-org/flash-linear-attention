# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM, RWKV7Model

AutoConfig.register(RWKV7Config.model_type, RWKV7Config, exist_ok=True)
AutoModel.register(RWKV7Config, RWKV7Model, exist_ok=True)
AutoModelForCausalLM.register(RWKV7Config, RWKV7ForCausalLM, exist_ok=True)


__all__ = ['RWKV7Config', 'RWKV7ForCausalLM', 'RWKV7Model']
