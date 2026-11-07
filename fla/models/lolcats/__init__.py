# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.lolcats.configuration_lolcats import LoLCATsConfig
from fla.models.lolcats.modeling_lolcats import LoLCATsForCausalLM, LoLCATsModel

AutoConfig.register(LoLCATsConfig.model_type, LoLCATsConfig, exist_ok=True)
AutoModel.register(LoLCATsConfig, LoLCATsModel, exist_ok=True)
AutoModelForCausalLM.register(LoLCATsConfig, LoLCATsForCausalLM, exist_ok=True)


__all__ = ['LoLCATsConfig', 'LoLCATsForCausalLM', 'LoLCATsModel']
