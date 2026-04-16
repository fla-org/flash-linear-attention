# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.rodimus.configuration_rodimus import RodimusConfig
from fla.models.rodimus.modeling_rodimus import RodimusForCausalLM, RodimusModel

AutoConfig.register(RodimusConfig.model_type, RodimusConfig, exist_ok=True)
AutoModel.register(RodimusConfig, RodimusModel, exist_ok=True)
AutoModelForCausalLM.register(RodimusConfig, RodimusForCausalLM, exist_ok=True)


__all__ = ['RodimusConfig', 'RodimusForCausalLM', 'RodimusModel']
