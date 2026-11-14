# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.griffin.configuration_griffin import GriffinConfig
from fla.models.griffin.modeling_griffin import GriffinForCausalLM, GriffinModel

AutoConfig.register(GriffinConfig.model_type, GriffinConfig, exist_ok=True)
AutoModel.register(GriffinConfig, GriffinModel, exist_ok=True)
AutoModelForCausalLM.register(GriffinConfig, GriffinForCausalLM, exist_ok=True)


__all__ = ['GriffinConfig', 'GriffinForCausalLM', 'GriffinModel']
