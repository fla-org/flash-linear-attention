# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.comba.configuration_comba import CombaConfig
from fla.models.comba.modeling_comba import CombaForCausalLM, CombaModel

AutoConfig.register(CombaConfig.model_type, CombaConfig, exist_ok=True)
AutoModel.register(CombaConfig, CombaModel, exist_ok=True)
AutoModelForCausalLM.register(CombaConfig, CombaForCausalLM, exist_ok=True)

__all__ = ['CombaConfig', 'CombaForCausalLM', 'CombaModel']
