# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.gka.configuration_gka import GKAConfig
from fla.models.gka.modeling_gka import GKAForCausalLM, GKAModel

AutoConfig.register(GKAConfig.model_type, GKAConfig, exist_ok=True)
AutoModel.register(GKAConfig, GKAModel, exist_ok=True)
AutoModelForCausalLM.register(GKAConfig, GKAForCausalLM, exist_ok=True)

__all__ = ['GKAConfig', 'GKAForCausalLM', 'GKAModel']
