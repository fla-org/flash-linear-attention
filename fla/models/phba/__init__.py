# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.phba.configuration_phba import PHBAConfig
from fla.models.phba.modeling_phba import PHBAForCausalLM, PHBAModel

AutoConfig.register(PHBAConfig.model_type, PHBAConfig, exist_ok=True)
AutoModel.register(PHBAConfig, PHBAModel, exist_ok=True)
AutoModelForCausalLM.register(PHBAConfig, PHBAForCausalLM, exist_ok=True)

__all__ = ['PHBAConfig', 'PHBAForCausalLM', 'PHBAModel']
