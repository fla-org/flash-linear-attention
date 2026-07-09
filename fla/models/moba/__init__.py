# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.moba.configuration_moba import MoBAConfig
from fla.models.moba.modeling_moba import MoBAForCausalLM, MoBAModel

AutoConfig.register(MoBAConfig.model_type, MoBAConfig, exist_ok=True)
AutoModel.register(MoBAConfig, MoBAModel, exist_ok=True)
AutoModelForCausalLM.register(MoBAConfig, MoBAForCausalLM, exist_ok=True)


__all__ = ['MoBAConfig', 'MoBAForCausalLM', 'MoBAModel']
