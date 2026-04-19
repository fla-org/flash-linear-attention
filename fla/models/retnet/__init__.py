# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.retnet.configuration_retnet import RetNetConfig
from fla.models.retnet.modeling_retnet import RetNetForCausalLM, RetNetModel

AutoConfig.register(RetNetConfig.model_type, RetNetConfig, exist_ok=True)
AutoModel.register(RetNetConfig, RetNetModel, exist_ok=True)
AutoModelForCausalLM.register(RetNetConfig, RetNetForCausalLM, exist_ok=True)


__all__ = ['RetNetConfig', 'RetNetForCausalLM', 'RetNetModel']
