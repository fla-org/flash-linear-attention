# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.delta_net.configuration_delta_net import DeltaNetConfig
from fla.models.delta_net.modeling_delta_net import DeltaNetForCausalLM, DeltaNetModel

AutoConfig.register(DeltaNetConfig.model_type, DeltaNetConfig, exist_ok=True)
AutoModel.register(DeltaNetConfig, DeltaNetModel, exist_ok=True)
AutoModelForCausalLM.register(DeltaNetConfig, DeltaNetForCausalLM, exist_ok=True)

__all__ = ['DeltaNetConfig', 'DeltaNetForCausalLM', 'DeltaNetModel']
