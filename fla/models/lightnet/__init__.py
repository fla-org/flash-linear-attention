# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.lightnet.configuration_lightnet import LightNetConfig
from fla.models.lightnet.modeling_lightnet import LightNetForCausalLM, LightNetModel

AutoConfig.register(LightNetConfig.model_type, LightNetConfig, exist_ok=True)
AutoModel.register(LightNetConfig, LightNetModel, exist_ok=True)
AutoModelForCausalLM.register(LightNetConfig, LightNetForCausalLM, exist_ok=True)


__all__ = ['LightNetConfig', 'LightNetForCausalLM', 'LightNetModel']
