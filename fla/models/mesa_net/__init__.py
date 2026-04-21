# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mesa_net.configuration_mesa_net import MesaNetConfig
from fla.models.mesa_net.modeling_mesa_net import MesaNetForCausalLM, MesaNetModel

AutoConfig.register(MesaNetConfig.model_type, MesaNetConfig, exist_ok=True)
AutoModel.register(MesaNetConfig, MesaNetModel, exist_ok=True)
AutoModelForCausalLM.register(MesaNetConfig, MesaNetForCausalLM, exist_ok=True)

__all__ = ['MesaNetConfig', 'MesaNetForCausalLM', 'MesaNetModel']
