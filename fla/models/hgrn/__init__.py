# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.hgrn.configuration_hgrn import HGRNConfig
from fla.models.hgrn.modeling_hgrn import HGRNForCausalLM, HGRNModel

AutoConfig.register(HGRNConfig.model_type, HGRNConfig, exist_ok=True)
AutoModel.register(HGRNConfig, HGRNModel, exist_ok=True)
AutoModelForCausalLM.register(HGRNConfig, HGRNForCausalLM, exist_ok=True)


__all__ = ['HGRNConfig', 'HGRNForCausalLM', 'HGRNModel']
