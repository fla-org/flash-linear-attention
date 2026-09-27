# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.iso_kdn.configuration_iso_kdn import IsoKDNConfig
from fla.models.iso_kdn.modeling_iso_kdn import IsoKDNForCausalLM, IsoKDNModel

AutoConfig.register(IsoKDNConfig.model_type, IsoKDNConfig, exist_ok=True)
AutoModel.register(IsoKDNConfig, IsoKDNModel, exist_ok=True)
AutoModelForCausalLM.register(IsoKDNConfig, IsoKDNForCausalLM, exist_ok=True)

__all__ = ['IsoKDNConfig', 'IsoKDNForCausalLM', 'IsoKDNModel']
