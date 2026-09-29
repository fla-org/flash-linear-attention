# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.diag_kdn.configuration_diag_kdn import DiagKDNConfig
from fla.models.diag_kdn.modeling_diag_kdn import DiagKDNForCausalLM, DiagKDNModel

AutoConfig.register(DiagKDNConfig.model_type, DiagKDNConfig, exist_ok=True)
AutoModel.register(DiagKDNConfig, DiagKDNModel, exist_ok=True)
AutoModelForCausalLM.register(DiagKDNConfig, DiagKDNForCausalLM, exist_ok=True)

__all__ = ['DiagKDNConfig', 'DiagKDNForCausalLM', 'DiagKDNModel']
