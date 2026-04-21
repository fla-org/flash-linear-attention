# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.transformer.configuration_transformer import TransformerConfig
from fla.models.transformer.modeling_transformer import TransformerForCausalLM, TransformerModel

AutoConfig.register(TransformerConfig.model_type, TransformerConfig, exist_ok=True)
AutoModel.register(TransformerConfig, TransformerModel, exist_ok=True)
AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)


__all__ = ['TransformerConfig', 'TransformerForCausalLM', 'TransformerModel']
