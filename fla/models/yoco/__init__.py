# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.yoco.configuration_yoco import YOCOConfig
from fla.models.yoco.modeling_yoco import YOCOForCausalLM, YOCOModel

AutoConfig.register(YOCOConfig.model_type, YOCOConfig, exist_ok=True)
AutoModel.register(YOCOConfig, YOCOModel, exist_ok=True)
AutoModelForCausalLM.register(YOCOConfig, YOCOForCausalLM, exist_ok=True)

__all__ = ['YOCOConfig', 'YOCOForCausalLM', 'YOCOModel']
