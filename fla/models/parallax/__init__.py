# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Parallax (parameterized local linear attention), contributed by
# Yifei Zuo et al. (https://arxiv.org/abs/2605.29157).

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.parallax.configuration_parallax import ParallaxConfig
from fla.models.parallax.modeling_parallax import (
    ParallaxForCausalLM,
    ParallaxModel,
)

AutoConfig.register(ParallaxConfig.model_type, ParallaxConfig, exist_ok=True)
AutoModel.register(ParallaxConfig, ParallaxModel, exist_ok=True)
AutoModelForCausalLM.register(ParallaxConfig, ParallaxForCausalLM, exist_ok=True)


__all__ = ['ParallaxConfig', 'ParallaxForCausalLM', 'ParallaxModel']
