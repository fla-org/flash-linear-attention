# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers.configuration_utils import PretrainedConfig

from fla.layers.attn import Attention
from fla.models.configuration_utils import get_attention_spec


def get_attention(config: PretrainedConfig, layer_idx: int) -> Attention | None:
    """Create standard attention for a selected layer, or return None for a native mixer."""
    attn_spec = get_attention_spec(config.attn, layer_idx=layer_idx)
    if attn_spec is None:
        return None
    return Attention(
        hidden_size=config.hidden_size,
        num_heads=attn_spec['num_heads'],
        num_kv_heads=attn_spec['num_kv_heads'],
        qkv_bias=attn_spec['qkv_bias'],
        window_size=attn_spec['window_size'],
        rope_theta=attn_spec['rope_theta'],
        max_position_embeddings=config.max_position_embeddings,
        layer_idx=layer_idx,
    )
