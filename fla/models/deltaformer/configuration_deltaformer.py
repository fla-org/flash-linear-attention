# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from transformers.configuration_utils import PreTrainedConfig

from fla.models.configuration_utils import (
    AttentionConfig,
    AttentionConfigBase,
    validate_attnres_config,
    validate_loss_config,
)


class DeltaFormerConfig(AttentionConfigBase, PreTrainedConfig):
    model_type = 'deltaformer'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 24,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        attn_mode: str = "chunk",
        hidden_act: str = "swish",
        max_position_embeddings: int = 2048,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        rope_theta: float = 10000.,
        rope_max_position_embeddings: int | None = None,
        attn: AttentionConfig = None,
        use_cache: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        attnres_block_size: int | None = None,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.attn_mode = attn_mode
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.rope_theta = rope_theta
        self.rope_max_position_embeddings = rope_max_position_embeddings
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.attnres_block_size = attnres_block_size

        validate_loss_config(fuse_cross_entropy=fuse_cross_entropy, fuse_linear_cross_entropy=fuse_linear_cross_entropy)

        validate_attnres_config(attnres_block_size=attnres_block_size)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
