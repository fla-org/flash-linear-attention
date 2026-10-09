# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

from transformers.configuration_utils import PretrainedConfig

from fla.models.hybrid import HybridAttentionConfig, _HybridAttentionConfigMixin


class GKAConfig(_HybridAttentionConfigMixin, PretrainedConfig):
    model_type = 'gka'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        expand_v: float = 1.,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        num_kv_heads: int | None = None,
        use_beta_gate: bool = True,
        use_alpha_connection: bool = True,
        use_v_conv: bool = True,
        use_forgetting_gate: bool = True,
        use_forgetting_gate_kk: bool = True,
        gla_rescale: bool = True,
        ridge_ratio: float = 0.02,
        num_iter: int = 30,
        use_gate: bool = True,
        conv_size: int = 4,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        num_hidden_layers: int = 24,
        norm_eps: float = 1e-6,
        attn: HybridAttentionConfig = None,
        use_cache: bool = True,
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
        attnres_block_size: int | None = None,
        **kwargs,
    ):
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads
        self.num_kv_heads = num_kv_heads
        self.use_beta_gate = use_beta_gate
        self.use_alpha_connection = use_alpha_connection
        self.use_v_conv = use_v_conv
        self.use_forgetting_gate = use_forgetting_gate
        self.use_forgetting_gate_kk = use_forgetting_gate_kk
        self.gla_rescale = gla_rescale
        self.ridge_ratio = ridge_ratio
        self.num_iter = num_iter
        self.use_gate = use_gate
        self.conv_size = conv_size
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size
        self.attnres_block_size = attnres_block_size

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.",
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improve memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )

        if attnres_block_size is not None and attnres_block_size != 1:
            if attnres_block_size < 2 or attnres_block_size % 2 != 0:
                raise ValueError(
                    "`attnres_block_size` must be `None`, `1` (full mode), or an even integer (one block "
                    f"contains `attnres_block_size // 2` transformer layers); got {attnres_block_size}."
                )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
