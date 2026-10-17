# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

from transformers.configuration_utils import PretrainedConfig

from fla.models.hybrid import HybridAttentionConfig, _HybridAttentionConfigMixin


class GSA2Config(_HybridAttentionConfigMixin, PretrainedConfig):
    model_type = 'gsa2'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        num_slots: int = 128,
        level2_axis: str = "head",
        w_rank: int | None = None,
        use_slot_proj: bool = True,
        w_scale: float = 1.0,
        w_clip: float | None = None,
        use_w_l2norm: bool = True,
        w_l2norm_eps: float = 1e-6,
        use_short_conv: bool = True,
        use_w_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        oja_out_act: str = "silu",
        oja_scale: float | None = None,
        gdn_scale: float | None = None,
        use_oja_q_l2norm: bool = True,
        use_oja_k_l2norm: bool = True,
        use_gdn_qk_l2norm_in_kernel: bool = True,
        oja_lower_bound: float = 5.0,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        num_hidden_layers: int = 20,
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
        self.num_slots = num_slots
        self.level2_axis = level2_axis
        self.w_rank = w_rank
        self.use_slot_proj = use_slot_proj
        self.w_scale = w_scale
        self.w_clip = w_clip
        self.use_w_l2norm = use_w_l2norm
        self.w_l2norm_eps = w_l2norm_eps
        self.use_short_conv = use_short_conv
        self.use_w_conv = use_w_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.oja_out_act = oja_out_act
        self.oja_scale = oja_scale
        self.gdn_scale = gdn_scale
        self.use_oja_q_l2norm = use_oja_q_l2norm
        self.use_oja_k_l2norm = use_oja_k_l2norm
        self.use_gdn_qk_l2norm_in_kernel = use_gdn_qk_l2norm_in_kernel
        self.oja_lower_bound = oja_lower_bound
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
                "`fuse_linear_cross_entropy` is enabled, which can improves memory efficiency "
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
