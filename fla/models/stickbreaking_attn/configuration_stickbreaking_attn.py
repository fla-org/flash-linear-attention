# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers.configuration_utils import PretrainedConfig


class StickBreakingAttentionConfig(PretrainedConfig):

    model_type = 'stickbreaking_attn'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attend_current: bool = False,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = 'swish',
        initializer_range: float = 0.02,
        norm_eps: float = 1e-6,
        use_cache: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        **kwargs,
    ):
        if use_cache:
            raise ValueError("Stick-breaking attention does not support KV-cache decoding. Use use_cache=False.")
        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError("fuse_cross_entropy and fuse_linear_cross_entropy cannot both be True.")

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.attend_current = attend_current
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
