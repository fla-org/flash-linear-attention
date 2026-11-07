# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers.configuration_utils import PreTrainedConfig

from fla.models.configuration_utils import validate_loss_config


class LoLCATsConfig(PreTrainedConfig):
    """
    Configuration of LoLCATs (https://arxiv.org/abs/2410.10254), a Llama-style model whose attention is linearized into
    sliding-window softmax attention and linear attention with Hedgehog feature maps.

    Llama checkpoints can be converted with `utils/convert_from_llama.py`; the feature maps and window factors are then
    trained with `LoLCATsAttention.attention_transfer_loss` before the rest of the model is fine-tuned.
    """

    model_type = 'lolcats'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        feature_dim: int = 64,
        window_size: int = 64,
        affine_attention_factors: bool = False,
        init_window_factor: float = 0.,
        rope_theta: float = 10000.,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        norm_eps: float = 1e-6,
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
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.affine_attention_factors = affine_attention_factors
        self.init_window_factor = init_window_factor
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        validate_loss_config(fuse_cross_entropy=fuse_cross_entropy, fuse_linear_cross_entropy=fuse_linear_cross_entropy)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
