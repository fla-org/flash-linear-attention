# -*- coding: utf-8 -*-

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class PaTHAttentionConfig(PretrainedConfig):

    model_type = 'path_attn'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        use_forget_gate: bool = False,
        use_w_shortconv: bool = True,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        self.use_forget_gate = use_forget_gate
        self.use_w_shortconv = use_w_shortconv

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
