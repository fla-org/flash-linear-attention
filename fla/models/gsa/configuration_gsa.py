# -*- coding: utf-8 -*-

from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig


class GSAConfig(PretrainedConfig):

    model_type = 'gsa'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        gate_logit_normalizer: Optional[int] = 8,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 24,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        num_slots: Optional[int] = 64,
        use_short_conv: bool = False,
        conv_size: int = 4,
        exapnd_k: float = 1,
        exapnd_v: float = 1,
        feature_map: str = 'swish',
        use_output_gate: bool = False,
        use_norm: bool = True,
        max_position_embeddings: int = 2048,
        hidden_act: str = "swish",
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-6,
        attn: Optional[Dict] = None,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.gate_logit_normalizer = gate_logit_normalizer
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_slots = num_slots
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.expand_k = exapnd_k
        self.expand_v = exapnd_v
        self.feature_map = feature_map
        self.use_output_gate = use_output_gate
        self.use_norm = use_norm
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        if attn is not None:
            if not isinstance(attn, Dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            attn['qkv_bias'] = attn.get('qkv_bias', False)
            attn['window_size'] = attn.get('window_size', None)
            attn['rope_theta'] = attn.get('rope_theta', 10000.)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
