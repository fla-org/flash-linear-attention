# -*- coding: utf-8 -*-

from typing import Dict, Optional, Union

from transformers.configuration_utils import PretrainedConfig


class RodimusConfig(PretrainedConfig):

    model_type = 'rodimus'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        block_type: str = 'rodimus_plus',
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        attn_mode: str = "chunk",
        residual_in_fp32: bool = True,
        block_residual_in_fp32: bool = False,
        expand_ratio: Optional[int] = 64,
        input_gate_low_rank: Optional[Union[float, str]] = 'auto',
        use_short_conv: bool = True,
        conv_size: int = 4,
        hidden_ratio: Optional[float] = 4/3,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        max_position_embeddings: int = 2048,
        norm_eps: float = 1e-5,
        k_norm_eps: Optional[float] = None,
        attn: Optional[Dict] = None,
        ska_attn: Optional[Dict] = None,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 126080,
        eos_token_id: int = 126081,
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        use_l2warp: bool = False,
        vocab_size: int = 126464,
        **kwargs
    ):
        self.block_type = block_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attn_mode = attn_mode
        self.residual_in_fp32 = residual_in_fp32
        self.block_residual_in_fp32 = block_residual_in_fp32
        self.expand_ratio = expand_ratio
        self.input_gate_low_rank = input_gate_low_rank

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.k_norm_eps = k_norm_eps

        self.attn = attn
        self.ska_attn = ska_attn
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
            attn['qk_norm'] = attn.get('qk_norm', False)
            attn['window_size'] = attn.get('window_size', None)
            attn['rope_theta'] = attn.get('rope_theta', 10000.)

        if ska_attn is not None:
            if not isinstance(ska_attn, Dict):
                raise ValueError("attn must be a dictionary")
            if 'num_heads' not in ska_attn:
                raise ValueError("Number of heads must be provided to initialize shared-key attention layers")
            ska_attn['qkv_bias'] = ska_attn.get('qkv_bias', False)
            ska_attn['qk_norm'] = ska_attn.get('qk_norm', False)
            ska_attn['window_size'] = ska_attn.get('window_size', 1024)
            ska_attn['rope_theta'] = ska_attn.get('rope_theta', 10000.)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
