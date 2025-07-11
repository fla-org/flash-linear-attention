# -*- coding: utf-8 -*-

import math
from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig


class SambaConfig(PretrainedConfig):

    model_type = "samba"

    def __init__(
        self,
        hidden_size: int = 2304,
        state_size: int = 16,
        num_hidden_layers: int = 18,
        norm_eps=1e-5,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        expand: int = 2,
        conv_kernel: int = 4,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        residual_in_fp32: bool = False,
        time_step_rank: str = "auto",
        time_step_scale: float = 1.0,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_init_scheme: str = "random",
        time_step_floor: float = 1e-4,
        max_position_embeddings: int = 2048,
        attn: Optional[Dict] = {
            'layers': (1, 3, 5, 7, 9, 11, 13, 15, 17),
            'num_heads': 18,
            'num_kv_heads': 18,
            'qkv_bias': False,
            'window_size': 2048,
            'rope_theta': 10000.
        },
        hidden_ratio: Optional[int] = 4,
        rescale_prenorm_residual: bool = False,
        use_cache: bool = True,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.intermediate_size = int(expand * self.hidden_size)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.time_step_rank = math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_scale = time_step_scale
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_init_scheme = time_step_init_scheme
        self.time_step_floor = time_step_floor
        self.max_position_embeddings = max_position_embeddings
        self.attn = attn
        self.hidden_ratio = hidden_ratio
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
