# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


from transformers.configuration_utils import PretrainedConfig


class YOCOConfig(PretrainedConfig):
    model_type = 'yoco'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        num_self_decoder_layers: int = 10,
        self_decoder_attn: dict | None = None,
        cross_decoder_attn: dict | None = None,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        num_hidden_layers: int = 21,
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
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.num_self_decoder_layers = num_self_decoder_layers
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        if self_decoder_attn is not None and not isinstance(self_decoder_attn, dict):
            raise ValueError("self_decoder_attn must be a dictionary")
        if cross_decoder_attn is not None and not isinstance(cross_decoder_attn, dict):
            raise ValueError("cross_decoder_attn must be a dictionary")

        self_decoder_attn = {} if self_decoder_attn is None else dict(self_decoder_attn)
        self_decoder_attn_type = self_decoder_attn.get('type', 'gated_deltanet')
        self_decoder_attn_num_heads = self_decoder_attn.get('num_heads', 6)
        self_decoder_attn_num_v_heads = self_decoder_attn.get('num_v_heads', self_decoder_attn_num_heads)
        self.self_decoder_attn = {
            'type': self_decoder_attn_type,
            'mode': self_decoder_attn.get('mode', 'chunk'),
            'use_gate': self_decoder_attn.get('use_gate', True),
            'expand_v': self_decoder_attn.get('expand_v', 2.0),
            'use_short_conv': self_decoder_attn.get('use_short_conv', True),
            'allow_neg_eigval': self_decoder_attn.get('allow_neg_eigval', False),
            'conv_size': self_decoder_attn.get('conv_size', 4),
            'head_dim': self_decoder_attn.get('head_dim', 256),
            'num_heads': self_decoder_attn_num_heads,
            'num_v_heads': self_decoder_attn_num_v_heads,
            'window_size': self_decoder_attn.get('window_size', None),
            'qkv_bias': self_decoder_attn.get('qkv_bias', False),
            'rope_theta': self_decoder_attn.get('rope_theta', 10000.0),
            'rope_inv_freq': self_decoder_attn.get('rope_inv_freq', 'fla'),
            'gate_logit_normalizer': self_decoder_attn.get('gate_logit_normalizer', 16.0),
        }

        cross_decoder_attn = {} if cross_decoder_attn is None else dict(cross_decoder_attn)
        cross_decoder_attn_num_heads = cross_decoder_attn.get('num_heads', 32)
        cross_decoder_attn_num_kv_heads = cross_decoder_attn.get('num_kv_heads', cross_decoder_attn_num_heads)
        self.cross_decoder_attn = {
            'num_heads': cross_decoder_attn_num_heads,
            'num_kv_heads': cross_decoder_attn_num_kv_heads,
            'qkv_bias': cross_decoder_attn.get('qkv_bias', False),
            'qk_norm': cross_decoder_attn.get('qk_norm', False),
            'window_size': cross_decoder_attn.get('window_size', None),
            'rope_theta': cross_decoder_attn.get('rope_theta', 10000.0),
            'rope_inv_freq': cross_decoder_attn.get('rope_inv_freq', 'fla'),
        }
        self.cross_decoder_attn.update({
            key: value for key, value in cross_decoder_attn.items() if key not in self.cross_decoder_attn
        })

        if self.self_decoder_attn['type'] not in {'gated_deltanet', 'gated_retention', 'swa'}:
            raise ValueError(
                "self_decoder_attn['type'] must be one of {'gated_deltanet', 'gated_retention', 'swa'}"
            )
        if self.self_decoder_attn['rope_inv_freq'] not in {'fla', 'yoco'}:
            raise ValueError("self_decoder_attn['rope_inv_freq'] must be one of {'fla', 'yoco'}")
        if self.cross_decoder_attn['rope_inv_freq'] not in {'fla', 'yoco'}:
            raise ValueError("cross_decoder_attn['rope_inv_freq'] must be one of {'fla', 'yoco'}")
        if self.num_self_decoder_layers < 0 or self.num_self_decoder_layers > self.num_hidden_layers:
            raise ValueError("num_self_decoder_layers must be between 0 and num_hidden_layers")
        if self.self_decoder_attn['window_size'] is not None and self.self_decoder_attn['window_size'] <= 0:
            raise ValueError("self_decoder_attn['window_size'] must be a positive integer")
        if self.cross_decoder_attn['window_size'] is not None and self.cross_decoder_attn['window_size'] <= 0:
            raise ValueError("cross_decoder_attn['window_size'] must be a positive integer")
        if self.self_decoder_attn['type'] == 'swa' and self.self_decoder_attn['window_size'] is None:
            raise ValueError("self_decoder_attn['window_size'] must be set when self_decoder_attn['type']='swa'")
        if self.self_decoder_attn['type'] == 'gated_retention' and self.self_decoder_attn['mode'] == 'fused_chunk':
            raise ValueError(
                "self_decoder_attn['mode']='fused_chunk' is not supported when self_decoder_attn['type']='gated_retention'"
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
