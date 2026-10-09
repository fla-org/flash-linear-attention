# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

from transformers.configuration_utils import PretrainedConfig


class MLSTMConfig(PretrainedConfig):

    model_type = 'mlstm'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_heads: int = 16,
        proj_factor: float = 2.0,
        qkv_proj_blocksize: int = 4,
        conv_size: int = 4,
        conv_bias: bool = True,
        attn_bias: bool = False,
        attn_dropout: float = 0.0,
        round_proj_up_to_multiple_of: int = 64,
        round_proj_up_dim_up: bool = True,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = 'swish',
        max_position_embeddings: int = 2048,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-5,
        attn: dict | None = None,
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
        legacy_attn_mode = kwargs.pop('attn_mode', 'chunk')
        legacy_kernel_mode = kwargs.pop('kernel_mode', 'chunk')
        legacy_chunk_size = kwargs.pop('chunk_size', None)
        kwargs.pop('distillation_kernel_name', None)
        kwargs.pop('distillation_kernel_kwargs', None)
        kwargs.pop('distillation_backend_path', None)
        kwargs.pop('distillation_repo_path', None)
        kwargs.pop('distillation_pad_to_chunk_size', None)

        if legacy_attn_mode != 'chunk':
            raise ValueError(
                "Only `attn_mode='chunk'` is supported because `fla.ops.mlstm` only provides `chunk_mlstm`.",
            )
        if legacy_kernel_mode != 'chunk':
            raise ValueError(
                "Only `kernel_mode='chunk'` is supported because `fla.ops.mlstm` only provides `chunk_mlstm`.",
            )
        if legacy_chunk_size not in (None, 64):
            warnings.warn(
                '`chunk_size` is ignored for `MLSTMConfig` because the local `chunk_mlstm` op does not '
                'expose a configurable chunk size.',
            )

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.proj_factor = proj_factor
        self.qkv_proj_blocksize = qkv_proj_blocksize
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.attn_bias = attn_bias
        self.attn_dropout = attn_dropout
        self.round_proj_up_to_multiple_of = round_proj_up_to_multiple_of
        self.round_proj_up_dim_up = round_proj_up_dim_up
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.elementwise_affine = elementwise_affine
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

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                '`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.',
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                '`fuse_linear_cross_entropy` is enabled, which can improves memory efficiency '
                'at the potential cost of reduced precision. '
                'If you observe issues like loss divergence, consider disabling this setting.',
            )

        if attn is not None:
            if not isinstance(attn, dict):
                raise ValueError('attn must be a dictionary')
            if 'layers' not in attn:
                raise ValueError('Layer indices must be provided to initialize hybrid attention layers')
            if 'num_heads' not in attn:
                raise ValueError('Number of heads must be provided to initialize hybrid attention layers')
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
