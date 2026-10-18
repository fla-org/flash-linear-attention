# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers.configuration_utils import PreTrainedConfig

from fla.models.configuration_utils import (
    AttentionConfig,
    AttentionConfigBase,
    validate_attnres_config,
    validate_loss_config,
)


class GLAConfig(AttentionConfigBase, PreTrainedConfig):

    model_type = 'gla'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_k: float = 0.5,
        expand_v: float = 1.,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 24,
        num_heads: int = 4,
        num_kv_heads: int | None = None,
        feature_map: str | None = None,
        attn_mode: str = "chunk",
        use_short_conv: bool = False,
        conv_size: int = 4,
        use_output_gate: bool = True,
        clamp_min: float | None = None,
        hidden_act: str = "swish",
        max_position_embeddings: int = 2048,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        use_gk: bool = True,
        use_gv: bool = False,
        attn: AttentionConfig = None,
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
        residual_mode: str | None = None,
        residual_kwargs: dict | None = None,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.feature_map = feature_map
        self.attn_mode = attn_mode
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_output_gate = use_output_gate
        self.clamp_min = clamp_min
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_gk = use_gk
        self.use_gv = use_gv
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        if residual_kwargs is not None and not isinstance(residual_kwargs, dict):
            raise TypeError('residual_kwargs must be a dict or None')
        self.residual_kwargs = dict(residual_kwargs or {})
        reserved = {'hidden_size', 'sub_layer_idx', 'num_sublayers'} & self.residual_kwargs.keys()
        if reserved:
            raise ValueError(f'{sorted(reserved)} are supplied by the model, not residual_kwargs')

        validate_attnres_config(attnres_block_size=attnres_block_size)
        if residual_mode is None:
            residual_mode = 'standard' if attnres_block_size is None else 'attnres'
        if attnres_block_size is not None:
            if residual_mode != 'attnres' or self.residual_kwargs.get('block_size', attnres_block_size) != attnres_block_size:
                raise ValueError('attnres_block_size conflicts with residual_mode or residual_kwargs')
            self.residual_kwargs['block_size'] = attnres_block_size
        self.residual_mode = residual_mode
        if num_hidden_layers < 0:
            raise ValueError('num_hidden_layers must be non-negative')

        validate_loss_config(fuse_cross_entropy=fuse_cross_entropy, fuse_linear_cross_entropy=fuse_linear_cross_entropy)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def attnres_block_size(self) -> int | None:
        """Expose the legacy option while serializing only the canonical residual fields."""
        return self.residual_kwargs.get('block_size', 1) if self.residual_mode == 'attnres' else None

    @attnres_block_size.setter
    def attnres_block_size(self, value: int | None) -> None:
        validate_attnres_config(attnres_block_size=value)
        self.residual_mode = 'standard' if value is None else 'attnres'
        self.residual_kwargs = dict(self.residual_kwargs)
        if value is None:
            self.residual_kwargs.pop('block_size', None)
        else:
            self.residual_kwargs['block_size'] = value

    def get_residual_kwargs(self, sub_layer_idx: int) -> dict:
        """Build constructor arguments for the selected residual and sublayer."""
        return dict(
            norm_eps=self.norm_eps,
            fuse_norm=self.fuse_norm,
        ) | self.residual_kwargs | dict(
            hidden_size=self.hidden_size,
            sub_layer_idx=sub_layer_idx,
            num_sublayers=2 * self.num_hidden_layers,
        )
