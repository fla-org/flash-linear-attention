# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math
import warnings

from transformers.configuration_utils import PretrainedConfig

from fla.models.hybrid import HybridAttentionConfig, _HybridAttentionConfigMixin

_DEFAULT_ATTN = {
    'layers': (1, 3, 5, 7, 9, 11, 13, 15, 17),
    'num_heads': 18,
    'num_kv_heads': 18,
    'qkv_bias': False,
    'window_size': 2048,
    'rope_theta': 10000.,
}


def _is_legacy_default_attn(attn: HybridAttentionConfig) -> bool:
    if not isinstance(attn, dict) or attn.keys() != _DEFAULT_ATTN.keys():
        return False
    layers = attn['layers']
    if not isinstance(layers, (list, tuple)) or tuple(layers) != _DEFAULT_ATTN['layers']:
        return False
    return all(attn[key] == value for key, value in _DEFAULT_ATTN.items() if key != 'layers')


def _adapt_default_attn(num_hidden_layers: int) -> dict:
    return {
        **_DEFAULT_ATTN,
        'layers': [layer_idx for layer_idx in _DEFAULT_ATTN['layers'] if layer_idx < num_hidden_layers],
    }


class SambaConfig(_HybridAttentionConfigMixin, PretrainedConfig):

    model_type = "samba"

    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs):
        config_dict = config_dict.copy()
        is_legacy_serialized_default = (
            config_dict.get('model_type') == cls.model_type
            and 'transformers_version' in config_dict
            and _is_legacy_default_attn(config_dict.get('attn'))
        )
        if is_legacy_serialized_default:
            # before hybrid-plan validation, shallow Samba JSON retained the
            # full depth-18 default. Adapt only that serialized legacy shape.
            num_hidden_layers = config_dict.get('num_hidden_layers', 18)
            config_dict['attn'] = _adapt_default_attn(num_hidden_layers)
        return super().from_dict(config_dict, **kwargs)

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
        attn: HybridAttentionConfig = _DEFAULT_ATTN,
        hidden_ratio: int | None = 4,
        rescale_prenorm_residual: bool = False,
        use_cache: bool = True,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        tie_word_embeddings: bool = False,
        attnres_block_size: int | None = None,
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
        if attn is _DEFAULT_ATTN:
            attn = _adapt_default_attn(num_hidden_layers)
        self.attn = attn
        self.hidden_ratio = hidden_ratio
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache

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
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
