# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

from transformers.configuration_utils import PretrainedConfig


class Mamba3Config(PretrainedConfig):
    """
    Configuration for the FLA port of Mamba-3.

    Mamba-3 differs from Mamba-2 in that it removes the 1D causal convolution,
    adds per-head biases on B and C, normalises B/C with RMSNorm, and applies
    a blockwise rotary transform to Q/K before the SSM scan. It optionally
    supports a low-rank MIMO projection for V and the output gate.
    """

    model_type = "mamba3"

    def __init__(
        self,
        head_dim: int = 64,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        state_size: int = 128,
        num_hidden_layers: int = 48,
        norm_eps: float = 1e-5,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        expand: int = 2,
        n_groups: int = 1,
        use_bias: bool = False,
        rope_fraction: float = 0.5,
        A_floor: float = 1e-4,
        is_outproj_norm: bool = False,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        residual_in_fp32: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        rescale_prenorm_residual: bool = True,
        use_cache: bool = True,
        chunk_size: int = 64,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.expand = expand
        self.n_groups = n_groups
        self.head_dim = head_dim
        self.num_heads = int(self.expand * self.hidden_size / self.head_dim)

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias

        self.rope_fraction = rope_fraction
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.residual_in_fp32 = residual_in_fp32
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.use_cache = use_cache
        self.chunk_size = chunk_size

        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.tie_word_embeddings = tie_word_embeddings

        if rope_fraction not in (0.5, 1.0):
            raise ValueError("`rope_fraction` must be 0.5 or 1.0.")
        if dt_min <= 0 or dt_max < dt_min:
            raise ValueError("`dt_min` and `dt_max` must satisfy 0 < dt_min <= dt_max.")
        if dt_init_floor <= 0:
            raise ValueError("`dt_init_floor` must be > 0.")
        if A_floor <= 0:
            raise ValueError("`A_floor` must be > 0.")
        if is_mimo and mimo_rank <= 0:
            raise ValueError("`mimo_rank` must be > 0 when `is_mimo=True`.")

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot both be True.",
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improve memory efficiency "
                "at the potential cost of reduced precision.",
            )

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
