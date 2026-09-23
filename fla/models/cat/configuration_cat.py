# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Contributed by: Jatin Prakash (bicycleman15)
# Controllably Efficient Language Models (https://arxiv.org/abs/2511.05313)

from __future__ import annotations

import warnings

from transformers.configuration_utils import PretrainedConfig


class CATConfig(PretrainedConfig):
    """
    Configuration class for CAT (Compress And Attend Transformers).

    Args:
        hidden_size (int, Optional):
            Decoder hidden size. Default: 2048.
        num_hidden_layers (int, Optional):
            Number of decoder layers. Default: 24.
        num_heads (int, Optional):
            Number of decoder attention heads. Default: 32.
        num_kv_heads (int, Optional):
            Number of decoder key/value heads. Default: `None`.
        max_chunk_size (int, Optional):
            Maximum chunk size supported by the compressor projection. Default: 16.
        dim_fx (int, Optional):
            Compressed chunk representation size. Must equal `hidden_size`. Default: `None`.
        compressor_hidden_size (int, Optional):
            Compressor hidden size. Default: `hidden_size // 2`.
        compressor_num_layers (int, Optional):
            Number of compressor layers. Default: `max(1, num_hidden_layers // 4)`.
        compressor_num_heads (int, Optional):
            Number of compressor attention heads. Default: `max(1, num_heads // 2)`.
        pad_to_multiple_of (int, Optional):
            Sequence-length bucket size used before CAT chunking. Default: 512.
        train_chunk_size_schedule (str, Optional):
            Training-only chunk-size schedule used when `chunk_size` is not passed. Default: "power_of_2".
        min_train_chunk_size (int, Optional):
            Smallest training chunk size for the power-of-two schedule. Default: 2.
    """

    model_type = 'cat'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        max_chunk_size: int = 16,
        dim_fx: int | None = None,
        compressor_hidden_size: int | None = None,
        compressor_num_layers: int | None = None,
        compressor_num_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        rope_theta: float | None = 10000.,
        max_position_embeddings: int = 2048,
        pad_to_multiple_of: int = 512,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        use_cache: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        train_chunk_size_schedule: str | None = "power_of_2",
        min_train_chunk_size: int = 2,
        vocab_size: int = 32000,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.max_chunk_size = max_chunk_size
        self.dim_fx = dim_fx if dim_fx is not None else hidden_size
        self.compressor_hidden_size = compressor_hidden_size if compressor_hidden_size is not None else hidden_size // 2
        self.compressor_num_layers = (
            compressor_num_layers if compressor_num_layers is not None else max(1, num_hidden_layers // 4)
        )
        self.compressor_num_heads = compressor_num_heads if compressor_num_heads is not None else max(1, num_heads // 2)

        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.pad_to_multiple_of = pad_to_multiple_of

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
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.train_chunk_size_schedule = train_chunk_size_schedule
        self.min_train_chunk_size = min_train_chunk_size
        self.vocab_size = vocab_size

        if self.max_chunk_size <= 0:
            raise ValueError(f"`max_chunk_size` must be positive, got {self.max_chunk_size}.")
        if self.pad_to_multiple_of <= 0:
            raise ValueError(f"`pad_to_multiple_of` must be positive, got {self.pad_to_multiple_of}.")
        valid_train_chunk_size_schedules = (None, "none", "fixed", "power_of_2")
        if self.train_chunk_size_schedule not in valid_train_chunk_size_schedules:
            raise ValueError(
                "`train_chunk_size_schedule` must be one of "
                f"{valid_train_chunk_size_schedules}, got {self.train_chunk_size_schedule!r}."
            )
        if self.min_train_chunk_size <= 0:
            raise ValueError(f"`min_train_chunk_size` must be positive, got {self.min_train_chunk_size}.")
        if self.min_train_chunk_size > self.max_chunk_size:
            raise ValueError(
                "`min_train_chunk_size` must not exceed `max_chunk_size`, "
                f"got {self.min_train_chunk_size} and {self.max_chunk_size}."
            )
        if (
            self.train_chunk_size_schedule == "power_of_2"
            and self.min_train_chunk_size & (self.min_train_chunk_size - 1)
        ):
            raise ValueError(
                "`min_train_chunk_size` must be a power of two when "
                f"`train_chunk_size_schedule='power_of_2'`, got {self.min_train_chunk_size}."
            )
        if self.max_position_embeddings < self.max_chunk_size:
            raise ValueError(
                "`max_position_embeddings` must be at least `max_chunk_size`, "
                f"got {self.max_position_embeddings} and {self.max_chunk_size}."
            )
        if self.dim_fx != self.hidden_size:
            raise ValueError(
                f"`dim_fx` ({self.dim_fx}) must equal `hidden_size` ({self.hidden_size}) because CAT feeds compressed "
                "chunk representations directly into the decoder."
            )
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads}).")
        if (self.hidden_size // self.num_heads) % 2 != 0:
            raise ValueError("CAT requires an even decoder head dimension for rotary embeddings.")
        if self.num_kv_heads is not None and self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"`num_heads` ({self.num_heads}) must be divisible by `num_kv_heads` ({self.num_kv_heads}).")
        if self.compressor_hidden_size % self.compressor_num_heads != 0:
            raise ValueError(
                f"`compressor_hidden_size` ({self.compressor_hidden_size}) must be divisible by "
                f"`compressor_num_heads` ({self.compressor_num_heads})."
            )
        if (self.compressor_hidden_size // self.compressor_num_heads) % 2 != 0:
            raise ValueError("CAT requires an even compressor head dimension for rotary embeddings.")

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError("`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.")
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improve memory efficiency at the potential cost of "
                "reduced precision. If you observe issues like loss divergence, consider disabling this setting.",
            )
        if use_cache:
            warnings.warn("CAT does not support KV-cache decoding; `use_cache` will be ignored by the model.")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
