# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

from transformers.configuration_utils import PreTrainedConfig

from fla.models.configuration_utils import validate_loss_config


class PHBAConfig(PreTrainedConfig):
    model_type = "phba"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float | None = 10000.0,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        elementwise_affine: bool = True,
        norm_eps: float = 1e-6,
        use_cache: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        moba_chunk_size: int = 128,
        moba_topk: int = 8,
        use_output_gate: bool = True,
        use_phba: bool = True,
        gsa_num_slots: int = 128,
        gate_logit_normalizer: float = 8,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        if type(elementwise_affine) is not bool:
            raise ValueError("PHBA elementwise_affine must be bool")
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size
        self.moba_chunk_size = moba_chunk_size
        self.moba_topk = moba_topk
        self.use_output_gate = use_output_gate
        self.use_phba = use_phba
        self.gsa_num_slots = gsa_num_slots
        self.gate_logit_normalizer = gate_logit_normalizer

        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("PHBA hidden_size must be positive and divisible by num_heads")
        if num_kv_heads not in (None, num_heads):
            raise ValueError("PHBA requires num_kv_heads == num_heads")
        if moba_chunk_size < 16 or (moba_chunk_size & (moba_chunk_size - 1)):
            raise ValueError("PHBA moba_chunk_size must be a power of two >= 16")
        if moba_topk < 1 or gsa_num_slots < 1:
            raise ValueError("PHBA moba_topk and gsa_num_slots must be positive")
        if not math.isfinite(gate_logit_normalizer) or gate_logit_normalizer <= 0:
            raise ValueError("PHBA gate_logit_normalizer must be finite and positive")
        if not use_phba:
            raise ValueError("PHBAConfig requires use_phba=True")
        if kwargs.pop("attn_type", "phba") != "phba":
            raise ValueError("PHBAConfig does not select other attention types")
        if kwargs.pop("attnres_block_size", None) is not None:
            raise ValueError("PHBA does not implement AttnRes")

        if not 16 <= hidden_size // num_heads <= 128 or (hidden_size // num_heads) % 8:
            raise ValueError("PHBA head dimensions must be multiples of 8 from 16 through 128")
        if moba_chunk_size not in (32, 64, 128, 256, 512) or moba_topk > 16 or gsa_num_slots > 128:
            raise ValueError("PHBA supports chunks32-512, topk1-16 and slots1-128")
        if use_cache:
            raise ValueError("The deterministic PHBA model requires use_cache=False")
        if window_size is not None:
            raise ValueError("PHBA does not apply an additional sliding window")
        validate_loss_config(fuse_cross_entropy=fuse_cross_entropy, fuse_linear_cross_entropy=fuse_linear_cross_entropy)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
