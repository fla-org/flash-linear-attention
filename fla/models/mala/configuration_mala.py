# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

"""
MALA (Magnitude-Aware Linear Attention) configuration.

Based on the paper:
"Rectifying Magnitude Neglect in Linear Attention"
ICCV 2025 (highlight)
https://arxiv.org/abs/2507.00698

Original implementation:
https://github.com/aldjalkdf/MAViT
"""


class MalaConfig(PretrainedConfig):
    r"""
    Configuration class for MALA (Magnitude-Aware Linear Attention).
    
    Based on the paper:
    "Rectifying Magnitude Neglect in Linear Attention"
    ICCV 2025 (highlight)
    https://arxiv.org/abs/2507.00698
    """

    model_type = "mala"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=None,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        expand_k=1.0,
        expand_v=1.0,
        use_lepe=True,
        lepe_kernel_size=5,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else 4 * hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.mlp_dropout = mlp_dropout
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.use_lepe = use_lepe
        self.lepe_kernel_size = lepe_kernel_size
