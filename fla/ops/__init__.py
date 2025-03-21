# -*- coding: utf-8 -*-

from .abc import chunk_abc
from .based import fused_chunk_based, parallel_based
from .delta_rule import chunk_delta_rule, fused_chunk_delta_rule, fused_recurrent_delta_rule
from .gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from .generalized_delta_rule import (
    chunk_dplr_delta_rule,
    chunk_iplr_delta_rule,
    fused_recurrent_dplr_delta_rule,
    fused_recurrent_iplr_delta_rule
)
from .gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from .gsa import chunk_gsa, fused_recurrent_gsa
from .hgrn import fused_recurrent_hgrn
from .lightning_attn import chunk_lightning_attn, fused_recurrent_lightning_attn
from .linear_attn import chunk_linear_attn, fused_chunk_linear_attn, fused_recurrent_linear_attn
from .nsa import parallel_nsa
from .retention import chunk_retention, fused_chunk_retention, fused_recurrent_retention, parallel_retention
from .rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6
from .rwkv7 import chunk_rwkv7, fused_recurrent_rwkv7
from .simple_gla import chunk_simple_gla, fused_recurrent_simple_gla, parallel_simple_gla

__all__ = [
    'chunk_abc',
    'fused_chunk_based', 'parallel_based',
    'chunk_delta_rule', 'fused_chunk_delta_rule', 'fused_recurrent_delta_rule',
    'chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule',
    'chunk_dplr_delta_rule', 'chunk_iplr_delta_rule',
    'fused_recurrent_dplr_delta_rule', 'fused_recurrent_iplr_delta_rule',
    'chunk_gla', 'fused_chunk_gla', 'fused_recurrent_gla',
    'chunk_gsa', 'fused_recurrent_gsa',
    'fused_recurrent_hgrn',
    'chunk_lightning_attn', 'fused_recurrent_lightning_attn',
    'chunk_linear_attn', 'fused_chunk_linear_attn', 'fused_recurrent_linear_attn',
    'parallel_nsa',
    'chunk_retention', 'fused_chunk_retention', 'fused_recurrent_retention', 'parallel_retention',
    'chunk_rwkv6', 'fused_recurrent_rwkv6',
    'chunk_rwkv7', 'fused_recurrent_rwkv7',
    'chunk_simple_gla', 'fused_recurrent_simple_gla', 'parallel_simple_gla',
]
