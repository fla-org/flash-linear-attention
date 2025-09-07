# -*- coding: utf-8 -*-

from .fused_chunk import delta_pre_attn
from .naive import delta_pre_attn_naive

__all__ = [
    'delta_pre_attn',
    'delta_pre_attn_naive',
]
