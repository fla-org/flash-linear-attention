# -*- coding: utf-8 -*-

from .naive import sb_attn_naive
from .parallel import sb_attn

__all__ = [
    'sb_attn',
    'sb_attn_naive',
]
