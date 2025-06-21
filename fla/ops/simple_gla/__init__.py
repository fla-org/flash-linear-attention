# -*- coding: utf-8 -*-

from .chunk import chunk_simple_gla
from .fused_chunk import fused_chunk_simple_gla
from .fused_recurrent import fused_recurrent_simple_gla
from .parallel import parallel_simple_gla

__all__ = [
    'chunk_simple_gla',
    'fused_chunk_simple_gla',
    'fused_recurrent_simple_gla',
    'parallel_simple_gla'
]
