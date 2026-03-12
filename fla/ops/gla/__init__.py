# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Device-aware operator imports

"""
GLA operators with automatic device-specific implementation selection.

This module automatically selects the appropriate implementation (GPU or NPU)
based on the current device.
"""

from fla.utils import get_available_device

# Detect current device backend
_device_backend = get_available_device()

# Import operators based on device type
if _device_backend == 'npu':
    # Use NPU-optimized implementations
    from .chunk_npu import chunk_gla_npu as chunk_gla
    # TODO: Add NPU versions for fused_chunk_gla and fused_recurrent_gla when available
    from .fused_chunk import fused_chunk_gla
    from .fused_recurrent import fused_recurrent_gla
else:
    # Use default (GPU) implementations
    from .chunk import chunk_gla
    from .fused_chunk import fused_chunk_gla
    from .fused_recurrent import fused_recurrent_gla

__all__ = [
    'chunk_gla',
    'fused_chunk_gla',
    'fused_recurrent_gla',
]
