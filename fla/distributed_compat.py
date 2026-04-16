# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
"""
Centralized compatibility module for torch.distributed imports.
All distributed-related imports should go through here to handle environments
where distributed tensor APIs are not available.
"""

import torch

# DeviceMesh
try:
    from torch.distributed import DeviceMesh
except ImportError:
    try:
        from torch.distributed.device_mesh import DeviceMesh
    except ImportError:
        DeviceMesh = None

# DTensor
try:
    from torch.distributed.tensor import DTensor
except (ImportError, AttributeError):
    DTensor = None

# Replicate, Shard, distribute_module, Placement
try:
    from torch.distributed.tensor import Placement, Replicate, Shard, distribute_module
except (ImportError, AttributeError):
    Placement = Replicate = Shard = distribute_module = None

# ParallelStyle
try:
    from torch.distributed.tensor.parallel import ParallelStyle
except (ImportError, AttributeError):
    ParallelStyle = None

# Convenience flag
HAS_DISTRIBUTED = all([
    DeviceMesh is not None,
    DTensor is not None,
    Placement is not None,
    Replicate is not None,
    Shard is not None,
    distribute_module is not None,
    ParallelStyle is not None,
])

__all__ = [
    'DeviceMesh',
    'DTensor',
    'Placement',
    'Replicate',
    'Shard',
    'distribute_module',
    'ParallelStyle',
    'HAS_DISTRIBUTED',
]
