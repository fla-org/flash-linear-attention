# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.utils import (
    IS_NPU,
    NPU_DMA_ALIGN_BYTES,
    npu_last_dim_elem_align,
    npu_supported_last_dim,
)


@pytest.fixture(autouse=True)
def _npu_byte_aligned_last_dims(request):
    """Skip parametrized ops tests whose K/V/D last-dim byte stride is not 32-byte aligned."""
    if not IS_NPU or not hasattr(request.node, 'callspec'):
        return
    params = request.node.callspec.params
    dtype = params.get('dtype', torch.float16)
    for name, val in params.items():
        if name in ('D', 'K', 'V') and isinstance(val, int) and not npu_supported_last_dim(val, dtype=dtype):
            align = npu_last_dim_elem_align(dtype)
            pytest.skip(
                f'NPU triton_ascend requires last-dim multiple of {align} elements '
                f'({NPU_DMA_ALIGN_BYTES}-byte stride for {dtype}), got {name}={val}',
            )
