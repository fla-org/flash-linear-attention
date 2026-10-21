# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Context Parallel test for the chunk_rwkv7 wrapper.

chunk_rwkv7 is a thin alias over chunk_dplr_delta_rule (r=q, w=gk). Reusing the
DPLR harness, this swaps in the wrapper as the kernel under test so that
cp_context is exercised end-to-end through the wrapper's kwarg passthrough.
"""

import pytest
import torch

from cp.test_dplr import _run_distributed
from fla.ops.rwkv7 import chunk_rwkv7


def _chunk_rwkv7(q, k, v, a, b, gk, scale=None, **kwargs):
    # chunk_rwkv7 defaults scale=1.0; align with chunk_dplr_delta_rule (scale=None
    # -> 1/sqrt(K)) so the wrapper matches the dplr_recurrence reference.
    return chunk_rwkv7(r=q, w=gk, k=k, v=v, a=a, b=b, scale=scale, **kwargs)


@pytest.mark.parametrize(
    ('world_size', 'lengths'),
    [
        pytest.param(2, [400, 624], id='cp2-sequence-cut'),
        pytest.param(4, [1024], id='cp4-single-sequence'),
    ],
)
def test_chunk(request, world_size, lengths):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"At least {world_size} devices required")

    _run_distributed(
        world_size=world_size,
        test_name=request.node.callspec.id,
        T=1024,
        H=4,
        D=64,
        lengths=lengths,
        dtype=torch.bfloat16,
        safe_gate=False,
        op=_chunk_rwkv7,
        op_chunk_size=64,
        assert_tilelang=False,
    )
