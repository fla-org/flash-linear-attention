# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Context Parallel test for the chunk_rwkv7 wrapper.

chunk_rwkv7 is a thin alias over chunk_dplr_delta_rule (r=q, w=gk). Reusing the
test_cp_dplr harness, this swaps in the wrapper as the kernel under test so that
cp_context is exercised end-to-end through the wrapper's kwarg passthrough.
"""

import pytest
import torch

# tests/context_parallel is on sys.path under pytest (see pyproject pythonpath)
from test_cp_dplr import run_cp_test_with_spawn

from fla.ops.rwkv7 import chunk_rwkv7


def _chunk_rwkv7(q, k, v, a, b, gk, scale=None, **kwargs):
    # chunk_rwkv7 defaults scale=1.0; align with chunk_dplr_delta_rule (scale=None
    # -> 1/sqrt(K)) so the wrapper matches the dplr_recurrence reference.
    return chunk_rwkv7(r=q, w=gk, k=k, v=v, a=a, b=b, scale=scale, **kwargs)


def test_cp2_wrapper_sequence_cut():
    """CP2 via chunk_rwkv7: sequences cut across the rank boundary."""
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")
    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_Wrapper_SequenceCut",
        T=1024, H=4, D=64,
        lengths=[400, 624],
        op=_chunk_rwkv7,
    )


def test_cp4_wrapper_single_sequence():
    """CP4 via chunk_rwkv7: single long sequence spanning all ranks."""
    if torch.cuda.device_count() < 4:
        pytest.skip("At least 4 GPUs required")
    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_Wrapper_SingleSequence",
        T=1024, H=4, D=64,
        lengths=[1024],
        op=_chunk_rwkv7,
    )
