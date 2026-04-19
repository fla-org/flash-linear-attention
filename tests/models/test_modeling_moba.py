# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import MoBAConfig

from .test_modeling_base import run_test_generation, run_test_model_forward_backward


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'qk_norm', 'use_output_gate', 'moba_topk', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-qkn{}-og{}-k{}-{}".format(*test))
        for test in [
            # baseline
            (4, 4, 1024, 4, 64, False, False, True, 3, torch.bfloat16),
            # D=128 + l2warp (exercises Hopper head-dim and the l2warp branch)
            (4, 4, 1024, 4, 128, True, False, True, 3, torch.bfloat16),
            # qk_norm + no output gate (covers both layer variants in one case)
            (2, 4, 1024, 4, 64, False, True, False, 3, torch.bfloat16),
            # topk=1 triggers the full-attn short-circuit inside `parallel_moba`
            (2, 4, 1024, 4, 64, False, False, True, 1, torch.bfloat16),
        ]
    ],
)
def test_modeling(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    use_l2warp: bool,
    qk_norm: bool,
    use_output_gate: bool,
    moba_topk: int,
    dtype: torch.dtype,
):
    run_test_model_forward_backward(
        L, B, T, H, D, MoBAConfig,
        use_l2warp=use_l2warp, dtype=dtype,
        qk_norm=qk_norm,
        use_output_gate=use_output_gate,
        moba_chunk_size=128,
        moba_topk=moba_topk,
    )


# ===================================================================================
# Test for Generation
# ===================================================================================
# NOTE: MoBA is listed in `GENERATION_UNSUPPORTED` and therefore this test is
# currently a skip. MoBA uses block-sparse attention during prefill but falls
# back to dense full attention during decoding (per the paper, Sec. 3.3), so
# the incremental-decode logits cannot bit-match a reference computed with
# full-sequence MoBA. Keeping the test wired up so it activates automatically
# if that constraint ever changes.
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 4, 2000, 8, 64, torch.float16),
        ]
    ],
)
def test_generation(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    run_test_generation(L, B, T, H, D, MoBAConfig, dtype)
