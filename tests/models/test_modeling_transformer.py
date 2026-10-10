# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import TransformerConfig, TransformerForCausalLM
from fla.utils import assert_close, device, find_spec_cached

from .test_modeling_base import run_test_generation, run_test_model_forward_backward

# Mark (not importorskip): a fully skipped module reports "no tests collected" and exits
# with code 5, which CI per-file loops treat as a failure.
pytestmark = pytest.mark.skipif(find_spec_cached("flash_attn") is None,
                                reason="Attention requires flash-attn (`pip install flash-attn --no-build-isolation`).")


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'attnres_block_size', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-bs{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64,  True,  None, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, None, torch.bfloat16),
            (4, 4, 1024, 4, 128, False, None, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 1,    torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 2,    torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 4,    torch.bfloat16),
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
    attnres_block_size: int | None,
    dtype: torch.dtype,
):
    run_test_model_forward_backward(
        L,
        B,
        T,
        H,
        D,
        TransformerConfig,
        use_l2warp=use_l2warp,
        attnres_block_size=attnres_block_size,
        dtype=dtype,
    )


# ===================================================================================
# Test for Generation
# ===================================================================================
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
    run_test_generation(L, B, T, H, D, TransformerConfig, dtype)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('window_size', [8, None])
@torch.no_grad()
def test_cached_chunk_continuation(dtype, window_size):
    torch.manual_seed(42)
    config = TransformerConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_heads=2,
        num_kv_heads=1,
        window_size=window_size,
        intermediate_size=256,
        vocab_size=256,
    )
    model = TransformerForCausalLM(config).to(device=device, dtype=dtype).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 22), device=device)
    ref = model(input_ids=input_ids, use_cache=False).logits
    cache = None
    outputs = []
    offset = 0
    for length in (3, 2, 7, 9, 1):
        output = model(input_ids=input_ids[:, offset:offset+length], past_key_values=cache, use_cache=True)
        outputs.append(output.logits)
        cache = output.past_key_values
        offset += length
        for layer_idx in range(config.num_hidden_layers):
            assert cache.get_seq_length(layer_idx) == offset
    actual = torch.cat(outputs, dim=1)
    assert torch.isfinite(actual).all()
    assert_close('logits', ref, actual, 0.005 if dtype == torch.float16 else 0.02)
