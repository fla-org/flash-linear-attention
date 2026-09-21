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


@pytest.mark.parametrize("batch_size", [1, 2, 7], ids=["flattened", "batched", "cross-row"])
@pytest.mark.parametrize(
    ("fuse_cross_entropy", "fuse_linear_cross_entropy"),
    [(False, False), (True, False), (False, True)],
    ids=["ce", "fused-ce", "fused-linear-ce"],
)
def test_packed_loss(batch_size: int, fuse_cross_entropy: bool, fuse_linear_cross_entropy: bool):
    torch.manual_seed(42)
    config = TransformerConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_heads=1,
        intermediate_size=128,
        vocab_size=64,
        use_cache=False,
        fuse_cross_entropy=fuse_cross_entropy,
        fuse_linear_cross_entropy=fuse_linear_cross_entropy,
    )
    model = TransformerForCausalLM(config).to(device=device, dtype=torch.float16).train()
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(2, 7), device=device)
    labels = input_ids.clone()
    labels[:, 2] = -100

    expected = model(input_ids=input_ids, labels=labels).loss
    expected.backward()
    expected_grads = {name: param.grad.float().clone() for name, param in model.named_parameters()}
    model.zero_grad(set_to_none=True)

    output = model(
        input_ids=input_ids.view(batch_size, -1),
        labels=labels.view(batch_size, -1),
        output_hidden_states=True,
        cu_seqlens=torch.tensor([0, 7, 14], dtype=torch.int32, device=device),
    )
    output.hidden_states[-1].retain_grad()
    actual = output.loss
    actual.backward()
    has_gradient = output.hidden_states[-1].grad.reshape(2, 7, -1).ne(0).any(dim=-1)
    assert torch.equal(has_gradient[:, :-1], labels[:, 1:].ne(-100))
    assert not has_gradient[:, -1].any()
    assert_close("loss", expected.float(), actual.float(), 2e-3)
    for name, param in model.named_parameters():
        assert_close(name, expected_grads[name], param.grad.float(), 2e-3)


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
