# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.mom import MomAttention
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('T', 'shared_mem'),
    [
        pytest.param(15, False, id='T15-shared_memFalse'),
        pytest.param(64, True, id='T64-shared_memTrue'),
        pytest.param(65, False, id='T65-shared_memFalse'),
        pytest.param(65, True, id='T65-shared_memTrue'),
    ],
)
def test_train_no_grad(T: int, shared_mem: bool):
    torch.manual_seed(42)
    layer = MomAttention(
        hidden_size=64,
        head_dim=32,
        num_heads=2,
        expand_v=1,
        num_memories=2,
        topk=1,
        shared_mem=shared_mem,
    ).to(device=device, dtype=torch.float32).train()
    hidden_states = torch.randn(2, T, 64, device=device, requires_grad=True)
    reference = layer(hidden_states)[0]
    reference.sum().backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    with torch.no_grad():
        actual = layer(hidden_states.detach())[0]
    assert torch.isfinite(actual).all()
    assert_close('o', reference.detach(), actual, 0.006)


@pytest.mark.parametrize(
    ('T', 'topk'),
    [
        pytest.param(32, 1, id='T32-topk1'),
        pytest.param(64, 2, id='T64-topk2'),
        pytest.param(65, 1, id='T65-topk1'),
        pytest.param(100, 2, id='T100-topk2'),
    ],
)
@torch.no_grad()
def test_causal(T: int, topk: int):
    torch.manual_seed(42)
    layer = MomAttention(
        hidden_size=64,
        head_dim=32,
        num_heads=2,
        expand_v=1,
        num_memories=4,
        topk=topk,
    ).to(device=device, dtype=torch.float32).eval()
    hidden_states = torch.randn(2, T, 64, device=device)
    reference = layer(hidden_states)[0]

    # changing a future token must not affect earlier outputs
    perturbed = hidden_states.clone()
    perturbed[:, -1] = torch.randn(2, 64, device=device)
    actual = layer(perturbed)[0]
    torch.testing.assert_close(actual[:, :-1], reference[:, :-1], rtol=0, atol=1e-5)

    # the prefix can use fused_recurrent while the full sequence uses chunk
    prefix = layer(hidden_states[:, :T // 2])[0]
    assert_close('prefix', reference[:, :T // 2], prefix, 1e-3)
