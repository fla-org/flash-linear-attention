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


@pytest.mark.parametrize('T', [15, 64, 65])
@pytest.mark.parametrize('shared_mem', [False, True])
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
