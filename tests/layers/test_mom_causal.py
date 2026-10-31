# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.mom import MomAttention
from fla.utils import device


@pytest.mark.parametrize('T', [32, 64, 100])
@pytest.mark.parametrize('topk', [1, 2])
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

    # tokens routed to the same memory must keep their time order, so later tokens cannot reach earlier outputs
    perturbed = hidden_states.clone()
    perturbed[:, -1] = torch.randn(2, 64, device=device)
    actual = layer(perturbed)[0]
    torch.testing.assert_close(actual[:, :-1], reference[:, :-1], rtol=0, atol=1e-5)

    prefix = layer(hidden_states[:, :T // 2])[0]
    torch.testing.assert_close(prefix, reference[:, :T // 2], rtol=1e-4, atol=1e-5)
