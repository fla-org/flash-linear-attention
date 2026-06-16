# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.utils import assert_close, device


def test_gated_deltanet_fused_qkv_conv_matches_fallback():
    torch.manual_seed(42)
    B, T, D = 1, 128, 512
    dtype = torch.bfloat16
    fast = GatedDeltaNet(
        hidden_size=D,
        head_dim=64,
        num_heads=6,
        expand_v=2,
        mode='chunk',
    ).to(device=device, dtype=dtype).train()
    fallback = GatedDeltaNet(
        hidden_size=D,
        head_dim=64,
        num_heads=6,
        expand_v=2,
        mode='chunk',
    ).to(device=device, dtype=dtype).train()
    fallback.load_state_dict(fast.state_dict())

    x = torch.randn(B, T, D, device=device, dtype=dtype)
    do = torch.randn_like(x)
    x_fast = x.detach().clone().requires_grad_(True)
    x_fallback = x.detach().clone().requires_grad_(True)

    y_fast = fast(x_fast)[0]
    mask = torch.ones(B, T, device=device, dtype=torch.long)
    y_fallback = fallback(x_fallback, attention_mask=mask)[0]

    (y_fast * do).sum().backward()
    (y_fallback * do).sum().backward()

    assert_close('y', y_fallback, y_fast, 0.0)
    assert_close('dx', x_fallback.grad, x_fast.grad, 0.0)
