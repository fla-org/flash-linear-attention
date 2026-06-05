# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch

from fla.layers.parallax_attn import ParallaxAttention
from fla.utils import check_shared_mem, device


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'qk_norm', 'window_size'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-qknorm{}-W{}".format(*test))
        for test in [
            (2, 256, 4, 4, 64, False, None),
            (2, 300, 2, 8, 64, True, None),     # GQA + qk_norm + non-pow2 T
            (2, 512, 4, 4, 128, False, 64),     # sliding window
        ]
    ],
)
def test_parallax_attention(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    qk_norm: bool,
    window_size: int | None,
    dtype: torch.dtype,
):
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shared mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    hidden_size = HQ * D
    layer = ParallaxAttention(
        hidden_size=hidden_size,
        num_heads=HQ,
        num_kv_heads=H,
        qk_norm=qk_norm,
        window_size=window_size,
        layer_idx=0,
    ).to(device).to(dtype)

    x = torch.randn(B, T, hidden_size, device=device, dtype=dtype, requires_grad=True)
    o, _, _ = layer(x)
    assert o.shape == (B, T, hidden_size)

    o.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"missing grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
