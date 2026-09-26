# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.lightnet import LightNetAttention
from fla.models.utils import Cache
from fla.utils import assert_close, device


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("T_prefill,T_continue", [(8, 4), (32, 80)])
def test_warm_cache_multi_token_continuation(dtype: torch.dtype, T_prefill: int, T_continue: int):
    B, D = 2, 64
    torch.manual_seed(42)
    layer = LightNetAttention(
        hidden_size=D,
        num_heads=2,
        expand_ratio=8,
        layer_idx=0,
    ).to(device=device, dtype=dtype).eval()
    prefill = torch.randn(B, T_prefill, D, device=device, dtype=dtype)
    continuation = torch.randn(B, T_continue, D, device=device, dtype=dtype)
    tol = 0.005 if dtype == torch.float16 else 0.02

    with torch.no_grad():
        expected, _, _ = layer(hidden_states=torch.cat([prefill, continuation], dim=1))
        cache = Cache()
        actual_prefill, _, returned_cache = layer(hidden_states=prefill, past_key_values=cache, use_cache=True)
        actual, _, _ = layer(hidden_states=continuation, past_key_values=cache, use_cache=True)

    assert returned_cache is cache
    assert_close("prefill", expected[:, :T_prefill], actual_prefill, tol)
    assert_close("continuation", expected[:, T_prefill:], actual, tol)
