# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.mamba import Mamba
from fla.layers.mamba2 import Mamba2
from fla.layers.rwkv6 import RWKV6Attention
from fla.models.utils import Cache
from fla.utils import assert_close, device


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "builder",
    [
        pytest.param(lambda layer_idx: Mamba(hidden_size=64, intermediate_size=64, layer_idx=layer_idx), id="mamba"),
        pytest.param(lambda layer_idx: Mamba2(hidden_size=64, num_heads=4, head_dim=32, layer_idx=layer_idx), id="mamba2"),
        pytest.param(lambda layer_idx: RWKV6Attention(hidden_size=64, num_heads=4, layer_idx=layer_idx), id="rwkv6"),
    ],
)
def test_t0_prefill_decode(builder, dtype: torch.dtype):
    torch.manual_seed(42)
    B, T, D = 2, 0, 64
    layer = builder(0).to(device=device, dtype=dtype).eval()
    empty = torch.randn(B, T, D, device=device, dtype=dtype)
    token = torch.randn(B, 1, D, device=device, dtype=dtype)
    tol = 0.005 if dtype == torch.float16 else 0.02

    with torch.no_grad():
        output, _, _ = layer(hidden_states=empty)
        assert output.shape == (B, T, D)

        expected, _, _ = layer(hidden_states=token)
        cache = Cache()
        output, _, returned_cache = layer(hidden_states=empty, past_key_values=cache, use_cache=True)
        assert output.shape == (B, T, D)
        assert returned_cache is cache
        assert cache.get_seq_length(0) == 0
        actual, _, _ = layer(hidden_states=token, past_key_values=cache, use_cache=True)

    assert cache.get_seq_length(0) == 1
    assert_close("decode", expected, actual, tol)
