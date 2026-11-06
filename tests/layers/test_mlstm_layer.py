# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.mlstm import MLSTM
from fla.models.cache_utils import Cache
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('mode', 'T', 'split', 'padding', 'dtype'),
    [
        pytest.param('chunk', 65, 64, 0, torch.float32, id='chunk-prefill-single-token-decode'),
        pytest.param('chunk', 17, 9, 2, torch.float16, id='chunk-padded-continuation'),
        pytest.param('fused_recurrent', 9, 8, 2, torch.float32, id='recurrent-padded-decode'),
        pytest.param('fused_recurrent', 17, 7, 0, torch.bfloat16, id='recurrent-dense-continuation'),
    ],
)
def test_cache(mode: str, T: int, split: int, padding: int, dtype: torch.dtype):
    torch.manual_seed(42)
    B, D = 2, 64
    layer = MLSTM(hidden_size=D, num_heads=2, proj_factor=1, mode=mode, layer_idx=0).to(device, dtype).eval()
    hidden_states = torch.randn(B, T, D, device=device, dtype=dtype)
    attention_mask = None
    if padding:
        attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
        attention_mask[0, :padding] = 0
    cache = Cache()
    with torch.no_grad():
        ref, _, _ = layer(hidden_states=hidden_states, attention_mask=attention_mask)
        prefill, _, returned_cache = layer(
            hidden_states=hidden_states[:, :split],
            attention_mask=attention_mask[:, :split] if attention_mask is not None else None,
            past_key_values=cache,
            use_cache=True,
        )
        assert returned_cache is cache
        assert cache.get_seq_length(0) == split
        continuation, _, returned_cache = layer(
            hidden_states=hidden_states[:, split:],
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
    assert returned_cache is cache
    assert cache.get_seq_length(0) == T
    assert_close('output', ref, torch.cat((prefill, continuation), dim=1), 0.005)


def test_forward():
    torch.manual_seed(42)
    layer = MLSTM(hidden_size=64, num_heads=2, proj_factor=1).to(device)
    hidden_states = torch.randn(1, 7, 64, device=device, requires_grad=True)
    output, _, _ = layer(hidden_states=hidden_states)
    output.square().mean().backward()
    assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()
    assert layer.fgate.bias.grad is not None and torch.isfinite(layer.fgate.bias.grad).all()


def test_cu_seqlens():
    torch.manual_seed(42)
    layer = MLSTM(hidden_size=64, num_heads=2, proj_factor=1).to(device)
    hidden_states = torch.randn(1, 15, 64, device=device, requires_grad=True)
    cu_seqlens = torch.tensor([0, 1, 8, 15], device=device, dtype=torch.int32)
    attention_mask = torch.ones(1, 15, device=device, dtype=torch.bool)
    attention_mask[:, :3] = False
    ref, _, _ = layer(hidden_states=hidden_states, cu_seqlens=cu_seqlens)
    out, _, _ = layer(hidden_states=hidden_states, attention_mask=attention_mask, cu_seqlens=cu_seqlens)
    do = torch.randn_like(out)
    ref_grad, = torch.autograd.grad((ref * do).sum(), hidden_states)
    grad, = torch.autograd.grad((out * do).sum(), hidden_states)
    assert_close('output', ref, out, 0.005)
    assert_close('dx', ref_grad, grad, 0.005)


def test_cache_requires_layer_idx():
    torch.manual_seed(42)
    layer = MLSTM(hidden_size=64, num_heads=2, proj_factor=1)
    with pytest.raises(ValueError, match='layer_idx'):
        layer(hidden_states=torch.zeros(1, 1, 64), past_key_values=Cache(), use_cache=True)
