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


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('T_continue', [4, 80])
@pytest.mark.parametrize('use_short_conv', [False, True])
def test_warm_cache_ignores_left_padding_content(dtype: torch.dtype, T_continue: int, use_short_conv: bool):
    torch.manual_seed(42)
    layer = LightNetAttention(
        hidden_size=64, num_heads=2, expand_ratio=8, use_short_conv=use_short_conv, layer_idx=0,
    ).to(device=device, dtype=dtype).eval()
    prefill = torch.randn(2, 8, 64, device=device, dtype=dtype)
    continuation = torch.randn(2, T_continue, 64, device=device, dtype=dtype)
    mask = torch.ones(2, T_continue, device=device, dtype=torch.bool)
    mask[:, :3] = False
    changed = continuation.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 5
    outputs, states = [], []
    with torch.no_grad():
        for current in (continuation, changed):
            cache = Cache()
            layer(hidden_states=prefill, past_key_values=cache, use_cache=True)
            output, _, _ = layer(hidden_states=current, attention_mask=mask, past_key_values=cache, use_cache=True)
            outputs.append(output[mask])
            states.append(cache[0]['ffn_state'].clone())
    assert torch.isfinite(outputs[0]).all()
    assert torch.isfinite(states[0]).all()
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(states[0], states[1])


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_warm_cache_all_padding_keeps_state(dtype: torch.dtype):
    torch.manual_seed(42)
    layer = LightNetAttention(hidden_size=64, num_heads=2, expand_ratio=8, layer_idx=0).to(device=device, dtype=dtype).eval()
    prefill = torch.randn(2, 8, 64, device=device, dtype=dtype)
    padding = torch.randn(2, 3, 64, device=device, dtype=dtype)
    continuation = torch.randn(2, 5, 64, device=device, dtype=dtype)
    mask = torch.zeros(2, 3, device=device, dtype=torch.bool)
    outputs = []
    with torch.no_grad():
        for skip_padding in (True, False):
            cache = Cache()
            layer(hidden_states=prefill, past_key_values=cache, use_cache=True)
            if not skip_padding:
                old_normalizer = cache[0]['ffn_state'].clone()
                old_recurrent_state = cache[0]['recurrent_state'].clone()
                layer(hidden_states=padding, attention_mask=mask, past_key_values=cache, use_cache=True)
                assert torch.equal(cache[0]['ffn_state'], old_normalizer)
                assert torch.equal(cache[0]['recurrent_state'], old_recurrent_state)
            output, _, _ = layer(hidden_states=continuation, past_key_values=cache, use_cache=True)
            outputs.append(output)
    assert torch.isfinite(outputs[0]).all()
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('T_continue', [4, 80])
def test_warm_cache_padding_has_zero_gradient(dtype: torch.dtype, T_continue: int):
    torch.manual_seed(42)
    layer = LightNetAttention(hidden_size=64, num_heads=2, expand_ratio=8, layer_idx=0).to(device=device, dtype=dtype).train()
    prefill = torch.randn(2, 8, 64, device=device, dtype=dtype)
    continuation = torch.randn(2, T_continue, 64, device=device, dtype=dtype, requires_grad=True)
    mask = torch.ones(2, T_continue, device=device, dtype=torch.bool)
    mask[:, :3] = False
    cache = Cache()
    with torch.no_grad():
        layer(hidden_states=prefill, past_key_values=cache, use_cache=True)
    output, _, _ = layer(hidden_states=continuation, attention_mask=mask, past_key_values=cache, use_cache=True)
    valid_output = output[mask]
    (valid_output.float() * torch.randn_like(valid_output)).sum().backward()
    assert torch.isfinite(continuation.grad).all()
    assert torch.count_nonzero(continuation.grad[~mask]) == 0
    assert torch.count_nonzero(continuation.grad[mask]) > 0


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('T_continue', [4, 80])
def test_warm_cache_padding_before_first_valid_token(dtype: torch.dtype, T_continue: int):
    torch.manual_seed(42)
    layer = LightNetAttention(hidden_size=64, num_heads=2, expand_ratio=8, layer_idx=0).to(device=device, dtype=dtype).train()
    prefill = torch.randn(2, 8, 64, device=device, dtype=dtype)
    continuation = torch.randn(2, T_continue, 64, device=device, dtype=dtype, requires_grad=True)
    mask = torch.ones(2, T_continue, device=device, dtype=torch.bool)
    mask[:, :3] = False
    cache = Cache()
    with torch.no_grad():
        layer(hidden_states=prefill, attention_mask=torch.zeros(2, 8, device=device, dtype=torch.bool),
              past_key_values=cache, use_cache=True)
        expected, _, _ = layer(hidden_states=continuation[:, 3:].detach(), use_cache=False)
    output, _, _ = layer(hidden_states=continuation, attention_mask=mask, past_key_values=cache, use_cache=True)
    valid_output = output[:, 3:]
    assert torch.isfinite(valid_output).all()
    assert torch.isfinite(cache[0]['recurrent_state']).all()
    assert_close('empty-prefix continuation', expected, valid_output, 2e-2 if dtype == torch.bfloat16 else 2e-3)
    (valid_output.float() * torch.randn_like(valid_output)).sum().backward()
    assert torch.isfinite(continuation.grad).all()
    assert torch.count_nonzero(continuation.grad[~mask]) == 0
    assert torch.count_nonzero(continuation.grad[mask]) > 0
