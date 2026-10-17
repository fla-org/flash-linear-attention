# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.attn import Attention
from fla.models.utils import Cache
from fla.utils import assert_close, device, find_spec_cached

pytestmark = pytest.mark.skipif(find_spec_cached('flash_attn') is None, reason='Attention requires flash-attn')


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('num_kv_heads', [1, 2])
@pytest.mark.parametrize('window_size', [1, 8, None])
@pytest.mark.parametrize('chunks', [(3, 2, 7, 9, 1), (9, 1, 1, 1), (9, 12, 1)])
@torch.no_grad()
def test_attention_cached_chunks(dtype, num_kv_heads, window_size, chunks):
    torch.manual_seed(42)
    layer = Attention(hidden_size=128, num_heads=2, num_kv_heads=num_kv_heads, window_size=window_size, layer_idx=0)
    layer = layer.to(device=device, dtype=dtype).eval()
    hidden_states = torch.randn(2, sum(chunks), 128, device=device, dtype=dtype)
    full_cache = Cache()
    ref = layer(hidden_states=hidden_states, past_key_values=full_cache, use_cache=True)[0]

    cache = Cache()
    outputs = []
    offset = 0
    for length in chunks:
        outputs.append(layer(hidden_states=hidden_states[:, offset:offset+length], past_key_values=cache, use_cache=True)[0])
        offset += length
        assert cache.get_seq_length(0) == offset
        assert cache[0]['attn_state'][0].shape[1] == (offset if window_size is None else min(offset, window_size))
    actual = torch.cat(outputs, dim=1)
    assert torch.isfinite(actual).all()
    tol = 0.002 if dtype == torch.float16 else 0.01
    assert_close('output', ref, actual, tol)
    for ref_state, state in zip(full_cache[0]['attn_state'], cache[0]['attn_state'], strict=True):
        assert torch.isfinite(state).all()
        assert_close('cache', ref_state, state, tol)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_attention_cached_chunks_backward(dtype):
    torch.manual_seed(42)
    layer = Attention(hidden_size=128, num_heads=2, num_kv_heads=1, window_size=8, layer_idx=0)
    layer = layer.to(device=device, dtype=dtype)
    hidden_states = torch.randn(2, 22, 128, device=device, dtype=dtype, requires_grad=True)
    grad = torch.randn_like(hidden_states)
    ref = layer(hidden_states=hidden_states)[0]
    ref.backward(grad)
    ref_grads = [hidden_states.grad.clone(), *(p.grad.clone() for p in layer.parameters())]
    hidden_states.grad = None
    layer.zero_grad(set_to_none=True)

    cache = Cache()
    outputs = []
    offset = 0
    for length in (3, 2, 7, 9, 1):
        outputs.append(layer(hidden_states=hidden_states[:, offset:offset+length], past_key_values=cache, use_cache=True)[0])
        offset += length
    actual = torch.cat(outputs, dim=1)
    actual.backward(grad)
    tol = 0.002 if dtype == torch.float16 else 0.01
    for ref_grad, actual_grad in zip(ref_grads, [hidden_states.grad, *(p.grad for p in layer.parameters())], strict=True):
        assert torch.isfinite(actual_grad).all()
        assert_close('gradient', ref_grad, actual_grad, tol)
