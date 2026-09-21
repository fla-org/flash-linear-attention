# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import copy

import pytest
import torch

from fla.layers import GatedDeltaNet, GatedDeltaNet2, KimiDeltaAttention
from fla.models.utils import Cache
from fla.utils import IS_NVIDIA, device


@pytest.mark.skipif(not IS_NVIDIA, reason='CUDA synchronization checks require NVIDIA')
@pytest.mark.parametrize('layer_cls', [KimiDeltaAttention, GatedDeltaNet, GatedDeltaNet2])
@torch.no_grad()
def test_short_conv_packed_decode(layer_cls):
    """Packed decode shares its length check and preserves dense outputs and cached states."""
    torch.manual_seed(42)
    layer = layer_cls(hidden_size=128, num_heads=2, head_dim=64, expand_v=1, layer_idx=0)
    layer = layer.to(device=device, dtype=torch.float16).eval()
    x = torch.randn(3, 1, 128, device=device, dtype=torch.float16)
    kwargs = {'cu_seqlens': torch.arange(4, device=device, dtype=torch.int32)}

    cache = Cache()
    layer(x, past_key_values=cache, use_cache=True)
    layer(x.reshape(1, 3, 128), past_key_values=copy.deepcopy(cache), use_cache=True, **kwargs)
    ref_cache, packed_cache = copy.deepcopy(cache), copy.deepcopy(cache)
    ref = layer(x, past_key_values=ref_cache, use_cache=True)[0]

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        actual = layer(x.reshape(1, 3, 128), past_key_values=packed_cache, use_cache=True, **kwargs)[0]

    scalar_reads = sum(event.count for event in prof.key_averages() if event.key == 'aten::_local_scalar_dense')
    assert scalar_reads == 1
    torch.testing.assert_close(actual.reshape_as(ref), ref, atol=0, rtol=0)
    for actual_state, ref_state in zip(packed_cache[0]['conv_state'], ref_cache[0]['conv_state']):
        torch.testing.assert_close(actual_state, ref_state, atol=0, rtol=0)
    torch.testing.assert_close(packed_cache[0]['recurrent_state'], ref_cache[0]['recurrent_state'], atol=0, rtol=0)
