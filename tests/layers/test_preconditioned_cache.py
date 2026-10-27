# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.precond_gated_deltanet import PrecondGatedDeltaNet
from fla.layers.precond_kda import PrecondKDA
from fla.models.cache_utils import Cache, FLACache
from fla.utils import assert_close, device_name


@pytest.mark.skipif(device_name == 'cpu', reason='Preconditioned layers require an accelerator.')
@pytest.mark.parametrize('layer_class', [PrecondGatedDeltaNet, PrecondKDA], ids=['gdn', 'kda'])
@pytest.mark.parametrize('packed', [False, True], ids=['dense', 'packed'])
@torch.no_grad()
def test_preconditioned_cache_continuation(layer_class, packed):
    """Composite recurrent state survives chunk/decode transitions, transfer, and cache conversion."""
    torch.manual_seed(42)
    batch_size, seq_len, hidden_size = 2, 131, 128
    layer = layer_class(
        hidden_size=hidden_size,
        num_heads=2,
        head_dim=64,
        expand_v=1,
        layer_idx=0,
    ).to(device=device_name, dtype=torch.float16).eval()
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device_name, dtype=torch.float16)

    def forward(inputs, cache):
        length = inputs.shape[1]
        cu_seqlens = None
        if packed:
            inputs = inputs.reshape(1, batch_size * length, hidden_size)
            cu_seqlens = torch.arange(0, (batch_size + 1) * length, length, device=device_name, dtype=torch.int32)
        output, _, _ = layer(hidden_states=inputs, past_key_values=cache, use_cache=True, cu_seqlens=cu_seqlens)
        return output.reshape(batch_size, length, hidden_size)

    reference_cache = Cache()
    expected = forward(hidden_states, reference_cache)
    cache = Cache()
    outputs = []
    for start, end in ((0, 65), (65, 130), (130, 131)):
        outputs.append(forward(hidden_states[:, start:end], cache))
        state = cache[0]
        assert set(state) == {'recurrent_state', 'attn_state', 'conv_state', 'ffn_state'}
        assert isinstance(state['recurrent_state'], tuple)
        assert len(state['recurrent_state']) == 2
        recurrent_state, preconditioner_state = state['recurrent_state']
        assert recurrent_state.shape == (batch_size, 2, 64, 64)
        assert preconditioner_state.shape == (batch_size, 2, 64)
        if isinstance(cache, FLACache):
            cache.layers[0].offload()
            cache.layers[0].prefetch()
            torch.testing.assert_close(cache[0]['recurrent_state'], (recurrent_state, preconditioner_state), rtol=0, atol=0)
        cache = Cache.from_legacy_cache(cache.to_legacy_cache(), seen_tokens=cache.get_seq_length())

    assert_close('output', expected, torch.cat(outputs, dim=1), 2e-3)
    for name, expected_state, actual_state in zip(
        ('recurrent_state', 'preconditioner_state'),
        reference_cache[0]['recurrent_state'],
        cache[0]['recurrent_state'],
        strict=True,
    ):
        assert_close(name, expected_state, actual_state, 2e-3)
    assert cache.get_seq_length() == reference_cache.get_seq_length()
