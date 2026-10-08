# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.mlstm import MLSTM
from fla.models.utils import Cache
from fla.utils import device


@pytest.mark.parametrize('mode', ['chunk', 'fused_recurrent'])
@pytest.mark.parametrize('scenario', ['dense_prefill', 'masked_prefill', 'padded_prefill', 'masked_decode'])
def test_mlstm_cache_offset(mode: str, scenario: str):
    torch.manual_seed(42)
    B, D = 2, 64
    T = 1 if scenario == 'masked_decode' else 4
    layer = MLSTM(
        hidden_size=D,
        num_heads=2,
        proj_factor=1,
        mode=mode,
        layer_idx=0,
    ).to(device=device, dtype=torch.float32).eval()
    hidden_states = torch.randn(B, T, D, device=device)
    attention_mask = None
    if scenario != 'dense_prefill':
        attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
        if scenario == 'padded_prefill':
            attention_mask[0, :2] = 0
    cache = Cache()

    with torch.no_grad():
        if scenario == 'masked_decode':
            layer(
                hidden_states=torch.randn(B, 3, D, device=device),
                past_key_values=cache,
                use_cache=True,
            )
            assert cache.get_seq_length(0) == 3
        previous_length = cache.get_seq_length(0)
        output, _, returned_cache = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )

    assert returned_cache is cache
    assert output.shape == hidden_states.shape
    assert cache.get_seq_length(0) == previous_length + T
