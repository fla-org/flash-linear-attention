# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.simple_gla import SimpleGatedLinearAttention
from fla.models.utils import Cache
from fla.utils import assert_close, device


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_forward_ignores_padded_gate_content(dtype: torch.dtype):
    B, T, H, D = 2, 128, 4, 64
    PAD = 3
    torch.manual_seed(42)
    layer = SimpleGatedLinearAttention(
        hidden_size=H * D,
        num_heads=H,
        use_short_conv=False,
        layer_idx=0,
    ).to(device=device, dtype=dtype).eval()

    hidden_states = torch.randn(B, T, H * D, device=device, dtype=dtype)
    # Two batches that are identical everywhere except inside the padded span: only
    # what the padding tokens happen to embed to differs.
    varied = hidden_states.clone()
    varied[:, 10:10 + PAD] = torch.randn(B, PAD, H * D, device=device, dtype=dtype)
    mask = torch.ones(B, T, dtype=torch.long, device=device)
    mask[:, 10:10 + PAD] = 0

    with torch.no_grad():
        out_a, *_ = layer(hidden_states=hidden_states, attention_mask=mask)
        out_b, *_ = layer(hidden_states=varied, attention_mask=mask)

    tol = 0.005 if dtype == torch.float16 else 0.02
    # v is masked to zero already, so only the gate can carry the padded content
    # forward. Every token from the pad onward (both inside the recurrent state and
    # in the outputs) must be identical once the gate is masked the same way.
    assert_close("post-pad", out_a[:, 10 + PAD:], out_b[:, 10 + PAD:], tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cached_state_ignores_padded_gate_content(dtype: torch.dtype):
    # A brand-new cache's recurrent_state is all zeros, so an unmasked-but-wrong decay
    # applied while the state is still zero has nothing to corrupt (0 * garbage == 0).
    # The leak only shows up once the state already holds real content, i.e. padding
    # that arrives after a warm prefill, so warm the cache first and feed the padded
    # span as its own call before continuing.
    B, H, D = 2, 4, 64
    PAD = 3
    torch.manual_seed(42)
    layer = SimpleGatedLinearAttention(
        hidden_size=H * D,
        num_heads=H,
        use_short_conv=False,
        layer_idx=0,
    ).to(device=device, dtype=dtype).eval()

    warm = torch.randn(B, 16, H * D, device=device, dtype=dtype)
    padded = torch.randn(B, PAD, H * D, device=device, dtype=dtype)
    mask = torch.zeros(B, PAD, dtype=torch.long, device=device)
    continuation = torch.randn(B, 5, H * D, device=device, dtype=dtype)

    with torch.no_grad():
        cache_a = Cache()
        layer(hidden_states=warm, past_key_values=cache_a, use_cache=True)
        layer(hidden_states=padded, attention_mask=mask, past_key_values=cache_a, use_cache=True)
        out_a, *_ = layer(hidden_states=continuation, past_key_values=cache_a, use_cache=True)

        # a second padded step that differs only in what the padding tokens embed to
        padded_b = torch.randn(B, PAD, H * D, device=device, dtype=dtype)
        cache_b = Cache()
        layer(hidden_states=warm, past_key_values=cache_b, use_cache=True)
        layer(hidden_states=padded_b, attention_mask=mask, past_key_values=cache_b, use_cache=True)
        out_b, *_ = layer(hidden_states=continuation, past_key_values=cache_b, use_cache=True)

    tol = 0.005 if dtype == torch.float16 else 0.02
    # the cached recurrent state after the padded step must not depend on what the
    # padding tokens happened to embed to, so a later continuation must match too.
    assert_close("continuation-after-padded-step", out_a, out_b, tol)
