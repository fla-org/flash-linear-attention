# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers import (
    Comba,
    DeltaNet,
    GatedDeltaNet,
    GatedDeltaNet2,
    GatedDeltaProduct,
    GatedLinearAttention,
    GatedSlotAttention,
    HGRN2Attention,
    KimiDeltaAttention,
    MesaNet,
    MultiScaleRetention,
    RodimusAttention,
    YOCOGatedRetention,
)
from fla.layers.attn import Attention
from fla.utils import device

try:
    import flash_attn  # noqa: F401
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False


@pytest.mark.parametrize("B,T,H,D", [(2, 8, 2, 64), (3, 7, 4, 32)])
def test_attention_varlen_accepts_batched_layout_with_cu_seqlens(B: int, T: int, H: int, D: int):
    if not HAS_FLASH:
        pytest.skip(reason="Skipping test because flash-attn is not installed")

    torch.manual_seed(0)
    hidden_size = H * D
    layer = Attention(
        hidden_size=hidden_size,
        num_heads=H,
        num_kv_heads=H,
        qkv_bias=False,
        qk_norm=False,
        window_size=None,
        rope_theta=10000.0,
        max_position_embeddings=None,
        layer_idx=0,
    ).to(device=device, dtype=torch.float16)
    layer.eval()

    hidden_states = torch.randn(B, T, hidden_size, device=device, dtype=torch.float16, requires_grad=True)
    cu_seqlens = torch.arange(0, B * T + 1, T, dtype=torch.int32, device=device)

    out, _, _ = layer(hidden_states=hidden_states, cu_seqlens=cu_seqlens)
    assert out.shape == (B, T, hidden_size)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    ("layer_cls", "layer_kwargs"),
    [
        pytest.param(
            Comba,
            dict(hidden_size=16, head_dim=8, num_heads=2, num_v_heads=2, expand_v=1, use_short_conv=False),
            id="comba",
        ),
        pytest.param(
            DeltaNet,
            dict(hidden_size=16, num_heads=2, expand_k=1, expand_v=1, use_short_conv=False),
            id="delta-net",
        ),
        pytest.param(
            GatedDeltaNet,
            dict(hidden_size=16, head_dim=8, num_heads=2, num_v_heads=2, expand_v=1, use_short_conv=False),
            id="gated-deltanet",
        ),
        pytest.param(
            GatedDeltaNet2,
            dict(hidden_size=16, head_dim=8, num_heads=2, num_v_heads=2, expand_v=1, use_short_conv=False),
            id="gated-deltanet-2",
        ),
        pytest.param(
            GatedDeltaProduct,
            dict(hidden_size=16, head_dim=8, num_heads=2, num_v_heads=2, expand_v=1, use_short_conv=False),
            id="gated-delta-product",
        ),
        pytest.param(
            GatedLinearAttention,
            dict(hidden_size=16, num_heads=2, num_kv_heads=2, expand_k=1, expand_v=1),
            id="gated-linear-attention",
        ),
        pytest.param(
            GatedSlotAttention,
            dict(hidden_size=16, num_heads=2, num_kv_heads=2, num_slots=4, expand_k=1, expand_v=1),
            id="gated-slot-attention",
        ),
        pytest.param(HGRN2Attention, dict(hidden_size=16, num_heads=2, expand_ratio=8), id="hgrn2"),
        pytest.param(
            KimiDeltaAttention,
            dict(hidden_size=16, num_heads=2, num_v_heads=2, head_dim=8, expand_v=1, use_short_conv=False),
            id="kimi-delta-attention",
        ),
        pytest.param(
            MesaNet,
            dict(hidden_size=16, num_heads=2, head_dim=8, use_short_conv=False),
            id="mesa-net",
        ),
        pytest.param(
            MultiScaleRetention,
            dict(hidden_size=16, num_heads=2, num_kv_heads=2, expand_k=1, expand_v=1),
            id="multiscale-retention",
        ),
        pytest.param(
            RodimusAttention,
            dict(hidden_size=16, expand_ratio=8, mode="fused_recurrent", use_short_conv=False),
            id="rodimus",
        ),
        pytest.param(YOCOGatedRetention, dict(hidden_size=16, num_heads=2), id="yoco-gated-retention"),
    ],
)
def test_layer_prefers_explicit_cu_seqlens_over_attention_mask(layer_cls: type, layer_kwargs: dict) -> None:
    torch.manual_seed(42)
    hidden_states = torch.randn(1, 6, 16, device=device, dtype=torch.float16)
    attention_mask = torch.ones(1, 6, device=device, dtype=torch.long)
    padding_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], device=device, dtype=torch.long)
    cu_seqlens = torch.tensor([0, 3, 6], device=device, dtype=torch.int32)
    layer = layer_cls(**layer_kwargs).to(device=device, dtype=torch.float16).eval()

    expected, _, _ = layer(hidden_states=hidden_states, cu_seqlens=cu_seqlens)
    actual, _, _ = layer(hidden_states=hidden_states, attention_mask=attention_mask, cu_seqlens=cu_seqlens)
    masked, _, _ = layer(hidden_states=hidden_states, attention_mask=padding_mask)

    torch.testing.assert_close(actual, expected)
    assert masked.shape == hidden_states.shape
