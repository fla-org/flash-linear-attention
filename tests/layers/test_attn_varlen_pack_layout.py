# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers import (
    BitAttention,
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
    MoBA,
    MultiheadLatentAttention,
    MultiScaleRetention,
    NativeSparseAttention,
    Parallax,
    PrecondGatedDeltaNet,
    PrecondKDA,
    RodimusAttention,
    SlidingWindowSharedKeyAttention,
    YOCOCrossAttention,
    YOCOGatedRetention,
    YOCOSharedKVBuilder,
)
from fla.layers.attn import Attention
from fla.models.cache_utils import Cache
from fla.utils import assert_close, device

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
            PrecondGatedDeltaNet,
            dict(hidden_size=16, head_dim=8, num_heads=2, num_v_heads=2, expand_v=1, use_short_conv=False),
            id="precond-gated-deltanet",
        ),
        pytest.param(
            PrecondKDA,
            dict(hidden_size=16, head_dim=16, num_heads=2, num_v_heads=2, expand_v=1, use_short_conv=False),
            id="precond-kda",
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


@pytest.mark.parametrize(
    ("layer_cls", "layer_kwargs"),
    [
        pytest.param(
            NativeSparseAttention,
            dict(hidden_size=64, num_heads=16, num_kv_heads=1, head_dim=64, block_size=32, block_counts=2, window_size=16),
            id="nsa",
        ),
        pytest.param(
            MultiheadLatentAttention,
            dict(hidden_size=64, num_heads=2, q_lora_rank=None, qk_rope_head_dim=16, kv_lora_rank=32,
                 v_head_dim=16, qk_nope_head_dim=16, qk_head_dim=None),
            id="mla",
        ),
        pytest.param(
            SlidingWindowSharedKeyAttention,
            dict(hidden_size=64, num_heads=2, window_size=4),
            id="sliding-window-shared-key",
        ),
        pytest.param(YOCOCrossAttention, dict(hidden_size=64, num_heads=2), id="yoco-cross"),
        pytest.param(Parallax, dict(hidden_size=64, num_heads=2, num_kv_heads=2), id="parallax"),
    ],
)
@torch.no_grad()
def test_attention_packed_with_mask_matches_per_document(layer_cls: type, layer_kwargs: dict) -> None:
    if layer_cls in (NativeSparseAttention, MultiheadLatentAttention, SlidingWindowSharedKeyAttention) and not HAS_FLASH:
        pytest.skip(reason="Skipping test because flash-attn is not installed")

    torch.manual_seed(42)
    hidden_states = torch.randn(1, 12, 64, device=device, dtype=torch.float16)
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device=device, dtype=torch.long)
    cu_seqlens = torch.tensor([0, 5, 12], device=device, dtype=torch.int32)
    layer = layer_cls(**layer_kwargs, layer_idx=0).to(device=device, dtype=torch.float16).eval()
    if layer_cls is YOCOCrossAttention:
        shared_k = torch.randn(1, 12, 2, 32, device=device, dtype=torch.float16)
        shared_v = torch.randn_like(shared_k)

    expected = []
    start = 0
    for length in (5, 7):
        if layer_cls is YOCOCrossAttention:
            output, _ = layer(
                hidden_states=hidden_states[:, start:start+length],
                shared_k=shared_k[:, start:start+length],
                shared_v=shared_v[:, start:start+length],
            )
        else:
            output, _, _ = layer(
                hidden_states=hidden_states[:, start:start+length],
                attention_mask=None,
                past_key_values=Cache(),
                use_cache=True,
            )
        expected.append(output)
        start += length
    expected = torch.cat(expected, dim=1)

    for mask in (None, attention_mask):
        if layer_cls is YOCOCrossAttention:
            actual, _ = layer(
                hidden_states=hidden_states,
                shared_k=shared_k,
                shared_v=shared_v,
                attention_mask=mask,
                cu_seqlens=cu_seqlens,
            )
        else:
            actual, _, _ = layer(
                hidden_states=hidden_states,
                attention_mask=mask,
                past_key_values=Cache(),
                use_cache=True,
                cu_seqlens=cu_seqlens,
            )
        assert_close("packed", expected, actual, 0.005)


@torch.no_grad()
def test_yoco_shared_kv_builder_packed_with_mask_matches_per_document() -> None:
    torch.manual_seed(42)
    hidden_states = torch.randn(1, 12, 64, device=device, dtype=torch.float16)
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device=device, dtype=torch.long)
    cu_seqlens = torch.tensor([0, 5, 12], device=device, dtype=torch.int32)
    builder = YOCOSharedKVBuilder(hidden_size=64, num_heads=2, layer_idx=0).to(device=device, dtype=torch.float16).eval()

    expected_k, expected_v = [], []
    for start, end in ((0, 5), (5, 12)):
        k, v, _ = builder(hidden_states[:, start:end], past_key_values=Cache(), use_cache=True)
        expected_k.append(k)
        expected_v.append(v)
    actual_k, actual_v, _ = builder(
        hidden_states,
        attention_mask=attention_mask,
        past_key_values=Cache(),
        use_cache=True,
        cu_seqlens=cu_seqlens,
    )

    assert_close("k", torch.cat(expected_k, dim=1), actual_k, 0.005)
    assert_close("v", torch.cat(expected_v, dim=1), actual_v, 0.005)


@pytest.mark.parametrize(
    ("layer_cls", "layer_kwargs"),
    [
        pytest.param(Attention, dict(hidden_size=64, num_heads=2, num_kv_heads=2), id="attn"),
        pytest.param(BitAttention, dict(hidden_size=64, num_heads=2, num_kv_heads=2), id="bitattn"),
        pytest.param(
            MoBA,
            dict(hidden_size=64, num_heads=2, num_kv_heads=2, moba_chunk_size=4, moba_topk=2),
            id="moba",
        ),
    ],
)
@torch.no_grad()
def test_rope_packed_with_mask_empty_cache_matches_without_mask(
    layer_cls: type,
    layer_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if layer_cls in (Attention, BitAttention, MoBA) and not HAS_FLASH:
        pytest.skip(reason="Skipping test because flash-attn is not installed")
    if layer_cls is MoBA:
        # Isolate the layer's RoPE path from the MoBA backend's private flash-attn API.
        monkeypatch.setattr("fla.layers.moba.parallel_moba", lambda q, *args, **kwargs: q)

    torch.manual_seed(42)
    hidden_states = torch.randn(1, 12, 64, device=device, dtype=torch.float16)
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device=device, dtype=torch.long)
    cu_seqlens = torch.tensor([0, 5, 12], device=device, dtype=torch.int32)
    layer = layer_cls(**layer_kwargs, layer_idx=0).to(device=device, dtype=torch.float16).eval()

    expected, _, _ = layer(
        hidden_states=hidden_states,
        attention_mask=None,
        past_key_values=Cache(),
        use_cache=True,
        cu_seqlens=cu_seqlens,
    )
    actual, _, _ = layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=Cache(),
        use_cache=True,
        cu_seqlens=cu_seqlens,
    )

    assert_close("output", expected, actual, 0.005)


@pytest.mark.parametrize(
    ("layer_cls", "layer_kwargs"),
    [
        pytest.param(
            MultiScaleRetention,
            dict(hidden_size=16, num_heads=2, num_kv_heads=2, expand_k=1, expand_v=1),
            id="multiscale-retention",
        ),
        pytest.param(YOCOGatedRetention, dict(hidden_size=16, num_heads=2), id="yoco-gated-retention"),
    ],
)
@torch.no_grad()
def test_retention_cached_packed_with_mask_matches_without_mask(layer_cls: type, layer_kwargs: dict) -> None:
    torch.manual_seed(42)
    prefill = torch.randn(2, 4, 16, device=device, dtype=torch.float16)
    hidden_states = torch.randn(1, 6, 16, device=device, dtype=torch.float16)
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1]], device=device, dtype=torch.long)
    cu_seqlens = torch.tensor([0, 2, 6], device=device, dtype=torch.int32)
    layer = layer_cls(**layer_kwargs, layer_idx=0).to(device=device, dtype=torch.float16).eval()
    expected_cache, actual_cache = Cache(), Cache()
    layer(hidden_states=prefill, past_key_values=expected_cache, use_cache=True)
    layer(hidden_states=prefill, past_key_values=actual_cache, use_cache=True)

    expected, _, _ = layer(
        hidden_states=hidden_states,
        attention_mask=None,
        past_key_values=expected_cache,
        use_cache=True,
        cu_seqlens=cu_seqlens,
    )
    actual, _, _ = layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=actual_cache,
        use_cache=True,
        cu_seqlens=cu_seqlens,
    )

    assert_close("output", expected, actual, 0.005)
    assert_close("state", expected_cache[0]['recurrent_state'], actual_cache[0]['recurrent_state'], 0.005)


@pytest.mark.parametrize(
    ("layer_cls", "layer_kwargs"),
    [
        pytest.param(
            MultiScaleRetention,
            dict(hidden_size=16, num_heads=2, num_kv_heads=2, expand_k=1, expand_v=1),
            id="multiscale-retention",
        ),
        pytest.param(YOCOGatedRetention, dict(hidden_size=16, num_heads=2), id="yoco-gated-retention"),
    ],
)
@torch.no_grad()
def test_retention_left_padded_cached_decode_matches_unpadded(layer_cls: type, layer_kwargs: dict) -> None:
    torch.manual_seed(42)
    prefill = torch.randn(2, 4, 16, device=device, dtype=torch.float16)
    next_hidden = torch.randn(2, 1, 16, device=device, dtype=torch.float16)
    prefill_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]], device=device, dtype=torch.long)
    decode_mask = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]], device=device, dtype=torch.long)
    layer = layer_cls(**layer_kwargs, layer_idx=0).to(device=device, dtype=torch.float16).eval()
    cache = Cache()
    layer(hidden_states=prefill, attention_mask=prefill_mask, past_key_values=cache, use_cache=True)
    actual, _, _ = layer(
        hidden_states=next_hidden,
        attention_mask=decode_mask,
        past_key_values=cache,
        use_cache=True,
    )

    expected, expected_state = [], []
    for batch, start in ((0, 1), (1, 0)):
        reference_cache = Cache()
        layer(hidden_states=prefill[batch:batch+1, start:], past_key_values=reference_cache, use_cache=True)
        output, _, _ = layer(
            hidden_states=next_hidden[batch:batch+1],
            past_key_values=reference_cache,
            use_cache=True,
        )
        expected.append(output)
        expected_state.append(reference_cache[0]['recurrent_state'])

    assert_close("output", torch.cat(expected, dim=0), actual, 0.005)
    assert_close("state", torch.cat(expected_state, dim=0), cache[0]['recurrent_state'], 0.005)
