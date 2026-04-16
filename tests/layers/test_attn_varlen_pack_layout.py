import pytest
import torch

from fla.layers.attn import Attention
from fla.utils import device


try:
    HAS_FLASH = True
except Exception:
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
