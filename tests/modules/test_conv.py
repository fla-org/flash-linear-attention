# -*- coding: utf-8 -*-

import os

import pytest
import torch

from fla.modules.convolution import ShortConvolution
from fla.ops.common.utils import prepare_lens
from fla.ops.utils.testing import assert_close
from fla.utils import device

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [100, 500, 1])
@pytest.mark.parametrize("H", [128])
@pytest.mark.parametrize("C", [4])
@pytest.mark.skipif(
    causal_conv1d_fn is None,
    reason="causal_conv1d is not installed"
)
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
def test_shortconv(B: int, T: int, H: int, C: int):
    torch.manual_seed(42)
    conv_slow = ShortConvolution(H, C, activation='silu', use_fast_conv1d=False).to(device)
    conv_fast = ShortConvolution(H, C, activation='silu', use_fast_conv1d=True).to(device)
    conv_fast.weight.data.copy_(conv_slow.weight.data)
    if conv_fast.bias is not None:
        conv_fast.bias.data.copy_(conv_slow.bias.data)

    x = torch.randn(B, T, H).to(device)
    y_slow, y_slow_cache = conv_slow(x, output_final_state=True)
    y_fast, y_fast_cache = conv_fast(x, output_final_state=True)
    assert y_slow.shape == x.shape
    assert y_fast.shape == x.shape
    assert torch.allclose(y_slow, y_fast), f"{y_slow}\n{y_fast}"
    assert torch.allclose(y_slow_cache, y_fast_cache), f"{y_slow_cache}\n{y_fast_cache}"


@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("T", [500, 1024])
@pytest.mark.parametrize("H", [128])
@pytest.mark.parametrize("C", [4])
@pytest.mark.skipif(
    causal_conv1d_fn is None,
    reason="causal_conv1d is not installed"
)
def test_shortconv_varlen(N: int, T: int, H: int, C: int):
    torch.manual_seed(42)
    conv = ShortConvolution(H, C, activation='silu', use_fast_conv1d=True).to(device)
    offsets = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 16)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]

    x = torch.randn(1, T, H).to(device)

    ref = torch.cat([conv(x[:, bos:eos].contiguous())[0] for bos, eos in zip(offsets[:-1], offsets[1:])], 1)
    ref_cache = x.new_empty(N, H, C)
    for i, (bos, eos) in enumerate(zip(offsets[:-1], offsets[1:])):
        ref_cache[i] = conv(x[:, bos:eos].contiguous(), output_final_state=True)[1].squeeze(0)
    tri, tri_cache = conv(x, cu_seqlens=offsets, output_final_state=True)
    assert_close("y", ref, tri, 1e-5)
    assert_close("cache", ref_cache, tri_cache, 1e-5)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [100])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("C", [4])
@pytest.mark.skipif(
    causal_conv1d_fn is None,
    reason="causal_conv1d is not installed"
)
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
def test_shortconv_cache(B: int, T: int, H: int, C: int):
    torch.manual_seed(42)
    conv_slow = ShortConvolution(H, C, use_fast_conv1d=False).to(device)
    conv_fast = ShortConvolution(H, C, use_fast_conv1d=True).to(device)
    conv_fast.weight.data.copy_(conv_slow.weight.data)
    if conv_fast.bias is not None:
        conv_fast.bias.data.copy_(conv_slow.bias.data)

    x = torch.randn(B, T, H).to(device)
    mask = torch.randint(T//3, T, (B,)).unsqueeze(-1).gt(torch.arange(T).flip(-1)).bool().to(device)
    y, _ = conv_slow(x, mask=mask)
    cache_slow, cache_fast = None, None
    for i in range(T):
        y_slow, cache_slow = conv_slow(x[:, i:i+1], mask=mask[:, i:i+1], cache=cache_slow, output_final_state=True)
        y_fast, cache_fast = conv_fast(x[:, i:i+1], mask=mask[:, i:i+1], cache=cache_fast, output_final_state=True)

        assert_close(f" slow {i:2}", y_slow, y[:, i:i+1], 1e-5)
        assert_close(f" fast {i:2}", y_fast, y[:, i:i+1], 1e-5)
        assert_close(f"cache {i:2}", cache_slow, cache_fast, 1e-5)


@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("T", [500, 1024, 2048])
@pytest.mark.parametrize("H", [128])
@pytest.mark.parametrize("C", [4])
@pytest.mark.skipif(
    causal_conv1d_fn is None,
    reason="causal_conv1d is not installed"
)
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
def test_shortconv_cache_varlen(N: int, T: int, H: int, C: int):
    torch.manual_seed(42)
    conv_slow = ShortConvolution(H, C, use_fast_conv1d=False).to(device)
    conv_fast = ShortConvolution(H, C, use_fast_conv1d=True).to(device)
    conv_fast.weight.data.copy_(conv_slow.weight.data)
    if conv_fast.bias is not None:
        conv_fast.bias.data.copy_(conv_slow.bias.data)
    offsets = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 16)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]

    x = torch.randn(1, T, H).to(device)
    y, _ = conv_fast(x, cu_seqlens=offsets)
    cache_slow, cache_fast = None, None
    for i in range(min(prepare_lens(offsets))):
        indices = offsets[:-1] + i
        y_slow, cache_slow = conv_slow(x[:, indices].view(-1, 1, H), cache=cache_slow, output_final_state=True)
        y_fast, cache_fast = conv_fast(x[:, indices].view(-1, 1, H), cache=cache_fast, output_final_state=True)

        assert_close(f" slow {i:2}", y_slow.squeeze(), y[:, indices].squeeze(), 1e-5)
        assert_close(f" fast {i:2}", y_fast.squeeze(), y[:, indices].squeeze(), 1e-5)
        assert_close(f"cache {i:2}", cache_slow, cache_fast, 1e-5)
