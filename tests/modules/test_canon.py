# -*- coding: utf-8 -*-

import os

import pytest
import torch
from einops import rearrange

from fla.modules.canon import canon
from fla.utils import assert_close, device

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


@pytest.mark.parametrize('B', [4])
@pytest.mark.parametrize('T', [1, 100, 500, 1024])
@pytest.mark.parametrize('D', [128, 1024])
@pytest.mark.parametrize('W', [4])
@pytest.mark.parametrize('activation', ['swish', None])
@pytest.mark.parametrize('residual', [True, False])
@pytest.mark.skipif(
    causal_conv1d_fn is None,
    reason="causal_conv1d is not installed"
)
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
def test_canon(B: int, T: int, D: int, W: int, activation: str, residual: bool):
    torch.manual_seed(42)

    x = torch.randn(B, T, D).to(device)
    weight = torch.randn(D, W).to(device)
    bias = torch.randn(D).to(device)

    ref = causal_conv1d_fn(
        x=rearrange(x, "b t d -> b d t"),
        weight=weight,
        bias=bias,
        activation=activation,
    )
    ref = rearrange(ref, "b d t -> b t d")
    if residual:
        ref += x

    tri = canon(x, weight, bias, residual=x if residual else None, activation=activation)
    assert_close(" y", ref, tri, 1e-3)


@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("T", [500, 1024])
@pytest.mark.parametrize('D', [128, 1024])
@pytest.mark.parametrize("W", [4])
@pytest.mark.parametrize("activation", ['swish', None])
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.skipif(
    causal_conv1d_fn is None,
    reason="causal_conv1d is not installed"
)
def test_canon_varlen(N: int, T: int, D: int, W: int, activation: str, residual: bool):
    torch.manual_seed(42)
    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 16)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]

    x = torch.randn(1, T, D).to(device)
    weight = torch.randn(D, W).to(device)
    bias = torch.randn(D).to(device)

    ref = torch.cat([
        rearrange(
            causal_conv1d_fn(
                x=rearrange(x[:, bos:eos].contiguous(), "b t d -> b d t"),
                weight=weight,
                bias=bias,
                activation=activation,
            ),
            "b t d -> b d t"
        ) + (torch.zeros_like(x[:, bos:eos]) if not residual else x[:, bos:eos])
        for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ], 1)

    tri = canon(x, weight, bias, residual=x if residual else None, activation=activation, cu_seqlens=cu_seqlens)
    assert_close("y", ref, tri, 1e-5)
