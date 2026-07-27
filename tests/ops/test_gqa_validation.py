# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.attn.naive import naive_attn_decoding, naive_parallel_attn
from fla.ops.attn.parallel import parallel_attn
from fla.ops.forgetting_attn.naive import naive_forgetting_attn
from fla.ops.parallax.naive import naive_parallax
from fla.ops.parallax.parallel import parallel_parallax
from fla.utils import device

# ── Non-divisible GQA heads must raise ──────────────────────────────────────


@pytest.mark.parametrize(
    ('op_name', 'op_fn', 'HQ', 'H', 'kw'),
    [
        pytest.param("naive_parallel_attn", naive_parallel_attn, 3, 2, {}, id="naive_parallel_attn"),
        pytest.param("parallel_attn", parallel_attn, 3, 2, {}, id="parallel_attn"),
    ],
)
def test_gqa_validation_attn(op_name, op_fn, HQ, H, kw):
    """Non-divisible HQ/H must raise ValueError."""
    B, T, D = 1, 8, 64
    q = torch.randn(B, T, HQ, D, device=device)
    k = torch.randn(B, T, H, D, device=device)
    v = torch.randn(B, T, H, D, device=device)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        op_fn(q=q, k=k, v=v, **kw)


@pytest.mark.parametrize(
    ('op_name', 'op_fn', 'HQ', 'H', 'kw'),
    [
        pytest.param("naive_forgetting_attn", naive_forgetting_attn, 3, 2, {}, id="naive_forgetting_attn"),
    ],
)
def test_gqa_validation_forgetting(op_name, op_fn, HQ, H, kw):
    B, T, D = 1, 8, 64
    q = torch.randn(B, T, HQ, D, device=device)
    k = torch.randn(B, T, H, D, device=device)
    v = torch.randn(B, T, H, D, device=device)
    g = torch.randn(B, T, HQ, device=device)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        op_fn(q=q, k=k, v=v, g=g, **kw)


@pytest.mark.parametrize(
    ('op_name', 'op_fn', 'HQ', 'H', 'kw'),
    [
        pytest.param("naive_parallax", naive_parallax, 3, 2, {}, id="naive_parallax"),
        pytest.param("parallel_parallax", parallel_parallax, 3, 2, {}, id="parallel_parallax"),
    ],
)
def test_gqa_validation_parallax(op_name, op_fn, HQ, H, kw):
    B, T, D = 1, 8, 64
    q = torch.randn(B, T, HQ, D, device=device, dtype=torch.float16)
    r = torch.randn(B, T, HQ, D, device=device, dtype=torch.float16)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.float16)
    v = torch.randn(B, T, H, D, device=device, dtype=torch.float16)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        op_fn(q=q, r=r, k=k, v=v, **kw)


# ── Decoding entry points ────────────────────────────────────────────────────

def test_gqa_validation_attn_decoding():
    """attn_decoding_one_step with non-divisible heads must raise."""
    B, T, HQ, H, K, V = 2, 32, 3, 2, 64, 64
    cu_seqlens = torch.tensor([0, 16, 32], dtype=torch.long, device=device)
    q = torch.randn(1, B, HQ, K, device=device)
    k = torch.randn(1, T, H, K, device=device)
    v = torch.randn(1, T, H, V, device=device)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        attn_decoding_one_step(q=q, k=k, v=v, cu_seqlens=cu_seqlens)


def test_gqa_validation_naive_attn_decoding():
    """naive_attn_decoding with non-divisible heads must raise."""
    B, T, HQ, H, D = 2, 32, 3, 2, 64
    cu_seqlens = torch.tensor([0, 16, 32], dtype=torch.long, device=device)
    q = torch.randn(1, B, HQ, D, device=device)
    k = torch.randn(1, T, H, D, device=device)
    v = torch.randn(1, T, H, D, device=device)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        naive_attn_decoding(q=q, k=k, v=v, cu_seqlens=cu_seqlens)


# ── Valid composite GQA configurations must still work ─────────────────────

@pytest.mark.parametrize(
    ('HQ', 'H', 'G'),
    [
        (4, 2, 2),   # G=2
        (8, 2, 4),   # G=4
    ],
)
def test_gqa_valid_attn_forward(HQ, H, G):
    """Valid divisible GQA should not raise for parallel_attn."""
    B, T, D = 1, 8, 64
    q = torch.randn(B, T, HQ, D, device=device, dtype=torch.float16)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.float16)
    v = torch.randn(B, T, H, D, device=device, dtype=torch.float16)
    o = parallel_attn(q=q, k=k, v=v)
    assert o.shape == (B, T, HQ, D), f"Expected {(B, T, HQ, D)}, got {o.shape}"
