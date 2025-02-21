# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.ops.titans.chunk import chunk_titans_linear
from fla.ops.titans.naive import chunk_titans_linear_ref


def get_abs_err(x, y):
    return (x - y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x - y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def assert_close(prefix, ref, tri, ratio):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    assert get_err_ratio(ref, tri) < ratio, msg


def initialize_chunked_param(B, H, T, BT, dtype=torch.float32):
    # Calculate number of complete chunks and remaining elements
    num_complete_chunks = T // BT
    remainder = T % BT

    # Initialize for complete chunks
    if num_complete_chunks > 0:
        theta_chunks = torch.rand(B, H, num_complete_chunks, 1, dtype=dtype)
        theta_main = theta_chunks.repeat_interleave(BT, dim=2)  # Shape: (B, H, num_complete_chunks*BT, 1)
    else:
        theta_main = torch.empty(B, H, 0, 1, dtype=dtype)

    # Handle remaining elements if any
    if remainder > 0:
        theta_remainder = torch.rand(B, H, 1, 1, dtype=dtype)
        theta_remainder = theta_remainder.repeat_interleave(remainder, dim=2)  # Shape: (B, H, remainder, 1)

        # Concatenate main chunks with remainder
        theta = torch.cat([theta_main, theta_remainder], dim=2)
    else:
        theta = theta_main

    return theta


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [15, 16, 63, 64, 256, 512])
@pytest.mark.parametrize("H", [2, 16])
@pytest.mark.parametrize("D", [30, 64, 100])
@pytest.mark.parametrize("scale", [1])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("head_first", [True, False])
def test_chunk_fwd(
        B: int,
        T: int,
        H: int,
        D: int,
        dtype: torch.dtype,
        scale: float,
        head_first: bool
):
    BT = 16
    # in titans paper, they use the same value for each of alpha, theta, and eta in each chunk, so we initialize them here
    theta = initialize_chunked_param(B, H, T, BT, dtype)
    alpha = initialize_chunked_param(B, H, T, BT, dtype)
    eta = initialize_chunked_param(B, H, T, BT, dtype)
    if head_first:
        # titans normalize queries and keys using ℓ2-normalization
        q = F.normalize(torch.randn(B, H, T, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
        k = F.normalize(torch.randn(B, H, T, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
        v = torch.randn(B, H, T, D, dtype=dtype)
        w = torch.randn(H, D, dtype=dtype)
        b = torch.randn(H, D, dtype=dtype)
        h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    else:
        q = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
        k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
        v = torch.randn(B, T, H, D, dtype=dtype)
        w = torch.randn(H, D, dtype=dtype)
        b = torch.randn(H, D, dtype=dtype)
        h0 = torch.randn(B, H, D, D, dtype=torch.float32)
        # we need to reshape here because head_first is True
        theta = theta.permute(0, 2, 1, 3)
        alpha = alpha.permute(0, 2, 1, 3)
        eta = eta.permute(0, 2, 1, 3)
    q, k, v, w, b, theta, alpha, eta = map(lambda x: x.cuda().requires_grad_(True),
                                           (q, k, v, w, b, theta, alpha, eta))
    # in titans paper, h0 is not learnable
    h0 = h0.cuda()

    tri, tri_ht = chunk_titans_linear(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        b.clone(),
        theta.clone(),
        alpha.clone(),
        eta.clone(),
        output_final_state=True,
        BT=BT,
        initial_state=h0.clone(),
        head_first=head_first
    )

    ref, ref_ht = chunk_titans_linear_ref(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        b.clone(),
        theta.clone(),
        alpha.clone(),
        eta.clone(),
        output_final_state=True,
        BT=BT,
        initial_state=h0.clone(),
        head_first=head_first
    )

    assert_close(" o", ref, tri, 0.006)
    assert_close("ht", ref_ht, tri_ht, 0.005)
