
import pytest
import torch
import torch.nn.functional as F

from fla.ops.titans.naive import chunk_titans_linear_ref
from fla.utils import assert_close, device


def initialize_chunked_param(B, H, T, BT, dtype=torch.float32):
    # Calculate number of complete chunks and remaining elements
    num_complete_chunks = T // BT
    remainder = T % BT

    # Initialize for complete chunks
    if num_complete_chunks > 0:
        theta_chunks = torch.rand(B, H, num_complete_chunks, 1, dtype=dtype)
        theta_main = theta_chunks.repeat_interleave(
            BT, dim=2,
        )  # Shape: (B, H, num_complete_chunks*BT, 1)
    else:
        theta_main = torch.empty(B, H, 0, 1, dtype=dtype)

    # Handle remaining elements if any
    if remainder > 0:
        theta_remainder = torch.rand(B, H, 1, 1, dtype=dtype)
        theta_remainder = theta_remainder.repeat_interleave(
            remainder, dim=2,
        )  # Shape: (B, H, remainder, 1)

        # Concatenate main chunks with remainder
        theta = torch.cat([theta_main, theta_remainder], dim=2)
    else:
        theta = theta_main

    return theta


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, torch.float16),
            (2, 100, 4, 60, torch.float16),
            (2, 1024, 3, 128, torch.float16),
            (3, 2000, 4, 128, torch.float16),
            (4, 2048, 8, 64, torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    True, reason='FIXME',
)
def test_naive_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    BT = 64
    # set seed
    torch.manual_seed(1)
    # we don't use such initialization in the original code
    # theta = initialize_chunked_param(B, H, T, BT, dtype)
    # alpha = initialize_chunked_param(B, H, T, BT, dtype)
    # eta = initialize_chunked_param(B, H, T, BT, dtype)
    theta = torch.rand(B, H, T, 1, dtype=dtype)
    alpha = torch.rand(B, H, T, 1, dtype=dtype)
    eta = torch.rand(B, H, T, 1, dtype=dtype)

    # titans normalize queries and keys using ℓ2-normalization
    q = F.normalize(torch.randn(B, H, T, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, H, T, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, H, T, D, dtype=dtype)
    w = torch.randn(H, D, dtype=dtype)
    b = torch.randn(H, D, dtype=dtype)
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    theta = theta.permute(0, 2, 1, 3)
    alpha = alpha.permute(0, 2, 1, 3)
    eta = eta.permute(0, 2, 1, 3)
    q, k, v, w, b, theta, alpha, eta = map(
        lambda x: x.to(device).requires_grad_(False), (q, k, v, w, b, theta, alpha, eta),
    )
    # in titans paper, h0 is not learnable
    h0 = h0.to(device)

    ref_naive, ref_ht_naive = chunk_titans_linear_ref(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        b.clone(),
        theta.clone(),
        alpha.clone(),
        eta.clone(),
        output_final_state=True,
        chunk_size=BT,
        initial_state=h0.clone(),
        use_chunk=False,
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
        chunk_size=BT,
        initial_state=h0.clone(),
        use_chunk=True,
    )

    assert_close(" o", ref, ref_naive, 0.006)
    assert_close("ht", ref_ht, ref_ht_naive, 0.005)
