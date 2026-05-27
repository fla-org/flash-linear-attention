# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Correctness tests for the GDN-2 (Gated DeltaNet 2) ops.
#
# These tests pin the Triton chunkwise / fused-recurrent kernels against the
# pure-PyTorch ``naive_recurrent_gdn2`` reference. The reference is the ground
# truth: it is a direct transcription of the equation
#     S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T
# with no chunking, no WY trick, no fused gates - so any kernel that disagrees
# with it is wrong, full stop.

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_net_v2 import chunk_gdn2, fused_recurrent_gdn2, naive_recurrent_gdn2


def _make_inputs(B, T, H, K, V, dtype, device, seed=0):
    """Generate well-conditioned inputs for the recurrence.

    L2-normalize q and k so the rank-1 updates don't compound to astronomical
    magnitudes across hundreds of tokens; clamp the log-decay to ``[-5, -0.1]``
    so the state actually contracts on every step rather than drifting.
    Without these, fp32 accumulation drift after a few hundred steps swamps any
    reasonable tolerance even though the two implementations agree to 0.1%.
    """
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, K, dtype=dtype, device=device), p=2, dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, dtype=dtype, device=device), p=2, dim=-1)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.5
    g = torch.empty(B, T, H, K, device=device, dtype=torch.float32).uniform_(-5.0, -0.1).to(dtype)
    b = torch.rand(B, T, H, K, dtype=dtype, device=device)
    w = torch.rand(B, T, H, V, dtype=dtype, device=device)
    return q, k, v, g, b, w


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("B,T,H,K,V", [(1, 64, 2, 32, 32), (2, 128, 2, 64, 64)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_fused_recurrent_matches_naive(B, T, H, K, V, dtype):
    """The fused-recurrent Triton kernel is the un-chunked update rule in
    Triton form, so it should match the naive reference to within fp32 noise."""
    device = "cuda"
    q, k, v, g, b, w = _make_inputs(B, T, H, K, V, dtype, device)

    o_ref, ht_ref = naive_recurrent_gdn2(
        q, k, v, g, b, w,
        output_final_state=True,
    )
    o, ht = fused_recurrent_gdn2(
        q, k, v, g, b, w,
        output_final_state=True,
    )

    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ht, ht_ref.to(ht.dtype), rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("B,T,H,K,V", [(1, 64, 2, 32, 32), (2, 128, 2, 64, 64)])
def test_chunk_matches_naive(B, T, H, K, V):
    """The chunkwise Triton kernel uses the WY blocked algorithm, which is a
    different way of computing the same recurrence. Looser tolerances because
    of the fp16/bf16 matmul precision in the WY solve."""
    device = "cuda"
    dtype = torch.float32
    q, k, v, g, b, w = _make_inputs(B, T, H, K, V, dtype, device)

    o_ref, ht_ref = naive_recurrent_gdn2(
        q, k, v, g, b, w,
        output_final_state=True,
    )
    o, ht = chunk_gdn2(
        q, k, v, g, b, w,
        output_final_state=True,
    )

    # The chunkwise path runs many of its matmuls in tf32/bf16 inside Triton;
    # use generous fp32 tolerances appropriate for a reference comparison.
    torch.testing.assert_close(o, o_ref, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(ht.to(torch.float32), ht_ref.to(torch.float32), rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_chunk_backward_runs():
    """End-to-end backward smoke test: we don't compare gradients to a numeric
    reference here (too costly), but the autograd path must run without raising
    and produce gradients of the right shape and dtype for every input."""
    device = "cuda"
    dtype = torch.float32
    B, T, H, K, V = 1, 64, 2, 32, 32
    q, k, v, g, b, w = _make_inputs(B, T, H, K, V, dtype, device)
    for t in (q, k, v, g, b, w):
        t.requires_grad_(True)

    o, _ = chunk_gdn2(q, k, v, g, b, w, output_final_state=False)
    o.sum().backward()

    for name, t in [("q", q), ("k", k), ("v", v), ("g", g), ("b", b), ("w", w)]:
        assert t.grad is not None, f"{name}.grad is None"
        assert t.grad.shape == t.shape, f"{name}.grad has wrong shape"
        assert torch.isfinite(t.grad).all(), f"{name}.grad contains non-finite values"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_recurrent_with_initial_state():
    """initial_state should be propagated correctly across the recurrence."""
    device = "cuda"
    dtype = torch.float32
    B, T, H, K, V = 1, 32, 2, 32, 32
    q, k, v, g, b, w = _make_inputs(B, T, H, K, V, dtype, device)
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)

    o_ref, ht_ref = naive_recurrent_gdn2(
        q, k, v, g, b, w,
        initial_state=h0, output_final_state=True,
    )
    o, ht = fused_recurrent_gdn2(
        q, k, v, g, b, w,
        initial_state=h0, output_final_state=True,
    )
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ht, ht_ref.to(ht.dtype), rtol=1e-3, atol=1e-3)
