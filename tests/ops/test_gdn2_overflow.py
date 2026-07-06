# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Overflow reproduction and regression tests for GDN-2 backward.
#
# Root-cause summary: a drifted `A_log` can make per-chunk decay vanish,
# letting the recurrent state grow across long sequences until backward
# `tl.dot` accumulators overflow. The layer-level floor bounds the normal path;
# kernel finite guards are a backstop for op-level paths that bypass the layer.
#
# Fixes:
#   (a) A_log floor (fla/layers/gdn2.py): clamp(min=a_log_min) on A_log
#       before exp(). Guarantees exp(A_log) >= 1, so per-chunk decay is
#       non-trivial and state growth is bounded over 2048 chunks. This
#       breaks the chain at link #2 regardless of what triggers it.
#   (c) Kernel overflow guards (chunk_bwd.py, chunk_delta_h.py,
#       kda/chunk_bwd.py, gdn2/chunk_intra.py): zero inf/nan after each
#       tl.dot in the backward kernels. Backstop only — masks residual
#       overflow that the floor misses. Cannot prevent overflow inside
#       tl.dot (the partial sum overflows before the guard sees the
#       result), but zeros the corrupted gradient so it does not
#       propagate further.
#
# Test structure:
#   A. Layer-level (real training path) — validates fix (a):
#      GatedDeltaNet2 with A_log=-200 (simulates drifted A_log), large do.
#      The floor keeps exp(A_log) >= 1, bounding the state. Without the
#      floor, the state grows and gradients overflow at do >= 1e37.
#   B. Op-level stress (bypasses layer floor) — validates fix (c):
#      chunk_gdn2 with use_gate_in_kernel=True, A_log=-200, extreme do.
#      Only kernel guards can keep grads finite. Requires guards in ALL
#      backward kernels (chunk_kda_bwd_dAv, chunk_delta_h, chunk_bwd,
#      chunk_intra) because inf propagates through the pipeline.
#   C. Regression: A_log floor behaviour — A_log.grad == 0 below floor.

import pytest
import torch

from fla.ops.gdn2 import chunk_gdn2
from fla.utils import device


def _assert_all_finite(named_tensors):
    """Assert every (name, tensor) pair is finite; report the first failure."""
    for name, t in named_tensors:
        if t is None:
            continue
        if not torch.isfinite(t).all():
            n_bad = int((~torch.isfinite(t)).sum().item())
            raise AssertionError(
                f"{name} has {n_bad}/{t.numel()} non-finite values"
            )


# =============================================================================
# A. Layer-level overflow (validates fix (a) — A_log floor)
#
# Simulates the training scenario: A_log has drifted to -200 (exp ≈ 0,
# zero decay), state would grow over 2048 chunks. The floor clamps
# A_log to 0.0, so exp(A_log) >= 1, decay is non-trivial, state is bounded.
#
# Without the floor, the layer overflows at do_scale >= 1e37 (fp32 max
# in the WY inverse VJP and state-backward dots). With the floor, it
# stays finite up to do_scale = 1e35 — well beyond the 1.57M× loss
# weight amplification (~1e6 × state magnitude) seen in real training.
# =============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_overflow_layer_fp32_128k():
    """T=131072 (2048 chunks), fp32, A_log=-200, do_scale=1e15.

    The floor clamps exp(A_log) >= 1, bounding state growth. Without the
    floor, the unbounded state × 1e15 do overflows the fp32 accumulator
    in the state-backward dots (dot(k, dh), dot(q, do)).
    """
    from fla.layers import GatedDeltaNet2

    torch.manual_seed(42)
    B, T, hidden_size, head_dim, num_heads = 1, 131072, 128, 32, 2
    do_scale = 1e15

    layer = GatedDeltaNet2(
        hidden_size=hidden_size, head_dim=head_dim, num_heads=num_heads,
        use_short_conv=False,
    ).to(device).to(torch.float32)
    layer.A_log.data.fill_(-200.0)
    layer.train()

    x = torch.randn(B, T, hidden_size, device=device, dtype=torch.float32, requires_grad=True)
    o, _, _ = layer(x)
    assert torch.isfinite(o).all(), "forward output has non-finite values"

    do = torch.randn_like(o) * do_scale
    (o * do).sum().backward()

    _assert_all_finite(
        [("x.grad", x.grad)]
        + [(f"param:{n}", p.grad) for n, p in layer.named_parameters()]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_overflow_layer_bf16_128k():
    """T=131072 (2048 chunks), bf16, A_log=-200, do_scale=1e15.

    bf16 shares fp32's exponent range, so the overflow threshold is the
    same. The floor bounds the state regardless of dtype.
    """
    from fla.layers import GatedDeltaNet2

    torch.manual_seed(42)
    B, T, hidden_size, head_dim, num_heads = 1, 131072, 128, 32, 2
    do_scale = 1e15

    layer = GatedDeltaNet2(
        hidden_size=hidden_size, head_dim=head_dim, num_heads=num_heads,
        use_short_conv=False,
    ).to(device).to(torch.bfloat16)
    layer.A_log.data.fill_(-200.0)
    layer.train()

    x = torch.randn(B, T, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    o, _, _ = layer(x)
    assert torch.isfinite(o).all(), "forward output has non-finite values"

    do = torch.randn_like(o) * do_scale
    (o * do).sum().backward()

    _assert_all_finite(
        [("x.grad", x.grad)]
        + [(f"param:{n}", p.grad) for n, p in layer.named_parameters()]
    )


# =============================================================================
# B. Op-level stress (validates fix (c) — kernel overflow guards)
#
# Bypasses the layer's Python gate floor by using use_gate_in_kernel=True
# with A_log=-200. Only the kernel guards can keep grads finite.
#
# The backward pipeline runs 4 kernels; inf from the first (chunk_kda_bwd_dAv)
# propagates through the rest. Guards in all 4 are needed to zero the inf
# before it propagates. This is an extreme stress test (do_scale=1e37);
# in real training the floor (fix a) prevents reaching this magnitude.
# =============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_overflow_op_128k(dtype):
    """chunk_gdn2 with use_gate_in_kernel=True, A_log=-200, T=131072, do_scale=1e37.

    Without kernel guards, dq has 128/16777216 non-finite values (inf
    from dot(do, v) in chunk_kda_bwd_dAv propagates through the pipeline).
    With guards in all backward kernels, gradients are zeroed instead
    of inf — finite but information-lossy (backstop, not a cure).
    """
    torch.manual_seed(42)
    B, T, H, K, V = 1, 131072, 2, 64, 64
    do_scale = 1e37

    q = torch.randn(B, T, H, K, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device, requires_grad=True)
    g = torch.randn(B, T, H, K, dtype=dtype, device=device, requires_grad=True)
    b = torch.rand(B, T, H, K, dtype=dtype, device=device, requires_grad=True)
    w = torch.rand(B, T, H, V, dtype=dtype, device=device, requires_grad=True)
    A_log = torch.full((H,), -200.0, dtype=torch.float32, device=device, requires_grad=True)
    dt_bias = torch.randn(H * K, dtype=torch.float32, device=device, requires_grad=True)
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device, requires_grad=True)

    o, ht = chunk_gdn2(
        q=q, k=k, v=v, g=g, b=b, w=w,
        A_log=A_log, dt_bias=dt_bias,
        initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
    )
    assert torch.isfinite(o).all(), "forward output has non-finite values"

    do = torch.randn_like(o) * do_scale
    dht = torch.randn_like(ht) * do_scale
    ((o * do).sum() + (ht * dht).sum()).backward()

    _assert_all_finite([
        ("dq", q.grad), ("dk", k.grad), ("dv", v.grad),
        ("dg", g.grad), ("db", b.grad), ("dw", w.grad),
        ("dA_log", A_log.grad), ("dt_bias", dt_bias.grad),
        ("dh0", h0.grad),
    ])


# =============================================================================
# C. Regression: A_log floor (fix a) behaviour
# =============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_a_log_floor_grad_zero_below():
    """When A_log is below the floor, the clamp blocks gradients — A_log.grad
    should be 0 for clamped entries, and the forward should still be finite."""
    from fla.layers import GatedDeltaNet2

    torch.manual_seed(42)
    layer = GatedDeltaNet2(
        hidden_size=128, head_dim=32, num_heads=2,
        use_short_conv=False, a_log_min=0.0,
    ).to(device).to(torch.float32)
    layer.A_log.data.fill_(-200.0)
    layer.train()

    x = torch.randn(1, 256, 128, device=device, dtype=torch.float32, requires_grad=True)
    o, _, _ = layer(x)
    assert torch.isfinite(o).all()

    o.sum().backward()
    assert layer.A_log.grad is not None, "A_log.grad is None"
    assert torch.all(layer.A_log.grad == 0), (
        f"A_log.grad should be 0 below floor (a_log_min=0.0), got max={layer.A_log.grad.max().item()}"
    )
