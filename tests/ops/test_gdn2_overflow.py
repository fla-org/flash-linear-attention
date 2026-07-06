# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Overflow reproduction and regression tests for GDN-2 backward.
#
# Root-cause summary: a very negative `A_log` can make the layer's decay nearly
# vanish, allowing long-sequence recurrent state growth to amplify backward
# dot-product inputs. The layer-level floor mitigates that failure mode in the
# normal path; kernel finite guards are a backstop for selected op-level
# backward accumulators.
#
# `do_scale` below is a synthetic upstream-gradient scale used only to stress
# backward numerics. It is not a training hyperparameter.
#
# Fixes:
#   (a) A_log floor (fla/layers/gdn2.py): clamp(min=a_log_min) on A_log
#       before exp(). With the default a_log_min=0.0, drifted negative
#       values use exp(0) instead of a near-zero decay rate in the layer path.
#   (c) Kernel overflow guards (chunk_bwd.py, chunk_delta_h.py,
#       kda/chunk_bwd.py, gdn2/chunk_intra.py): zero inf/nan in selected
#       backward accumulators after overflow-prone tl.dot sites. These guards
#       do not prevent overflow inside tl.dot; they only stop non-finite values
#       from propagating on the guarded paths.
#
# Test structure:
#   A. Layer-level finite regression — exercises fix (a):
#      GatedDeltaNet2 with A_log=-200 (simulates drifted A_log), large do.
#      This did not reproduce the pre-fix failure at the selected scale on A100;
#      it guards that the floored public layer path remains finite.
#   B. Op-level controlled overflow repro — exercises fix (c):
#      chunk_gdn2 with use_gate_in_kernel=True, A_log=-200, extreme do.
#      On the pre-fix baseline, this stress case produced non-finite dq values
#      in both fp32 and bf16. With the guards, the returned gradients are finite.
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
# A. Layer-level finite regression (fix (a) — A_log floor)
#
# Simulates a drifted A_log=-200 in a long-sequence layer run. The test asserts
# that the default floor keeps this public layer path finite for the selected
# T=131072, do_scale=1e15 stress case. This is regression coverage for the
# layer mitigation, not the controlled pre-fix overflow reproduction.
# =============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_overflow_layer_fp32_128k():
    """T=131072 (2048 chunks), fp32, A_log=-200, synthetic do_scale=1e15.

    The floor clamps the layer's A_log contribution before exp(). This test
    checks that the selected long-sequence stress case returns finite
    gradients.
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
    """T=131072 (2048 chunks), bf16, A_log=-200, synthetic do_scale=1e15.

    Checks the same long-sequence layer stress case in bf16.
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
# B. Op-level controlled overflow repro (fix (c) — kernel overflow guards)
#
# Bypasses the layer's Python gate floor by using use_gate_in_kernel=True
# with A_log=-200. This path is intentionally more extreme than the layer tests:
# on the pre-fix baseline, this controlled stress produced
# `dq has 128/16777216 non-finite values` in both fp32 and bf16 on A100.
# =============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_overflow_op_128k(dtype):
    """chunk_gdn2 with use_gate_in_kernel=True, A_log=-200, T=131072, synthetic do_scale=1e37.

    This bypasses the layer floor and checks that the guarded backward path
    returns finite gradients for the selected op-level stress case.
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
