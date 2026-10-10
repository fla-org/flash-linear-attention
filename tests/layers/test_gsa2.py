# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from unittest import mock

import pytest
import torch

from fla.layers.gsa2 import GatedSlotAttention2
from fla.utils import IS_AMD, IS_NPU, IS_NVIDIA, device

# the layer runs the GDN-2 kernels, which tests/ops/test_gdn2.py guards the same way
pytestmark = pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_NPU), reason="CUDA/ROCm or Ascend NPU required")


@pytest.mark.parametrize(
    ('num_heads', 'num_v_heads', 'use_short_conv', 'num_slots', 'level2_axis'),
    [
        pytest.param(*case, id="H{}-HV{}-conv{}-M{}-{}".format(*case))
        for case in [
            (2, 2, False, 64, 'slot'),
            (2, 4, False, 64, 'head'),
            (2, 2, True, 128, 'head'),
        ]
    ],
)
def test_gsa2_layer(num_heads: int, num_v_heads: int, use_short_conv: bool, num_slots: int, level2_axis: str):
    """The layer runs forward and backward with finite gradients for every parameter, covering GVA, the slot-to-head
    projections of the head axis, and the short-conv path that the model tests do not reach."""
    torch.manual_seed(42)
    hidden_size, head_dim, B, T = 128, 32, 2, 128
    layer = GatedSlotAttention2(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        num_slots=num_slots,
        level2_axis=level2_axis,
        use_short_conv=use_short_conv,
        use_w_conv=use_short_conv,
        layer_idx=0,
    ).to(device=device, dtype=torch.float32).train()

    x = torch.randn(B, T, hidden_size, device=device, dtype=torch.float32, requires_grad=True)
    o, _, _ = layer(x)
    assert o.shape == (B, T, hidden_size)
    assert torch.isfinite(o).all()

    o.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in layer.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name}.grad is None"
            assert torch.isfinite(p.grad).all(), f"{name}.grad has non-finite values"


@pytest.mark.parametrize('mode', ['chunk', 'fused_recurrent'])
def test_gsa2_eval_with_grad_uses_chunk(mode: str):
    torch.manual_seed(42)
    layer = GatedSlotAttention2(
        hidden_size=128,
        head_dim=32,
        num_heads=2,
        num_slots=64,
        mode=mode,
        layer_idx=0,
    ).to(device=device, dtype=torch.bfloat16).eval()
    x = torch.randn(1, 32, 128, device=device, dtype=torch.bfloat16, requires_grad=True)

    message = "eval with autograd must not use a forward-only fused recurrent kernel"
    with mock.patch('fla.layers.gsa2.fused_recurrent_gated_oja_rule2', side_effect=AssertionError(message)) as oja2, \
            mock.patch('fla.layers.gsa2.fused_recurrent_gdn2', side_effect=AssertionError(message)) as gdn2:
        y = layer(x)[0]
        y.sum().backward()

    oja2.assert_not_called()
    gdn2.assert_not_called()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
