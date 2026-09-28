# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from unittest.mock import patch

import pytest
import torch

from fla.modules import fused_bitlinear
from fla.modules.fused_bitlinear import FusedBitLinear
from fla.utils import device


def _record_eps(recorded: dict):
    def spy(*args, **kwargs):
        recorded.update(kwargs)
        return torch.zeros(1, device=device)
    return spy


@pytest.mark.parametrize("eps", [1e-8, 1e-5], ids=["eps=1e-8", "eps=1e-5"])
def test_bit_linear_forwards_eps(eps: float):
    # a dropped eps leaves the norm on the callee default, so the layer would quietly
    # normalize with an epsilon the caller never asked for
    torch.manual_seed(42)

    recorded = {}
    x = torch.randn(4, 16, device=device)
    weight = torch.randn(16, 16, device=device)
    norm_weight = torch.randn(16, device=device)

    with patch.object(fused_bitlinear, "layer_norm_linear_quant_fn", _record_eps(recorded)):
        fused_bitlinear.bit_linear(x, weight, None, norm_weight, None, eps=eps)

    # exact equality, not assert_close: a missing eps records 0.0, and assert_close's
    # default err_atol=1e-6 would pass that
    assert "eps" in recorded, "bit_linear dropped its eps argument"
    assert recorded["eps"] == eps


def test_fused_bitlinear_forwards_norm_eps():
    # FusedBitLinear advertises its epsilon through __repr__, so the kernel must see it too
    torch.manual_seed(42)

    recorded = {}
    layer = FusedBitLinear(16, 16)
    layer.norm.eps = 1e-4

    with patch.object(fused_bitlinear, "layer_norm_linear_quant_fn", _record_eps(recorded)):
        layer(torch.randn(4, 16, device=device))

    assert "eps" in recorded, "FusedBitLinear.forward dropped the configured norm_eps"
    assert recorded["eps"] == 1e-4
