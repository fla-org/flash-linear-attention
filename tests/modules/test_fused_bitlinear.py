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


@pytest.mark.parametrize("eps", [1e-8, 1e-5], ids=["eps1e-8", "eps1e-5"])
def test_bit_linear(eps: float):
    torch.manual_seed(42)
    x = torch.randn(4, 16, device=device)
    weight = torch.randn(16, 16, device=device)
    norm_weight = torch.randn(16, device=device)

    with patch.object(fused_bitlinear, "layer_norm_linear_quant", return_value=x) as mock:
        fused_bitlinear.bit_linear(x=x, weight=weight, norm_weight=norm_weight, eps=eps)

    mock.assert_called_once()
    assert mock.call_args.kwargs["eps"] == eps


def test_fused_bitlinear():
    torch.manual_seed(42)
    x = torch.randn(4, 16, device=device)
    layer = FusedBitLinear(in_features=16, out_features=16)
    layer.norm.eps = 1e-4

    with patch.object(fused_bitlinear, "layer_norm_linear_quant", return_value=x) as mock:
        layer(x=x)

    mock.assert_called_once()
    assert mock.call_args.kwargs["eps"] == layer.norm.eps
