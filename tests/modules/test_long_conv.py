# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.modules import ImplicitLongConvolution, LongConvolution
from fla.utils import device


@pytest.mark.parametrize("module", [LongConvolution, ImplicitLongConvolution])
@pytest.mark.parametrize("T", [1, 10, 37, 64, 100])
@pytest.mark.parametrize("max_len", [64])
@torch.no_grad()
def test_long_conv_causal(module: type, T: int, max_len: int):
    torch.manual_seed(42)
    B, D = 2, 16
    conv = module(D, max_len).to(device=device, dtype=torch.float32)
    x = torch.randn(B, T, D, device=device, dtype=torch.float32)
    y = conv(x)

    # y[t] = x[t] + sum_{j <= t} k[j] * x[t - j], using the filter's first taps only
    k = conv.filter if module is LongConvolution else conv.filter(T)[0]
    ref = x.clone()
    for t in range(T):
        for j in range(min(t + 1, k.shape[-1])):
            ref[:, t] += k[:, j] * x[:, t - j]
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)

    # changing the last token must leave every earlier output unchanged
    x2 = x.clone()
    x2[:, -1] += 1
    torch.testing.assert_close(conv(x2)[:, :-1], y[:, :-1], rtol=0, atol=1e-5)
