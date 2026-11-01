# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.modules import ImplicitLongConvolution, LongConvolution
from fla.utils import device


@pytest.mark.parametrize(
    ('module', 'T', 'max_len'),
    [
        pytest.param(LongConvolution, 1, 64, id='single-token'),
        pytest.param(LongConvolution, 37, 64, id='short-sequence'),
        pytest.param(LongConvolution, 64, 64, id='full-filter'),
        pytest.param(LongConvolution, 100, 64, id='short-filter'),
        pytest.param(ImplicitLongConvolution, 37, 64, id='implicit-short-sequence'),
    ],
)
def test_long_conv_causal(module: type, T: int, max_len: int):
    torch.manual_seed(42)
    B, D = 2, 16
    conv = module(hidden_size=D, max_len=max_len).to(device=device, dtype=torch.float32)
    x = torch.randn(B, T, D, device=device, dtype=torch.float32, requires_grad=True)
    y = conv(x)

    k = conv.filter if module is LongConvolution else conv.filter(T)[0]
    # slice-based windows keep the singleton reference differentiable on NPU.
    padded = F.pad(x.transpose(1, 2), (k.shape[-1] - 1, 0))
    windows = torch.stack([padded[..., t:t + k.shape[-1]] for t in range(T)], dim=-2)
    ref = x + (windows * k.flip(-1)[None, :, None, :]).sum(-1).transpose(1, 2)
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)

    inputs = (x, *(p for p in conv.parameters() if p.requires_grad))
    do = torch.randn_like(y)
    grads = torch.autograd.grad(outputs=y, inputs=inputs, grad_outputs=do)
    ref_grads = torch.autograd.grad(outputs=ref, inputs=inputs, grad_outputs=do)
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(grad, ref_grad, rtol=1e-4, atol=1e-4)
    if module is LongConvolution and max_len > T:
        assert torch.count_nonzero(grads[1][:, T:]) == 0

    # changing the last token must leave every earlier output unchanged
    with torch.no_grad():
        x2 = x.clone()
        x2[:, -1] += 1
        torch.testing.assert_close(conv(x2)[:, :-1], y[:, :-1], rtol=0, atol=1e-4)
