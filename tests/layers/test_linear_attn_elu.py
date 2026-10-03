# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.linear_attn import LinearAttention
from fla.utils import device


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_elu_feature_map_values(dtype):
    layer = LinearAttention(
        hidden_size=128,
        num_heads=1,
        feature_map='elu',
    ).to(device)

    x = torch.tensor(
        [-10, -8, -6, -1, 0, 1, 100],
        dtype=dtype,
        device=device,
    )

    actual = layer.feature_map_q(x)

    x64 = x.double()

    expected = torch.where(
        x64 >= 0,
        x64 + 1,
        x64.exp(),
    ).to(dtype)

    rtol = {
        torch.bfloat16: 1e-2,
        torch.float16: 2e-3,
        torch.float32: 1e-6,
    }[dtype]

    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_elu_feature_map_gradient(dtype):
    layer = LinearAttention(
        hidden_size=128,
        num_heads=1,
        feature_map='elu',
    ).to(device=device)

    x = torch.tensor(
        [-10, -8, -6, -1, 0, 1, 100],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )

    y = layer.feature_map_q(x)

    dx, = torch.autograd.grad(y.float().sum(), x)

    x64 = x.detach().double()

    expected_grad = torch.where(
        x64 >= 0,
        torch.ones_like(x64),
        x64.exp(),
    ).to(dtype)

    rtol = {
        torch.bfloat16: 1e-2,
        torch.float16: 2e-3,
        torch.float32: 1e-6,
    }[dtype]

    torch.testing.assert_close(dx, expected_grad, rtol=rtol, atol=0)


@pytest.mark.parametrize("seq_len", [1, 65])
def test_elu_linear_attention_bf16(seq_len):
    layer = LinearAttention(
        hidden_size=128,
        num_heads=1,
        feature_map='elu',
        output_norm='identity',
        do_feature_map_norm=True,
    ).to(device=device, dtype=torch.bfloat16)

    eye = torch.eye(
        128,
        device=device,
        dtype=torch.bfloat16,
    )

    with torch.no_grad():
        layer.q_proj.weight.copy_(-8 * eye)
        layer.k_proj.weight.zero_()
        layer.v_proj.weight.copy_(eye)
        layer.o_proj.weight.copy_(eye)

    x = torch.ones(
        1, seq_len, 128,
        device=device,
        dtype=torch.bfloat16,
    )

    output = layer(x)[0]

    expected = torch.ones_like(output)

    torch.testing.assert_close(
        output,
        expected,
        rtol=1e-2,
        atol=1e-2,
    )

    output.float().mean().backward()

    grad = layer.q_proj.weight.grad

    assert grad is not None
    assert torch.isfinite(grad).all()
