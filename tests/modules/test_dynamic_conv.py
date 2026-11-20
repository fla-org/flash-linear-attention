# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.modules.dynamic_conv import dynamic_conv1d
from fla.modules.dynamic_conv.naive import naive_dynamic_conv1d
from fla.utils import IS_AMD, IS_NVIDIA, assert_close, device

pytestmark = pytest.mark.skipif(not (IS_NVIDIA or IS_AMD), reason="Dynamic convolution requires CUDA/HIP")


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'W', 'R', 'dynamic_activation', 'activation', 'has_residual', 'dtype'),
    [
        pytest.param(*test, id="B{}_T{}_D{}_W{}_R{}_dynamic{}_activation{}_residual{}_{}".format(*test))
        for test in [
            (2, 64, 128, 3, 16, None, None, False, torch.float32),
            (2, 65, 200, 4, 17, "silu", "swish", True, torch.float32),
            (2, 128, 128, 4, 32, None, None, False, torch.float16),
            (1, 500, 128, 3, 16, "swish", None, True, torch.float16),
        ]
    ],
)
def test_dynamic_conv(
    B: int,
    T: int,
    D: int,
    W: int,
    R: int,
    dynamic_activation: str | None,
    activation: str | None,
    has_residual: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    x = torch.randn(B, T, D).to(device, dtype).requires_grad_(True)
    h = torch.randn(B, T, D).to(device, dtype).requires_grad_(True)
    w1 = (torch.randn(R, D) / D ** 0.5).to(device, dtype).requires_grad_(True)
    w2 = (torch.randn(D * W, R) / R ** 0.5).to(device, dtype).requires_grad_(True)
    bias = torch.randn(D * W).to(device, dtype).requires_grad_(True)
    residual = x.detach().clone().requires_grad_(True) if has_residual else None
    dy = torch.randn_like(x)

    ref, _ = naive_dynamic_conv1d(
        x=x,
        h=h,
        dynamic_proj_w1=w1,
        dynamic_proj_w2=w2,
        dynamic_proj_bias=bias,
        dynamic_activation=dynamic_activation,
        residual=residual,
        activation=activation,
    )
    ref.backward(dy)
    ref_dx, x.grad = x.grad, None
    ref_dh, h.grad = h.grad, None
    ref_dw1, w1.grad = w1.grad, None
    ref_dw2, w2.grad = w2.grad, None
    ref_db, bias.grad = bias.grad, None
    if has_residual:
        ref_dr, residual.grad = residual.grad, None

    tri, _ = dynamic_conv1d(
        x=x,
        h=h,
        dynamic_proj_w1=w1,
        dynamic_proj_w2=w2,
        dynamic_proj_bias=bias,
        dynamic_activation=dynamic_activation,
        residual=residual,
        activation=activation,
    )
    tri.backward(dy)

    assert_close(" y", ref.float(), tri.float(), 1e-3)
    assert_close("dx", ref_dx.float(), x.grad.float(), 1e-3)
    assert_close("dh", ref_dh.float(), h.grad.float(), 1e-3)
    assert_close("dw1", ref_dw1.float(), w1.grad.float(), 1e-3)
    assert_close("dw2", ref_dw2.float(), w2.grad.float(), 1e-3)
    assert_close("db", ref_db.float(), bias.grad.float(), 1e-3)
    if has_residual:
        assert_close("dr", ref_dr.float(), residual.grad.float(), 1e-3)


@pytest.mark.parametrize(
    ('N', 'T', 'D', 'W', 'R', 'dynamic_activation', 'activation', 'has_residual', 'dtype'),
    [
        pytest.param(*test, id="N{}_T{}_D{}_W{}_R{}_dynamic{}_activation{}_residual{}_{}".format(*test))
        for test in [
            (4, 257, 128, 3, 16, None, None, False, torch.float32),
            (4, 500, 200, 4, 17, "swish", "silu", True, torch.float32),
            (4, 1024, 128, 4, 32, None, None, False, torch.float16),
            (4, 65, 128, 3, 16, "silu", None, True, torch.float16),
        ]
    ],
)
def test_dynamic_conv_varlen(
    N: int,
    T: int,
    D: int,
    W: int,
    R: int,
    dynamic_activation: str | None,
    activation: str | None,
    has_residual: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(1, T)[torch.randperm(T - 1)[:N-1]],
        torch.tensor([T], dtype=torch.long),
    ]).to(device).sort()[0]
    x = torch.randn(1, T, D).to(device, dtype).requires_grad_(True)
    h = torch.randn(1, T, D).to(device, dtype).requires_grad_(True)
    w1 = (torch.randn(R, D) / D ** 0.5).to(device, dtype).requires_grad_(True)
    w2 = (torch.randn(D * W, R) / R ** 0.5).to(device, dtype).requires_grad_(True)
    bias = torch.randn(D * W).to(device, dtype).requires_grad_(True)
    residual = x.detach().clone().requires_grad_(True) if has_residual else None
    dy = torch.randn_like(x)

    ref = torch.cat([
        naive_dynamic_conv1d(
            x=x[:, bos:eos],
            h=h[:, bos:eos],
            dynamic_proj_w1=w1,
            dynamic_proj_w2=w2,
            dynamic_proj_bias=bias,
            dynamic_activation=dynamic_activation,
            residual=residual[:, bos:eos] if has_residual else None,
            activation=activation,
        )[0]
        for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)
    ], dim=1)
    ref.backward(dy)
    ref_dx, x.grad = x.grad, None
    ref_dh, h.grad = h.grad, None
    ref_dw1, w1.grad = w1.grad, None
    ref_dw2, w2.grad = w2.grad, None
    ref_db, bias.grad = bias.grad, None
    if has_residual:
        ref_dr, residual.grad = residual.grad, None

    tri, _ = dynamic_conv1d(
        x=x,
        h=h,
        dynamic_proj_w1=w1,
        dynamic_proj_w2=w2,
        dynamic_proj_bias=bias,
        dynamic_activation=dynamic_activation,
        residual=residual,
        activation=activation,
        cu_seqlens=cu_seqlens,
    )
    tri.backward(dy)

    assert_close(" y", ref.float(), tri.float(), 1e-3)
    assert_close("dx", ref_dx.float(), x.grad.float(), 1e-3)
    assert_close("dh", ref_dh.float(), h.grad.float(), 1e-3)
    assert_close("dw1", ref_dw1.float(), w1.grad.float(), 1e-3)
    assert_close("dw2", ref_dw2.float(), w2.grad.float(), 1e-3)
    assert_close("db", ref_db.float(), bias.grad.float(), 1e-3)
    if has_residual:
        assert_close("dr", ref_dr.float(), residual.grad.float(), 1e-3)
