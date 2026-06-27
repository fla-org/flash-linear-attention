# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
OJ repro for Ascend NPU causal_conv1d coreDim (2-D grid) overflow.

https://github.com/fla-org/flash-linear-attention/issues/981

On triton-ascend / A2, the causal_conv1d fwd/bwd kernels launch a 3-D grid
``(cdiv(D, BD), NT, B)``. The launch is rejected once the effective grid
volume exceeds 65535. ``_clamp_bd_for_grid`` only grows ``BD`` (capped at 64)
and never tiles the time (``NT``) dimension, so long sequences with large
feature dims still overflow. In distributed training this surfaces as a hang.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from fla.modules.convolution import causal_conv1d
from fla.utils import device


def _ref(x, weight, bias=None, residual=None, activation=None):
    # x: (b, t, d); weight: (d, w)
    dtype_in = x.dtype
    xc = x.float().transpose(1, 2)  # (b, d, t)
    out = F.conv1d(
        xc,
        weight.float().unsqueeze(1),
        bias.float() if bias is not None else None,
        padding=weight.shape[1] - 1,
        groups=weight.shape[0],
    )[..., : x.shape[1]]
    out = out.transpose(1, 2)
    if activation in ("silu", "swish"):
        out = F.silu(out)
    out = out.to(dtype_in)
    if residual is not None:
        out = out + residual
    return out


@pytest.mark.parametrize(
    ("B", "T", "D", "W"),
    [
        # Matches issue #981 (seq_len=4265) with a feature dim large enough
        # that the bwd grid volume still exceeds 65535 after the BD clamp.
        pytest.param(1, 4265, 8192, 3, id="B1_T4265_D8192"),
    ],
)
def test_causal_conv1d_npu_coredim_overflow(B, T, D, W):
    torch.manual_seed(42)
    dtype = torch.bfloat16

    x = torch.randn(B, T, D, device=device, dtype=dtype).requires_grad_(True)
    weight = torch.randn(D, W, device=device, dtype=dtype).requires_grad_(True)
    bias = torch.randn(D, device=device, dtype=dtype).requires_grad_(True)
    residual = x.detach().clone().requires_grad_(True)
    dy = torch.randn(B, T, D, device=device, dtype=dtype)

    NT = math.ceil(T / 8)  # bwd BT for D >= 2048
    print(f"\n[conv1d-npu-coredim] B={B} T={T} D={D} W={W} "
          f"bwd grid volume upper-bound cdiv(D,64)*NT={math.ceil(D/64) * NT} (limit 65535)")

    ref = _ref(x, weight, bias, residual=residual)
    ref.backward(dy)
    ref_dx, x.grad = x.grad, None
    ref_dw, weight.grad = weight.grad, None
    ref_db, bias.grad = bias.grad, None
    ref_dr, residual.grad = residual.grad, None

    tri, _ = causal_conv1d(x, weight, bias, residual=residual, backend="triton")
    tri.backward(dy)
    tri_dx, x.grad = x.grad, None
    tri_dw, weight.grad = weight.grad, None
    tri_db, bias.grad = bias.grad, None
    tri_dr, residual.grad = residual.grad, None

    torch.testing.assert_close(tri, ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(tri_dx, ref_dx, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(tri_dw, ref_dw, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(tri_db, ref_db, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(tri_dr, ref_dr, atol=1e-2, rtol=1e-2)
