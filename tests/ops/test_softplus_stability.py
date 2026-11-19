# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Regression tests for the negative tail of the fused GDN softplus (#1776).

``log(1 + exp(x))`` returns zero once ``exp(x)`` drops below half an fp32 ULP
at 1, and ``lg2.approx`` is inaccurate on the small residual that is left for
x around -16. Both the PTX and the generic Triton path lost the finite gate,
which then accumulated into a visibly wrong recurrent state.
"""

import math

import pytest
import torch
import torch.nn.functional as F
import triton
from triton import language as tl

from fla.ops.gated_delta_rule.gate import gdn_gate_fwd
from fla.ops.utils.softplus import softplus_nv, softplus_triton
from fla.utils import IS_NVIDIA, device


@triton.jit
def _softplus_kernel(X, Y, MODE: tl.constexpr, N: tl.constexpr):
    idx = tl.arange(0, N)
    x = tl.load(X + idx)
    if MODE == 0:
        y = softplus_nv(x)
    else:
        y = softplus_triton(x)
    tl.store(Y + idx, y)


@pytest.mark.parametrize(("mode", "name", "skip"), [(0, "softplus_nv", not IS_NVIDIA), (1, "softplus_triton", False)])
def test_softplus_negative_tail(mode, name, skip):
    if skip:
        pytest.skip(f"{name} is NVIDIA-only")
    x = torch.tensor([-16.5, -17.0, -18.0, -20.0, -10.0, 0.0, 1.0, 20.0], device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    _softplus_kernel[(1,)](x, y, mode, x.numel())
    ref = F.softplus(x.double()).float()
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-12)


def test_gdn_gate_preserves_small_negative_softplus():
    g = torch.full((8192, 1), -17.0, device=device)
    A_log = torch.full((1,), math.log(16), device=device)

    gate = gdn_gate_fwd(g, A_log)

    single_step = -16.0 * F.softplus(torch.tensor(-17.0, device=device, dtype=torch.float64)).float()
    torch.testing.assert_close(gate[0], single_step.expand(1), rtol=1e-4, atol=1e-12)

    # q = k = v = 0 leaves only the gate decay, so the state is exp(sum(gate)).
    state = torch.exp(gate.double().sum(dim=0))
    expected = math.exp(8192 * (-16.0 * math.log1p(math.exp(-17.0))))
    torch.testing.assert_close(state, torch.full_like(state, expected), rtol=1e-4, atol=1e-9)
