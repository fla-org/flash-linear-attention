# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from unittest import mock

import pytest
import torch

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'dtype'),
    [
        pytest.param(*case, id="B{}-T{}-D{}-{}".format(*case))
        for case in [
            (1, 128, 512, torch.bfloat16),
            (2, 100, 512, torch.bfloat16),
            (2, 256, 512, torch.float16),
        ]
    ],
)
def test_gated_deltanet_fused_qkv_conv_matches_fallback(B: int, T: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    layer = GatedDeltaNet(
        hidden_size=D,
        head_dim=64,
        num_heads=6,
        expand_v=2,
        mode='chunk',
    ).to(device=device, dtype=dtype).train()

    x = torch.randn(B, T, D, device=device, dtype=dtype)
    do = torch.randn_like(x)

    # Count calls into the per-projection ShortConvolution modules so the test can
    # prove which path ran: the fused path bypasses them and issues a single conv.
    conv_calls = {'n': 0}

    def bump(*_):
        conv_calls['n'] += 1
    handles = [m.register_forward_hook(bump) for m in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d)]
    try:
        # Dense, no cache, no varlen: the guard holds and q/k/v run as a single conv.
        x_fused = x.detach().clone().requires_grad_(True)
        conv_calls['n'] = 0
        y_fused = layer(x_fused)[0]
        fused_calls = conv_calls['n']
        (y_fused * do).sum().backward()
        g_fused = {n: p.grad.detach().clone() for n, p in layer.named_parameters()}
        dx_fused = x_fused.grad.detach().clone()
        layer.zero_grad(set_to_none=True)

        # Reference: disable fusion via the guard predicate, keeping the same dense
        # input, so the three q/k/v convs run as separate calls.
        x_sep = x.detach().clone().requires_grad_(True)
        conv_calls['n'] = 0
        with mock.patch.object(GatedDeltaNet, '_use_fused_qkv_conv', return_value=False):
            y_sep = layer(x_sep)[0]
        sep_calls = conv_calls['n']
        (y_sep * do).sum().backward()
        g_sep = {n: p.grad.detach().clone() for n, p in layer.named_parameters()}
        dx_sep = x_sep.grad.detach().clone()
    finally:
        for h in handles:
            h.remove()

    assert fused_calls == 0, f"fused path should bypass the conv modules, saw {fused_calls} call(s)"
    assert sep_calls == 3, f"separate path should call all three conv modules, saw {sep_calls} call(s)"

    assert_close('y', y_sep, y_fused, 0.0)
    assert_close('dx', dx_sep, dx_fused, 0.0)
    # The fused path concatenates the q/k/v conv weights into one conv, which regroups
    # the backward accumulation for those weights, so the conv-weight grads can differ
    # by tiny accumulation-order noise.
    for name in g_fused:
        assert_close(f'd{name}', g_sep[name], g_fused[name], 5e-3)
