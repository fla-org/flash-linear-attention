# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.utils import assert_close, device


@pytest.mark.parametrize('mode', ['chunk', 'fused_recurrent'])
def test_gated_deltanet_eval_with_grad_uses_chunk(mode: str):
    torch.manual_seed(42)
    layer = GatedDeltaNet(
        hidden_size=512,
        expand_v=2,
        head_dim=64,
        num_heads=6,
        mode=mode,
    ).to(device=device, dtype=torch.bfloat16).eval()
    x = torch.randn(1, 32, 512, device=device, dtype=torch.bfloat16, requires_grad=True)

    with mock.patch(
        'fla.layers.gated_deltanet.fused_recurrent_gated_delta_rule',
        side_effect=AssertionError("eval with autograd must not use the forward-only fused recurrent kernel"),
    ) as fused_recurrent:
        y = layer(x)[0]
        y.sum().backward()

    fused_recurrent.assert_not_called()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


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
        expand_v=2,
        head_dim=64,
        num_heads=6,
        mode='chunk',
    ).to(device=device, dtype=dtype).train()

    x = torch.randn(B, T, D, device=device, dtype=dtype)
    do = torch.randn_like(x)

    # the fused path bypasses the per-projection modules, so their hooks distinguish it from the fallback.
    conv_calls = {'n': 0}

    def bump(*_):
        conv_calls['n'] += 1

    handles = [m.register_forward_hook(bump) for m in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d)]
    try:
        x_fused = x.detach().clone().requires_grad_(True)
        conv_calls['n'] = 0
        y_fused = layer(x_fused)[0]
        fused_calls = conv_calls['n']
        (y_fused * do).sum().backward()
        g_fused = {n: p.grad.detach().clone() for n, p in layer.named_parameters()}
        dx_fused = x_fused.grad.detach().clone()
        layer.zero_grad(set_to_none=True)

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
    # concatenating q/k/v convolution weights regroups backward accumulation,
    # so their gradients can differ by small accumulation-order noise.
    for name in g_fused:
        assert_close(f'd{name}', g_sep[name], g_fused[name], 5e-3)


@pytest.mark.parametrize('activation', [None, 'swish', 'silu', 'sigmoid'], ids=['default', 'swish', 'silu', 'sigmoid'])
@pytest.mark.parametrize('use_gate', [False, True], ids=['ungated', 'gated'])
@pytest.mark.parametrize(
    ('T', 'dtype'),
    [
        pytest.param(1, torch.float32, id='T1-float32'),
        pytest.param(63, torch.bfloat16, id='T63-bfloat16'),
        pytest.param(128, torch.float16, id='T128-float16'),
    ],
)
def test_gated_deltanet_output_gate(activation: str | None, use_gate: bool, T: int, dtype: torch.dtype):
    torch.manual_seed(42)
    kwargs = {} if activation is None else {'output_gate_activation': activation}
    layer = GatedDeltaNet(
        hidden_size=128,
        expand_v=1,
        head_dim=32,
        num_heads=2,
        use_gate=use_gate,
        **kwargs,
    ).to(device=device, dtype=dtype).train()
    x = torch.randn(2, T, 128, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn_like(x)
    tri = layer(x)[0]
    tri.backward(do)
    tri_dx = x.grad.clone()
    tri_grads = {name: parameter.grad.clone() for name, parameter in layer.named_parameters()}
    x.grad = None
    layer.zero_grad(set_to_none=True)

    def reference_norm(x, g=None):
        output = F.rms_norm(
            input=x.float(),
            normalized_shape=(layer.head_v_dim,),
            weight=layer.o_norm.weight.float(),
            eps=layer.o_norm.eps,
        )
        if use_gate:
            gate = g.float().sigmoid() if activation == 'sigmoid' else F.silu(g.float())
            output = output * gate
        return output.to(x.dtype)

    with mock.patch.object(layer.o_norm, 'forward', side_effect=reference_norm):
        ref = layer(x)[0]
        ref.backward(do)

    assert torch.isfinite(tri).all()
    assert torch.isfinite(tri_dx).all()
    assert_close('o', ref, tri, 5e-3)
    assert_close('dx', x.grad, tri_dx, 5e-3)
    for name, parameter in layer.named_parameters():
        assert torch.isfinite(tri_grads[name]).all()
        assert_close(f'd{name}', parameter.grad, tri_grads[name], 5e-3)


@pytest.mark.parametrize('use_gate', [False, True], ids=['ungated', 'gated'])
def test_gated_deltanet_output_gate_validation(use_gate: bool):
    with pytest.raises(ValueError, match='output_gate_activation'):
        GatedDeltaNet(hidden_size=128, expand_v=1, head_dim=32, num_heads=2, use_gate=use_gate, output_gate_activation='relu')
