# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import (
    FusedLayerNormGated,
    FusedLayerNormSwishGateLinear,
    FusedRMSNormGated,
    FusedRMSNormSwishGateLinear,
)
from fla.modules.norm.layernorm_gated import layernorm_fn
from fla.utils import IS_NVIDIA_BLACKWELL, assert_close, device


@pytest.mark.parametrize(
    ('B', 'H', 'T', 'D', 'elementwise_affine', 'activation', 'bias'),
    [
        pytest.param(*test, id=f"B{test[0]}_H{test[1]}_T{test[2]}_D{test[3]}_affine{test[4]}_{test[5]}_bias{test[6]}")
        for test in [
            (2, 2, 1,    64,  False, "silu",   False),
            (2, 2, 512,  128, True,  "silu",   True),
            (2, 2, 2048, 1200, True,  "sigmoid", False),
            (2, 2, 50,   50,  False, "sigmoid", False),
        ]
    ],
)
def test_layernorm_gated(B: int, H: int, T: int, D: int, elementwise_affine: bool, activation: str, bias: bool):
    torch.manual_seed(42)
    x = torch.randn(B, H, T, D).to(device).requires_grad_(True)
    g = torch.randn(B, H, T, D).to(device).requires_grad_(True)

    ref = nn.LayerNorm(D, elementwise_affine=elementwise_affine, bias=bias).to(device)
    tri = FusedLayerNormGated(D, elementwise_affine=elementwise_affine, bias=bias, activation=activation).to(device)
    if ref.weight is not None:
        nn.init.normal_(ref.weight)
        tri.weight.data.copy_(ref.weight.data)
    if ref.bias is not None:
        nn.init.normal_(ref.bias)
        tri.bias.data.copy_(ref.bias.data)

    act_fn = F.silu if activation == "silu" else F.sigmoid
    ref_y = ref(x) * act_fn(g)
    tri_y = tri(x, g)
    ref_dx, ref_dg = torch.autograd.grad((ref(x) * act_fn(g)).sum(), (x, g))
    tri_dx, tri_dg = torch.autograd.grad(tri_y.sum(), (x, g))

    if ref.weight is not None:
        ref_dw = torch.autograd.grad((ref(x) * act_fn(g)).sum(), ref.weight)[0]
        tri_dw = torch.autograd.grad(tri(x, g).sum(), tri.weight)[0]
    if ref.bias is not None:
        ref_db = torch.autograd.grad((ref(x) * act_fn(g)).sum(), ref.bias)[0]
        tri_db = torch.autograd.grad(tri(x, g).sum(), tri.bias)[0]

    assert_close(' y', ref_y, tri_y, 1e-3)
    assert_close('dx', ref_dx, tri_dx, 1e-3)
    assert_close('dg', ref_dg, tri_dg, 1e-3)
    if ref.weight is not None:
        assert_close('dw', ref_dw, tri_dw, 1e-3)
    if ref.bias is not None:
        assert_close('db', ref_db, tri_db, 1e-3)


@pytest.mark.parametrize(
    ('B', 'H', 'T', 'D', 'activation'),
    [
        pytest.param(*test, id=f"B{test[0]}_H{test[1]}_T{test[2]}_D{test[3]}_{test[4]}")
        for test in [
            (2, 2, 1,    64,  "silu"),
            (2, 2, 512,  128, "sigmoid"),
            (2, 2, 2048, 1200, "silu"),
            (2, 2, 50,   50,  "sigmoid"),
        ]
    ],
)
def test_rmsnorm_gated(B: int, H: int, T: int, D: int, activation: str):
    torch.manual_seed(42)
    x = torch.randn(B, H, T, D).to(device).requires_grad_(True)
    g = torch.randn(B, H, T, D).to(device).requires_grad_(True)
    ref = nn.RMSNorm(D, eps=0).to(device)
    tri = FusedRMSNormGated(D, eps=0, activation=activation).to(device)
    nn.init.normal_(ref.weight)
    tri.weight.data.copy_(ref.weight.data)

    act_fn = F.silu if activation == "silu" else F.sigmoid
    ref_y = ref(x) * act_fn(g)
    tri_y = tri(x, g)
    ref_dx, ref_dg = torch.autograd.grad((ref(x) * act_fn(g)).sum(), (x, g))
    tri_dx, tri_dg = torch.autograd.grad(tri_y.sum(), (x, g))

    ref_dw = torch.autograd.grad((ref(x) * act_fn(g)).sum(), ref.weight)[0]
    tri_dw = torch.autograd.grad(tri(x, g).sum(), tri.weight)[0]

    assert_close(' y', ref_y, tri_y, 1e-3)
    assert_close('dx', ref_dx, tri_dx, 1e-3)
    assert_close('dg', ref_dg, tri_dg, 1e-3)
    assert_close('dw', ref_dw, tri_dw, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'O', 'is_rms_norm', 'linear_bias'),
    [
        # D <= 512 and D > 512 select the two different backward kernels
        pytest.param(*test, id=f"B{test[0]}_T{test[1]}_D{test[2]}_O{test[3]}_rms{test[4]}_bias{test[5]}")
        for test in [
            (2, 64,  64,   32,  False, False),
            (2, 64,  64,   32,  True,  True),
            (2, 500, 1024, 256, False, True),
            (2, 500, 1024, 256, True,  False),
        ]
    ],
)
def test_norm_swish_gate_linear(B: int, T: int, D: int, O: int, is_rms_norm: bool, linear_bias: bool):
    torch.manual_seed(42)
    eps = 1e-5
    x = torch.randn(B, T, D).to(device)
    g = torch.randn(B, T, D).to(device)
    nw = torch.randn(D).to(device)
    lw = (torch.randn(O, D) / D ** 0.5).to(device)
    lb = torch.randn(O).to(device) if linear_bias else None
    do = torch.randn(B, T, O).to(device)

    def leaves():
        return [t.clone().requires_grad_(True) if t is not None else None for t in (x, g, lw, lb)]

    tri_x, tri_g, tri_lw, tri_lb = leaves()
    tri = (FusedRMSNormSwishGateLinear if is_rms_norm else FusedLayerNormSwishGateLinear)(D, eps=eps).to(device)
    tri.weight.data.copy_(nw)
    tri(tri_x, tri_g, tri_lw, tri_lb).backward(do)

    ref_x, ref_g, ref_lw, ref_lb = leaves()
    ref_nw = nw.clone().requires_grad_(True)
    if is_rms_norm:
        ref_norm = ref_x * torch.rsqrt(ref_x.pow(2).mean(-1, keepdim=True) + eps) * ref_nw
    else:
        ref_norm = F.layer_norm(ref_x, (D,), ref_nw, None, eps)
    F.linear(ref_norm * F.silu(ref_g), ref_lw, ref_lb).backward(do)

    assert_close('            dx', ref_x.grad, tri_x.grad, 1e-3)
    assert_close('            dg', ref_g.grad, tri_g.grad, 1e-3)
    assert_close('  dnorm_weight', ref_nw.grad, tri.weight.grad, 1e-3)
    assert_close('dlinear_weight', ref_lw.grad, tri_lw.grad, 1e-3)
    if linear_bias:
        assert_close('  dlinear_bias', ref_lb.grad, tri_lb.grad, 1e-3)


@pytest.mark.skipif(not IS_NVIDIA_BLACKWELL, reason="large-offset repro requires a Blackwell/B200-class CUDA GPU")
def test_rmsnorm_gated_large_batch_offsets():
    torch.manual_seed(42)
    B, H, T, D = 256, 12, 6144, 128
    x = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16).requires_grad_(True)
    g = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16).requires_grad_(True)
    weight = torch.ones(D, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        ref = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        ref = (ref * weight.float() * F.silu(g.float())).to(torch.bfloat16)
    tri = FusedRMSNormGated(D, eps=1e-6, activation="silu").to(device, dtype=torch.bfloat16)
    tri.weight.data.copy_(weight)
    y = tri(x, g)

    assert_close(' y', ref, y, 6.3e-2)
    del ref

    y.float().sum().backward()
    assert x.grad is not None
    assert g.grad is not None
    assert tri.weight.grad is not None


@pytest.mark.skipif(not IS_NVIDIA_BLACKWELL, reason="large-offset repro requires a Blackwell/B200-class CUDA GPU")
def test_rmsnorm_gated_large_batch_offsets_large_d():
    torch.manual_seed(42)
    B, H, T, D = 256, 1, 8200, 1024
    x = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16).requires_grad_(True)
    g = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16).requires_grad_(True)
    tri = FusedRMSNormGated(D, eps=1e-6, activation="silu").to(device, dtype=torch.bfloat16)
    tri.weight.data.fill_(1)

    y = tri(x, g)
    with torch.no_grad():
        ref = torch.cat(
            [tri(x[start:start + 128], g[start:start + 128]) for start in range(0, B, 128)],
            dim=0,
        )
    assert_close(' y', ref, y, 6.3e-2)
    del ref

    y.float().sum().backward()
    assert x.grad is not None
    assert g.grad is not None
    assert tri.weight.grad is not None


@pytest.mark.parametrize(
    ('T', 'D', 'group_size', 'dtype', 'gate_mode'),
    [
        pytest.param(*test, id="T{}-D{}-group{}-{}-{}".format(*test))
        for test in [
            (1, 64, None, torch.float32, "none"),
            (63, 200, 50, torch.float32, "none"),
            (65, 1024, 256, torch.float16, "before"),
            (100, 1028, 257, torch.bfloat16, "after"),
            (33, 513, None, torch.float32, "before"),
            (33, 513, None, torch.bfloat16, "after"),
        ]
    ],
)
@pytest.mark.parametrize("is_rms_norm", [False, True], ids=["layer", "rms"])
def test_groupnorm_gated(T: int, D: int, group_size: int | None, dtype: torch.dtype, gate_mode: str, is_rms_norm: bool):
    torch.manual_seed(42)
    eps = 1e-5
    x = torch.randn(2, T, D * 2, device=device, dtype=dtype)[..., ::2].requires_grad_(True)
    weight = torch.randn(D, device=device, requires_grad=True)
    bias = torch.randn(D, device=device, requires_grad=True) if not is_rms_norm else None
    z = torch.randn_like(x, requires_grad=True) if gate_mode != 'none' else None
    inputs = [t for t in (x, weight, bias, z) if t is not None]
    do = torch.randn_like(x)

    tri_y = layernorm_fn(
        x=x,
        weight=weight,
        bias=bias,
        z=z,
        eps=eps,
        group_size=group_size,
        norm_before_gate=gate_mode != 'before',
        is_rms_norm=is_rms_norm,
    )
    tri_grads = torch.autograd.grad(tri_y, inputs, do)

    ref_x = x.float()
    if gate_mode == 'before':
        ref_x = ref_x * F.silu(z.float())
    grouped = ref_x.reshape(2, T, -1, group_size or D)
    if not is_rms_norm:
        grouped = grouped - grouped.mean(-1, keepdim=True)
    ref_y = (grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)).reshape_as(x) * weight
    if bias is not None:
        ref_y = ref_y + bias
    if gate_mode == 'after':
        ref_y = ref_y * F.silu(z.float())
    ref_y = ref_y.to(dtype)
    ref_grads = torch.autograd.grad(ref_y, inputs, do)

    tolerance = 5e-3 if dtype == torch.bfloat16 else 1e-3
    assert_close(' y', ref_y, tri_y, tolerance)
    names = [name for name, t in zip(('dx', 'dw', 'db', 'dz'), (x, weight, bias, z)) if t is not None]
    for name, expected, actual in zip(names, ref_grads, tri_grads):
        assert_close(name, expected, actual, tolerance)
