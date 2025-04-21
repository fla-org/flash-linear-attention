# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedLayerNormGated, FusedRMSNormGated
from fla.utils import assert_close, device


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("H", [2])
@pytest.mark.parametrize("T", [1, 50, 512, 1000, 2048])
@pytest.mark.parametrize("D", [50, 64, 128, 1200])
@pytest.mark.parametrize("elementwise_affine", [False, True])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("bias", [False])
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


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("H", [2])
@pytest.mark.parametrize("T", [1, 50, 512, 1000, 2048])
@pytest.mark.parametrize("D", [50, 64, 128, 1200])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
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
