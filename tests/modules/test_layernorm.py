# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn as nn
from einops import rearrange
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from fla.modules import (GroupNorm, GroupNormLinear, LayerNorm,
                         LayerNormLinear, RMSNorm, RMSNormLinear)
from fla.utils import device


@pytest.mark.parametrize("B", [1, 4, 8])
@pytest.mark.parametrize("H", [1, 4])
@pytest.mark.parametrize("T", [1, 50, 2048])
@pytest.mark.parametrize("D", [50, 64, 128])
@pytest.mark.parametrize("elementwise_affine", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_layernorm(B: int, H: int, T: int, D: int, elementwise_affine: bool, bias: bool):
    x = torch.randn(B, H, T, D).to(device).requires_grad_(True)
    ref = nn.LayerNorm(D, elementwise_affine=elementwise_affine, bias=bias).to(device)
    tri = LayerNorm(D, elementwise_affine=elementwise_affine, bias=bias).to(device)
    if ref.weight is not None:
        nn.init.normal_(ref.weight)
        tri.weight.data.copy_(ref.weight.data)
    if ref.bias is not None:
        nn.init.normal_(ref.bias)
        tri.bias.data.copy_(ref.bias.data)

    ref_y = ref(x)
    tri_y = tri(x)
    ref_dx = torch.autograd.grad(ref(x).sum(), x)[0]
    tri_dx = torch.autograd.grad(tri(x).sum(), x)[0]

    if ref.weight is not None:
        ref_dw = torch.autograd.grad(ref(x).sum(), ref.weight)[0]
        tri_dw = torch.autograd.grad(tri(x).sum(), tri.weight)[0]
    if ref.bias is not None:
        ref_db = torch.autograd.grad(ref(x).sum(), ref.bias)[0]
        tri_db = torch.autograd.grad(tri(x).sum(), tri.bias)[0]

    torch.testing.assert_close(ref_y, tri_y, rtol=0, atol=1e-4)
    torch.testing.assert_close(ref_dx, tri_dx, rtol=0, atol=1e-4)
    if ref.weight is not None:
        torch.testing.assert_close(ref_dw, tri_dw, rtol=0, atol=1e-3)
    if ref.bias is not None:
        torch.testing.assert_close(ref_db, tri_db, rtol=0, atol=1e-4)


@pytest.mark.parametrize("B", [1, 4, 8])
@pytest.mark.parametrize("T", [1, 50, 2048])
@pytest.mark.parametrize("D", [64, 128, 512, 1024, 2048])
@pytest.mark.parametrize("G", [1, 4])
def test_groupnorm(B: int, T: int, D: int, G: int):
    x = torch.randn(B, T, D).to(device).requires_grad_(True)
    ref = nn.GroupNorm(G, D).to(device)
    tri = GroupNorm(G, D, bias=True).to(device)
    nn.init.normal_(ref.weight)
    nn.init.normal_(ref.bias)
    tri.weight.data.copy_(ref.weight.data)
    tri.bias.data.copy_(ref.bias.data)

    ref_x = rearrange(x, 'b t d -> (b t) d')
    ref_y = rearrange(ref(ref_x), '(b t) d -> b t d', b=B)
    tri_y = tri(x)
    ref_dx = torch.autograd.grad(ref(ref_x).sum(), x)[0]
    tri_dx = torch.autograd.grad(tri(x).sum(), x)[0]
    ref_dw = torch.autograd.grad(ref(ref_x).sum(), ref.weight)[0]
    tri_dw = torch.autograd.grad(tri(x).sum(), tri.weight)[0]
    ref_db = torch.autograd.grad(ref(ref_x).sum(), ref.bias)[0]
    tri_db = torch.autograd.grad(tri(x).sum(), tri.bias)[0]

    torch.testing.assert_close(ref_y, tri_y, rtol=0, atol=1e-4)
    torch.testing.assert_close(ref_dx, tri_dx, rtol=0, atol=1e-4)
    torch.testing.assert_close(ref_dw, tri_dw, rtol=0, atol=1e-3)
    torch.testing.assert_close(ref_db, tri_db, rtol=0, atol=1e-4)


@pytest.mark.parametrize("B", [1, 4, 8])
@pytest.mark.parametrize("H", [1, 4])
@pytest.mark.parametrize("T", [1, 50, 2048])
@pytest.mark.parametrize("D", [50, 64, 128])
def test_rmsnorm(B: int, H: int, T: int, D: int):
    x = torch.randn(B, H, T, D).to(device).requires_grad_(True)
    ref = LlamaRMSNorm(D, eps=0).to(device)
    tri = RMSNorm(D, eps=0).to(device)
    nn.init.normal_(ref.weight)
    tri.weight.data.copy_(ref.weight.data)

    ref_y = ref(x)
    tri_y = tri(x)
    ref_dx = torch.autograd.grad(ref(x).sum(), x)[0]
    tri_dx = torch.autograd.grad(tri(x).sum(), x)[0]

    ref_dw = torch.autograd.grad(ref(x).sum(), ref.weight)[0]
    tri_dw = torch.autograd.grad(tri(x).sum(), tri.weight)[0]

    torch.testing.assert_close(ref_y, tri_y, rtol=0, atol=1e-4)
    torch.testing.assert_close(ref_dx, tri_dx, rtol=0, atol=1e-4)
    torch.testing.assert_close(ref_dw, tri_dw, rtol=0, atol=1e-3)


@pytest.mark.parametrize("N", [1, 16, 128])
@pytest.mark.parametrize("D", [50, 64, 128])
def test_layernorm_linear(N: int, D: int):
    torch.manual_seed(1)
    x = torch.randn(N, D).to(device).requires_grad_(True)
    ref = nn.Sequential(nn.LayerNorm(D, elementwise_affine=True, bias=True), nn.Linear(D, D)).to(device)
    tri = LayerNormLinear(D, elementwise_affine=True, bias=True).to(device)
    nn.init.normal_(ref[0].weight)
    nn.init.normal_(ref[0].bias)
    nn.init.normal_(ref[1].weight, mean=0.0, std=0.01)
    nn.init.normal_(ref[1].bias, mean=0.0, std=0.01)
    tri.weight.data.copy_(ref[0].weight.data)
    tri.bias.data.copy_(ref[0].bias.data)
    weight, bias = ref[1].weight.clone(), ref[1].bias.clone()

    ref_y = ref(x)
    tri_y = tri(x, weight, bias)
    ref_dx = torch.autograd.grad(ref(x).sum(), x)[0]
    tri_dx = torch.autograd.grad(tri(x, weight, bias).sum(), x)[0]
    ref_dw = torch.autograd.grad(ref(x).sum(), ref[0].weight)[0]
    tri_dw = torch.autograd.grad(tri(x, weight, bias).sum(), tri.weight)[0]
    ref_db = torch.autograd.grad(ref(x).sum(), ref[0].bias)[0]
    tri_db = torch.autograd.grad(tri(x, weight, bias).sum(), tri.bias)[0]
    ref_dlw = torch.autograd.grad(ref(x).sum(), ref[1].weight)[0]
    tri_dlw = torch.autograd.grad(tri(x, weight, bias).sum(), weight)[0]
    ref_dlb = torch.autograd.grad(ref(x).sum(), ref[1].bias)[0]
    tri_dlb = torch.autograd.grad(tri(x, weight, bias).sum(), bias)[0]

    torch.testing.assert_close(ref_y, tri_y, rtol=0, atol=1e-4)
    torch.testing.assert_close(ref_dx, tri_dx, rtol=0, atol=1e-4)
    torch.testing.assert_close(ref_dw, tri_dw, rtol=0, atol=1e-3)
    torch.testing.assert_close(ref_db, tri_db, rtol=0, atol=1e-3)
    torch.testing.assert_close(ref_dlw, tri_dlw, rtol=0, atol=1e-4)
    torch.testing.assert_close(ref_dlb, tri_dlb, rtol=0, atol=1e-4)


@pytest.mark.parametrize("N", [1, 16, 128])
@pytest.mark.parametrize("D", [64, 128, 512])
@pytest.mark.parametrize("G", [1, 4])
def test_groupnorm_linear(N: int, D: int, G: int):
    torch.manual_seed(1)
    x = torch.randn(N, D).to(device).requires_grad_(True)
    ref = nn.Sequential(nn.GroupNorm(G, D), nn.Linear(D, D)).to(device)
    tri = GroupNormLinear(G, D, bias=True).to(device)
    nn.init.normal_(ref[0].weight)
    nn.init.normal_(ref[0].bias)
    nn.init.normal_(ref[1].weight, mean=0.0, std=0.01)
    nn.init.normal_(ref[1].bias, mean=0.0, std=0.01)
    tri.weight.data.copy_(ref[0].weight.data)
    tri.bias.data.copy_(ref[0].bias.data)
    weight, bias = ref[1].weight.clone(), ref[1].bias.clone()

    ref_y = ref(x)
    tri_y = tri(x, weight, bias)
    ref_dx = torch.autograd.grad(ref(x).sum(), x)[0]
    tri_dx = torch.autograd.grad(tri(x, weight, bias).sum(), x)[0]
    ref_dw = torch.autograd.grad(ref(x).sum(), ref[0].weight)[0]
    tri_dw = torch.autograd.grad(tri(x, weight, bias).sum(), tri.weight)[0]
    ref_db = torch.autograd.grad(ref(x).sum(), ref[0].bias)[0]
    tri_db = torch.autograd.grad(tri(x, weight, bias).sum(), tri.bias)[0]
    ref_dlw = torch.autograd.grad(ref(x).sum(), ref[1].weight)[0]
    tri_dlw = torch.autograd.grad(tri(x, weight, bias).sum(), weight)[0]
    ref_dlb = torch.autograd.grad(ref(x).sum(), ref[1].bias)[0]
    tri_dlb = torch.autograd.grad(tri(x, weight, bias).sum(), bias)[0]

    torch.testing.assert_close(ref_y, tri_y, rtol=0, atol=1e-4)
    torch.testing.assert_close(ref_dx, tri_dx, rtol=0, atol=1e-4)
    torch.testing.assert_close(ref_dw, tri_dw, rtol=0, atol=1e-3)
    torch.testing.assert_close(ref_db, tri_db, rtol=0, atol=1e-3)
    torch.testing.assert_close(ref_dlw, tri_dlw, rtol=0, atol=1e-4)
    torch.testing.assert_close(ref_dlb, tri_dlb, rtol=0, atol=1e-4)


@pytest.mark.parametrize("N", [1, 16, 128])
@pytest.mark.parametrize("D", [50, 64, 128])
def test_rmsnorm_linear(N: int, D: int):
    torch.manual_seed(1)
    x = torch.randn(N, D).to(device).requires_grad_(True)
    ref = nn.Sequential(LlamaRMSNorm(D, eps=0), nn.Linear(D, D)).to(device)
    tri = RMSNormLinear(D, eps=0).to(device)
    nn.init.normal_(ref[0].weight)
    nn.init.normal_(ref[1].weight, mean=0.0, std=0.01)
    nn.init.normal_(ref[1].bias, mean=0.0, std=0.01)
    tri.weight.data.copy_(ref[0].weight.data)
    weight, bias = ref[1].weight.clone(), ref[1].bias.clone()

    ref_y = ref(x)
    tri_y = tri(x, weight, bias)
    ref_dx = torch.autograd.grad(ref(x).sum(), x)[0]
    tri_dx = torch.autograd.grad(tri(x, weight, bias).sum(), x)[0]
    ref_dw = torch.autograd.grad(ref(x).sum(), ref[0].weight)[0]
    tri_dw = torch.autograd.grad(tri(x, weight, bias).sum(), tri.weight)[0]
    ref_dlw = torch.autograd.grad(ref(x).sum(), ref[1].weight)[0]
    tri_dlw = torch.autograd.grad(tri(x, weight, bias).sum(), weight)[0]
    ref_dlb = torch.autograd.grad(ref(x).sum(), ref[1].bias)[0]
    tri_dlb = torch.autograd.grad(tri(x, weight, bias).sum(), bias)[0]

    torch.testing.assert_close(ref_y, tri_y, rtol=0, atol=1e-4)
    torch.testing.assert_close(ref_dx, tri_dx, rtol=0, atol=1e-4)
    torch.testing.assert_close(ref_dw, tri_dw, rtol=0, atol=1e-3)
    torch.testing.assert_close(ref_dlw, tri_dlw, rtol=0, atol=1e-4)
    torch.testing.assert_close(ref_dlb, tri_dlb, rtol=0, atol=1e-4)
