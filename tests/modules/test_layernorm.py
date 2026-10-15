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
from einops import rearrange
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from fla.modules import GroupNorm, GroupNormLinear, LayerNorm, LayerNormLinear, RMSNorm, RMSNormLinear
from fla.modules.fused_bitlinear import activation_quant, layer_norm_linear_quant, weight_quant
from fla.modules.layernorm import GroupNormRef
from fla.utils import assert_close, device


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("H", [2])
@pytest.mark.parametrize("T", [512])
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

    assert_close(' y', ref_y, tri_y, 1e-3)
    assert_close('dx', ref_dx, tri_dx, 1e-3)
    if ref.weight is not None:
        assert_close('dw', ref_dw, tri_dw, 1e-3)
    if ref.bias is not None:
        assert_close('db', ref_db, tri_db, 1e-3)


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [512])
@pytest.mark.parametrize("D", [64, 128, 512, 1024, 2048, 2052])
@pytest.mark.parametrize("G", [1, 4])
@pytest.mark.parametrize("is_rms_norm", [True, False])
def test_groupnorm(B: int, T: int, D: int, G: int, is_rms_norm: bool):
    torch.manual_seed(42)
    x = torch.randn(B, T, D).to(device).requires_grad_(True)
    if is_rms_norm:
        ref = GroupNormRef(num_groups=G, hidden_size=D, bias=True, is_rms_norm=True).to(device)
    else:
        ref = nn.GroupNorm(G, D).to(device)
    tri = GroupNorm(G, D, bias=True, is_rms_norm=is_rms_norm).to(device)
    nn.init.normal_(ref.weight)
    nn.init.normal_(ref.bias)
    tri.weight.data.copy_(ref.weight.data)
    tri.bias.data.copy_(ref.bias.data)
    ref = ref.to(dtype=torch.float32)

    ref_x = rearrange(x, 'b t d -> (b t) d').to(dtype=torch.float32)
    ref_y = rearrange(ref(ref_x), '(b t) d -> b t d', b=B)
    tri_y = tri(x)
    ref_dx = torch.autograd.grad(ref(ref_x).sum(), x)[0]
    tri_dx = torch.autograd.grad(tri(x).sum(), x)[0]
    ref_dw = torch.autograd.grad(ref(ref_x).sum(), ref.weight)[0]
    tri_dw = torch.autograd.grad(tri(x).sum(), tri.weight)[0]
    ref_db = torch.autograd.grad(ref(ref_x).sum(), ref.bias)[0]
    tri_db = torch.autograd.grad(tri(x).sum(), tri.bias)[0]

    assert_close(' y', ref_y, tri_y, 1e-3)
    assert_close('dx', ref_dx, tri_dx, 1e-3)
    assert_close('dw', ref_dw, tri_dw, 1e-3)
    assert_close('db', ref_db, tri_db, 1e-3)


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("H", [2])
@pytest.mark.parametrize("T", [512])
@pytest.mark.parametrize("D", [50, 64, 128, 256])
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

    assert_close(' y', ref_y, tri_y, 1e-3)
    assert_close('dx', ref_dx, tri_dx, 1e-3)
    assert_close('dw', ref_dw, tri_dw, 1e-3)


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

    assert_close('  y', ref_y, tri_y, 1e-3)
    assert_close(' dx', ref_dx, tri_dx, 1e-3)
    assert_close(' dw', ref_dw, tri_dw, 1e-3)
    assert_close(' db', ref_db, tri_db, 1e-3)
    assert_close('dlw', ref_dlw, tri_dlw, 1e-3)
    assert_close('dlb', ref_dlb, tri_dlb, 1e-3)


@pytest.mark.parametrize(
    ('T', 'D', 'is_rms_norm', 'affine', 'has_residual', 'prenorm', 'residual_in_fp32', 'has_linear_bias', 'contiguous'),
    [
        (1, 64, False, True, False, False, False, True, True),
        (7, 50, True, True, False, False, False, True, True),
        (33, 128, False, True, True, True, False, True, True),
        (257, 128, True, True, True, True, True, True, True),
        (17, 64, False, False, True, False, True, True, True),
        (32, 128, True, False, False, True, True, True, True),
        (65, 257, False, True, True, True, True, False, True),
        (129, 2048, True, True, False, False, False, False, True),
        (35, 63, False, True, True, True, True, False, False),
    ],
    ids=[
        'single-row', 'partial-row', 'residual', 'fp32-residual', 'no-affine', 'prenorm',
        'partial-residual', 'bitlinear', 'non-contiguous',
    ],
)
@pytest.mark.parametrize(
    ('dtype', 'amp_dtype'),
    [(torch.float32, None), (torch.float16, None), (torch.bfloat16, None),
     (torch.float32, torch.float16), (torch.float32, torch.bfloat16)],
    ids=['fp32', 'fp16', 'bf16', 'amp-fp16', 'amp-bf16'],
)
def test_layernorm_linear_quant(
    T: int,
    D: int,
    is_rms_norm: bool,
    affine: bool,
    has_residual: bool,
    prenorm: bool,
    residual_in_fp32: bool,
    has_linear_bias: bool,
    contiguous: bool,
    dtype: torch.dtype,
    amp_dtype: torch.dtype | None,
):
    torch.manual_seed(42)
    x = torch.randn(T, D, device=device, dtype=dtype)
    if not contiguous:
        x = x.t().contiguous().t()
    x.requires_grad_()
    w = torch.randn(D, device=device, dtype=dtype).requires_grad_() if affine else None
    b = torch.randn(D, device=device, dtype=dtype).requires_grad_() if affine else None
    linear_weight = torch.randn(32, D, device=device, dtype=dtype).requires_grad_()
    linear_bias = torch.randn(32, device=device, dtype=dtype).requires_grad_() if has_linear_bias else None
    residual = torch.randn_like(x, dtype=torch.float32 if residual_in_fp32 else dtype) if has_residual else None
    if residual is not None:
        residual.requires_grad_()
    inputs = {
        name: tensor
        for name, tensor in zip(('dx', 'dw', 'db', 'dlw', 'dlb', 'dresidual'), (x, w, b, linear_weight, linear_bias, residual))
        if tensor is not None
    }
    eps = 1e-6
    out_dtype = amp_dtype or dtype

    ref_residual = x.float() + residual.float() if has_residual else x.float()
    ref_norm = ref_residual if is_rms_norm else ref_residual - ref_residual.mean(-1, keepdim=True)
    ref_norm = ref_norm * torch.rsqrt(ref_norm.square().mean(-1, keepdim=True) + eps)
    if affine:
        ref_norm = ref_norm * w.float() + b.float()
    ref_quant = ref_norm + (activation_quant(ref_norm) - ref_norm).detach()
    ref_weight = linear_weight + (weight_quant(linear_weight) - linear_weight).detach()
    ref = F.linear(
        input=ref_quant.to(out_dtype),
        weight=ref_weight.to(out_dtype),
        bias=linear_bias.to(out_dtype) if linear_bias is not None else None,
    )
    if prenorm:
        ref_residual = ref_residual.to(residual.dtype if has_residual else torch.float32 if residual_in_fp32 else dtype)
        ref = (ref, ref_residual)

    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
        tri = layer_norm_linear_quant(
            x=x,
            norm_weight=w,
            norm_bias=b,
            linear_weight=linear_weight,
            linear_bias=linear_bias,
            residual=residual,
            eps=eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=is_rms_norm,
        )
    ref, tri = (ref, tri) if prenorm else ((ref,), (tri,))
    do = tuple(torch.randn_like(output) for output in ref)
    ref_grads = torch.autograd.grad(ref, tuple(inputs.values()), do)
    tri_grads = torch.autograd.grad(tri, tuple(inputs.values()), do)
    for name, expected, actual in zip(('o', 'residual'), ref, tri, strict=False):
        assert torch.isfinite(actual).all()
        assert_close(name, expected, actual, 0.01)
    for name, expected, actual in zip(inputs, ref_grads, tri_grads, strict=True):
        assert torch.isfinite(actual).all()
        assert_close(name, expected, actual, 0.01)


@pytest.mark.parametrize("N", [1, 16, 128])
@pytest.mark.parametrize("D", [64, 128, 512])
@pytest.mark.parametrize("G", [1, 4])
@pytest.mark.parametrize("is_rms_norm", [True, False])
def test_groupnorm_linear(N: int, D: int, G: int, is_rms_norm: bool):
    torch.manual_seed(1)
    x = torch.randn(N, D).to(device).requires_grad_(True)
    if is_rms_norm:
        ref = nn.Sequential(
            GroupNormRef(num_groups=G, hidden_size=D, bias=True, is_rms_norm=True),
            nn.Linear(D, D),
        ).to(device)
    else:
        ref = nn.Sequential(nn.GroupNorm(G, D), nn.Linear(D, D)).to(device)
    tri = GroupNormLinear(G, D, bias=True, is_rms_norm=is_rms_norm).to(device)
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

    assert_close('  y', ref_y, tri_y, 1e-3)
    assert_close(' dx', ref_dx, tri_dx, 1e-3)
    assert_close(' dw', ref_dw, tri_dw, 1e-3)
    assert_close(' db', ref_db, tri_db, 1e-3)
    assert_close('dlw', ref_dlw, tri_dlw, 1e-3)
    assert_close('dlb', ref_dlb, tri_dlb, 1e-3)


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

    assert_close('  y', ref_y, tri_y, 1e-3)
    assert_close(' dx', ref_dx, tri_dx, 1e-3)
    assert_close(' dw', ref_dw, tri_dw, 1e-3)
    assert_close('dlw', ref_dlw, tri_dlw, 1e-3)
    assert_close('dlb', ref_dlb, tri_dlb, 1e-3)


# ============================================================
# Regression tests: layer_norm_bwd_kernel with few tokens
# ============================================================
#
# On GPUs with many SMs (e.g., Blackwell B200 with 160+ SMs),
# when T (total tokens) is small relative to the SM count,
# some Triton programs in layer_norm_bwd_kernel have no work
# (i_sg * BS >= T // G). Without an early-exit guard, these
# idle programs access invalid memory via out-of-bounds tile loads,
# causing "CUDA error: illegal memory access."
#
# The bug triggers when:
#   NS = cdiv(SM_count, G) * G > T
# i.e., more programs launched than tokens to process.
#
# These tests use small T values to ensure the backward kernel
# handles idle programs correctly on any GPU.


@pytest.mark.parametrize("T", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("D", [128, 256, 512])
def test_rmsnorm_small_t(T: int, D: int):
    """RMSNorm backward must handle T < SM_count without illegal memory access."""
    x = torch.randn(T, D).to(device).requires_grad_(True)
    ref = LlamaRMSNorm(D, eps=0).to(device)
    tri = RMSNorm(D, eps=0).to(device)
    nn.init.normal_(ref.weight)
    tri.weight.data.copy_(ref.weight.data)

    ref_y = ref(x)
    tri_y = tri(x)
    assert_close(' y', ref_y, tri_y, 1e-3)

    ref_dx = torch.autograd.grad(ref(x).sum(), x)[0]
    tri_dx = torch.autograd.grad(tri(x).sum(), x)[0]
    assert_close('dx', ref_dx, tri_dx, 1e-3)

    ref_dw = torch.autograd.grad(ref(x).sum(), ref.weight)[0]
    tri_dw = torch.autograd.grad(tri(x).sum(), tri.weight)[0]
    assert_close('dw', ref_dw, tri_dw, 1e-3)


@pytest.mark.parametrize("T", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("D", [128, 256, 512])
def test_layernorm_small_t(T: int, D: int):
    """LayerNorm backward must handle T < SM_count without illegal memory access."""
    x = torch.randn(T, D).to(device).requires_grad_(True)
    ref = nn.LayerNorm(D, elementwise_affine=True, bias=True).to(device)
    tri = LayerNorm(D, elementwise_affine=True, bias=True).to(device)
    nn.init.normal_(ref.weight)
    nn.init.normal_(ref.bias)
    tri.weight.data.copy_(ref.weight.data)
    tri.bias.data.copy_(ref.bias.data)

    ref_y = ref(x)
    tri_y = tri(x)
    assert_close(' y', ref_y, tri_y, 1e-3)

    ref_dx = torch.autograd.grad(ref(x).sum(), x)[0]
    tri_dx = torch.autograd.grad(tri(x).sum(), x)[0]
    assert_close('dx', ref_dx, tri_dx, 1e-3)

    ref_dw = torch.autograd.grad(ref(x).sum(), ref.weight)[0]
    tri_dw = torch.autograd.grad(tri(x).sum(), tri.weight)[0]
    assert_close('dw', ref_dw, tri_dw, 1e-3)

    ref_db = torch.autograd.grad(ref(x).sum(), ref.bias)[0]
    tri_db = torch.autograd.grad(tri(x).sum(), tri.bias)[0]
    assert_close('db', ref_db, tri_db, 1e-3)


@pytest.mark.parametrize("T", [1, 4, 16])
@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("G", [1, 4])
@pytest.mark.parametrize("is_rms_norm", [True, False])
def test_groupnorm_small_t(T: int, D: int, G: int, is_rms_norm: bool):
    """GroupNorm backward must handle T < SM_count without illegal memory access."""
    torch.manual_seed(42)
    x = torch.randn(1, T, D).to(device).requires_grad_(True)
    if is_rms_norm:
        ref = GroupNormRef(num_groups=G, hidden_size=D, bias=True, is_rms_norm=True).to(device)
    else:
        ref = nn.GroupNorm(G, D).to(device)
    tri = GroupNorm(G, D, bias=True, is_rms_norm=is_rms_norm).to(device)
    nn.init.normal_(ref.weight)
    nn.init.normal_(ref.bias)
    tri.weight.data.copy_(ref.weight.data)
    tri.bias.data.copy_(ref.bias.data)
    ref = ref.to(dtype=torch.float32)

    ref_x = x.reshape(T, D).to(dtype=torch.float32)
    ref_y = ref(ref_x).reshape(1, T, D)
    tri_y = tri(x)
    assert_close(' y', ref_y, tri_y, 1e-3)

    ref_dx = torch.autograd.grad(ref(ref_x).sum(), x)[0]
    tri_dx = torch.autograd.grad(tri(x).sum(), x)[0]
    assert_close('dx', ref_dx, tri_dx, 1e-3)

    ref_dw = torch.autograd.grad(ref(ref_x).sum(), ref.weight)[0]
    tri_dw = torch.autograd.grad(tri(x).sum(), tri.weight)[0]
    assert_close('dw', ref_dw, tri_dw, 1e-3)

    ref_db = torch.autograd.grad(ref(ref_x).sum(), ref.bias)[0]
    tri_db = torch.autograd.grad(tri(x).sum(), tri.bias)[0]
    assert_close('db', ref_db, tri_db, 1e-3)
