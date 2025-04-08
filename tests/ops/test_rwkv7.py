# -*- coding: utf-8 -*-

import os

import pytest
import torch

from fla.ops.rwkv7.channel_mixing import channel_mixing_rwkv7, channel_mixing_rwkv7_torch
from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7, torch_addcmul_rwkv7
from fla.ops.utils.testing import assert_close
from fla.utils import device, is_intel_alchemist


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [1024])
@pytest.mark.parametrize("n_embd", [512, 1024])
@pytest.mark.parametrize("dim_ffn", [2048, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("xprevdim", [2, 3])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
def test_channel_mixing_gradients(B, T, n_embd, dim_ffn, dtype, inplace, xprevdim):
    torch.manual_seed(42)
    torch._dynamo.config.cache_size_limit = 512

    x = torch.randn(
        B, T, n_embd, device=device, dtype=dtype, requires_grad=True
    )
    if xprevdim == 3:
        x_prev = torch.randn(
            B, 1, n_embd, device=device, dtype=dtype, requires_grad=True
        )
    else:
        x_prev = torch.randn(
            B, n_embd, device=device, dtype=dtype, requires_grad=True
        )
    x_k = torch.randn(1, 1, n_embd, device=device, dtype=dtype, requires_grad=True)
    K_ = torch.randn(n_embd, dim_ffn, device=device, dtype=dtype, requires_grad=True)
    V_ = torch.randn(dim_ffn, n_embd, device=device, dtype=dtype, requires_grad=True)

    x2 = x.clone().detach().requires_grad_(True)
    x_prev2 = x_prev.clone().detach().requires_grad_(True)
    x_k2 = x_k.clone().detach().requires_grad_(True)
    K_2 = K_.clone().detach().requires_grad_(True)
    V_2 = V_.clone().detach().requires_grad_(True)

    out1, last1 = channel_mixing_rwkv7_torch(
        x.to(torch.float32),
        x_prev.to(torch.float32),
        x_k.to(torch.float32),
        K_.to(torch.float32),
        V_.to(torch.float32),
    )
    loss1 = out1.mean() + last1.mean()
    loss1.backward()

    out2, last2 = channel_mixing_rwkv7(x2, x_prev2, x_k2, K_2, V_2, inplace)
    loss2 = out2.mean() + last2.mean()
    loss2.backward()

    assert_close(" dx", x.grad, x2.grad, ratio=5e-3)
    assert_close(" dxprev", x_prev.grad, x_prev2.grad, ratio=5e-3)
    assert_close(" dx_k", x_k.grad, x_k2.grad, ratio=5e-3)
    assert_close(" dK_", K_.grad, K_2.grad, ratio=5e-3)
    assert_close(" dV_", V_.grad, V_2.grad, ratio=5e-3)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [4096])
@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("use_g", [True, False])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
def test_fused_rwkv7_addcmul(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    use_g: bool
):
    hidden_size = H*D
    if is_intel_alchemist:
        pytest.skip("Skip test because Alchemist does not have enough global shared memory")
    hidden_states = torch.randn(B, T, hidden_size).uniform_(-8, 8).to(device).to(dtype).requires_grad_()
    xx = torch.randn(B, T, hidden_size).uniform_(-8, 8).to(device).to(dtype).requires_grad_()
    x_r = torch.randn(1, 1, hidden_size).uniform_(-8, 8).to(device).to(dtype).requires_grad_()
    x_w = torch.randn(1, 1, hidden_size).uniform_(-8, 8).to(device).to(dtype).requires_grad_()
    x_k = torch.randn(1, 1, hidden_size).uniform_(-8, 8).to(device).to(dtype).requires_grad_()
    x_v = torch.randn(1, 1, hidden_size).uniform_(-8, 8).to(device).to(dtype).requires_grad_()
    x_a = torch.randn(1, 1, hidden_size).uniform_(-8, 8).to(device).to(dtype).requires_grad_()
    if use_g:
        x_g = torch.randn(1, 1, hidden_size).uniform_(-8, 8).to(device).to(dtype).requires_grad_()
    else:
        x_g = None
    xr0, xw0, xk0, xv0, xa0, xg0 = fused_addcmul_rwkv7(hidden_states, xx, x_r, x_w, x_k, x_v, x_a, x_g)
    xr1, xw1, xk1, xv1, xa1, xg1 = torch_addcmul_rwkv7(hidden_states, xx, x_r, x_w, x_k, x_v, x_a, x_g)
    torch.testing.assert_close(xr0, xr1, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(xw0, xw1, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(xk0, xk1, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(xv0, xv1, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(xa0, xa1, rtol=1e-3, atol=1e-3)
    if use_g:
        torch.testing.assert_close(xg0, xg1, rtol=1e-3, atol=1e-3)
        (xr0 + xw0 + xk0 + xv0 + xa0 + xg0).sum().backward()
    else:
        (xr0 + xw0 + xk0 + xv0 + xa0).sum().backward()
    d_ixr = x_r.grad.clone()
    d_ixw = x_w.grad.clone()
    d_ixk = x_k.grad.clone()
    d_ixv = x_v.grad.clone()
    d_ixa = x_a.grad.clone()
    d_hidden = hidden_states.grad.clone()
    d_xx = xx.grad.clone()

    x_r.grad.zero_()
    x_w.grad.zero_()
    x_k.grad.zero_()
    x_v.grad.zero_()
    x_a.grad.zero_()
    if use_g:
        d_ixg = x_g.grad.clone()
        x_g.grad.zero_()
    hidden_states.grad.zero_()
    xx.grad.zero_()

    if use_g:
        (xr1 + xw1 + xk1 + xv1 + xa1 + xg1).sum().backward()
    else:
        (xr1 + xw1 + xk1 + xv1 + xa1).sum().backward()
    d_ixr1 = x_r.grad.clone()
    d_ixw1 = x_w.grad.clone()
    d_ixk1 = x_k.grad.clone()
    d_ixv1 = x_v.grad.clone()
    d_ixa1 = x_a.grad.clone()
    if use_g:
        d_ixg1 = x_g.grad.clone()
    d_hidden1 = hidden_states.grad.clone()
    d_xx1 = xx.grad.clone()

    torch.testing.assert_close(d_ixr, d_ixr1, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(d_ixw, d_ixw1, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(d_ixk, d_ixk1, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(d_ixv, d_ixv1, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(d_ixa, d_ixa1, rtol=1e-3, atol=1e-3)
    if use_g:
        torch.testing.assert_close(d_ixg, d_ixg1, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(d_hidden, d_hidden1, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(d_xx, d_xx1, rtol=1e-3, atol=1e-3)
