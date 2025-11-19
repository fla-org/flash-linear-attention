# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.generalized_delta_rule.dplr.fused_recurrent import fused_recurrent_dplr_delta_rule
from fla.ops.rwkv7.channel_mixing import channel_mixing_rwkv7, channel_mixing_rwkv7_torch
from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7, torch_addcmul_rwkv7
from fla.ops.rwkv7.fused_k_update import fused_k_rwkv7, k_update_ref
from fla.ops.rwkv7.fused_recurrent import fused_mul_recurrent_rwkv7
from fla.ops.rwkv7.gate_output_correction import gate_output_correction, gate_output_correction_ref
from fla.utils import IS_NVIDIA_HOPPER, assert_close, device


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [1024])
@pytest.mark.parametrize("n_embd", [1024])
@pytest.mark.parametrize("dim_ffn", [4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("xprevdim", [2, 3])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled",
)
def test_channel_mixing_gradients(B, T, n_embd, dim_ffn, dtype, inplace, xprevdim):
    torch.manual_seed(42)
    torch._dynamo.config.cache_size_limit = 512

    x = torch.randn(
        B, T, n_embd, device=device, dtype=dtype, requires_grad=True,
    )
    if xprevdim == 3:
        x_prev = torch.randn(
            B, 1, n_embd, device=device, dtype=dtype, requires_grad=True,
        )
    else:
        x_prev = torch.randn(
            B, n_embd, device=device, dtype=dtype, requires_grad=True,
        )
    x_k = torch.randn(1, 1, n_embd, device=device, dtype=dtype, requires_grad=True)
    K_ = torch.randn(n_embd, dim_ffn, device=device, dtype=dtype, requires_grad=True)
    V_ = torch.randn(dim_ffn, n_embd, device=device, dtype=dtype, requires_grad=True)

    x2 = x.clone().detach().requires_grad_(True)
    x_prev2 = x_prev.clone().detach().requires_grad_(True)
    x_k2 = x_k.clone().detach().requires_grad_(True)
    K_2 = K_.clone().detach().requires_grad_(True)
    V_2 = V_.clone().detach().requires_grad_(True)

    o1, last1 = channel_mixing_rwkv7_torch(
        x.to(torch.float32),
        x_prev.to(torch.float32),
        x_k.to(torch.float32),
        K_.to(torch.float32),
        V_.to(torch.float32),
    )
    loss1 = o1.mean() + last1.mean()
    loss1.backward()

    o2, last2 = channel_mixing_rwkv7(x2, x_prev2, x_k2, K_2, V_2, inplace)
    loss2 = o2.mean() + last2.mean()
    loss2.backward()

    assert_close(" dx", x.grad, x2.grad, ratio=5e-3)
    assert_close(" dxprev", x_prev.grad, x_prev2.grad, ratio=5e-3)
    assert_close(" dx_k", x_k.grad, x_k2.grad, ratio=5e-3)
    assert_close(" dK_", K_.grad, K_2.grad, ratio=5e-3)
    assert_close(" dV_", V_.grad, V_2.grad, ratio=5e-3)


@pytest.mark.parametrize('B', [2])
@pytest.mark.parametrize('T', [1, 1024])
@pytest.mark.parametrize('H', [1])
@pytest.mark.parametrize('D', [64])
@pytest.mark.parametrize('scale', [None, 1])
@pytest.mark.parametrize('dtype', [torch.float32])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled',
)
def test_fused_mul_recurrent_fwd(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    r = torch.empty(B, T, H, D, device=device).uniform_(-8, -6).to(dtype=dtype)
    k = torch.empty(B, T, H, D, device=device).uniform_(-8, -6).to(dtype=dtype)
    v = torch.empty(B, T, H, D, device=device).uniform_(-8, -6).to(dtype=dtype)
    w = torch.empty(B, T, H, D, device=device).uniform_(-8, -6).to(dtype=dtype)

    kk = torch.empty(B, T, H, D, device=device).uniform_(-1, 1)
    kk = F.normalize(kk, dim=-1).to(dtype=dtype)

    a = -kk.clone()
    a_scale = torch.empty(B, T, H, D, device=device).uniform_(0, 0.1).to(dtype=dtype)
    b = (kk * a_scale).requires_grad_(False)  # kk*a
    h0 = torch.randn(B, H, D, D, dtype=torch.float)
    r, k, v, a, a_scale, b, w, h0 = map(lambda x: x.to(device).requires_grad_(False),
                                        (r, k, v, a, a_scale, b, w, h0))
    ref, ref_ht = fused_recurrent_dplr_delta_rule(
        q=r.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=w.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    tri, tri_ht = fused_mul_recurrent_rwkv7(
        r=r.clone(),
        w=w.clone(),
        k=k.clone(),
        v=v.clone(),
        kk=kk.clone(),
        a=a_scale.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    assert_close('o', ref, tri, 0.002)
    assert_close('ht', ref_ht, tri_ht, 0.002)


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("T", [20, 1024, 4100, 131072])
@pytest.mark.parametrize("H", [2])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_g", [True, False])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled",
)
def test_fused_rwkv7_addcmul(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    use_g: bool,
):
    if T == 128 * 1024 and not IS_NVIDIA_HOPPER:
        pytest.skip("Skipping test for T=131072 on non-Hopper GPUs")
    hidden_size = H*D
    hidden_states = torch.randn(B, T, hidden_size).to(device).to(dtype).requires_grad_()
    xx = torch.randn(B, T, hidden_size).to(device).to(dtype).requires_grad_()
    x_r = torch.randn(1, 1, hidden_size).to(device).to(dtype).requires_grad_()
    x_w = torch.randn(1, 1, hidden_size).to(device).to(dtype).requires_grad_()
    x_k = torch.randn(1, 1, hidden_size).to(device).to(dtype).requires_grad_()
    x_v = torch.randn(1, 1, hidden_size).to(device).to(dtype).requires_grad_()
    x_a = torch.randn(1, 1, hidden_size).to(device).to(dtype).requires_grad_()
    if use_g:
        x_g = torch.randn(1, 1, hidden_size).to(device).to(dtype).requires_grad_()
    else:
        x_g = None
    xr0, xw0, xk0, xv0, xa0, xg0 = fused_addcmul_rwkv7(hidden_states, xx, x_r, x_w, x_k, x_v, x_a, x_g)
    xr1, xw1, xk1, xv1, xa1, xg1 = torch_addcmul_rwkv7(hidden_states.float(),
                                                       xx.float(), x_r.float(),
                                                       x_w.float(), x_k.float(),
                                                       x_v.float(), x_a.float(),
                                                       x_g.float() if use_g else None)
    ratio = 1e-5 if dtype == torch.float32 else 0.002
    assert_close("xr0", xr0, xr1, ratio=ratio)
    assert_close("xw0", xw0, xw1, ratio=ratio)
    assert_close("xk0", xk0, xk1, ratio=ratio)
    assert_close("xv0", xv0, xv1, ratio=ratio)
    assert_close("xa0", xa0, xa1, ratio=ratio)
    if use_g:
        assert_close("xg0", xg0, xg1, ratio=ratio)
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

    assert_close("d_ixr", d_ixr, d_ixr1, ratio=ratio)
    assert_close("d_ixw", d_ixw, d_ixw1, ratio=ratio)
    assert_close("d_ixk", d_ixk, d_ixk1, ratio=ratio)
    assert_close("d_ixv", d_ixv, d_ixv1, ratio=ratio)
    assert_close("d_ixa", d_ixa, d_ixa1, ratio=ratio)
    if use_g:
        assert_close("d_ixg", d_ixg, d_ixg1, ratio=ratio)
    assert_close("d_hidden", d_hidden, d_hidden1, ratio=ratio)
    assert_close("d_xx", d_xx, d_xx1, ratio=ratio)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [13, 4096, 8000])
@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ka_shape", [1, 3])
def test_fused_k_update(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    ka_shape: int,
):
    k = torch.randn(B, T, H*D).uniform_(-8, 8).to(device).to(dtype).requires_grad_()
    a = torch.randn(B, T, H*D).uniform_(-8, 8).to(device).to(dtype).requires_grad_()
    if ka_shape == 1:
        ka = torch.randn(H*D).uniform_(-8, 8).to(device).to(dtype).requires_grad_()
    else:
        ka = torch.randn(1, 1, H*D).uniform_(-8, 8).to(device).to(dtype).requires_grad_()

    ref = k_update_ref(k.float(), a.float(), ka.float())
    ref.sum().backward()
    ref_dk, k.grad = k.grad.clone(), None
    ref_da, a.grad = a.grad.clone(), None
    ref_dka, ka.grad = ka.grad.clone(), None
    tri = fused_k_rwkv7(k, a, ka)
    tri.sum().backward()
    ratio = 5e-5 if dtype == torch.float32 else 0.002
    assert_close("  o", tri, ref, ratio=ratio)
    assert_close(" dk", ref_dk, k.grad, ratio=ratio)
    assert_close(" da", ref_da, a.grad, ratio=ratio)
    assert_close("dka", ref_dka, ka.grad, ratio=ratio)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [4096])
@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gate_output_correction(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    value_dim = H * D
    torch.manual_seed(0)

    o_ref = torch.randn(B, T, value_dim, device=device, dtype=dtype, requires_grad=True)
    r_ref = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    k_ref = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    r_k_ref = torch.randn(H, D, device=device, dtype=dtype, requires_grad=True)
    v_ref = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    g_ref = torch.randn(B, T, value_dim, device=device, dtype=dtype, requires_grad=True)

    tensors_cus = [t.clone().detach().requires_grad_(True) for t in [o_ref, r_ref, k_ref, r_k_ref, v_ref, g_ref]]
    o_cus, r_cus, k_cus, r_k_cus, v_cus, g_cus = tensors_cus

    output_ref = gate_output_correction_ref(o_ref.float(), r_ref.float(), k_ref.float(),
                                            r_k_ref.float(), v_ref.float(), g_ref.float())
    output_ref.sum().backward()

    output_cus = gate_output_correction(o_cus, r_cus, k_cus, r_k_cus, v_cus, g_cus)
    output_cus.sum().backward()

    assert_close(" o", output_ref, output_cus, 0.002)
    assert_close("do", o_ref.grad, o_cus.grad, 0.002)
    assert_close("dr", r_ref.grad, r_cus.grad, 0.002)
    assert_close("dk", k_ref.grad, k_cus.grad, 0.002)
    assert_close("drk", r_k_ref.grad, r_k_cus.grad, 0.002)
    assert_close("dv", v_ref.grad, v_cus.grad, 0.002)
    assert_close("dg", g_ref.grad, g_cus.grad, 0.002)
