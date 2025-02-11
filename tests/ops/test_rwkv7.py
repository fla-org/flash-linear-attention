# -*- coding: utf-8 -*-


import pytest
import torch
import torch.nn.functional as F


from fla.utils import device
from fla.ops.rwkv7.recurrent_naive import (naive_recurrent_rwkv7,
                                           naive_recurrent_rwkv7_2,
                                           native_recurrent_rwkv7)
from fla.ops.rwkv7.channel_mixing import (
    rwkv_mix_fwd,
    rwkv_mix_torch,
    rwkv_relu_and_square_fwd,
    rwkv_relu_and_square_torch,
    channel_mixing_rwkv7_torch,
    channel_mixing_rwkv7,
)
from fla.ops.rwkv7.fused_recurrent import fused_recurrent_rwkv7
from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7, torch_addcmul_rwkv7
import torch.test
from utils import assert_close
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [20])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_naive_recurrent_rwkv7(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype
):

    require_grad = True
    torch.manual_seed(42)

    def get_err_ratio(x, y):
        err = (x-y).flatten().square().mean().sqrt().item()
        base = (x).flatten().square().mean().sqrt().item()
        return err / (base + 1e-20)
    q = torch.empty(B, H, T, D, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    k = torch.empty(B, H, T, D, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    v = torch.randn(B, H, T, D, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    w = (torch.randn(B, H, T, D, device=device).uniform_(-8, -6).to(dtype)).requires_grad_(require_grad)
    a = torch.rand(B, H, T, D, device=device, dtype=dtype).clamp(0, 0.1).requires_grad_(require_grad)
    b = torch.randn(B, H, T, D, device=device, dtype=dtype).clamp(-0.2, 0.2).requires_grad_(require_grad)

    do = torch.rand_like(v, device=device).fill_(torch.rand(1).item())
    h = torch.zeros(B, H, D, D, device=device, dtype=torch.float32).requires_grad_(require_grad)
    with torch.no_grad():
        ref_o, _, _ = naive_recurrent_rwkv7(q=q,  k=k, v=v, w=w, a=a, b=b, scale=1.0, initial_state=h)
        ref_o1, _, _ = naive_recurrent_rwkv7_2(q=q,  k=k, v=v, w=w, a=a, b=b, scale=1.0, initial_state=h)

        assert get_err_ratio(ref_o, ref_o1) < 1e-4
        print("Forward pass test passed")

    ref_o, _, _ = naive_recurrent_rwkv7(q=q,  k=k, v=v, w=w, a=a, b=b, scale=1.0, initial_state=h)
    ref_o.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_da, a.grad = a.grad.clone(), None
    ref_db, b.grad = b.grad.clone(), None

    tri_o, _ = native_recurrent_rwkv7(q=q,  k=k, v=v, w=w, a=a, b=b, scale=1.0, initial_state=h)
    tri_o.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_da, a.grad = a.grad.clone(), None
    tri_db, b.grad = b.grad.clone(), None

    atol = 1e-2

    assert get_err_ratio(ref_o, tri_o) < atol
    assert get_err_ratio(ref_dq, tri_dq) < atol
    assert get_err_ratio(ref_dk, tri_dk) < atol
    assert get_err_ratio(ref_dv, tri_dv) < atol
    assert get_err_ratio(ref_dw, tri_dw) < atol
    assert get_err_ratio(ref_da, tri_da) < atol
    assert get_err_ratio(ref_db, tri_db) < atol


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("hidden_dim", [512, 1024, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_rwkv_mix(batch_size, seq_len, hidden_dim, dtype):
    torch.manual_seed(13)

    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    x_prev = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)
    x_k = torch.randn(1, 1, hidden_dim, device=device, dtype=dtype)

    torch_output = rwkv_mix_torch(
        x.to(torch.float32), x_prev.to(torch.float32), x_k.to(torch.float32)
    )
    triton_output = rwkv_mix_fwd(x, x_prev, x_k)
    rtol = 1e-5 if dtype == torch.float32 else 1e-2
    atol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(
        torch_output, triton_output.to(torch.float32), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("seq_len", [1024, 2048, 4096])
@pytest.mark.parametrize("hidden_dim", [512, 1024, 2048])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("inplace", [True, False])
def test_rwkv_relu_and_square(seq_len, hidden_dim, dtype, inplace):
    torch.manual_seed(42)

    x = torch.randn(seq_len, hidden_dim, device=device, dtype=dtype)

    torch_output = rwkv_relu_and_square_torch(x)
    triton_output = rwkv_relu_and_square_fwd(x, inplace=inplace)

    torch.testing.assert_close(torch_output, triton_output, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("n_embd", [512, 1024])
@pytest.mark.parametrize("dim_ffn", [2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_channel_mixing_gradients(batch_size, seq_len, n_embd, dim_ffn, dtype):
    torch.manual_seed(42)
    torch._dynamo.config.cache_size_limit = 512

    x = torch.randn(
        batch_size, seq_len, n_embd, device=device, dtype=dtype, requires_grad=True
    )
    x_prev = torch.randn(
        batch_size, n_embd, device=device, dtype=dtype, requires_grad=True
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

    out2, last2 = channel_mixing_rwkv7(x2, x_prev2, x_k2, K_2, V_2)
    loss2 = out2.mean() + last2.mean()
    loss2.backward()

    # Test gradients
    rtol = 1e-3 if dtype == torch.float32 else 0.025
    atol = 1e-3 if dtype == torch.float32 else 0.025

    torch.testing.assert_close(x.grad, x2.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(x_prev.grad, x_prev2.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(x_k.grad, x_k2.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(K_.grad, K_2.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(V_.grad, V_2.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [4096])
@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_fused_recurrent_rwkv7(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype
):
    require_grad = True
    torch.manual_seed(44)

    def get_err_ratio(x, y):
        err = (x-y).flatten().square().mean().sqrt().item()
        base = (x).flatten().square().mean().sqrt().item()
        return err / (base + 1e-20)
    q = torch.empty(B, H, T, D, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    k = torch.empty(B, H, T, D, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    v = torch.empty(B, H, T, D, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    w = torch.empty(B, H, T, D, device=device).uniform_(-8, -6).to(dtype=dtype).requires_grad_(True)

    kk = torch.empty(B, H, T, D, device=device).uniform_(-1, 1)
    kk = torch.nn.functional.normalize(kk, dim=-1).to(dtype=dtype)

    a = -kk.clone().requires_grad_(True)  # -kk
    a_scale = torch.empty(B, H, T, D, device=device).uniform_(0, 0.1).to(dtype=dtype)
    b = (kk * a_scale).requires_grad_(True)  # kk*a

    do = torch.rand_like(v).to(device).fill_(torch.rand(1).item())
    h = torch.rand(B, H, D, D, device=device, dtype=torch.float32).requires_grad_(require_grad)

    with torch.no_grad():
        q, k, v, w, a, b, h = (x.to(dtype=torch.float64).to('cpu') for x in (q, k, v, w, a, b, h))
        ref_o, ref_state, _ = naive_recurrent_rwkv7(q, k, v, w, a, b, scale=1.0, initial_state=h)
        q, k, v, w, a, b, h = (x.to(dtype=dtype).to(device) for x in (q, k, v, w, a, b, h))
        result, state = fused_recurrent_rwkv7(q, k, v, a, b, initial_state=h.transpose(-1, -2),
                                              w=w, head_first=True)
        if torch.isnan(result).any():
            raise ValueError("NaN detected in output")
        if torch.isnan(ref_o).any():
            raise ValueError("NaN detected in reference output")
        ref_o = ref_o.to(dtype=torch.float32).to(device)
        result = result.to(dtype=torch.float32).to(device)
        ref_state = ref_state.to(dtype=torch.float32).to(device)
        state = state.to(dtype=torch.float32).to(device)
        tol = 1e-3 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(result, ref_o, atol=tol, rtol=tol)
        diff = torch.abs(result - ref_o)
        diff_state = torch.abs(state - ref_state.transpose(-1, -2))
        print("Max error:", diff.max().item(), diff_state.max().item())
        print("Mean error:", diff.mean().item(), diff_state.mean().item())
        print("Forward pass test passed", (ref_o - result).abs().max().item())
        assert get_err_ratio(ref_o, result) < 5e-4
    print('test passed')


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [4096])
@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("use_g", [True, False])
def test_fused_rwkv7_addcmul(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    use_g: bool
):
    T = 4096
    B = 4
    hidden_size = H*D
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
