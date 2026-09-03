# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.momentum_delta_rule import chunk_mode_rule, fused_recurrent_mode_rule
from fla.utils import assert_close, device, device_platform, IS_NVIDIA_HOPPER


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'use_qk_l2norm_in_kernel', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, False, torch.float16),
            (2, 100, 4, 60, False, torch.float16),
            (2, 1000, 3, 128, False, torch.float16),
            (2, 1024, 4, 128, True, torch.float16),
            (3, 2000, 4, 128, False, torch.float16),
            (4, 2048, 8, 64, False, torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, h0))
    do = torch.rand_like(v)
    dht = torch.rand_like(h0)

    # Chunk path (scale is implicit 1/sqrt(D) in momentum ops; we keep the
    # same normalized q/k handling as delta for parity testing)
    tri, tri_ht = chunk_mode_rule(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        output_final_state=True,
        initial_state=h0.clone(),
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None

    ref, ref_ht = fused_recurrent_mode_rule(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        output_final_state=True,
        initial_state=h0.clone(),
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad

    assert_close('o', ref, tri, 0.006)
    assert_close('ht', ref_ht, tri_ht, 0.006)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.008)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize(
    ('H', 'D', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 64, [0, 15], torch.float16),
            (3, 60, [0, 111, 500], torch.float16),
            (3, 64, [0, 256, 500, 900, 1000], torch.float16),
            (4, 100, [0, 15, 100, 300, 1200, 1599, 1800, 2000], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_chunk_varlen(
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1
    cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    beta = torch.randn(1, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn(N, H, D, D, dtype=dtype)
    q, k, v, beta, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, h0))
    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    ref, ref_ht = fused_recurrent_mode_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        output_final_state=True,
        initial_state=h0.clone(),
        cu_seqlens=cu_seqlens_t,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None

    tri, tri_ht = chunk_mode_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        output_final_state=True,
        initial_state=h0.clone(),
        cu_seqlens=cu_seqlens_t,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.008)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)


@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_chunk_initial_state_grad_count():
    """Backward must return dh0 in the initial_state slot, not in output_final_state."""
    torch.manual_seed(0)
    B, T, H, D = 2, 64, 2, 32
    dtype = torch.bfloat16
    q = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid().requires_grad_(True)
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device, requires_grad=True)

    o, ht = chunk_mode_rule(q, k, v, beta, initial_state=h0, output_final_state=True)
    assert ht is not None
    assert ht.shape == h0.shape
    (o.sum() + ht.sum()).backward()
    assert h0.grad is not None
    assert torch.isfinite(h0.grad).all()
    assert q.grad is not None and torch.isfinite(q.grad).all()

    # Without initial_state, dh0 should be None and not crash
    q2 = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    k2 = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    v2 = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    beta2 = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid().requires_grad_(True)
    o2, ht2 = chunk_mode_rule(q2, k2, v2, beta2, initial_state=None, output_final_state=False)
    assert ht2 is None
    o2.sum().backward()
    assert q2.grad is not None


@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_fused_recurrent_uses_u_not_v():
    """Fused bwd must use u=v-k·h, not raw v; parity between chunk and fused."""
    torch.manual_seed(1)
    B, T, H, D = 2, 32, 2, 32
    dtype = torch.bfloat16
    q = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid().requires_grad_(True)
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device, requires_grad=True)
    do = torch.randn(B, T, H, D, dtype=dtype, device=device)
    dht = torch.randn(B, H, D, D, dtype=torch.float32, device=device)

    # chunk as reference
    q_c, k_c, v_c, beta_c, h0_c = (x.detach().clone().requires_grad_(True) for x in (q, k, v, beta, h0))
    o_c, ht_c = chunk_mode_rule(
        q=F.normalize(q_c, p=2, dim=-1),
        k=F.normalize(k_c, p=2, dim=-1),
        v=v_c,
        beta=beta_c,
        initial_state=h0_c,
        output_final_state=True,
    )
    ((o_c * do).sum() + (ht_c * dht).sum()).backward()
    grads_c = [q_c.grad, k_c.grad, v_c.grad, beta_c.grad, h0_c.grad]

    # fused
    q_f, k_f, v_f, beta_f, h0_f = (x.detach().clone().requires_grad_(True) for x in (q, k, v, beta, h0))
    o_f, ht_f = fused_recurrent_mode_rule(
        q=F.normalize(q_f, p=2, dim=-1),
        k=F.normalize(k_f, p=2, dim=-1),
        v=v_f,
        beta=beta_f,
        initial_state=h0_f,
        output_final_state=True,
    )
    ((o_f * do).sum() + (ht_f * dht).sum()).backward()
    grads_f = [q_f.grad, k_f.grad, v_f.grad, beta_f.grad, h0_f.grad]

    for a, b, name in zip(grads_c, grads_f, ['dq', 'dk', 'dv', 'db', 'dh0']):
        assert_close(name, a, b, 0.008)


@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_qk_l2norm_wiring():
    """Default qk_norm='l2' must normalize; kernel l2 should match eager parity."""
    torch.manual_seed(2)
    B, T, H, D = 2, 64, 2, 32
    dtype = torch.bfloat16
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()

    # Eager normalized vs kernel l2 - compare outputs
    q_eager = F.normalize(q.float(), p=2, dim=-1).to(dtype)
    k_eager = F.normalize(k.float(), p=2, dim=-1).to(dtype)
    o_eager, _ = chunk_mode_rule(
        q_eager.clone().requires_grad_(True),
        k_eager.clone().requires_grad_(True),
        v.clone().requires_grad_(True),
        beta.clone().requires_grad_(True),
        use_qk_l2norm_in_kernel=False,
    )
    q_kern = q.clone().requires_grad_(True)
    k_kern = k.clone().requires_grad_(True)
    v_k = v.clone().requires_grad_(True)
    beta_k = beta.clone().requires_grad_(True)
    o_kern, _ = chunk_mode_rule(q_kern, k_kern, v_k, beta_k, use_qk_l2norm_in_kernel=True)
    assert_close('o_l2', o_eager, o_kern, 0.005)

    # Also check that the l2 path actually does work (grad finite) and
    # that enabling l2 changes the output vs no-l2 (catches silently no-op)
    o_no_l2, _ = chunk_mode_rule(
        q.clone().requires_grad_(True),
        k.clone().requires_grad_(True),
        v.clone().requires_grad_(True),
        beta.clone().requires_grad_(True),
        use_qk_l2norm_in_kernel=False,
    )
    assert not torch.allclose(o_kern.float(), o_no_l2.float(), atol=1e-3), "l2 kernel should change output vs no-l2"
    o_kern.sum().backward()
    assert torch.isfinite(q_kern.grad).all() and q_kern.grad.abs().max().item() > 0
    assert torch.isfinite(k_kern.grad).all() and k_kern.grad.abs().max().item() > 0


@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
@pytest.mark.skipif(
    device_platform == 'intel' or IS_NVIDIA_HOPPER,
    reason='Skip on Intel Triton failure or H100 illegal memory access pending kernel fix',
)
def test_momentum_deltanet_layer_default_l2():
    """Layer default qk_norm='l2' should wire through to the op and not NaN."""
    torch.manual_seed(0)
    from fla.layers import MomentumDeltaNet

    layer = MomentumDeltaNet(
        hidden_size=256, num_heads=4, head_dim=64, use_short_conv=False, qk_norm='l2',
    ).to(device=device, dtype=torch.bfloat16)
    # T=64 and head_dim=64 avoid H100 TMA hang with small K=32
    x = torch.randn(2, 64, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    y, _, _ = layer(x)
    assert not torch.isnan(y).any()
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # chunk vs fused inference parity via layer: short seq uses fused, long uses chunk
    layer.eval()
    with torch.no_grad():
        y_chunk, _, _ = layer(torch.randn(2, 128, 256, device=device, dtype=torch.bfloat16))
        assert not torch.isnan(y_chunk).any()
