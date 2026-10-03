# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.iso_kdn import chunk_iso_kdn, fused_recurrent_iso_kdn, iso_kdn_gain
from fla.ops.iso_kdn.naive import naive_iso_kdn_gain, naive_recurrent_iso_kdn
from fla.ops.kda.gate import naive_kda_gate
from fla.utils import assert_close, device


def make_noise(*shape, lower=0.05):
    return lower + F.softplus(torch.randn(*shape, dtype=torch.float))


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "gate_logit_normalizer", "info_scale", "use_initial_state"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-gate_logit_normalizer{}-info_scale{}-h0{}".format(*test),
        )
        for test in [
            (1, 1, 1, 64, 1, None, False),
            (2, 63, 3, 60, 1, None, True),
            (2, 500, 4, 64, 0.1, None, True),
            (3, 1024, 4, 128, 1, 32, True),
            (2, 2000, 8, 128, 10, None, False),
            (1, 4096, 2, 64, 1, 256, True),
        ]
    ],
)
def test_gain(
    B: int,
    T: int,
    H: int,
    D: int,
    gate_logit_normalizer: float,
    info_scale: float | None,
    use_initial_state: bool,
):
    torch.manual_seed(42)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float), p=2, dim=-1)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float)) / gate_logit_normalizer
    omega = make_noise(B, T, H)
    r = make_noise(B, T, H)
    c0 = make_noise(B, H, lower=0.1) if use_initial_state else None
    k, g, omega, r = map(lambda x: x.to(device), (k, g, omega, r))
    g, omega, r = map(lambda x: x.requires_grad_(True), (g, omega, r))
    if use_initial_state:
        c0 = c0.to(device).requires_grad_(True)

    dbeta = torch.randn(B, T, H, dtype=torch.float, device=device)
    dct = torch.randn(B, H, dtype=torch.float, device=device)

    ref, ref_ct = naive_iso_kdn_gain(
        k=k,
        g=g,
        omega=omega,
        r=r,
        initial_state=c0,
        output_final_state=True,
        info_scale=info_scale,
    )
    ((ref * dbeta).sum() + (ref_ct * dct).sum()).backward()
    ref_dg, ref_domega, ref_dr = g.grad, omega.grad, r.grad
    g.grad = omega.grad = r.grad = None
    if use_initial_state:
        ref_dc0, c0.grad = c0.grad, None

    tri, tri_ct = iso_kdn_gain(
        g=g,
        omega=omega,
        r=r,
        initial_state=c0,
        output_final_state=True,
        info_scale=info_scale,
    )
    ((tri * dbeta).sum() + (tri_ct * dct).sum()).backward()
    tri_dg, tri_domega, tri_dr = g.grad, omega.grad, r.grad
    if use_initial_state:
        tri_dc0 = c0.grad

    assert tri.dtype == torch.float32
    assert_close("beta", ref, tri, 1e-4)
    assert_close("ct", ref_ct, tri_ct, 1e-4)
    assert_close("dg", ref_dg, tri_dg, 1e-4)
    assert_close("domega", ref_domega, tri_domega, 1e-4)
    assert_close("dr", ref_dr, tri_dr, 1e-4)
    if use_initial_state:
        assert_close("dc0", ref_dc0, tri_dc0, 1e-4)


@pytest.mark.parametrize(
    ("cu_seqlens", "H", "D"),
    [
        pytest.param(*test, id="cu_seqlens{}-H{}-D{}".format(*test))
        for test in [
            ([0, 15], 2, 64),
            ([0, 64, 128, 129], 3, 60),
            ([0, 256, 500, 1000], 4, 128),
            ([0, 15, 100, 300, 1200, 2000], 4, 64),
        ]
    ],
)
def test_gain_varlen(cu_seqlens: list[int], H: int, D: int):
    torch.manual_seed(42)
    cu_seqlens_cpu = torch.LongTensor(cu_seqlens)
    cu_seqlens = cu_seqlens_cpu.to(device)
    T, N = cu_seqlens_cpu[-1].item(), len(cu_seqlens_cpu) - 1

    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float), p=2, dim=-1)
    g = F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float))
    omega = make_noise(1, T, H)
    r = make_noise(1, T, H)
    c0 = make_noise(N, H, lower=0.1)
    k = k.to(device)
    g, omega, r, c0 = map(lambda x: x.to(device).requires_grad_(True), (g, omega, r, c0))
    dbeta = torch.randn(1, T, H, dtype=torch.float, device=device)
    dct = torch.randn(N, H, dtype=torch.float, device=device)

    tri, tri_ct = iso_kdn_gain(
        g=g,
        omega=omega,
        r=r,
        initial_state=c0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    ((tri * dbeta).sum() + (tri_ct * dct).sum()).backward()
    tri_dg, tri_domega, tri_dr, tri_dc0 = g.grad, omega.grad, r.grad, c0.grad
    g.grad = omega.grad = r.grad = c0.grad = None

    ref, ref_ct = [], []
    for i in range(N):
        bos, eos = cu_seqlens_cpu[i], cu_seqlens_cpu[i + 1]
        ref_i, ref_ct_i = naive_iso_kdn_gain(
            k=k[:, bos:eos],
            g=g[:, bos:eos],
            omega=omega[:, bos:eos],
            r=r[:, bos:eos],
            initial_state=c0[i:i+1],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ct.append(ref_ct_i)
    ref, ref_ct = torch.cat(ref, 1), torch.cat(ref_ct, 0)
    ((ref * dbeta).sum() + (ref_ct * dct).sum()).backward()
    ref_dg, ref_domega, ref_dr, ref_dc0 = g.grad, omega.grad, r.grad, c0.grad

    assert_close("beta", ref, tri, 1e-4)
    assert_close("ct", ref_ct, tri_ct, 1e-4)
    assert_close("dg", ref_dg, tri_dg, 1e-4)
    assert_close("domega", ref_domega, tri_domega, 1e-4)
    assert_close("dr", ref_dr, tri_dr, 1e-4)
    assert_close("dc0", ref_dc0, tri_dc0, 1e-4)


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "scale", "use_qk_l2norm_in_kernel", "use_gate_in_kernel", "state_v_first", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-HV{}-D{}-scale{}-use_qk_l2norm{}-use_gate{}-state_v_first{}-{}".format(*test),
        )
        for test in [
            (1, 63, 1, 1, 64, 1, False, False, False, torch.float16),
            (2, 500, 3, 3, 60, 1, True, True, False, torch.float16),
            (3, 1024, 4, 4, 128, 0.1, True, False, True, torch.float16),
            (2, 1024, 2, 4, 64, 0.1, False, True, False, torch.bfloat16),
            (2, 1500, 4, 8, 128, 0.1, True, True, True, torch.bfloat16),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    state_v_first: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, HV, D, dtype=dtype)
    if use_gate_in_kernel:
        g = torch.randn(B, T, HV, D, dtype=dtype)
        A_log = torch.randn(HV, dtype=torch.float)
        dt_bias = torch.randn(HV * D, dtype=torch.float)
    else:
        g = F.logsigmoid(torch.randn(B, T, HV, D, dtype=torch.float))
    omega = make_noise(B, T, HV)
    r = make_noise(B, T, HV)
    h0 = torch.randn(B, HV, D, D, dtype=torch.float)
    c0 = make_noise(B, HV, lower=0.1)
    inputs = [q, k, v, g, omega, r, h0, c0] + ([A_log, dt_bias] if use_gate_in_kernel else [])
    inputs = [x.to(device).requires_grad_(True) for x in inputs]
    q, k, v, g, omega, r, h0, c0 = inputs[:8]
    A_log, dt_bias = inputs[8:] if use_gate_in_kernel else (None, None)

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    dct = torch.randn_like(c0)

    ref, (ref_ht, ref_ct) = naive_recurrent_iso_kdn(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=naive_kda_gate(g, A_log, dt_bias) if use_gate_in_kernel else g.clone(),
        omega=omega.clone(),
        r=r.clone(),
        scale=scale,
        initial_state=(h0.clone(), c0.clone()),
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum() + (ref_ct * dct).sum()).backward()
    ref_grads = [x.grad for x in inputs]
    for x in inputs:
        x.grad = None

    tri, (tri_ht, tri_ct) = chunk_iso_kdn(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=g.clone(),
        omega=omega.clone(),
        r=r.clone(),
        scale=scale,
        initial_state=(h0.transpose(-1, -2).contiguous() if state_v_first else h0.clone(), c0.clone()),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        state_v_first=state_v_first,
    )
    if state_v_first:
        tri_ht = tri_ht.transpose(-1, -2)
    ((tri * do).sum() + (tri_ht * dht).sum() + (tri_ct * dct).sum()).backward()
    tri_grads = [x.grad for x in inputs]

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("ct", ref_ct, tri_ct, 0.005)
    for name, ref_grad, tri_grad, ratio in zip(
        ("dq", "dk", "dv", "dg", "domega", "dr", "dh0", "dc0", "dA", "dbias"),
        ref_grads,
        tri_grads,
        (0.008, 0.008, 0.008, 0.02, 0.02, 0.02, 0.008, 0.02, 0.02, 0.02),
    ):
        assert_close(name, ref_grad, tri_grad, ratio)


@pytest.mark.parametrize(
    ("H", "HV", "D", "cu_seqlens", "dtype"),
    [
        pytest.param(*test, id="H{}-HV{}-D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 4, 60, [0, 15], torch.float16),
            (4, 4, 64, [0, 256, 500, 1000], torch.float16),
            (2, 4, 128, [0, 15, 100, 300, 1200, 2000], torch.bfloat16),
        ]
    ],
)
def test_chunk_varlen(H: int, HV: int, D: int, cu_seqlens: list[int], dtype: torch.dtype):
    torch.manual_seed(42)
    cu_seqlens_cpu = torch.LongTensor(cu_seqlens)
    cu_seqlens = cu_seqlens_cpu.to(device)
    T, N = cu_seqlens_cpu[-1].item(), len(cu_seqlens_cpu) - 1

    q = torch.randn(1, T, H, D, dtype=dtype)
    k = torch.randn(1, T, H, D, dtype=dtype)
    v = torch.randn(1, T, HV, D, dtype=dtype)
    g = F.logsigmoid(torch.randn(1, T, HV, D, dtype=torch.float))
    omega = make_noise(1, T, HV)
    r = make_noise(1, T, HV)
    h0 = torch.randn(N, HV, D, D, dtype=torch.float)
    c0 = make_noise(N, HV, lower=0.1)
    q, k, v, g, omega, r, h0, c0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, g, omega, r, h0, c0))

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    dct = torch.randn_like(c0)

    tri, (tri_ht, tri_ct) = chunk_iso_kdn(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        omega=omega.clone(),
        r=r.clone(),
        initial_state=(h0.clone(), c0.clone()),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    ((tri * do).sum() + (tri_ht * dht).sum() + (tri_ct * dct).sum()).backward()
    tri_grads = [x.grad for x in (q, k, v, g, omega, r, h0, c0)]
    for x in (q, k, v, g, omega, r, h0, c0):
        x.grad = None

    ref, ref_ht, ref_ct = [], [], []
    for i in range(N):
        bos, eos = cu_seqlens_cpu[i], cu_seqlens_cpu[i + 1]
        ref_i, (ref_ht_i, ref_ct_i) = naive_recurrent_iso_kdn(
            q=F.normalize(q[:, bos:eos], p=2, dim=-1),
            k=F.normalize(k[:, bos:eos], p=2, dim=-1),
            v=v[:, bos:eos],
            g=g[:, bos:eos],
            omega=omega[:, bos:eos],
            r=r[:, bos:eos],
            initial_state=(h0[i:i+1], c0[i:i+1]),
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
        ref_ct.append(ref_ct_i)
    ref, ref_ht, ref_ct = torch.cat(ref, 1), torch.cat(ref_ht, 0), torch.cat(ref_ct, 0)
    ((ref * do).sum() + (ref_ht * dht).sum() + (ref_ct * dct).sum()).backward()
    ref_grads = [x.grad for x in (q, k, v, g, omega, r, h0, c0)]

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("ct", ref_ct, tri_ct, 0.005)
    for name, ref_grad, tri_grad, ratio in zip(
        ("dq", "dk", "dv", "dg", "domega", "dr", "dh0", "dc0"),
        ref_grads,
        tri_grads,
        (0.008, 0.008, 0.008, 0.02, 0.02, 0.02, 0.008, 0.02),
    ):
        assert_close(name, ref_grad, tri_grad, ratio)


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "use_gate_in_kernel", "state_v_first", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HV{}-D{}-use_gate{}-state_v_first{}-{}".format(*test))
        for test in [
            (1, 1, 1, 1, 64, False, False, torch.float),
            (2, 63, 3, 3, 60, True, True, torch.float),
            (4, 500, 4, 8, 128, False, False, torch.bfloat16),
            (2, 1024, 2, 4, 64, True, True, torch.float16),
        ]
    ],
)
@torch.inference_mode()
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    use_gate_in_kernel: bool,
    state_v_first: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, HV, D, dtype=dtype)
    if use_gate_in_kernel:
        g = torch.randn(B, T, HV, D, dtype=dtype)
        A_log = torch.randn(HV, dtype=torch.float, device=device)
        dt_bias = torch.randn(HV * D, dtype=torch.float, device=device)
    else:
        g = F.logsigmoid(torch.randn(B, T, HV, D, dtype=torch.float))
        A_log, dt_bias = None, None
    omega = make_noise(B, T, HV)
    r = make_noise(B, T, HV)
    h0 = torch.randn(B, HV, D, D, dtype=torch.float)
    c0 = make_noise(B, HV, lower=0.1)
    q, k, v, g, omega, r, h0, c0 = map(lambda x: x.to(device), (q, k, v, g, omega, r, h0, c0))

    ref, (ref_ht, ref_ct) = naive_recurrent_iso_kdn(
        q=F.normalize(q, p=2, dim=-1),
        k=F.normalize(k, p=2, dim=-1),
        v=v,
        g=naive_kda_gate(g, A_log, dt_bias) if use_gate_in_kernel else g,
        omega=omega,
        r=r,
        initial_state=(h0, c0),
        output_final_state=True,
    )
    tri, (tri_ht, tri_ct) = fused_recurrent_iso_kdn(
        q=q,
        k=k,
        v=v,
        g=g,
        omega=omega,
        r=r,
        initial_state=(h0.transpose(-1, -2).contiguous() if state_v_first else h0, c0),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        state_v_first=state_v_first,
    )
    if state_v_first:
        tri_ht = tri_ht.transpose(-1, -2)
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("ct", ref_ct, tri_ct, 0.005)


@torch.inference_mode()
def test_chunk_prefill_decode():
    torch.manual_seed(42)
    B, T, H, HV, D = 2, 300, 2, 4, 64
    dtype = torch.bfloat16
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(B, T, HV, D, dtype=torch.float, device=device))
    omega = make_noise(B, T, HV).to(device)
    r = make_noise(B, T, HV).to(device)
    c0 = make_noise(B, HV, lower=0.1).to(device)

    ref, _ = chunk_iso_kdn(q, k, v, g, omega, r, initial_state=(None, c0), use_qk_l2norm_in_kernel=True)
    # prefill with the chunk kernel, then decode token by token from the cached (memory, precision) state
    P = 200
    tri = [None] * (T - P + 1)
    tri[0], state = chunk_iso_kdn(
        q[:, :P], k[:, :P], v[:, :P], g[:, :P], omega[:, :P], r[:, :P],
        initial_state=(None, c0),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    for i in range(P, T):
        s = slice(i, i + 1)
        tri[i - P + 1], state = fused_recurrent_iso_kdn(
            q[:, s], k[:, s], v[:, s], g[:, s], omega[:, s], r[:, s],
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    assert_close("o", ref, torch.cat(tri, 1), 0.005)
