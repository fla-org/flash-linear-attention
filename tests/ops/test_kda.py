# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate, naive_kda_gate
from fla.ops.kda.naive import naive_chunk_kda, naive_recurrent_kda
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, device


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "scale", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test),
        )
        for test in [
            (1, 64, 1, 64, 1, 1, torch.float),
            (2, 512, 3, 60, 1, 1, torch.float),
            (4, 1024, 4, 128, 0.1, 1, torch.float),
            (4, 1024, 4, 128, 1, 10, torch.float),
        ]
    ],
)
def test_naive_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason="chunk_gated_delta_rule is not supported on alchemist for D>128")

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float)) / gate_logit_normalizer
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, g, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, g, beta, h0))

    ref, ref_ht = naive_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    tri, tri_ht = naive_chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "scale", "gate_logit_normalizer", "use_qk_l2norm_in_kernel", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-use_qk_l2norm_in_kernel{}-{}".format(*test),
        )
        for test in [
            (1, 64, 1, 64, 1, 1, False, torch.float),
            (2, 512, 3, 60, 1, 1, False, torch.float),
            (3, 1000, 4, 100, 0.1, 1, True, torch.float),
            (4, 1024, 4, 128, 0.1, 1, False, torch.float),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason="chunk_gated_delta_rule is not supported on alchemist for D>128")

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float)) / gate_logit_normalizer
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, g, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, g, beta, h0))

    ref, ref_ht = naive_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    tri, tri_ht = fused_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    (
        "B",
        "T",
        "H",
        "D",
        "scale",
        "gate_logit_normalizer",
        "mask_p",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "dtype",
        "tma",
    ),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-qk_l2norm{}-gate{}-dtype{}-tma{}".format(*test),
        )
        for test in [
            (1, 63, 1, 64, 1, 1, 0, False, False, torch.float16, True),
            (2, 500, 3, 60, 1, 1, 0, False, False, torch.float16, True),
            (2, 1000, 3, 64, 0.1, 1, 0.5, False, False, torch.float16, False),
            (3, 1024, 4, 100, 1, 0.1, 0, False, False, torch.float16, False),
            (4, 1024, 4, 128, 0.1, 1, 0, False, False, torch.float16, True),
            (4, 1024, 4, 128, 0.1, 1, 0, True, False, torch.float16, True),
            (2, 1500, 4, 128, 0.1, 10, 0, False, True, torch.float16, False),
            (4, 2048, 8, 64, 0.1, 1, 0, False, True, torch.float16, True),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    dtype: torch.dtype,
    tma: bool,
):
    torch.manual_seed(42)
    if not tma:
        os.environ["FLA_USE_TMA"] = "0"
    else:
        os.environ["FLA_USE_TMA"] = "1"
    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = torch.randn(B, T, H, D, dtype=torch.float)
    if use_gate_in_kernel:
        A_log = torch.randn(H, dtype=torch.float)
        dt_bias = torch.randn(H * D, dtype=torch.float)
    else:
        g = F.logsigmoid(g) / gate_logit_normalizer
        g = g * (torch.rand_like(g) > mask_p)
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    if use_gate_in_kernel:
        A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(True), (A_log, dt_bias))
    q, k, v, g, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, g, beta, h0))

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    ref, ref_ht = naive_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=(naive_kda_gate(g, A_log, dt_bias) if use_gate_in_kernel else g.clone()),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    if use_gate_in_kernel:
        ref_dA, A_log.grad = A_log.grad, None
        ref_dbias, dt_bias.grad = dt_bias.grad, None
    ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None

    tri, tri_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    if use_gate_in_kernel:
        tri_dA, A_log.grad = A_log.grad, None
        tri_dbias, dt_bias.grad = dt_bias.grad, None
    tri_dq, tri_dk, tri_dv, tri_dg, tri_db, tri_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.008)
    assert_close("dk", ref_dk, tri_dk, 0.008)
    assert_close("dv", ref_dv, tri_dv, 0.008)
    assert_close("dg", ref_dg, tri_dg, 0.02)
    assert_close("db", ref_db, tri_db, 0.02)
    if use_gate_in_kernel:
        assert_close("dA", ref_dA, tri_dA, 0.001)
        assert_close("dbias", ref_dbias, tri_dbias, 0.001)
    assert_close("dh0", ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens", "dtype", "use_tma"),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}-tma{}".format(*test))
        for test in [
            (4, 60, 0.1, [0, 15], torch.float16, False),
            (4, 64, 0.9, [0, 256, 500, 1000], torch.float16, False),
            (4, 128, 0.5, [0, 256, 500, 1000], torch.float16, False),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.float16, False),
            (4, 256, 0, [0, 100, 300, 1200, 3000, 4096], torch.float16, False),
        ]
    ],
)
def test_chunk_varlen(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
    use_tma: bool,
):
    if not use_tma:
        os.environ["FLA_USE_TMA"] = "0"
    else:
        os.environ["FLA_USE_TMA"] = "1"
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    # randomly split the sequence into N segments
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    g = F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float))
    mask = torch.rand_like(g) > mask_p
    g = g * mask + (~mask) * (-1000)

    beta = torch.rand(1, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=torch.float32)

    q, k, v, g, beta, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, g, beta, h0))
    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    tri, tri_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=k.clone(),  # k is already normalized
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dg, tri_db, tri_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = naive_recurrent_kda(
            q=F.normalize(q[:, cu_seqlens[i]: cu_seqlens[i + 1]], p=2, dim=-1),
            k=k[:, cu_seqlens[i]: cu_seqlens[i + 1]],  # k is already normalized
            v=v[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            beta=beta[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            g=g[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.007)
    assert_close("dk", ref_dk, tri_dk, 0.008)
    assert_close("dv", ref_dv, tri_dv, 0.007)
    assert_close("dg", ref_dg, tri_dg, 0.015)
    assert_close("db", ref_db, tri_db, 0.015)
    assert_close("dh0", ref_dh0, tri_dh0, 0.007)
    os.environ["FLA_USE_TMA"] = "0"


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "HAS_BIAS"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-bias{}".format(*test))
        for test in [
            (1, 2, 2, 12, False),
            (1, 32, 2, 16, False),
            (2, 64, 4, 32, False),
            (4, 128, 8, 64, False),
            (4, 128, 8, 128, False),
            (1, 2, 2, 12, True),
            (1, 32, 2, 16, True),
            (2, 64, 4, 32, True),
            (4, 128, 8, 64, True),
            (4, 128, 8, 128, True),
        ]
    ],
)
def test_gate(
    B: int,
    T: int,
    H: int,
    D: int,
    HAS_BIAS: bool,
):
    torch.manual_seed(42)
    g = torch.randn(B, T, H, D, dtype=torch.float32) * 10
    A_log = torch.log(torch.randn(1, 1, H, 1, dtype=torch.float32).uniform_(1, 16))
    dt_bias = torch.randn(H * D, dtype=torch.float32) if HAS_BIAS else None
    g, A_log = map(lambda x: x.to(device).requires_grad_(True), (g, A_log))
    if dt_bias is not None:
        dt_bias = dt_bias.to(device).requires_grad_(True)
    do = torch.randn_like(g).view(B, T, H, D)

    ref = naive_kda_gate(
        g.clone(), A_log.clone(), dt_bias.clone() if dt_bias is not None else None,
    )
    tri = fused_kda_gate(
        g.clone(), A_log.clone(), dt_bias.clone() if dt_bias is not None else None,
    )
    (ref * do).sum().backward(retain_graph=True)

    ref_dg, ref_dA = g.grad, A_log.grad
    ref_dbias = dt_bias.grad if dt_bias is not None else None
    g.grad = A_log.grad = None
    if dt_bias is not None:
        dt_bias.grad = None

    ((tri * do).sum()).backward(retain_graph=True)
    tri_dg, tri_dA = g.grad, A_log.grad
    tri_dbias = dt_bias.grad if dt_bias is not None else None
    g.grad = A_log.grad = None
    if dt_bias is not None:
        dt_bias.grad = None

    assert_close("o", ref, tri, 1e-4)
    assert_close("dg", ref_dg, tri_dg, 1e-4)
    assert_close("dA", ref_dA, tri_dA, 1e-4)
    if HAS_BIAS:
        assert_close("dbias", ref_dbias, tri_dbias, 1e-4)
