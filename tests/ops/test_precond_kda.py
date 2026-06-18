# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.kda.gate import fused_kda_gate, naive_kda_gate
from fla.ops.precond_kda import chunk_precond_kda, fused_recurrent_precond_kda
from fla.ops.precond_kda.naive import naive_recurrent_precond_kda
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ("B", "T", "H", "K", "V", "scale", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-K{}-V{}-scale{}-gate_logit_normalizer{}-{}".format(*test),
        )
        for test in [
            (1, 64, 1, 64, 64, 1, 1, torch.float),
            (2, 512, 3, 60, 60, 1, 1, torch.float),
            (4, 1024, 4, 128, 128, 0.1, 1, torch.float),
            (4, 1024, 4, 128, 128, 1, 10, torch.float),
        ]
    ],
)
def test_naive_chunk(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    """Test chunk forward pass against naive recurrent reference."""
    torch.manual_seed(42)

    q = torch.rand(B, T, H, K, dtype=dtype)
    k = torch.rand(B, T, H, K, dtype=dtype)
    v = torch.rand(B, T, H, V, dtype=dtype)
    gk = F.logsigmoid(torch.randn(B, T, H, K, dtype=torch.float)) / gate_logit_normalizer
    g_atk = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float))
    beta_atk = torch.randn(B, T, H, dtype=dtype).sigmoid()
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    log_atk_scale = torch.full((H,), -0.2, dtype=torch.float32)
    h0 = torch.randn(B, H, K, V, dtype=torch.float32)

    q, k, v, gk, g_atk, beta_atk, beta, h0, log_atk_scale = map(
        lambda x: x.to(device).requires_grad_(True) if x.requires_grad else x.to(device),
        (q, k, v, gk, g_atk, beta_atk, beta, h0, log_atk_scale)
    )
    q, k, v, gk, g_atk, beta_atk, beta, h0 = map(
        lambda x: x.requires_grad_(True),
        (q, k, v, gk, g_atk, beta_atk, beta, h0)
    )

    ref, ref_ht, ref_at = naive_recurrent_precond_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=gk.clone(),
        g_atk=g_atk.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        x=1.5,
        log_atk_scale=log_atk_scale.clone(),
    )

    tri, tri_ht, tri_at = chunk_precond_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=gk.clone(),
        g_atk=g_atk.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        x=1.5,
        log_atk_scale=log_atk_scale.clone(),
    )

    assert_close("o", ref, tri, 0.005)
    if tri_ht is not None:
        assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "H", "K", "V", "scale", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-K{}-V{}-scale{}-gate_logit_normalizer{}-{}".format(*test),
        )
        for test in [
            (1, 64, 1, 64, 64, 1, 1, torch.float),
            (2, 512, 3, 60, 60, 1, 1, torch.float),
            (3, 1000, 4, 100, 100, 0.1, 1, torch.float),
            (4, 1024, 4, 128, 128, 0.1, 1, torch.float),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    """Test fused_recurrent forward against naive recurrent reference."""
    torch.manual_seed(42)

    q = torch.rand(B, T, H, K, dtype=dtype)
    k = torch.rand(B, T, H, K, dtype=dtype)
    v = torch.rand(B, T, H, V, dtype=dtype)
    gk = F.logsigmoid(torch.randn(B, T, H, K, dtype=torch.float)) / gate_logit_normalizer
    g_atk = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float))
    beta_atk = torch.randn(B, T, H, dtype=dtype).sigmoid()
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    log_atk_scale = torch.full((H,), -0.2, dtype=torch.float32)
    h0 = torch.randn(B, H, K, V, dtype=torch.float32)

    q, k, v, gk, g_atk, beta_atk, beta, log_atk_scale, h0 = map(
        lambda x: x.to(device).requires_grad_(False),
        (q, k, v, gk, g_atk, beta_atk, beta, log_atk_scale, h0)
    )

    ref, ref_ht, ref_at = naive_recurrent_precond_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=gk.clone(),
        g_atk=g_atk.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        x=1.5,
        log_atk_scale=log_atk_scale.clone(),
    )

    tri, tri_ht, tri_at = fused_recurrent_precond_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=gk.clone(),
        g_atk=g_atk.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        x=1.5,
        log_atk_scale=log_atk_scale.clone(),
    )

    assert_close("o", ref, tri, 0.005)
    if tri_ht is not None:
        assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "H", "K", "V", "scale", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-K{}-V{}-scale{}-gate_logit_normalizer{}-{}".format(*test),
        )
        for test in [
            (1, 64, 1, 64, 64, 1, 1, torch.float),
            (2, 512, 3, 60, 60, 1, 1, torch.float),
            (3, 1000, 4, 100, 100, 0.1, 1, torch.float),
            (4, 1024, 4, 128, 128, 0.1, 1, torch.float),
        ]
    ],
)
def test_fused_recurrent_transpose_state(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    """Test fused_recurrent with transpose_state_layout=True vs False."""
    torch.manual_seed(42)
    q = torch.rand(B, T, H, K, dtype=dtype)
    k = torch.rand(B, T, H, K, dtype=dtype)
    v = torch.rand(B, T, H, V, dtype=dtype)
    gk = F.logsigmoid(torch.randn(B, T, H, K, dtype=torch.float)) / gate_logit_normalizer
    g_atk = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float))
    beta_atk = torch.randn(B, T, H, dtype=dtype).sigmoid()
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    log_atk_scale = torch.full((H,), -0.2, dtype=torch.float32)
    h0_kv = torch.randn(B, H, K, V, dtype=torch.float32)
    h0_vk = h0_kv.transpose(-1, -2).contiguous()

    q, k, v, gk, g_atk, beta_atk, beta, h0_kv, h0_vk, log_atk_scale = map(
        lambda x: x.to(device),
        (q, k, v, gk, g_atk, beta_atk, beta, h0_kv, h0_vk, log_atk_scale)
    )

    ref, ref_ht, _ = fused_recurrent_precond_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=gk.clone(),
        g_atk=g_atk.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0_kv.clone(),
        output_final_state=True,
        transpose_state_layout=False,
        x=1.5,
        log_atk_scale=log_atk_scale.clone(),
    )
    tri, tri_ht, _ = fused_recurrent_precond_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=gk.clone(),
        g_atk=g_atk.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0_vk.clone(),
        output_final_state=True,
        transpose_state_layout=True,
        x=1.5,
        log_atk_scale=log_atk_scale.clone(),
    )
    assert_close("o", ref, tri, 1e-4)
    assert_close("ht", ref_ht, tri_ht.transpose(-1, -2), 1e-4)


@pytest.mark.parametrize(
    (
        "B",
        "T",
        "H",
        "K",
        "V",
        "scale",
        "mask_p",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "solve_tril_precision",
        "dtype",
    ),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-K{}-V{}-scale{}-mask_p{}-qk_l2norm{}-gate{}-solve_tril{}-{}".format(*test),
        )
        for test in [
            (1, 63, 1, 64, 64, 1, 0, False, False, 'tf32x3', torch.float),
            (2, 500, 3, 60, 60, 1, 0, False, False, 'tf32x3', torch.float16),
            (2, 1000, 3, 64, 64, 0.1, 0.5, False, False, 'tf32', torch.float16),
            (3, 1024, 4, 100, 100, 1, 0, False, False, 'tf32', torch.float16),
            (4, 1024, 4, 128, 128, 0.1, 0, False, False, 'tf32x3', torch.float16),
            (4, 1024, 4, 128, 128, 0.1, 0, True, False, 'tf32', torch.bfloat16),
            (2, 1500, 4, 128, 128, 0.1, 0, False, True, 'tf32x3', torch.float16),
            (4, 2048, 8, 64, 64, 0.1, 0, False, True, 'tf32', torch.float16),
            # High masking + gate: mirrors the failing varlen config in non-varlen form
            (2, 1000, 4, 64, 64, 0.1, 0.9, False, True, 'tf32x3', torch.float16),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    scale: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    solve_tril_precision: str,
    dtype: torch.dtype,
):
    """Test chunk forward + backward against naive reference."""
    torch.manual_seed(42)

    q = torch.rand(B, T, H, K, dtype=dtype)
    k = torch.rand(B, T, H, K, dtype=dtype)
    v = torch.rand(B, T, H, V, dtype=dtype)
    gk = torch.randn(B, T, H, K, dtype=torch.float if not use_gate_in_kernel else dtype)
    if use_gate_in_kernel:
        A_log = torch.randn(H, dtype=torch.float)
        dt_bias = torch.randn(H * K, dtype=torch.float)
        if mask_p > 0:
            # Mask gk to -1000 for strong decay (matches varlen test behavior for gate_in_kernel=True)
            mask = torch.rand_like(gk) > mask_p
            gk = gk * mask + (~mask).float() * (-1000)
    else:
        gk = F.logsigmoid(gk)
        gk = gk * (torch.rand_like(gk) > mask_p)
    g_atk = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float))
    beta_atk = torch.randn(B, T, H, dtype=dtype).sigmoid()
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    log_atk_scale = torch.full((H,), -0.2, dtype=torch.float32)
    h0 = torch.randn(B, H, K, V, dtype=torch.float32)

    if use_gate_in_kernel:
        A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(True), (A_log, dt_bias))
    q, k, v, gk, g_atk, beta_atk, beta, h0, log_atk_scale = map(
        lambda x: x.to(device).requires_grad_(True),
        (q, k, v, gk, g_atk, beta_atk, beta, h0, log_atk_scale)
    )

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    # Naive reference: manually L2 normalize q, k
    ref, ref_ht, ref_at = naive_recurrent_precond_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=(naive_kda_gate(gk, A_log, dt_bias) if use_gate_in_kernel else gk.clone()),
        g_atk=g_atk.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        x=1.5,
        log_atk_scale=log_atk_scale.clone(),
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    if use_gate_in_kernel:
        ref_dA, A_log.grad = A_log.grad, None
        ref_dbias, dt_bias.grad = dt_bias.grad, None
    ref_dq, ref_dk, ref_dv = q.grad, k.grad, v.grad
    ref_dg, ref_dg_atk = gk.grad, g_atk.grad
    ref_dbeta_atk, ref_dbeta = beta_atk.grad, beta.grad
    ref_dlog_atk_scale = log_atk_scale.grad
    ref_dh0 = h0.grad
    q.grad = k.grad = v.grad = gk.grad = g_atk.grad = None
    beta_atk.grad = beta.grad = h0.grad = log_atk_scale.grad = None

    tri, tri_ht, tri_at = chunk_precond_kda(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=gk.clone(),
        g_atk=g_atk.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        solve_tril_precision=solve_tril_precision,
        x=1.5,
        log_atk_scale=log_atk_scale.clone(),
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    if use_gate_in_kernel:
        tri_dA, A_log.grad = A_log.grad, None
        tri_dbias, dt_bias.grad = dt_bias.grad, None
    tri_dq, tri_dk, tri_dv = q.grad, k.grad, v.grad
    tri_dg, tri_dg_atk = gk.grad, g_atk.grad
    tri_dbeta_atk, tri_dbeta = beta_atk.grad, beta.grad
    tri_dlog_atk_scale = log_atk_scale.grad
    tri_dh0 = h0.grad

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.008)
    assert_close("dk", ref_dk, tri_dk, 0.008)
    assert_close("dv", ref_dv, tri_dv, 0.008)
    assert_close("dg", ref_dg, tri_dg, 0.02)
    assert_close("dg_atk", ref_dg_atk, tri_dg_atk, 0.02)
    assert_close("dbeta_atk", ref_dbeta_atk, tri_dbeta_atk, 0.02)
    assert_close("dbeta", ref_dbeta, tri_dbeta, 0.02)
    if use_gate_in_kernel:
        assert_close("dA", ref_dA, tri_dA, 0.003, warning=True)
        assert_close("dbias", ref_dbias, tri_dbias, 0.008)
    if tri_dlog_atk_scale is not None and ref_dlog_atk_scale is not None:
        assert_close("dlog_atk_scale", ref_dlog_atk_scale, tri_dlog_atk_scale, 0.02)
    assert_close("dh0", ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize(
    ("B", "T", "H", "K", "V", "scale", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-K{}-V{}-scale{}-gate_logit_normalizer{}-{}".format(*test),
        )
        for test in [
            (1, 63, 1, 64, 64, 1, 1, torch.float16),
            (2, 500, 3, 60, 60, 1, 1, torch.float16),
            (3, 1024, 4, 128, 128, 0.1, 1, torch.float16),
            (4, 2048, 8, 64, 64, 0.1, 1, torch.float16),
        ]
    ],
)
def test_chunk_transpose_state(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    """Test chunk with transpose_state_layout=True vs False."""
    torch.manual_seed(42)
    q = torch.rand(B, T, H, K, dtype=dtype)
    k = torch.rand(B, T, H, K, dtype=dtype)
    v = torch.rand(B, T, H, V, dtype=dtype)
    gk = F.logsigmoid(torch.randn(B, T, H, K, dtype=torch.float)) / gate_logit_normalizer
    g_atk = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float))
    beta_atk = torch.randn(B, T, H, dtype=dtype).sigmoid()
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    log_atk_scale = torch.full((H,), -0.2, dtype=torch.float32)
    h0_kv = torch.randn(B, H, K, V, dtype=torch.float32)
    h0_vk = h0_kv.transpose(-1, -2).contiguous()

    q, k, v, gk, g_atk, beta_atk, beta, h0_kv, h0_vk, log_atk_scale = map(
        lambda x: x.to(device).requires_grad_(True),
        (q, k, v, gk, g_atk, beta_atk, beta, h0_kv, h0_vk, log_atk_scale)
    )

    do = torch.randn_like(v)
    dht_vk = torch.randn(B, H, K, V, dtype=torch.float32, device=device)
    dht_kv = dht_vk.transpose(-1, -2).contiguous()

    tri, tri_ht, _ = chunk_precond_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=gk.clone(),
        g_atk=g_atk.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0_vk.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        transpose_state_layout=True,
        x=1.5,
        log_atk_scale=log_atk_scale.clone(),
    )
    ((tri * do).sum() + (tri_ht * dht_vk).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv = q.grad, k.grad, v.grad
    tri_dg, tri_dg_atk = gk.grad, g_atk.grad
    tri_dbeta_atk, tri_dbeta = beta_atk.grad, beta.grad
    tri_dh0 = h0_vk.grad
    q.grad = k.grad = v.grad = gk.grad = g_atk.grad = None
    beta_atk.grad = beta.grad = h0_vk.grad = log_atk_scale.grad = None

    ref, ref_ht, _ = chunk_precond_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=gk.clone(),
        g_atk=g_atk.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0_kv.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        transpose_state_layout=False,
        x=1.5,
        log_atk_scale=log_atk_scale.clone(),
    )
    ((ref * do).sum() + (ref_ht * dht_kv).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv = q.grad, k.grad, v.grad
    ref_dg, ref_dg_atk = gk.grad, g_atk.grad
    ref_dbeta_atk, ref_dbeta = beta_atk.grad, beta.grad
    ref_dh0 = h0_kv.grad

    assert_close("o", ref, tri, 1e-4)
    assert_close("ht", ref_ht, tri_ht.transpose(-1, -2), 1e-4)
    assert_close("dq", ref_dq, tri_dq, 1e-4)
    assert_close("dk", ref_dk, tri_dk, 1e-4)
    assert_close("dv", ref_dv, tri_dv, 1e-4)
    assert_close("dg", ref_dg, tri_dg, 1e-4)
    assert_close("dg_atk", ref_dg_atk, tri_dg_atk, 1e-4)
    assert_close("dbeta_atk", ref_dbeta_atk, tri_dbeta_atk, 1e-4)
    assert_close("dbeta", ref_dbeta, tri_dbeta, 1e-4)
    assert_close("dh0", ref_dh0, tri_dh0.transpose(-1, -2), 1e-4)


@pytest.mark.parametrize(
    ("H", "K", "V", "mask_p", "cu_seqlens", "dtype", "use_gate_in_kernel", "solve_tril_precision", "safe_gate", "disable_recompute"),
    [
        pytest.param(*test, id="H{}-K{}-V{}-mask_p{}-cu_seqlens{}-{}-gate{}-solve_tril{}-safe_gate{}-disable_recompute{}".format(*test))
        for test in [
            (4, 60, 60, 0.1, [0, 15], torch.float16, True, 'tf32x3', False, False),
            (4, 64, 64, 0.9, [0, 256, 500, 1000], torch.float16, True, 'tf32', False, False),
            (4, 128, 128, 0.5, [0, 256, 500, 1000], torch.float16, False, 'tf32x3', False, False),
            (4, 100, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.float16, True, 'tf32', False, False),
            (4, 64, 64, 0, [0, 100, 300, 1200, 3000, 4096], torch.float16, False, 'tf32x3', True, True),
        ]
    ],
)
def test_chunk_varlen(
    H: int,
    K: int,
    V: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
    use_gate_in_kernel: bool,
    solve_tril_precision: str,
    safe_gate: bool,
    disable_recompute: bool,
):
    """Test chunk with variable length sequences."""
    torch.manual_seed(42)
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.rand((1, T, H, K), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.rand((1, T, H, V), dtype=dtype)
    gk = torch.randn(1, T, H, K, dtype=torch.float if not use_gate_in_kernel else dtype)
    if use_gate_in_kernel:
        A_log = torch.log(torch.randn(1, 1, H, 1, dtype=torch.float32, device=device).uniform_(1, 16))
        dt_bias = torch.randn(H * K, dtype=torch.float32, device=device)
    else:
        gk = F.logsigmoid(gk)
        gk = gk * (torch.rand_like(gk) > mask_p)
    mask = torch.rand_like(gk) > mask_p
    gk = gk * mask + (~mask) * (-1000)
    if safe_gate:
        assert not use_gate_in_kernel
        gk = gk.clamp(-5, 0)

    g_atk = F.logsigmoid(torch.randn(1, T, H, dtype=torch.float))
    beta_atk = torch.rand(1, T, H, dtype=dtype).sigmoid()
    beta = torch.rand(1, T, H, dtype=dtype).sigmoid()
    log_atk_scale = torch.full((H,), -0.2, dtype=torch.float32)
    h0 = torch.randn((N, H, K, V), dtype=torch.float32)

    q, k, v, gk, g_atk, beta_atk, beta, h0, log_atk_scale = map(
        lambda x: x.to(device).requires_grad_(True),
        (q, k, v, gk, g_atk, beta_atk, beta, h0, log_atk_scale)
    )
    if use_gate_in_kernel:
        A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(), (A_log, dt_bias))
    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    tri, tri_ht, tri_at = chunk_precond_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=gk.clone(),
        g_atk=g_atk.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        disable_recompute=disable_recompute,
        solve_tril_precision=solve_tril_precision,
        x=1.5,
        log_atk_scale=log_atk_scale.clone(),
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv = q.grad, k.grad, v.grad
    tri_dg, tri_dg_atk = gk.grad, g_atk.grad
    tri_dbeta_atk, tri_dbeta = beta_atk.grad, beta.grad
    tri_dh0 = h0.grad
    tri_dlog_atk_scale = log_atk_scale.grad
    q.grad = k.grad = v.grad = gk.grad = g_atk.grad = None
    beta_atk.grad = beta.grad = h0.grad = log_atk_scale.grad = None
    if use_gate_in_kernel:
        tri_dA, A_log.grad = A_log.grad, None
        tri_dbias, dt_bias.grad = dt_bias.grad, None

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i, ref_at_i = naive_recurrent_precond_kda(
            q=F.normalize(q[:, cu_seqlens[i]: cu_seqlens[i + 1]], p=2, dim=-1),
            k=F.normalize(k[:, cu_seqlens[i]: cu_seqlens[i + 1]], p=2, dim=-1),
            v=v[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            g=(naive_kda_gate(
                gk[:, cu_seqlens[i]: cu_seqlens[i + 1]].to(torch.float),
                A_log.to(torch.float),
                dt_bias.to(torch.float)
            ) if use_gate_in_kernel else gk[:, cu_seqlens[i]: cu_seqlens[i + 1]]),
            g_atk=g_atk[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            beta_atk=beta_atk[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            beta=beta[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            initial_state=h0[i],
            output_final_state=True,
            x=1.5,
            log_atk_scale=log_atk_scale.clone(),
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i.squeeze(0))
    ref = torch.cat(ref, 1)
    ref_ht = torch.stack(ref_ht, 0)

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv = q.grad, k.grad, v.grad
    ref_dg, ref_dg_atk = gk.grad, g_atk.grad
    ref_dbeta_atk, ref_dbeta = beta_atk.grad, beta.grad
    ref_dh0 = h0.grad
    ref_dlog_atk_scale = log_atk_scale.grad
    if use_gate_in_kernel:
        ref_dA, A_log.grad = A_log.grad, None
        ref_dbias, dt_bias.grad = dt_bias.grad, None

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.007)
    assert_close("dk", ref_dk, tri_dk, 0.008)
    assert_close("dv", ref_dv, tri_dv, 0.007)
    assert_close("dg", ref_dg, tri_dg, 0.015)
    assert_close("dg_atk", ref_dg_atk, tri_dg_atk, 0.015)
    assert_close("dbeta_atk", ref_dbeta_atk, tri_dbeta_atk, 0.015)
    assert_close("dbeta", ref_dbeta, tri_dbeta, 0.015)
    assert_close("dh0", ref_dh0, tri_dh0, 0.007)
    if tri_dlog_atk_scale is not None and ref_dlog_atk_scale is not None:
        assert_close("d_log_atk_scale", ref_dlog_atk_scale, tri_dlog_atk_scale, 0.02)
    if use_gate_in_kernel:
        # dA/dbias tolerance is looser than KDA (0.005) because the asymmetric k_precond
        # backward (b_dk2 * k - b_dkt * k_precond vs KDA's (b_dk2 - b_dkt) * k) accumulates
        # more fp16 rounding error under extreme masking with sparse gradient signal.
        assert_close("dA", ref_dA, tri_dA, 0.025, warning=True)
        assert_close("dbias", ref_dbias, tri_dbias, 0.02)


@pytest.mark.parametrize(
    ("H", "K", "V", "mask_p", "cu_seqlens", "dtype", "use_gate_in_kernel", "safe_gate", "disable_recompute"),
    [
        pytest.param(*test, id="H{}-K{}-V{}-mask_p{}-cu_seqlens{}-{}-gate{}-safe_gate{}-disable_recompute{}".format(*test))
        for test in [
            (4, 60, 60, 0.1, [0, 8192], torch.float16, True, False, False),
            (4, 64, 64, 0.9, [0, 256, 500, 1000], torch.float16, True, False, False),
            (4, 128, 128, 0.5, [0, 256, 500, 1000], torch.float16, False, False, False),
            (4, 100, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.float16, True, False, False),
            (4, 64, 64, 0, [0, 100, 300, 1200, 3000, 4096], torch.float16, False, True, True),
        ]
    ],
)
@torch.inference_mode()
def test_chunk_varlen_prefill(
    H: int,
    K: int,
    V: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    disable_recompute: bool,
):
    """Test chunk varlen inference (forward only, no backward)."""
    torch.manual_seed(42)
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.randn((1, T, H, K), dtype=dtype).to(device)
    k = F.normalize(torch.randn(1, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype).to(device)
    v = torch.randn((1, T, H, V), dtype=dtype).to(device)
    gk = torch.randn(1, T, H, K, dtype=torch.float if not use_gate_in_kernel else dtype).to(device)
    if use_gate_in_kernel:
        A_log = torch.log(torch.randn(1, 1, H, 1, dtype=torch.float32, device=device).uniform_(1, 16)).to(device)
        dt_bias = torch.randn(H * K, dtype=torch.float32, device=device).to(device)
    else:
        gk = F.logsigmoid(gk)
        gk = gk * (torch.rand_like(gk) > mask_p)
    mask = torch.rand_like(gk) > mask_p
    gk = gk * mask + (~mask) * (-1000)
    if safe_gate:
        assert not use_gate_in_kernel
        gk = gk.clamp(-5, 0)

    g_atk = F.logsigmoid(torch.randn(1, T, H, dtype=torch.float)).to(device)
    beta_atk = torch.rand(1, T, H, dtype=dtype).sigmoid().to(device)
    beta = torch.rand(1, T, H, dtype=dtype).sigmoid().to(device)
    log_atk_scale = torch.full((H,), -0.2, dtype=torch.float32).to(device)
    h0 = torch.randn((N, H, K, V), dtype=torch.float32).to(device)

    tri, tri_ht, _ = chunk_precond_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=k.clone(),
        v=v.clone(),
        g=gk.clone(),
        g_atk=g_atk.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        disable_recompute=disable_recompute,
        x=1.5,
        log_atk_scale=log_atk_scale.clone(),
    )

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i, _ = naive_recurrent_precond_kda(
            q=F.normalize(q[:, cu_seqlens[i]: cu_seqlens[i + 1]], p=2, dim=-1),
            k=k[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            v=v[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            g=(naive_kda_gate(
                gk[:, cu_seqlens[i]: cu_seqlens[i + 1]].to(torch.float),
                A_log.to(torch.float),
                dt_bias.to(torch.float)
            ) if use_gate_in_kernel else gk[:, cu_seqlens[i]: cu_seqlens[i + 1]]),
            g_atk=g_atk[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            beta_atk=beta_atk[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            beta=beta[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            initial_state=h0[i],
            output_final_state=True,
            x=1.5,
            log_atk_scale=log_atk_scale.clone(),
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i.squeeze(0))
    ref = torch.cat(ref, 1)
    ref_ht = torch.stack(ref_ht, 0)

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "H", "K", "HAS_BIAS"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-bias{}".format(*test))
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
    K: int,
    HAS_BIAS: bool,
):
    """Test gate forward + backward."""
    torch.manual_seed(42)
    g = torch.randn(B, T, H, K, dtype=torch.float32) * 10
    A_log = torch.log(torch.randn(1, 1, H, 1, dtype=torch.float32).uniform_(1, 16))
    dt_bias = torch.randn(H * K, dtype=torch.float32) if HAS_BIAS else None
    g, A_log = map(lambda x: x.to(device).requires_grad_(True), (g, A_log))
    if dt_bias is not None:
        dt_bias = dt_bias.to(device).requires_grad_(True)
    do = torch.randn_like(g).view(B, T, H, K)

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
