# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import pytest
import torch
import torch.nn.functional as F

from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd
from fla.ops.kda.gate import fused_kda_gate, naive_kda_gate, naive_kda_lowerbound_gate
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
    ("B", "H", "D", "scale", "gate_logit_normalizer", "use_qk_l2norm_in_kernel", "use_gate_in_kernel", "safe_gate", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-H{}-D{}-scale{}-norm{}-qk_l2{}-gate{}-safe_gate{}-dtype{}".format(*test),
        )
        for test in [
            (16, 16, 128, 0.1, 1.0, True, False, False, torch.bfloat16),
            (32, 8, 64, 1.0, 1.0, False, False, False, torch.float16),
            (7, 32, 128, 0.5, 0.5, True, False, False, torch.bfloat16),  # Odd batch size
            (16, 16, 128, 0.1, 1.0, True, True, False, torch.bfloat16),
            (32, 8, 64, 1.0, 1.0, False, True, False, torch.float16),
            (7, 32, 128, 0.5, 0.5, True, True, True, torch.bfloat16),  # Odd batch size
        ]
    ],
)
def test_fused_recurrent_vllm_decode(
    B: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    dtype: torch.dtype,
):
    """Test vLLM-style decoding with continuous batching and paged state storage."""
    torch.manual_seed(42)
    device = torch.device("cuda")

    # Setup cache pool and inputs
    max_cache_slots = B * 3
    state_pool = torch.randn(max_cache_slots, H, D, D, dtype=torch.float32, device=device)
    state_indices = torch.randperm(max_cache_slots, device=device)[:B].int()

    # Fill unaccessed slots with a huge value to detect out-of-bound access
    HUGE_VALUE = 1e30
    mask = torch.ones(max_cache_slots, dtype=torch.bool, device=device)
    mask[state_indices.long()] = False
    state_pool[mask] = HUGE_VALUE

    T = 1
    total_tokens = B * T

    q = torch.rand(1, total_tokens, H, D, dtype=dtype, device=device)
    k = torch.rand(1, total_tokens, H, D, dtype=dtype, device=device)
    v = torch.rand(1, total_tokens, H, D, dtype=dtype, device=device)
    g = torch.randn(1, total_tokens, H, D, dtype=torch.float if not use_gate_in_kernel else dtype, device=device)

    if use_gate_in_kernel:
        A_log = torch.log(torch.randn(1, 1, H, 1, dtype=torch.float32, device=device).uniform_(1, 16)).squeeze()
        dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)
        lower_bound = -5.0 if safe_gate else None
        naive_kda_gate_fn = naive_kda_lowerbound_gate if safe_gate else naive_kda_gate
    else:
        g = F.logsigmoid(g) / gate_logit_normalizer
        A_log = None
        dt_bias = None
        lower_bound = None
        naive_kda_gate_fn = None

    beta = torch.randn(1, total_tokens, H, dtype=dtype, device=device).sigmoid()

    cu_seqlens = torch.arange(0, total_tokens + 1, step=T, device=device, dtype=torch.int32)
    ref_state_pool = state_pool.clone()
    tri_state_pool = state_pool.clone()

    # Reference implementation (loop over batch)
    ref_outputs = []
    for i in range(B):
        start, end = i, i + 1
        slot_idx = state_indices[i].item()

        q_i = q[:, start:end].clone()
        k_i = k[:, start:end].clone()
        v_i = v[:, start:end].clone()
        g_i = g[:, start:end].clone()
        beta_i = beta[:, start:end].clone()

        h_init = ref_state_pool[slot_idx].clone().unsqueeze(0)
        ref_o_i, ref_ht_i = naive_recurrent_kda(
            q=F.normalize(q_i, p=2, dim=-1),
            k=F.normalize(k_i, p=2, dim=-1),
            v=v_i,
            g=(naive_kda_gate_fn(g_i, A_log, dt_bias) if use_gate_in_kernel else g_i),
            beta=beta_i,
            scale=scale,
            initial_state=h_init,
            output_final_state=True
        )
        ref_outputs.append(ref_o_i)
        ref_state_pool[slot_idx] = ref_ht_i.squeeze(0)

    ref_out = torch.cat(ref_outputs, dim=1)

    # Triton kernel
    q_in = q.clone()
    k_in = k.clone()
    if not use_qk_l2norm_in_kernel:
        q_in = F.normalize(q_in, p=2, dim=-1)
        k_in = F.normalize(k_in, p=2, dim=-1)

    tri_out, _ = fused_recurrent_kda_fwd(
        q=q_in,
        k=k_in,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=tri_state_pool,
        scale=scale,
        output_final_state=False,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_exp2=False,
        lower_bound=lower_bound
    )

    # Verify results
    assert_close("o", ref_out, tri_out, 0.005)
    assert_close("ht", ref_state_pool[state_indices.long()], tri_state_pool[state_indices.long()], 0.005)

    mask = torch.ones(max_cache_slots, dtype=torch.bool, device=device)
    mask[state_indices.long()] = False
    assert_close("Untouched ht", ref_state_pool[mask], tri_state_pool[mask], 0.0)


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
        "safe_gate",
        "disable_recompute",
    ),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-qk_l2norm{}-gate{}-dtype{}-safe_gate{}-disable_recompute{}".format(
                *test),
        )
        for test in [
            (1, 63, 1, 64, 1, 1, 0, False, False, torch.float16, True, False),
            (2, 500, 3, 60, 1, 1, 0, False, False, torch.float16, True, True),
            (2, 1000, 3, 64, 0.1, 1, 0.5, False, False, torch.float16, False, True),
            (3, 1024, 4, 100, 1, 0.1, 0, False, False, torch.float16, False, False),
            (4, 1024, 4, 128, 0.1, 1, 0, False, False, torch.float16, True, True),
            (4, 1024, 4, 128, 0.1, 1, 0, True, False, torch.float16, True, False),
            (2, 1500, 4, 128, 0.1, 10, 0, False, True, torch.float16, False, True),
            (4, 2048, 8, 64, 0.1, 1, 0, False, True, torch.float16, True, True),
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
    safe_gate: bool,
    disable_recompute: bool,
):
    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = torch.randn(B, T, H, D, dtype=torch.float if not use_gate_in_kernel else dtype)
    if use_gate_in_kernel:
        A_log = torch.randn(H, dtype=torch.float)
        dt_bias = torch.randn(H * D, dtype=torch.float)
    else:
        g = F.logsigmoid(g) / gate_logit_normalizer
        g = g * (torch.rand_like(g) > mask_p)
    if safe_gate:
        lower_bound = -5.0
        if not use_gate_in_kernel:
            g = g.clamp(-5, 0)
        naive_kda_gate_fn = naive_kda_lowerbound_gate
    else:
        lower_bound = None
        naive_kda_gate_fn = naive_kda_gate

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
        g=(naive_kda_gate_fn(g, A_log, dt_bias) if use_gate_in_kernel else g.clone()),
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
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        disable_recompute=disable_recompute
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
        assert_close("dA", ref_dA, tri_dA, 0.003, warning=True)
        assert_close("dbias", ref_dbias, tri_dbias, 0.008)
    assert_close("dh0", ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens", "dtype", "use_gate_in_kernel", "safe_gate", "disable_recompute"),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}-gate{}-safe_gate{}-disable_recompute{}".format(*test))
        for test in [
            (4, 60, 0.1, [0, 15], torch.float16, True, False, False),
            (4, 64, 0.9, [0, 256, 500, 1000], torch.float16, True, False, False),
            (4, 128, 0.5, [0, 256, 500, 1000], torch.float16, False, False, False),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.float16, True, False, False),
            (4, 256, 0, [0, 100, 300, 1200, 3000, 4096], torch.float16, False, True, True),
        ]
    ],
)
def test_chunk_varlen(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    disable_recompute: bool,
):
    torch.manual_seed(42)
    # randomly split the sequence into N segments
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    cu_seqlens_cpu = cu_seqlens.cpu()
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    g = torch.randn(1, T, H, D, dtype=torch.float if not use_gate_in_kernel else dtype)
    if use_gate_in_kernel:
        A_log = torch.log(torch.randn(1, 1, H, 1, dtype=torch.float32, device=device).uniform_(1, 16))
        dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)
    else:
        g = F.logsigmoid(g)
        g = g * (torch.rand_like(g) > mask_p)
    mask = torch.rand_like(g) > mask_p
    g = g * mask + (~mask) * (-1000)
    if safe_gate:
        assert use_gate_in_kernel is False
        g = g.clamp(-5, 0)

    beta = torch.rand(1, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=torch.float32)

    q, k, v, g, beta, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, g, beta, h0))
    if use_gate_in_kernel:
        A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(), (A_log, dt_bias))
    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    tri, tri_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=k.clone(),  # k is already normalized
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        disable_recompute=disable_recompute
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dg, tri_db, tri_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None
    if use_gate_in_kernel:
        tri_dA, A_log.grad = A_log.grad, None
        tri_dbias, dt_bias.grad = dt_bias.grad, None

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = naive_recurrent_kda(
            q=F.normalize(q[:, cu_seqlens[i]: cu_seqlens[i + 1]], p=2, dim=-1),
            k=k[:, cu_seqlens[i]: cu_seqlens[i + 1]],  # k is already normalized
            v=v[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            beta=beta[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            g=(naive_kda_gate(g[:, cu_seqlens[i]: cu_seqlens[i + 1]].to(torch.float), A_log.to(torch.float),
               dt_bias.to(torch.float)) if use_gate_in_kernel else g[:, cu_seqlens[i]: cu_seqlens[i + 1]]),
            initial_state=h0[i],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad
    if use_gate_in_kernel:
        ref_dA, A_log.grad = A_log.grad, None
        ref_dbias, dt_bias.grad = dt_bias.grad, None
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.007)
    assert_close("dk", ref_dk, tri_dk, 0.008)
    assert_close("dv", ref_dv, tri_dv, 0.007)
    assert_close("dg", ref_dg, tri_dg, 0.015)
    assert_close("db", ref_db, tri_db, 0.015)
    assert_close("dh0", ref_dh0, tri_dh0, 0.007)
    if use_gate_in_kernel:
        assert_close("dA", ref_dA, tri_dA, 0.008, warning=True)
        assert_close("dbias", ref_dbias, tri_dbias, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "HAS_BIAS", "LOWER_BOUND"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-bias{}-lowerbound{}".format(*test))
        for test in [
            (1, 2, 2, 12, False, -5.0),
            (1, 32, 2, 16, False, -5.0),
            (2, 64, 4, 32, False, -5.0),
            (4, 128, 8, 64, False, -5.0),
            (4, 128, 8, 128, False, None),
            (1, 2, 2, 12, True, None),
            (1, 32, 2, 16, True, None),
            (2, 64, 4, 32, True, None),
            (4, 128, 8, 64, True, None),
            (4, 128, 8, 128, True, None),
        ]
    ],
)
def test_gate(
    B: int,
    T: int,
    H: int,
    D: int,
    HAS_BIAS: bool,
    LOWER_BOUND: float | None,
):
    torch.manual_seed(42)
    g = torch.randn(B, T, H, D, dtype=torch.float32) * 10
    A_log = torch.log(torch.randn(1, 1, H, 1, dtype=torch.float32).uniform_(1, 16))
    dt_bias = torch.randn(H * D, dtype=torch.float32) if HAS_BIAS else None
    g, A_log = map(lambda x: x.to(device).requires_grad_(True), (g, A_log))
    if dt_bias is not None:
        dt_bias = dt_bias.to(device).requires_grad_(True)
    do = torch.randn_like(g).view(B, T, H, D)

    if LOWER_BOUND is not None:
        ref = naive_kda_lowerbound_gate(
            g.clone(), A_log.clone(), dt_bias.clone() if dt_bias is not None else None, LOWER_BOUND
        )
    else:
        ref = naive_kda_gate(
            g.clone(), A_log.clone(), dt_bias.clone() if dt_bias is not None else None,
        )
    tri = fused_kda_gate(
        g.clone(), A_log.clone(), dt_bias.clone() if dt_bias is not None else None,
        lower_bound=LOWER_BOUND
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


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_chunk_return_intermediate_states(dtype):
    """Test that return_intermediate_states=True works in inference mode and returns h with correct shape."""
    torch.manual_seed(42)
    B, T, H, D = 2, 1024, 4, 128
    chunk_size = 64

    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    g = torch.randn(B, T, H, D, dtype=dtype, device=device)
    beta = torch.rand(B, T, H, dtype=dtype, device=device)

    with torch.inference_mode():
        # Test equal-length sequences
        o, final_state, h = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=True,
            return_intermediate_states=True,
            disable_recompute=False  # Should not cause issues in inference mode
        )

        # Verify shapes
        assert o.shape == (B, T, H, D), f"Output shape mismatch: {o.shape}"
        assert final_state.shape == (B, H, D, D), f"Final state shape mismatch: {final_state.shape}"

        # Calculate expected NT (number of chunks)
        expected_nt = (T + chunk_size - 1) // chunk_size
        assert h.shape == (B, expected_nt, H, D, D), f"h shape mismatch: {h.shape}, expected: {(B, expected_nt, H, D, D)}"
        assert h.dtype == dtype, f"h dtype should be bfloat16, got: {h.dtype}"

        # Test variable-length sequences with proper flattened inputs
        total_tokens = 1024
        N = 2  # Number of sequences
        # Create cu_seqlens for varlen: [0, len1, len1+len2, ..., total_tokens]
        # Simple case: two sequences of equal length
        seq_len = total_tokens // N
        cu_seqlens = torch.tensor([0, seq_len, total_tokens], dtype=torch.long, device=device)

        # Generate new tensors for varlen test (flattened batch size = 1)
        q_varlen = torch.randn(1, total_tokens, H, D, dtype=dtype, device=device)
        k_varlen = torch.randn(1, total_tokens, H, D, dtype=dtype, device=device)
        v_varlen = torch.randn(1, total_tokens, H, D, dtype=dtype, device=device)
        g_varlen = torch.randn(1, total_tokens, H, D, dtype=dtype, device=device)
        beta_varlen = torch.rand(1, total_tokens, H, dtype=dtype, device=device)

        o_varlen, final_state_varlen, h_varlen = chunk_kda(
            q=q_varlen,
            k=k_varlen,
            v=v_varlen,
            g=g_varlen,
            beta=beta_varlen,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            return_intermediate_states=True,
            disable_recompute=False
        )

        # Verify varlen shapes - B should be 1 (flattened), sequence length is total_tokens
        assert o_varlen.shape == (1, total_tokens, H, D), f"Varlen output shape mismatch: {o_varlen.shape}"
        assert final_state_varlen.shape == (N, H, D, D), f"Varlen final state shape mismatch: {final_state_varlen.shape}"

        # NT for varlen is total number of chunks across all sequences
        assert h_varlen.shape[0] == 1, f"Varlen h batch dim should be 1, got: {h_varlen.shape[0]}"
        assert h_varlen.shape[2:] == (H, D, D), f"Varlen h dims mismatch: {h_varlen.shape[2:]}"
        assert h_varlen.dtype == dtype, f"Varlen h dtype should be {dtype}, got: {h_varlen.dtype}"
