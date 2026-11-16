# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from itertools import accumulate

import pytest
import torch
import torch.nn.functional as F

from fla.ops.kda.naive import naive_recurrent_kda
from fla.ops.triadic_gdn import naive_recurrent_triadic_gdn
from fla.ops.triadic_kda import chunk_triadic_kda, naive_chunk_triadic_kda, naive_recurrent_triadic_kda
from fla.utils import assert_close, device_name

# some PyTorch ops used by the FP64 references are unsupported on Ascend.
REFERENCE_DEVICE = 'cpu' if device_name == 'npu' else device_name


def _make_inputs(B: int, T: int, H: int, K: int, V: int, E: int, gate_scale: float, dtype: torch.dtype, *, device=device_name):
    q = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, V, dtype=dtype)
    k2 = F.normalize(F.softplus(torch.randn(B, T, H, E, dtype=torch.float32)), p=2, dim=-1)
    q2 = F.normalize(F.softplus(torch.randn(B, T, H, E, dtype=torch.float32)), p=2, dim=-1)
    gk = F.logsigmoid(torch.rand(B, T, H, K, dtype=torch.float32)) * gate_scale
    g2 = F.logsigmoid(torch.rand(B, T, H, E, dtype=torch.float32)) * gate_scale
    beta = torch.rand(B, T, H, dtype=torch.float32).sigmoid()
    return [x.to(device) for x in (q, k, v, k2, q2, gk, g2, beta)]


def _run_reference(op, *, q: torch.Tensor, **kwargs):
    if q.device.type == REFERENCE_DEVICE:
        return op(q=q, **kwargs)

    # preserve autograd across the copies so the reference checks the same inputs and gradients.
    ref_kwargs = {name: x.to(REFERENCE_DEVICE) if isinstance(x, torch.Tensor) else x for name, x in kwargs.items()}
    o, ht = op(q=q.to(REFERENCE_DEVICE), **ref_kwargs)
    return o.to(q.device), ht.to(q.device) if ht is not None else None


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'E', 'gate_scale', 'chunk_size'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-V{}-E{}-gate_scale{}-chunk_size{}".format(*test))
        for test in [
            (2, 64, 2, 32, 32, 1, 0.5, 64),
            (2, 100, 2, 32, 48, 4, 0.5, 32),
            (1, 63, 3, 16, 16, 12, 0.5, 64),
            (1, 130, 2, 32, 32, 16, 2.0, 64),
        ]
    ],
)
def test_naive(B: int, T: int, H: int, K: int, V: int, E: int, gate_scale: float, chunk_size: int):
    torch.manual_seed(42)
    q, k, v, k2, q2, gk, g2, beta = _make_inputs(B, T, H, K, V, E, gate_scale, torch.float32, device=REFERENCE_DEVICE)
    h0 = torch.randn(B, H, E, K, V, dtype=torch.float32, device=REFERENCE_DEVICE)
    scale = K ** -0.5
    # the references return outputs in v.dtype, so keep FP64 for the direct comparison
    v_fp64 = v.double()

    ref, ref_ht = naive_recurrent_kda(
        q=(q2[..., None] * q[..., None, :]).flatten(-2),
        k=(k2[..., None] * k[..., None, :]).flatten(-2),
        v=v,
        g=(g2[..., None] + gk[..., None, :]).flatten(-2),
        beta=beta,
        scale=scale,
        initial_state=h0.flatten(2, 3),
        output_final_state=True,
    )
    tri, tri_ht = naive_recurrent_triadic_kda(
        q=q,
        k=k,
        v=v_fp64,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    assert_close('o', ref, tri, 1e-5)
    assert_close('ht', ref_ht, tri_ht.flatten(2, 3), 1e-5)

    chunk_o, chunk_ht = naive_chunk_triadic_kda(
        q=q,
        k=k,
        v=v_fp64,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    assert_close('chunk_o', ref, chunk_o, 1e-5)
    assert_close('chunk_ht', ref_ht, chunk_ht.flatten(2, 3), 1e-5)
    assert tri.dtype == chunk_o.dtype == torch.float64
    assert tri_ht.dtype == chunk_ht.dtype == torch.float32
    torch.testing.assert_close(chunk_o, tri, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(chunk_ht, tri_ht, rtol=1e-6, atol=1e-7)

    # gk = 0 recovers Triadic Gated DeltaNet
    gdn_o, gdn_ht = naive_recurrent_triadic_gdn(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        g=g2,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    zero_gk = torch.zeros_like(gk)
    tkda_gdn_o, tkda_gdn_ht = naive_chunk_triadic_kda(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        gk=zero_gk,
        g2=g2,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    assert_close('gdn_o', gdn_o, tkda_gdn_o, 1e-5)
    assert_close('gdn_ht', gdn_ht, tkda_gdn_ht, 1e-5)


@pytest.mark.parametrize(
    ('lengths', 'gate_scale', 'cu_dtype'),
    [
        pytest.param((13, 0, 50), 0.5, torch.int32, id='int32-short'),
        pytest.param((63, 0, 67), 2.0, torch.int64, id='int64-multi-chunk'),
    ],
)
def test_naive_varlen(lengths: tuple[int, ...], gate_scale: float, cu_dtype: torch.dtype):
    torch.manual_seed(42)
    H, K, V, E = 2, 32, 32, 16
    inputs = _make_inputs(1, sum(lengths), H, K, V, E, gate_scale, torch.float32, device=REFERENCE_DEVICE)
    h0 = torch.randn(len(lengths), H, E, K, V, dtype=torch.float32, device=REFERENCE_DEVICE)
    cu_seqlens = torch.tensor(list(accumulate(lengths, initial=0)), dtype=cu_dtype, device=REFERENCE_DEVICE)

    ref_outputs, ref_states = [], []
    segments = zip(*(x.split(lengths, dim=1) for x in inputs))
    for i, segment_inputs in enumerate(segments):
        state = h0[i:i+1]
        if lengths[i] == 0:
            ref_states.append(state)
            continue
        ref_o, ref_ht = naive_recurrent_triadic_kda(*segment_inputs, initial_state=state, output_final_state=True)
        ref_outputs.append(ref_o)
        ref_states.append(ref_ht)
    ref_o, ref_ht = torch.cat(ref_outputs, dim=1), torch.cat(ref_states)

    for op in (naive_recurrent_triadic_kda, naive_chunk_triadic_kda):
        o, ht = op(*inputs, initial_state=h0, output_final_state=True, cu_seqlens=cu_seqlens)
        torch.testing.assert_close(o, ref_o, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(ht, ref_ht, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(ht[1], h0[1], rtol=0, atol=0)


@pytest.mark.parametrize(
    'op', [naive_recurrent_triadic_kda, naive_chunk_triadic_kda], ids=['recurrent', 'chunk'],
)
def test_naive_beta_default(op):
    torch.manual_seed(42)
    *inputs, beta = _make_inputs(1, 35, 2, 8, 8, 4, 0.5, torch.float64, device=REFERENCE_DEVICE)
    o, ht = op(*inputs, output_final_state=True)
    ref_o, ref_ht = op(*inputs, beta=torch.ones_like(beta), output_final_state=True)
    torch.testing.assert_close(o, ref_o, rtol=0, atol=0)
    torch.testing.assert_close(ht, ref_ht, rtol=0, atol=0)


@pytest.mark.parametrize(
    'op', [naive_recurrent_triadic_kda, naive_chunk_triadic_kda], ids=['recurrent', 'chunk'],
)
def test_naive_varlen_rejects_multiple_batches(op):
    torch.manual_seed(42)
    inputs = _make_inputs(2, 3, 1, 4, 4, 2, 0.5, torch.float32, device=REFERENCE_DEVICE)
    with pytest.raises(ValueError, match='The batch size is expected to be 1'):
        op(
            *inputs,
            cu_seqlens=torch.tensor([0, 1, 3], dtype=torch.int32, device=REFERENCE_DEVICE),
        )


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'E', 'scale', 'gate_scale', 'use_state', 'disable_recompute', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-K{}-V{}-E{}-scale{}-gate_scale{}-use_state{}-disable_recompute{}-{}".format(*test),
        )
        for test in [
            (1, 63, 1, 64, 64, 1, 1.0, 0.2, True, False, torch.bfloat16),
            (1, 200, 2, 64, 64, 4, 1.0, 0.2, True, False, torch.float32),
            (2, 129, 3, 60, 64, 2, 1.0, 0.2, True, True, torch.bfloat16),
            (2, 512, 2, 128, 128, 4, 0.1, 1.5, False, False, torch.bfloat16),
            (2, 512, 2, 128, 128, 8, 1.0, 0.2, False, True, torch.bfloat16),
            (1, 257, 2, 100, 96, 12, 1.0, 0.1, True, False, torch.bfloat16),
            (1, 130, 2, 64, 64, 16, 1.0, 0.2, True, True, torch.bfloat16),
            (2, 100, 2, 8, 12, 3, 1.0, 0.2, True, False, torch.bfloat16),
            (1, 193, 2, 256, 128, 4, 1.0, 0.2, True, False, torch.bfloat16),
            (1, 129, 2, 128, 128, 6, 1.0, 0.2, True, True, torch.bfloat16),
            (1, 131, 2, 64, 64, 20, 1.0, 0.2, True, False, torch.bfloat16),
            (1, 131, 2, 64, 48, 20, 1.0, 0.2, True, True, torch.float16),
            (1, 129, 2, 32, 32, 72, None, 0.2, True, False, torch.bfloat16),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    E: int,
    scale: float | None,
    gate_scale: float,
    use_state: bool,
    disable_recompute: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    state_dtype = dtype if E in (6, 20) else torch.float32
    h0 = torch.randn(B, H, E, K, V, dtype=state_dtype, device=device_name) if use_state else None
    raw = _make_inputs(B, T, H, K, V, E, gate_scale, dtype)
    if E == 20:
        raw[3] = raw[3].to(dtype)
        raw[4] = raw[4].to(dtype)
    q, k, v, k2, q2, gk, g2, beta = raw
    if scale is None:
        beta = None
        inputs = [x.requires_grad_(True) for x in (q, k, v, k2, q2, gk, g2)]
        grad_specs = (
            ('dq', 0.008), ('dk', 0.008), ('dv', 0.008), ('dk2', 0.008), ('dq2', 0.008), ('dgk', 0.02), ('dg2', 0.02),
        )
    else:
        inputs = [x.requires_grad_(True) for x in (q, k, v, k2, q2, gk, g2, beta)]
        grad_specs = (
            ('dq', 0.008), ('dk', 0.008), ('dv', 0.008), ('dk2', 0.008),
            ('dq2', 0.008), ('dgk', 0.02), ('dg2', 0.02), ('dbeta', 0.02),
        )
    if use_state:
        inputs.append(h0.requires_grad_(True))
        grad_specs = (*grad_specs, ('dh0', 0.008))
    do = torch.randn_like(v)
    dht = torch.randn(B, H, E, K, V, dtype=torch.float32, device=device_name) if use_state else None

    oracle = naive_recurrent_triadic_kda if T <= 256 else naive_chunk_triadic_kda
    ref, ref_ht = _run_reference(
        oracle,
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=use_state,
    )
    ((ref * do).sum() + ((ref_ht * dht).sum() if use_state else 0)).backward()
    ref_grads = [x.grad for x in inputs]
    for x in inputs:
        x.grad = None

    tri, tri_ht = chunk_triadic_kda(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=use_state,
        disable_recompute=disable_recompute,
    )
    ((tri * do).sum() + ((tri_ht * dht).sum() if use_state else 0)).backward()
    tri_grads = [x.grad for x in inputs]

    assert_close('o', ref, tri, 0.005)
    if use_state:
        assert_close('ht', ref_ht, tri_ht, 0.005)
        assert tri_grads[-1].dtype == h0.dtype
    for (name, ratio), ref_grad, tri_grad in zip(grad_specs, ref_grads, tri_grads):
        assert tri_grad.dtype == ref_grad.dtype
        assert_close(name, ref_grad, tri_grad, ratio)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'E', 'gk_scale', 'g2_scale', 'disable_recompute', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-K{}-V{}-E{}-gk_scale{}-g2_scale{}-disable_recompute{}-{}".format(*test),
        )
        for test in [
            (2, 130, 2, 64, 64, 4, 10.0, 0.5, False, torch.bfloat16),
            (2, 130, 2, 64, 64, 4, 25.0, 0.5, False, torch.bfloat16),
            (2, 130, 2, 64, 64, 4, 0.5, 25.0, False, torch.bfloat16),
            (2, 130, 2, 64, 64, 20, 32.0, 32.0, False, torch.float32),
            (1, 257, 2, 100, 96, 12, 32.0, 32.0, True, torch.bfloat16),
        ]
    ],
)
def test_chunk_strong_decay(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    E: int,
    gk_scale: float,
    g2_scale: float,
    disable_recompute: bool,
    dtype: torch.dtype,
):
    # per-token log decays average about -0.5 * scale, so a 16-token diagonal block spans up to about 240
    torch.manual_seed(42)
    q, k, v, k2, q2, gk, g2, beta = _make_inputs(B, T, H, K, V, E, 1.0, dtype)
    h0 = torch.randn(B, H, E, K, V, dtype=torch.float32, device=device_name)
    inputs = [x.requires_grad_(True) for x in (q, k, v, k2, q2, gk * gk_scale, g2 * g2_scale, beta, h0)]
    q, k, v, k2, q2, gk, g2, beta, h0 = inputs
    do, dht = torch.randn_like(v), torch.randn_like(h0)

    ref, ref_ht = _run_reference(
        naive_recurrent_triadic_kda,
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
    )
    ref_grads = torch.autograd.grad((ref * do).sum() + (ref_ht * dht).sum(), inputs)

    tri, tri_ht = chunk_triadic_kda(
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        gk=gk,
        g2=g2,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        disable_recompute=disable_recompute,
    )
    tri_grads = torch.autograd.grad((tri * do).sum() + (tri_ht * dht).sum(), inputs)

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    for name, ref_grad, tri_grad, ratio in zip(
        ('dq', 'dk', 'dv', 'dk2', 'dq2', 'dgk', 'dg2', 'dbeta', 'dh0'),
        ref_grads,
        tri_grads,
        (0.008, 0.008, 0.008, 0.008, 0.008, 0.02, 0.02, 0.02, 0.008),
    ):
        assert_close(name, ref_grad, tri_grad, ratio)


@pytest.mark.parametrize(
    ('H', 'K', 'V', 'E', 'cu_seqlens', 'in_kernel_gates', 'lower_bound', 'disable_recompute'),
    [
        pytest.param(
            *test,
            id="H{}-K{}-V{}-E{}-cu_seqlens{}-in_kernel{}-lb{}-save{}".format(*test),
        )
        for test in [
            (2, 64, 64, 4, [0, 15, 80, 145], False, None, False),
            (2, 128, 128, 4, [0, 1, 65, 130], True, None, True),
            (3, 64, 96, 12, [0, 33, 128, 210], True, -5.0, False),
            (2, 64, 64, 20, [0, 0, 19, 83], True, -5.0, True),
        ]
    ],
)
def test_chunk_varlen_and_in_kernel_gates(
    H: int,
    K: int,
    V: int,
    E: int,
    cu_seqlens: list[int],
    in_kernel_gates: bool,
    lower_bound: float | None,
    disable_recompute: bool,
):
    torch.manual_seed(42)
    N, T = len(cu_seqlens) - 1, cu_seqlens[-1]
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device_name)
    h0 = torch.randn(N, H, E, K, V, dtype=torch.float32, device=device_name, requires_grad=True)

    if in_kernel_gates:
        use_a_log = E != 20
        q = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=device_name, requires_grad=True)
        k = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=device_name, requires_grad=True)
        v = torch.randn(1, T, H, V, dtype=torch.bfloat16, device=device_name, requires_grad=True)
        k2 = F.normalize(
            F.softplus(torch.randn(1, T, H, E, dtype=torch.float32, device=device_name)), p=2, dim=-1,
        ).detach().requires_grad_(True)
        q2 = F.normalize(
            F.softplus(torch.randn(1, T, H, E, dtype=torch.float32, device=device_name)), p=2, dim=-1,
        ).detach().requires_grad_(True)
        gk = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=device_name, requires_grad=True)
        g2 = torch.randn(1, T, H, E, dtype=torch.float32, device=device_name, requires_grad=True)
        beta = torch.randn(1, T, H, dtype=torch.bfloat16, device=device_name, requires_grad=True)
        A_log = (
            torch.empty(H, dtype=torch.float32, device=device_name).uniform_(1.0, 8.0).log().detach().requires_grad_(True)
            if use_a_log else None
        )
        dt_bias = torch.full((H * K,), -3.0, dtype=torch.float32, device=device_name, requires_grad=True)
        A2_log = (
            torch.empty(H, dtype=torch.float32, device=device_name).uniform_(1.0, 8.0).log().detach().requires_grad_(True)
            if use_a_log else None
        )
        dt2_bias = torch.full((H * E,), -3.0, dtype=torch.float32, device=device_name, requires_grad=True)
        if use_a_log:
            inputs = [q, k, v, k2, q2, gk, g2, beta, A_log, dt_bias, A2_log, dt2_bias, h0]
            names = ('dq', 'dk', 'dv', 'dk2', 'dq2', 'dgk', 'dg2', 'dbeta', 'dA', 'ddt', 'dA2', 'ddt2', 'dh0')
        else:
            inputs = [q, k, v, k2, q2, gk, g2, beta, dt_bias, dt2_bias, h0]
            names = ('dq', 'dk', 'dv', 'dk2', 'dq2', 'dgk', 'dg2', 'dbeta', 'ddt', 'ddt2', 'dh0')

        tri, tri_ht = chunk_triadic_kda(
            q=q,
            k=k,
            v=v,
            k2=k2,
            q2=q2,
            gk=gk,
            g2=g2,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            A2_log=A2_log,
            dt2_bias=dt2_bias,
            initial_state=h0,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=True,
            lower_bound=lower_bound,
            cu_seqlens=cu,
            disable_recompute=disable_recompute,
        )
        do = torch.randn_like(v)
        dht = torch.randn_like(h0)
        tri_grads = torch.autograd.grad((tri * do).sum() + (tri_ht * dht).sum(), inputs)

        q_n = (q.float() * torch.rsqrt((q.float() ** 2).sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
        k_n = (k.float() * torch.rsqrt((k.float() ** 2).sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
        gk_pre = gk.float() + dt_bias.view(H, K)
        g2_pre = g2.float() + dt2_bias.view(H, E)
        if lower_bound is None:
            gk_eff = -A_log.view(H, 1).exp() * F.softplus(gk_pre)
            g2_eff = -A2_log.view(H, 1).exp() * F.softplus(g2_pre)
        else:
            a1 = A_log.view(H, 1).exp() if A_log is not None else 1.0
            a2 = A2_log.view(H, 1).exp() if A2_log is not None else 1.0
            gk_eff = lower_bound * torch.sigmoid(a1 * gk_pre)
            g2_eff = lower_bound * torch.sigmoid(a2 * g2_pre)
        beta_eff = 2.0 * beta.float().sigmoid()
        ref, ref_ht = _run_reference(
            naive_recurrent_triadic_kda,
            q=q_n,
            k=k_n,
            v=v,
            k2=k2,
            q2=q2,
            gk=gk_eff,
            g2=g2_eff,
            beta=beta_eff,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
        )
        ref_grads = torch.autograd.grad((ref * do).sum() + (ref_ht * dht).sum(), inputs)
    else:
        base_inputs = [x.requires_grad_(True) for x in _make_inputs(1, T, H, K, V, E, 0.2, torch.bfloat16)]
        q, k, v, k2, q2, gk, g2, beta = base_inputs
        inputs = [*base_inputs, h0]
        names = ('dq', 'dk', 'dv', 'dk2', 'dq2', 'dgk', 'dg2', 'dbeta', 'dh0')
        do = torch.randn_like(v)
        dht = torch.randn_like(h0)
        tri, tri_ht = chunk_triadic_kda(
            q=q,
            k=k,
            v=v,
            k2=k2,
            q2=q2,
            gk=gk,
            g2=g2,
            beta=beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
            disable_recompute=disable_recompute,
        )
        tri_grads = torch.autograd.grad((tri * do).sum() + (tri_ht * dht).sum(), inputs)
        seg_os, seg_hts = [], []
        for idx in range(N):
            bos, eos = cu_seqlens[idx], cu_seqlens[idx + 1]
            seg_o, seg_ht = _run_reference(
                naive_recurrent_triadic_kda,
                q=q[:, bos:eos],
                k=k[:, bos:eos],
                v=v[:, bos:eos],
                k2=k2[:, bos:eos],
                q2=q2[:, bos:eos],
                gk=gk[:, bos:eos],
                g2=g2[:, bos:eos],
                beta=beta[:, bos:eos],
                initial_state=h0[idx:idx + 1],
                output_final_state=True,
            )
            seg_os.append(seg_o)
            seg_hts.append(seg_ht)
        ref, ref_ht = torch.cat(seg_os, dim=1), torch.cat(seg_hts, dim=0)
        ref_grads = torch.autograd.grad((ref * do).sum() + (ref_ht * dht).sum(), inputs)

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    for name, ref_grad, tri_grad in zip(names, ref_grads, tri_grads):
        assert_close(name, ref_grad, tri_grad, 0.02)


def test_chunk_validation():
    q, k, v, k2, q2, gk, g2, beta = _make_inputs(1, 16, 2, 32, 32, 4, 0.2, torch.bfloat16)
    with pytest.raises(ValueError, match='E >= 1'):
        chunk_triadic_kda(q, k, v, k2[..., :0], q2[..., :0], gk, g2[..., :0], beta)
    with pytest.raises(ValueError, match=r'\[1, 256\]'):
        q_big = torch.randn(1, 16, 2, 512, dtype=torch.bfloat16, device=device_name)
        chunk_triadic_kda(q_big, q_big, v, k2, q2, q_big.float(), g2, beta)
    with pytest.raises(ValueError, match='lower_bound'):
        chunk_triadic_kda(q, k, v, k2, q2, gk, g2, beta, use_gate_in_kernel=True, lower_bound=0.0)
    with pytest.raises(ValueError, match='Packed inputs require B=1'):
        q2b, k2b, vb, k2_2b, q2_2b, gk2b, g2_2b, beta2b = _make_inputs(2, 16, 2, 32, 32, 4, 0.2, torch.bfloat16)
        cu = torch.tensor([0, 16, 32], dtype=torch.int32, device=device_name)
        chunk_triadic_kda(q2b, k2b, vb, k2_2b, q2_2b, gk2b, g2_2b, beta2b, cu_seqlens=cu)
