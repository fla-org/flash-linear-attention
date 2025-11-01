
import os

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule, fused_recurrent_dplr_delta_rule
from fla.utils import assert_close, device, device_platform


def recurrent_dplr_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    q, k, v, a, b, gk = map(lambda x: x.transpose(1, 2).to(torch.float), (q, k, v, a, b, gk))

    B, H, T, K, V = *q.shape, v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        S = initial_state
    if scale is None:
        scale = K ** -0.5
    q = q * scale

    for i in range(T):
        _q = q[:, :, i]
        _k = k[:, :, i]
        _v = v[:, :, i].clone()
        a_i = a[:, :, i]
        b_i = b[:, :, i]
        # first matmul then decay in DPLR.
        _v2 = (S.clone() * a_i[..., None]).sum(-2)
        S = S.clone() * gk[:, :, i].exp()[..., None]
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2) + b_i.unsqueeze(-1) * _v2.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    if not output_final_state:
        S = None
    o = o.transpose(1, 2)
    return o, S


def chunk_dplr_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    scale: float = None,
    chunk_size: int = 64,
):
    q, k, v, a, b, gk = map(lambda x: x.transpose(1, 2).to(torch.float), (q, k, v, a, b, gk))
    BT = chunk_size
    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT

    q, k, v, a, b, gk = map(lambda x: F.pad(x, (0, 0, 0, pad_len)).to(torch.float), [q, k, v, a, b, gk])
    B, H, _, K, V = *q.shape, v.shape[-1]
    NT = q.shape[-2] // BT
    if scale is None:
        scale = K ** -0.5
    q = q * scale

    S = k.new_zeros(B, H, K, V)
    if initial_state is not None:
        S += initial_state

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, a, b, gk = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BT), [q, k, v, a, b, gk])
    gk_cumsum = gk.cumsum(-2)
    A_ab = torch.zeros(B, H, NT, BT, BT).to(q.device)
    A_qk = torch.zeros(B, H, NT, BT, BT).to(q.device)
    A_ak = torch.zeros(B, H, NT, BT, BT).to(q.device)
    A_qb = torch.zeros(B, H, NT, BT, BT).to(q.device)

    for i in range(BT):
        a_i = a[:, :, :, i, None]
        q_i = q[:, :, :, i, None]
        gk_i = gk_cumsum[:, :, :, i, None]
        mask = (torch.arange(BT) <= i).to(q.device)
        attn_i = (gk_i - gk_cumsum).masked_fill(~mask.unsqueeze(-1), float('-inf')).exp()
        A_qk[:, :, :, i, :] = (q_i * k * attn_i).sum(-1).clone()
        A_qb[:, :, :, i, :] = (q_i * b * attn_i).sum(-1).clone()
        mask = (torch.arange(BT) < i).to(q.device)
        # shift by one.
        attn_i = (gk_i - gk[:, :, :, i, None] - gk_cumsum).masked_fill(~mask.unsqueeze(-1), float('-inf')).exp()
        A_ab[:, :, :, i, :] = (a_i * b * attn_i).sum(-1).clone()
        A_ak[:, :, :, i, :] = (a_i * k * attn_i).sum(-1).clone()

    A_ab = A_ab
    for i in range(1, BT):
        A_ab[..., i, :i] = A_ab[..., i, :i].clone() + (A_ab[..., i, :, None].clone() * A_ab[..., :, :i].clone()).sum(-2)

    A_ab = A_ab + torch.eye(BT, dtype=torch.float, device=q.device)
    u = A_ab @ (A_ak @ v)
    w = A_ab @ ((gk_cumsum-gk).exp() * a)

    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, NT):
        q_i, k_i, v_i, u_i, w_i, b_i = q[:, :, i], k[:, :, i], v[:, :, i], u[:, :, i], w[:, :, i], b[:, :, i]
        v2_i = u_i + w_i @ S
        o_1 = A_qk[:, :, i] @ v_i
        o_2 = A_qb[:, :, i] @ v2_i
        o_3 = (q_i * gk_cumsum[:, :, i].exp()) @ S
        o[:, :, i] = o_1 + o_2 + o_3
        decay = (gk_cumsum[:, :, i, -1, None] - gk_cumsum[:, :, i]).exp()
        S = S*gk_cumsum[:, :, i, -1, :, None].exp() + (k_i * decay).transpose(-1, -2) @ v_i + \
            (b_i * decay).transpose(-1, -2) @ v2_i

    S = None if output_final_state is False else S
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T].transpose(1, 2)
    return o, S


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, torch.float),
            (2, 1024, 4, 60, 1, torch.float),
            (2, 1024, 8, 128, 1, torch.float),
            (2, 1024, 8, 128, 0.1, torch.float),
            (4, 2048, 8, 64, 0.1, torch.float),
            (2, 1024, 8, 128, 1, torch.float16),
        ]
    ],
)
def test_recurrent_fwd(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    os.environ['TORCH_CUDA_MATMUL_PRECISION'] = 'highest'
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    a = torch.rand(B, T, H, D, dtype=dtype)
    gk = torch.randn(B, T, H, D, dtype=torch.float)

    a = F.normalize(a, p=2, dim=-1)
    b = -a
    gk = F.logsigmoid(gk) / 16

    h0 = torch.randn(B, H, D, D, dtype=torch.float)
    q, k, v, a, b, gk, h0 = map(lambda x: x.to(device).requires_grad_(False), (q, k, v, a, b, gk, h0))
    ref, ref_ht = chunk_dplr_delta_rule_ref(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=gk.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    tri, tri_ht = recurrent_dplr_delta_rule_ref(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=gk.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    assert_close('o', ref, tri, 0.001)
    assert_close('ht', ref_ht, tri_ht, 0.001)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, torch.float),
            (2, 1024, 4, 60, 1, torch.float),
            (2, 1024, 8, 100, 1, torch.float),
            (2, 1024, 8, 128, 0.1, torch.float),
            (4, 2048, 8, 64, 0.1, torch.float),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    a = torch.rand(B, T, H, D, dtype=dtype)
    gk = torch.randn(B, T, H, D, dtype=torch.float)

    a = F.normalize(a, p=2, dim=-1)
    b = -a
    gk = F.logsigmoid(gk) / 4

    h0 = torch.randn(B, H, D, D, dtype=torch.float)
    q, k, v, a, b, gk, h0 = map(lambda x: x.to(device).requires_grad_(False), (q, k, v, a, b, gk, h0))
    ref, ref_ht = recurrent_dplr_delta_rule_ref(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=gk.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    tri, tri_ht = fused_recurrent_dplr_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=gk.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    assert_close('o', ref, tri, 0.002)
    assert_close('ht', ref_ht, tri_ht, 0.002)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'mask_p', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, 1, 0, torch.float16),
            (2, 1000, 3, 60, 1, 1, 0, torch.float16),
            (2, 1024, 3, 64, 0.1, 1, 0.5, torch.float16),
            (2, 1024, 4, 100, 1, 0.1, 0, torch.float16),
            (2, 1024, 4, 128, 0.1, 1, 0, torch.float16),
            (2, 1024, 4, 128, 0.1, 1, 0.5, torch.float16),
            (2, 1024, 4, 128, 0.1, 10, 0, torch.float16),
            (4, 2048, 8, 64, 0.1, 1, 0, torch.float16),
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
    scale: float,
    gate_logit_normalizer: float,
    mask_p: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    a = torch.rand(B, T, H, D, dtype=dtype)
    gk = torch.randn(B, T, H, D, dtype=torch.float)

    a = F.normalize(a, p=2, dim=-1)
    b = -a
    gk = F.logsigmoid(gk)
    gk = gk / gate_logit_normalizer
    gk = gk * (torch.rand_like(gk) > mask_p)

    h0 = torch.randn(B, H, D, D, dtype=torch.float)
    q, k, v, a, b, gk, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, a, b, gk, h0))
    ref, ref_ht = chunk_dplr_delta_rule_ref(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=gk.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_da, ref_db, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, a.grad, b.grad, gk.grad, h0.grad
    q.grad = k.grad = v.grad = a.grad = b.grad = gk.grad = h0.grad = None

    tri, tri_ht = chunk_dplr_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=gk.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_da, tri_db, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, a.grad, b.grad, gk.grad, h0.grad
    q.grad = k.grad = v.grad = a.grad = b.grad = gk.grad = h0.grad = None

    assert_close('o', ref, tri, 0.007)
    assert_close('ht', ref_ht, tri_ht, 0.008)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('da', ref_da, tri_da, 0.008)
    assert_close('db', ref_db, tri_db, 0.008)
    if gate_logit_normalizer >= 1 and ref_dg.norm() > 0.01:  # otherwise it is meaningless
        assert_close('dg', ref_dg, tri_dg, 0.008)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize(
    ('H', 'D', 'mask_p', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 64, 0, [0, 15], torch.float16),
            (4, 64, 0, [0, 256, 500, 1000], torch.float16),
            (4, 64, 0.5, [0, 256, 500, 1000], torch.float16),
            (4, 100, 0, [0, 15, 100, 300, 1111, 1599, 2000], torch.float16),
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
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    # seq-first required for inputs with variable lengths
    q = torch.randn(1, T, H, D, dtype=dtype)
    k = torch.randn(1, T, H, D, dtype=dtype)
    v = torch.randn(1, T, H, D, dtype=dtype)
    a = torch.rand(1, T, H, D, dtype=dtype)
    gk = torch.randn(1, T, H, D, dtype=torch.float)
    a = F.normalize(a, p=2, dim=-1)
    b = -a
    gk = F.logsigmoid(gk)
    gk = gk * (torch.rand_like(gk) > mask_p)
    h0 = torch.randn(N, H, D, D, dtype=torch.float)
    q, k, v, a, b, gk, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, a, b, gk, h0))

    tri, tri_ht = chunk_dplr_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=gk.clone(),
        output_final_state=True,
        initial_state=h0.clone(),
        cu_seqlens=cu_seqlens,
    )
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_da, tri_db, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, a.grad, b.grad, gk.grad, h0.grad
    q.grad = k.grad = v.grad = a.grad = b.grad = gk.grad = h0.grad = None

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = chunk_dplr_delta_rule_ref(
            q=q[:, cu_seqlens[i]:cu_seqlens[i+1]],
            k=k[:, cu_seqlens[i]:cu_seqlens[i+1]],
            v=v[:, cu_seqlens[i]:cu_seqlens[i+1]],
            a=a[:, cu_seqlens[i]:cu_seqlens[i+1]],
            b=b[:, cu_seqlens[i]:cu_seqlens[i+1]],
            gk=gk[:, cu_seqlens[i]:cu_seqlens[i+1]],
            initial_state=h0[i, None],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)

    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_da, ref_db, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, a.grad, b.grad, gk.grad, h0.grad

    assert_close('o', ref, tri, 0.007)
    assert_close('ht', ref_ht, tri_ht, 0.008)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('da', ref_da, tri_da, 0.008)
    assert_close('db', ref_db, tri_db, 0.008)
    assert_close('dg', ref_dg, tri_dg, 0.008)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)
