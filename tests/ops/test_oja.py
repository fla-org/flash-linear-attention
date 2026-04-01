
import os

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from fla.ops.gated_oja_rule import chunk_gated_oja_rule, fused_recurrent_gated_oja_rule
from fla.utils import assert_close, device, is_intel_alchemist


def recurrent_oja_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i]  # B H D
        g_i = g[:, :, i]
        # breakpoint()
        h = h * g_i.exp()[:, :, None, :]
        b_beta = beta[:, :, i]
        b_k = b_k - (h * b_v[:, :, None, :]).sum(-1)
        b_v = b_v * b_beta[..., None]
        h = h + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)
    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


def chunk_oja_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    BT = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    # Calculate padding needed to make T a multiple of BT
    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])

    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        # Pad all tensors
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, 0, 0, pad_len))
    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    chunk_size = BT
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale  # B H T D
    assert l % chunk_size == 0
    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, beta, g = map(
        lambda x: rearrange(x, 'b h (n c) ... -> b h n c ...', c=chunk_size),
        [q, k, v, beta, g]
    )
    g = g.cumsum(-2)  # b h n c d

    """
    vector decay for attention
    qkvg (B H N C D)
    """
    attn = torch.zeros(*q.shape[:-1], chunk_size, dtype=torch.float, device=q.device)  # B H N C C
    # attn = -((v_beta @ v.transpose(-1, -2))* L_mask).masked_fill(mask, 0) # B H N C C
    for i in range(BT):
        v_i = v[..., i, :]  # B H N D
        g_i = g[..., i:i+1, :]  # B H N 1 Dv
        attn[..., i] = torch.einsum('... c d, ... d -> ... c', v * (g - g_i).exp(), v_i)  # B H N C
    attn = attn * beta[..., None]  # B H N C C

    attn = -attn.masked_fill(mask, 0)

    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = (attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)) * beta[..., None, :]

    W = attn @ (v * g.exp())  # B H N C Dv
    U = attn @ k  # B H N C Dk

    S = k.new_zeros(b, h, d_k, d_v)

    if initial_state is not None:
        S = initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, l // chunk_size):
        q_i, u_i, v_i, g_i = q[:, :, i], U[:, :, i], v[:, :, i], g[:, :, i]  # B H C Dv
        k_new = u_i - W[:, :, i] @ S.transpose(-1, -2)  # b h c dk - b h c dv @ b h dv dk
        attn = q_i @ k_new.transpose(-1, -2)  # b h c d @ b h d c -> b h c c
        attn = attn.masked_fill(mask, 0)  # b h c c
        g_last = g_i[:, :, -1, :]  # b h dv
        o_inter = (q_i @ S)  # b h c dv
        vg_i = v_i / g_i.exp()
        o[:, :, i] = (o_inter + (attn @ vg_i)) * g_i.exp()    # b h c dv + b h c c @ b h c dv
        S = S * g_last[:, :, None, :].exp()  # B H Dk Dv @ B H 1 D
        S += k_new.transpose(-1, -2) @ (v_i * (g_last[:, :, None, :] - g_i).exp())
    if not output_final_state:
        S = None
    # unpad
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T]
    o = o.transpose(1, 2)
    return o, S


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (4, 2048, 8, 128, 1, 10, torch.float),
            (4, 1024, 4, 64, 0.1, 10, torch.float),
            (2, 1536, 4, 128, 1, 100, torch.float16),
            (4, 2048, 4, 256, 1, 100, torch.float16),
        ]
    ]
)
def test_naive_chunk_oja(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=torch.float32)
    k = torch.randn(B, T, H, D, dtype=torch.float32)
    v = torch.randn(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, D, dtype=torch.float32)) / gate_logit_normalizer
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0))
    ref, ref_ht = recurrent_oja_ref(
        q=q.clone(),
        k=k.clone(),
        v=F.normalize(v.clone(), p=2, dim=-1),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    tri, tri_ht = chunk_oja_ref(
        q=q.clone(),
        k=k.clone(),
        v=F.normalize(v.clone(), p=2, dim=-1),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    assert_close('ht', ref_ht, tri_ht, 0.002)
    assert_close('o', ref, tri, 0.002)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, 1, torch.float),
            (2, 500, 4, 60, 1, 1, torch.float),
            (2, 1000, 8, 128, 1, 0.1, torch.float),
            (3, 1024, 4, 128, 0.1, 1, torch.float),
            (4, 1024, 8, 128, 1, 10, torch.float),
            (4, 2048, 8, 64, 0.1, 1, torch.float)
        ]
    ]
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=torch.float32)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, D, dtype=torch.float32)) / gate_logit_normalizer
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0))
    ref, ref_ht = recurrent_oja_ref(
        q=q.clone(),
        k=k.clone(),
        v=F.normalize(v.clone(), p=2, dim=-1).to(dtype),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    tri, tri_ht = fused_recurrent_gated_oja_rule(
        q=q.clone(),
        k=k.clone(),
        v=F.normalize(v.clone(), p=2, dim=-1).to(dtype),
        beta=beta.clone(),
        gv=g.clone(),
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
            (1, 63, 1, 64, 1, 0.01, 0, torch.float16),
            (2, 500, 3, 60, 1, 1, 0, torch.float16),
            (2, 1000, 3, 64, 0.1, 1, 0.5, torch.float16),
            (3, 1024, 4, 100, 1, 0.1, 0, torch.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, torch.float16),
        ]
    ]
)
def test_chunk_forward(
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
    v = torch.randn(B, T, H, D, dtype=torch.float32)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, D, dtype=torch.float32)) / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0))
    ref, ref_ht = recurrent_oja_ref(
        q=q.clone(),
        k=k.clone(),
        v=F.normalize(v.clone(), p=2, dim=-1).to(dtype),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    tri, tri_ht = chunk_gated_oja_rule(
        q=q.clone(),
        k=k.clone(),
        v=F.normalize(v.clone(), p=2, dim=-1).to(dtype),
        beta=beta.clone(),
        gv=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'mask_p', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-{}".format(*test)
        )
        for test in [
            (4, 4096, 4, 128, 1, 1, 0, torch.float16),
            (1, 4096, 1, 64, 1, 1, 0, torch.float16),
            (2, 4096, 3, 60, 1, 1, 0, torch.float16),
        ]
    ]
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

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=torch.float)
    beta = torch.rand(B, T, H, dtype=torch.float).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, D, dtype=torch.float)) / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, g, h0))
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    print('================== Running forward and backward ==================')

    ref, ref_ht = recurrent_oja_ref(
        q=q.clone(),
        k=k.clone(),
        v=F.normalize(v.clone(), p=2, dim=-1),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    tri, tri_ht = chunk_gated_oja_rule(
        q=q.clone(),
        k=k.clone(),
        v=F.normalize(v.clone(), p=2, dim=-1).to(dtype),
        beta=beta.clone(),
        gv=g.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )

    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None

    # breakpoint()

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    assert_close('dg', ref_dg, tri_dg, 0.005)


@pytest.mark.parametrize(
    ('H', 'D', 'mask_p', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 60, 0, [0, 96, 177], torch.float16),
            (16, 128, 0, [0, 256, 500, 1000], torch.float16),
            (4, 64, 0.5, [0, 256, 500, 1000], torch.float16),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ]
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set'
)
def test_chunk_varlen(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    if is_intel_alchemist and D > 128:
        pytest.skip(reason='chunk_gated_oja_rule is not supported on alchemist for D>128')
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    # randomly split the sequence into N segments
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype)
    k = torch.randn((1, T, H, D), dtype=dtype)
    v = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    g = F.logsigmoid(torch.rand(1, T, H, D, dtype=torch.float))
    g = g * (torch.rand_like(g) > mask_p)
    beta = torch.rand(1, T, H, dtype=torch.float32).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=torch.float32)

    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0))
    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    tri, tri_ht = chunk_gated_oja_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        gv=g.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = recurrent_oja_ref(
            q=q[:, cu_seqlens[i]:cu_seqlens[i+1]],
            k=k[:, cu_seqlens[i]:cu_seqlens[i+1]],
            v=v[:, cu_seqlens[i]:cu_seqlens[i+1]],
            beta=beta[:, cu_seqlens[i]:cu_seqlens[i+1]],
            g=g[:, cu_seqlens[i]:cu_seqlens[i+1]],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.007)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.007)
    assert_close('db', ref_dbeta, tri_dbeta, 0.015)
    assert_close('dg', ref_dg, tri_dg, 0.015)
    assert_close('dh0', ref_dh0, tri_dh0, 0.007)
