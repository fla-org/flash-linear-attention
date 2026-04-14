# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch

from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.attn.naive import naive_parallel_attn
from fla.ops.attn.parallel import parallel_attn
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ("window_size", "cu_seqlens"),
    [
        pytest.param(None, None, id="full"),
        pytest.param(64, None, id="swa"),
        pytest.param(64, [0, 97, 173, 300], id="varlen_swa"),
    ],
)
def test_parallel_attn_sink_matches_reference(window_size, cu_seqlens):
    torch.manual_seed(123)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    dtype = torch.float16
    B, T, H, HQ, D = 2, 192, 2, 8, 64
    if cu_seqlens is not None:
        B, T = 1, cu_seqlens[-1]
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    sinks = torch.randn((HQ,), dtype=torch.float32, device=device)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    q_ref = q.float().detach().clone().requires_grad_(True)
    k_ref = k.float().detach().clone().requires_grad_(True)
    v_ref = v.float().detach().clone().requires_grad_(True)
    sinks_ref = sinks.detach().clone().requires_grad_(True)

    if cu_seqlens is None:
        o_ref, _ = naive_parallel_attn(
            q_ref, k_ref, v_ref, sinks=sinks_ref, scale=0.1, window_size=window_size
        )
    else:
        outputs = []
        for bos, eos in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False):
            o_i, _ = naive_parallel_attn(
                q_ref[:, bos:eos], k_ref[:, bos:eos], v_ref[:, bos:eos],
                sinks=sinks_ref, scale=0.1, window_size=window_size
            )
            outputs.append(o_i)
        o_ref = torch.cat(outputs, dim=1)
    o_ref = o_ref.to(dtype)
    o_ref.backward(do)

    q_sg = q.detach().clone().requires_grad_(True)
    k_sg = k.detach().clone().requires_grad_(True)
    v_sg = v.detach().clone().requires_grad_(True)
    sinks_sg = sinks.detach().clone().requires_grad_(True)
    o_sg = parallel_attn(
        q=q_sg, k=k_sg, v=v_sg, sinks=sinks_sg, scale=0.1, window_size=window_size, cu_seqlens=cu_seqlens
    )
    o_sg.backward(do)

    assert_close(" o_ref_vs_sg", o_ref, o_sg, 0.01)
    assert_close("dq_ref_vs_sg", q_ref.grad.to(dtype), q_sg.grad, 0.02)
    assert_close("dk_ref_vs_sg", k_ref.grad.to(dtype), k_sg.grad, 0.02)
    assert_close("dv_ref_vs_sg", v_ref.grad.to(dtype), v_sg.grad, 0.02)
    assert_close("ds_ref_vs_sg", sinks_ref.grad, sinks_sg.grad, 0.02)


def test_parallel_attn_sink_empty_row_matches_reference():
    torch.manual_seed(987)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    dtype = torch.float16
    B, T, H, HQ, D = 2, 96, 2, 8, 64
    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    sinks = torch.randn((HQ,), dtype=torch.float32, device=device)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    q_ref = q.float().detach().clone().requires_grad_(True)
    k_ref = k.float().detach().clone().requires_grad_(True)
    v_ref = v.float().detach().clone().requires_grad_(True)
    sinks_ref = sinks.detach().clone().requires_grad_(True)
    o_ref, _ = naive_parallel_attn(
        q_ref, k_ref, v_ref, sinks=sinks_ref, scale=0.1, window_size=0
    )
    o_ref = o_ref.to(dtype)
    o_ref.backward(do)

    q_tri = q.detach().clone().requires_grad_(True)
    k_tri = k.detach().clone().requires_grad_(True)
    v_tri = v.detach().clone().requires_grad_(True)
    sinks_tri = sinks.detach().clone().requires_grad_(True)
    o_tri = parallel_attn(
        q=q_tri, k=k_tri, v=v_tri, sinks=sinks_tri, scale=0.1, window_size=0
    )
    o_tri.backward(do)

    assert_close(" o_ref_vs_tri", o_ref, o_tri, 0.01)
    assert_close("dq_ref_vs_tri", q_ref.grad.to(dtype), q_tri.grad, 0.02)
    assert_close("dk_ref_vs_tri", k_ref.grad.to(dtype), k_tri.grad, 0.02)
    assert_close("dv_ref_vs_tri", v_ref.grad.to(dtype), v_tri.grad, 0.02)
    assert_close("ds_ref_vs_tri", sinks_ref.grad, sinks_tri.grad, 0.02)


@pytest.mark.parametrize(
    ("window_size", "cu_seqlens"),
    [
        pytest.param(None, None, id="full"),
        pytest.param(64, None, id="swa"),
        pytest.param(64, [0, 97, 173, 300], id="varlen_swa"),
    ],
)
def test_parallel_attn_sink_with_g_matches_reference(window_size, cu_seqlens):
    torch.manual_seed(321)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    dtype = torch.float16
    B, T, H, HQ, D = 2, 192, 2, 8, 64
    if cu_seqlens is not None:
        B, T = 1, cu_seqlens[-1]
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    g = torch.empty((B, T, HQ), dtype=dtype, device=device).uniform_(-0.1, -0.01)
    sinks = torch.randn((HQ,), dtype=torch.float32, device=device)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    q_ref = q.float().detach().clone().requires_grad_(True)
    k_ref = k.float().detach().clone().requires_grad_(True)
    v_ref = v.float().detach().clone().requires_grad_(True)
    g_ref = g.float().detach().clone().requires_grad_(True)
    sinks_ref = sinks.detach().clone().requires_grad_(True)

    if cu_seqlens is None:
        o_ref, _ = naive_parallel_attn(
            q=q_ref,
            k=k_ref,
            v=v_ref,
            g=g_ref,
            sinks=sinks_ref,
            scale=0.1,
            window_size=window_size,
        )
    else:
        outputs = []
        for bos, eos in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False):
            o_i, _ = naive_parallel_attn(
                q=q_ref[:, bos:eos],
                k=k_ref[:, bos:eos],
                v=v_ref[:, bos:eos],
                g=g_ref[:, bos:eos],
                sinks=sinks_ref,
                scale=0.1,
                window_size=window_size,
            )
            outputs.append(o_i)
        o_ref = torch.cat(outputs, dim=1)
    o_ref = o_ref.to(dtype)
    o_ref.backward(do)

    q_tri = q.detach().clone().requires_grad_(True)
    k_tri = k.detach().clone().requires_grad_(True)
    v_tri = v.detach().clone().requires_grad_(True)
    g_tri = g.detach().clone().requires_grad_(True)
    sinks_tri = sinks.detach().clone().requires_grad_(True)
    o_tri = parallel_attn(
        q=q_tri,
        k=k_tri,
        v=v_tri,
        g=g_tri,
        sinks=sinks_tri,
        scale=0.1,
        window_size=window_size,
        cu_seqlens=cu_seqlens,
    )
    o_tri.backward(do)

    assert_close(" o_ref_vs_tri", o_ref, o_tri, 0.01)
    assert_close("dq_ref_vs_tri", q_ref.grad.to(dtype), q_tri.grad, 0.02)
    assert_close("dk_ref_vs_tri", k_ref.grad.to(dtype), k_tri.grad, 0.02)
    assert_close("dv_ref_vs_tri", v_ref.grad.to(dtype), v_tri.grad, 0.02)
    assert_close("dg_ref_vs_tri", g_ref.grad.to(dtype), g_tri.grad, 0.02)
    assert_close("ds_ref_vs_tri", sinks_ref.grad, sinks_tri.grad, 0.02)


def test_attn_decoding_sink_matches_reference():
    torch.manual_seed(456)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    B, T, H, HQ, D = 3, 128, 2, 8, 64
    dtype = torch.float16
    q = torch.randn((1, B, HQ, D), dtype=dtype, device=device)
    k = torch.randn((1, T * B, H, D), dtype=dtype, device=device)
    v = torch.randn((1, T * B, H, D), dtype=dtype, device=device)
    sinks = torch.randn((HQ,), dtype=torch.float32, device=device)
    cu_seqlens = torch.tensor([i * T for i in range(B + 1)], dtype=torch.int32, device=device)

    refs = []
    for bos, eos in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False):
        query_idx = torch.tensor([eos - bos - 1], dtype=torch.long, device=device)
        o_i, _ = naive_parallel_attn(
            q=q[:, len(refs):len(refs)+1].float(),
            k=k[:, bos:eos].float(),
            v=v[:, bos:eos].float(),
            query_indices=query_idx,
            sinks=sinks,
            scale=0.1,
        )
        refs.append(o_i)
    ref = torch.cat(refs, dim=1).to(dtype)

    tri = attn_decoding_one_step(q=q, k=k, v=v, sinks=sinks, scale=0.1, cu_seqlens=cu_seqlens)
    assert_close("o_decode_ref_vs_sg", ref, tri, 0.01)


@pytest.mark.parametrize(
    "do_gate_scale",
    [
        pytest.param(False, id="no_gate_scale"),
        pytest.param(True, id="with_gate_scale"),
    ],
)
def test_attn_decoding_sink_with_g_matches_reference(do_gate_scale):
    torch.manual_seed(654)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    B, T, H, HQ, D = 3, 128, 2, 8, 64
    dtype = torch.float16
    q = torch.randn((1, B, HQ, D), dtype=dtype, device=device)
    k = torch.randn((1, T * B, H, D), dtype=dtype, device=device)
    v = torch.randn((1, T * B, H, D), dtype=dtype, device=device)
    g = torch.empty((1, T * B, HQ), dtype=dtype, device=device).uniform_(-0.1, -0.01)
    sinks = torch.randn((HQ,), dtype=torch.float32, device=device) * 0.7
    cu_seqlens = torch.tensor([i * T for i in range(B + 1)], dtype=torch.int32, device=device)

    refs = []
    gate_scale = 0.1 if do_gate_scale else 1.0
    for bos, eos in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False):
        q_i = q[:, len(refs):len(refs) + 1].float()
        g_i = g[:, bos:eos].float()
        query_idx = torch.tensor([eos - bos - 1], dtype=torch.long, device=device)
        o_i, _ = naive_parallel_attn(
            q=q_i,
            k=k[:, bos:eos].float(),
            v=v[:, bos:eos].float(),
            g=g_i,
            g_scale=gate_scale,
            query_indices=query_idx,
            sinks=sinks,
            scale=0.1,
        )
        refs.append(o_i)
    ref = torch.cat(refs, dim=1).to(dtype)
    tri = attn_decoding_one_step(
        q=q,
        k=k,
        v=v,
        g=g,
        sinks=sinks,
        scale=0.1,
        cu_seqlens=cu_seqlens,
        do_gate_scale=do_gate_scale,
    )
    assert_close("o_decode_g_ref_vs_tri", ref, tri, 0.01)
