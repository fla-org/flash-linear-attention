# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.attn.naive import naive_attn_decoding, naive_parallel_attn
from fla.ops.attn.parallel import parallel_attn
from fla.utils import assert_close, device


def _repeat_kv_for_gpt_oss(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    B, H, T, D = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(B, H, n_rep, T, D)
    return hidden_states.reshape(B, H * n_rep, T, D)


def _gpt_oss_eager_sink_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink_bias: torch.Tensor,
    *,
    scale: float,
    window_size: int | None = None,
):
    """
    Mirrors the sink path in GPT-OSS eager attention:
    concat sink logits as an extra column, softmax once, then drop the sink column.
    """
    B, T, HQ, _ = q.shape
    H = k.shape[2]
    G = HQ // H

    query_states = q.transpose(1, 2)
    key_states = _repeat_kv_for_gpt_oss(k.transpose(1, 2), G)
    value_states = _repeat_kv_for_gpt_oss(v.transpose(1, 2), G)

    attn_logits = torch.matmul(query_states, key_states.transpose(2, 3)) * scale

    row_idx = torch.arange(T, device=q.device)[None, :, None]
    col_idx = torch.arange(T, device=q.device)[None, None, :]
    invalid = col_idx > row_idx
    if window_size is not None:
        invalid = invalid | (row_idx - col_idx >= window_size)
    attn_logits = attn_logits.masked_fill(invalid[:, None], float("-inf"))

    sink_bias_logits = sink_bias.view(1, HQ, 1, 1).expand(B, HQ, T, 1)
    combined_logits = torch.cat((attn_logits, sink_bias_logits), dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    attn_probs = probs[..., :-1]
    output = torch.matmul(attn_probs, value_states).transpose(1, 2).contiguous()
    return output, attn_probs


@pytest.mark.parametrize(
    "window_size",
    [
        pytest.param(None, id="full"),
        pytest.param(64, id="swa"),
    ],
)
def test_attn_sink_ref_matches_gpt_oss_eager(window_size):
    torch.manual_seed(777)
    dtype = torch.float64

    B, T, H, HQ, D = 2, 96, 2, 8, 64

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    sink_bias = torch.randn((HQ,), dtype=dtype, device=device)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    sink_bias_ref = sink_bias.detach().clone().requires_grad_(True)
    o_ref, _ = naive_parallel_attn(
        q=q_ref,
        k=k_ref,
        v=v_ref,
        sink_bias=sink_bias_ref,
        scale=0.1,
        window_size=window_size,
    )
    o_ref.backward(do)

    q_gpt = q.detach().clone().requires_grad_(True)
    k_gpt = k.detach().clone().requires_grad_(True)
    v_gpt = v.detach().clone().requires_grad_(True)
    sink_bias_gpt = sink_bias.detach().clone().requires_grad_(True)
    o_gpt, _ = _gpt_oss_eager_sink_reference(
        q=q_gpt,
        k=k_gpt,
        v=v_gpt,
        sink_bias=sink_bias_gpt,
        scale=0.1,
        window_size=window_size,
    )
    o_gpt.backward(do)

    assert_close(" o_ref_vs_gpt", o_ref, o_gpt, 1e-10, err_atol=1e-10)
    assert_close("dq_ref_vs_gpt", q_ref.grad, q_gpt.grad, 1e-10, err_atol=1e-10)
    assert_close("dk_ref_vs_gpt", k_ref.grad, k_gpt.grad, 1e-10, err_atol=1e-10)
    assert_close("dv_ref_vs_gpt", v_ref.grad, v_gpt.grad, 1e-10, err_atol=1e-10)
    assert_close("ds_ref_vs_gpt", sink_bias_ref.grad, sink_bias_gpt.grad, 1e-10, err_atol=1e-10)


def test_attn_sink_empty_row_ref_matches_gpt_oss_eager():
    torch.manual_seed(778)
    dtype = torch.float64

    B, T, H, HQ, D = 2, 48, 2, 8, 64
    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    sink_bias = torch.randn((HQ,), dtype=dtype, device=device)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    sink_bias_ref = sink_bias.detach().clone().requires_grad_(True)
    o_ref, _ = naive_parallel_attn(
        q=q_ref,
        k=k_ref,
        v=v_ref,
        sink_bias=sink_bias_ref,
        scale=0.1,
        window_size=0,
    )
    o_ref.backward(do)

    q_gpt = q.detach().clone().requires_grad_(True)
    k_gpt = k.detach().clone().requires_grad_(True)
    v_gpt = v.detach().clone().requires_grad_(True)
    sink_bias_gpt = sink_bias.detach().clone().requires_grad_(True)
    o_gpt, _ = _gpt_oss_eager_sink_reference(
        q=q_gpt,
        k=k_gpt,
        v=v_gpt,
        sink_bias=sink_bias_gpt,
        scale=0.1,
        window_size=0,
    )
    o_gpt.backward(do)

    assert_close(" o_empty_ref_vs_gpt", o_ref, o_gpt, 1e-10, err_atol=1e-10)
    assert_close("dq_empty_ref_vs_gpt", q_ref.grad, q_gpt.grad, 1e-10, err_atol=1e-10)
    assert_close("dk_empty_ref_vs_gpt", k_ref.grad, k_gpt.grad, 1e-10, err_atol=1e-10)
    assert_close("dv_empty_ref_vs_gpt", v_ref.grad, v_gpt.grad, 1e-10, err_atol=1e-10)
    assert_close("ds_empty_ref_vs_gpt", sink_bias_ref.grad, sink_bias_gpt.grad, 1e-10, err_atol=1e-10)


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
    sink_bias = torch.randn((HQ,), dtype=torch.float32, device=device)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    q_ref = q.float().detach().clone().requires_grad_(True)
    k_ref = k.float().detach().clone().requires_grad_(True)
    v_ref = v.float().detach().clone().requires_grad_(True)
    sink_bias_ref = sink_bias.detach().clone().requires_grad_(True)

    if cu_seqlens is None:
        o_ref, _ = naive_parallel_attn(
            q=q_ref,
            k=k_ref,
            v=v_ref,
            sink_bias=sink_bias_ref,
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
                sink_bias=sink_bias_ref,
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
    sink_bias_tri = sink_bias.detach().clone().requires_grad_(True)
    o_tri = parallel_attn(
        q=q_tri,
        k=k_tri,
        v=v_tri,
        sink_bias=sink_bias_tri,
        scale=0.1,
        window_size=window_size,
        cu_seqlens=cu_seqlens,
    )
    o_tri.backward(do)

    assert_close(" o_ref_vs_tri", o_ref, o_tri, 0.01)
    assert_close("dq_ref_vs_tri", q_ref.grad.to(dtype), q_tri.grad, 0.02)
    assert_close("dk_ref_vs_tri", k_ref.grad.to(dtype), k_tri.grad, 0.02)
    assert_close("dv_ref_vs_tri", v_ref.grad.to(dtype), v_tri.grad, 0.02)
    assert_close("ds_ref_vs_tri", sink_bias_ref.grad, sink_bias_tri.grad, 0.02)


def test_parallel_attn_sink_empty_row_matches_reference():
    torch.manual_seed(987)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    dtype = torch.float16
    B, T, H, HQ, D = 2, 96, 2, 8, 64
    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    sink_bias = torch.randn((HQ,), dtype=torch.float32, device=device)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    q_ref = q.float().detach().clone().requires_grad_(True)
    k_ref = k.float().detach().clone().requires_grad_(True)
    v_ref = v.float().detach().clone().requires_grad_(True)
    sink_bias_ref = sink_bias.detach().clone().requires_grad_(True)
    o_ref, _ = naive_parallel_attn(
        q=q_ref,
        k=k_ref,
        v=v_ref,
        sink_bias=sink_bias_ref,
        scale=0.1,
        window_size=0,
    )
    o_ref = o_ref.to(dtype)
    o_ref.backward(do)

    q_tri = q.detach().clone().requires_grad_(True)
    k_tri = k.detach().clone().requires_grad_(True)
    v_tri = v.detach().clone().requires_grad_(True)
    sink_bias_tri = sink_bias.detach().clone().requires_grad_(True)
    o_tri = parallel_attn(
        q=q_tri,
        k=k_tri,
        v=v_tri,
        sink_bias=sink_bias_tri,
        scale=0.1,
        window_size=0,
    )
    o_tri.backward(do)

    assert_close(" o_ref_vs_tri", o_ref, o_tri, 0.01)
    assert_close("dq_ref_vs_tri", q_ref.grad.to(dtype), q_tri.grad, 0.02)
    assert_close("dk_ref_vs_tri", k_ref.grad.to(dtype), k_tri.grad, 0.02)
    assert_close("dv_ref_vs_tri", v_ref.grad.to(dtype), v_tri.grad, 0.02)
    assert_close("ds_ref_vs_tri", sink_bias_ref.grad, sink_bias_tri.grad, 0.02)


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
    sink_bias = torch.randn((HQ,), dtype=torch.float32, device=device)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    q_ref = q.float().detach().clone().requires_grad_(True)
    k_ref = k.float().detach().clone().requires_grad_(True)
    v_ref = v.float().detach().clone().requires_grad_(True)
    g_ref = g.float().detach().clone().requires_grad_(True)
    sink_bias_ref = sink_bias.detach().clone().requires_grad_(True)

    if cu_seqlens is None:
        o_ref, _ = naive_parallel_attn(
            q=q_ref,
            k=k_ref,
            v=v_ref,
            g=g_ref,
            sink_bias=sink_bias_ref,
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
                sink_bias=sink_bias_ref,
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
    sink_bias_tri = sink_bias.detach().clone().requires_grad_(True)
    o_tri = parallel_attn(
        q=q_tri,
        k=k_tri,
        v=v_tri,
        g=g_tri,
        sink_bias=sink_bias_tri,
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
    assert_close("ds_ref_vs_tri", sink_bias_ref.grad, sink_bias_tri.grad, 0.02)


def test_attn_decoding_sink_matches_reference():
    torch.manual_seed(456)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    B, T, H, HQ, D = 3, 128, 2, 8, 64
    dtype = torch.float16
    q = torch.randn((1, B, HQ, D), dtype=dtype, device=device)
    k = torch.randn((1, T * B, H, D), dtype=dtype, device=device)
    v = torch.randn((1, T * B, H, D), dtype=dtype, device=device)
    sink_bias = torch.randn((HQ,), dtype=torch.float32, device=device)
    cu_seqlens = torch.tensor([i * T for i in range(B + 1)], dtype=torch.int32, device=device)

    ref = naive_attn_decoding(
        q=q.float(),
        k=k.float(),
        v=v.float(),
        sink_bias=sink_bias,
        scale=0.1,
        cu_seqlens=cu_seqlens,
    ).to(dtype)

    tri = attn_decoding_one_step(
        q=q,
        k=k,
        v=v,
        sink_bias=sink_bias,
        scale=0.1,
        cu_seqlens=cu_seqlens,
    )
    assert_close("o_decode_ref_vs_tri", ref, tri, 0.01)


def test_attn_decoding_sink_empty_row_matches_reference():
    torch.manual_seed(457)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    B, H, HQ, D = 3, 2, 8, 64
    lengths = [0, 128, 73]
    dtype = torch.float16
    q = torch.randn((1, B, HQ, D), dtype=dtype, device=device)
    total_t = sum(lengths)
    k = torch.randn((1, total_t, H, D), dtype=dtype, device=device)
    v = torch.randn((1, total_t, H, D), dtype=dtype, device=device)
    sink_bias = torch.randn((HQ,), dtype=torch.float32, device=device)
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)

    ref = naive_attn_decoding(
        q=q.float(),
        k=k.float(),
        v=v.float(),
        sink_bias=sink_bias,
        scale=0.1,
        cu_seqlens=cu_seqlens,
    ).to(dtype)

    tri = attn_decoding_one_step(
        q=q,
        k=k,
        v=v,
        sink_bias=sink_bias,
        scale=0.1,
        cu_seqlens=cu_seqlens,
    )
    assert torch.isfinite(tri).all()
    assert_close("o_decode_empty_row_ref_vs_tri", ref, tri, 0.01)


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
    sink_bias = torch.randn((HQ,), dtype=torch.float32, device=device) * 0.7
    cu_seqlens = torch.tensor([i * T for i in range(B + 1)], dtype=torch.int32, device=device)

    ref = naive_attn_decoding(
        q=q.float(),
        k=k.float(),
        v=v.float(),
        g=g.float(),
        sink_bias=sink_bias,
        scale=0.1,
        cu_seqlens=cu_seqlens,
        do_gate_scale=do_gate_scale,
    ).to(dtype)
    tri = attn_decoding_one_step(
        q=q,
        k=k,
        v=v,
        g=g,
        sink_bias=sink_bias,
        scale=0.1,
        cu_seqlens=cu_seqlens,
        do_gate_scale=do_gate_scale,
    )
    assert_close("o_decode_g_ref_vs_tri", ref, tri, 0.01)
