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
from fla.ops.attn.naive import naive_parallel_attn
from fla.ops.attn.parallel import parallel_attn
from fla.utils import assert_close, device


def _repeat_kv_for_gpt_oss(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)


def _gpt_oss_eager_sink_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    *,
    scale: float,
    window_size: int | None = None,
    query_indices: torch.Tensor | None = None,
):
    """
    Mirrors the sink path in transformers' GPT-OSS eager attention:
    concat sink logits as an extra column, softmax once, then drop the sink column.
    """
    batch_size, q_len, num_heads, _ = q.shape
    kv_len = k.shape[1]
    num_kv_heads = k.shape[2]
    num_key_value_groups = num_heads // num_kv_heads

    query_states = q.transpose(1, 2)
    key_states = _repeat_kv_for_gpt_oss(k.transpose(1, 2), num_key_value_groups)
    value_states = _repeat_kv_for_gpt_oss(v.transpose(1, 2), num_key_value_groups)

    attn_logits = torch.matmul(query_states, key_states.transpose(2, 3)) * scale

    if query_indices is None:
        query_positions = torch.arange(q_len, device=q.device).unsqueeze(0).expand(batch_size, q_len)
    else:
        if query_indices.ndim == 1:
            query_positions = query_indices.unsqueeze(0).expand(batch_size, q_len)
        else:
            query_positions = query_indices
        query_positions = query_positions.to(device=q.device, dtype=torch.long)
        assert query_positions.shape == (batch_size, q_len), "query_indices must have shape [TQ] or [B, TQ]"

    row_idx = query_positions[:, :, None]
    col_idx = torch.arange(kv_len, device=q.device)[None, None, :]
    invalid = col_idx > row_idx
    if window_size is not None:
        invalid = invalid | (row_idx - col_idx >= window_size)
    attn_logits = attn_logits.masked_fill(invalid[:, None], float("-inf"))

    sink_logits = sinks.view(1, num_heads, 1, 1).expand(batch_size, num_heads, q_len, 1)
    combined_logits = torch.cat((attn_logits, sink_logits), dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    attn_probs = probs[..., :-1]
    output = torch.matmul(attn_probs, value_states).transpose(1, 2).contiguous()
    return output, attn_probs


@pytest.mark.parametrize(
    ("window_size", "query_indices"),
    [
        pytest.param(None, None, id="full"),
        pytest.param(64, None, id="swa"),
        pytest.param(None, [0, 31, 63, 95], id="indexed"),
        pytest.param(32, [31, 63, 95, 95], id="indexed_swa"),
    ],
)
def test_attn_sink_ref_matches_gpt_oss_eager(window_size, query_indices):
    torch.manual_seed(777)
    dtype = torch.float64

    batch_size, kv_len, q_len, num_kv_heads, num_heads, head_dim = 2, 96, 96, 2, 8, 64
    if query_indices is not None:
        q_len = len(query_indices)
        query_indices = torch.tensor(query_indices, device=device)

    q = torch.randn((batch_size, q_len, num_heads, head_dim), dtype=dtype, device=device)
    k = torch.randn((batch_size, kv_len, num_kv_heads, head_dim), dtype=dtype, device=device)
    v = torch.randn((batch_size, kv_len, num_kv_heads, head_dim), dtype=dtype, device=device)
    sinks = torch.randn((num_heads,), dtype=dtype, device=device)
    do = torch.randn((batch_size, q_len, num_heads, head_dim), dtype=dtype, device=device)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    sinks_ref = sinks.detach().clone().requires_grad_(True)
    o_ref, _ = naive_parallel_attn(
        q=q_ref,
        k=k_ref,
        v=v_ref,
        sinks=sinks_ref,
        scale=0.1,
        window_size=window_size,
        query_indices=query_indices,
    )
    o_ref.backward(do)

    q_gpt = q.detach().clone().requires_grad_(True)
    k_gpt = k.detach().clone().requires_grad_(True)
    v_gpt = v.detach().clone().requires_grad_(True)
    sinks_gpt = sinks.detach().clone().requires_grad_(True)
    o_gpt, _ = _gpt_oss_eager_sink_reference(
        q=q_gpt,
        k=k_gpt,
        v=v_gpt,
        sinks=sinks_gpt,
        scale=0.1,
        window_size=window_size,
        query_indices=query_indices,
    )
    o_gpt.backward(do)

    assert_close(" o_ref_vs_gpt", o_ref, o_gpt, 1e-10, err_atol=1e-10)
    assert_close("dq_ref_vs_gpt", q_ref.grad, q_gpt.grad, 1e-10, err_atol=1e-10)
    assert_close("dk_ref_vs_gpt", k_ref.grad, k_gpt.grad, 1e-10, err_atol=1e-10)
    assert_close("dv_ref_vs_gpt", v_ref.grad, v_gpt.grad, 1e-10, err_atol=1e-10)
    assert_close("ds_ref_vs_gpt", sinks_ref.grad, sinks_gpt.grad, 1e-10, err_atol=1e-10)


def test_attn_sink_empty_row_ref_matches_gpt_oss_eager():
    torch.manual_seed(778)
    dtype = torch.float64

    batch_size, seq_len, num_kv_heads, num_heads, head_dim = 2, 48, 2, 8, 64
    q = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=dtype, device=device)
    k = torch.randn((batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype, device=device)
    v = torch.randn((batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype, device=device)
    sinks = torch.randn((num_heads,), dtype=dtype, device=device)
    do = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=dtype, device=device)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    sinks_ref = sinks.detach().clone().requires_grad_(True)
    o_ref, _ = naive_parallel_attn(q=q_ref, k=k_ref, v=v_ref, sinks=sinks_ref, scale=0.1, window_size=0)
    o_ref.backward(do)

    q_gpt = q.detach().clone().requires_grad_(True)
    k_gpt = k.detach().clone().requires_grad_(True)
    v_gpt = v.detach().clone().requires_grad_(True)
    sinks_gpt = sinks.detach().clone().requires_grad_(True)
    o_gpt, _ = _gpt_oss_eager_sink_reference(
        q=q_gpt,
        k=k_gpt,
        v=v_gpt,
        sinks=sinks_gpt,
        scale=0.1,
        window_size=0,
    )
    o_gpt.backward(do)

    assert_close(" o_empty_ref_vs_gpt", o_ref, o_gpt, 1e-10, err_atol=1e-10)
    assert_close("dq_empty_ref_vs_gpt", q_ref.grad, q_gpt.grad, 1e-10, err_atol=1e-10)
    assert_close("dk_empty_ref_vs_gpt", k_ref.grad, k_gpt.grad, 1e-10, err_atol=1e-10)
    assert_close("dv_empty_ref_vs_gpt", v_ref.grad, v_gpt.grad, 1e-10, err_atol=1e-10)
    assert_close("ds_empty_ref_vs_gpt", sinks_ref.grad, sinks_gpt.grad, 1e-10, err_atol=1e-10)


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

    q_tri = q.detach().clone().requires_grad_(True)
    k_tri = k.detach().clone().requires_grad_(True)
    v_tri = v.detach().clone().requires_grad_(True)
    sinks_tri = sinks.detach().clone().requires_grad_(True)
    o_tri = parallel_attn(
        q=q_tri, k=k_tri, v=v_tri, sinks=sinks_tri, scale=0.1, window_size=window_size, cu_seqlens=cu_seqlens
    )
    o_tri.backward(do)

    assert_close(" o_ref_vs_tri", o_ref, o_tri, 0.01)
    assert_close("dq_ref_vs_tri", q_ref.grad.to(dtype), q_tri.grad, 0.02)
    assert_close("dk_ref_vs_tri", k_ref.grad.to(dtype), k_tri.grad, 0.02)
    assert_close("dv_ref_vs_tri", v_ref.grad.to(dtype), v_tri.grad, 0.02)
    assert_close("ds_ref_vs_tri", sinks_ref.grad, sinks_tri.grad, 0.02)


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
    sinks = torch.randn((HQ,), dtype=torch.float32, device=device)
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)

    refs = []
    for bos, eos in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False):
        query_idx = max(eos - bos - 1, 0)
        o_i, _ = _gpt_oss_eager_sink_reference(
            q=q[:, len(refs):len(refs)+1].float(),
            k=k[:, bos:eos].float(),
            v=v[:, bos:eos].float(),
            sinks=sinks,
            scale=0.1,
            query_indices=torch.tensor([query_idx], dtype=torch.long, device=device),
        )
        refs.append(o_i)
    ref = torch.cat(refs, dim=1).to(dtype)

    tri = attn_decoding_one_step(q=q, k=k, v=v, sinks=sinks, scale=0.1, cu_seqlens=cu_seqlens)
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
