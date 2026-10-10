# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib.util
import inspect
import logging
import os

import pytest
import torch
import torch.nn.functional as F
import triton

from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd
from fla.ops.kda.gate import fused_kda_gate, naive_kda_gate, naive_kda_lowerbound_gate
from fla.ops.kda.naive import naive_chunk_kda, naive_recurrent_kda
from fla.ops.utils.cache import FLA_CACHE_MODE
from fla.utils import IS_INTEL_ALCHEMIST, IS_NPU, IS_NVIDIA, assert_close, device


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "scale", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HV{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 64, 1, 1, 64, 1, 1, torch.float),
            (2, 512, 3, 3, 60, 1, 1, torch.float),
            (4, 1024, 4, 4, 128, 0.1, 1, torch.float),
            (4, 1024, 4, 4, 128, 1, 10, torch.float),

            (1, 64, 1, 2, 64, 1, 1, torch.float),
            (2, 512, 2, 4, 60, 1, 1, torch.float),
        ]
    ],
)
def test_naive_chunk(B: int, T: int, H: int, HV: int, D: int, scale: float, gate_logit_normalizer: float, dtype: torch.dtype):
    torch.manual_seed(42)
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason="chunk_gated_delta_rule is not supported on alchemist for D>128")

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, HV, D, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, HV, D, dtype=torch.float)) / gate_logit_normalizer
    beta = torch.randn(B, T, HV, dtype=dtype).sigmoid()
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32)
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


def test_chunk_invalid_chunk_size():
    B, T, H, D = 1, 64, 1, 64
    q = torch.randn(B, T, H, D, dtype=torch.float, device=device)
    k = torch.randn(B, T, H, D, dtype=torch.float, device=device)
    v = torch.randn(B, T, H, D, dtype=torch.float, device=device)
    g = torch.randn(B, T, H, D, dtype=torch.float, device=device)
    beta = torch.randn(B, T, H, dtype=torch.float, device=device)

    with pytest.raises(ValueError, match=r"`chunk_size` must be either 32 or 64"):
        chunk_kda(q, k, v, g, beta, chunk_size=16)


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "scale", "gate_logit_normalizer", "use_qk_l2norm_in_kernel", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HV{}-D{}-scale{}-gate_logit_normalizer{}-use_qk_l2norm{}-{}".format(*test))
        for test in [
            (1, 64, 1, 1, 64, 1, 1, False, torch.float),
            (2, 512, 3, 3, 60, 1, 1, False, torch.float),
            (3, 1000, 4, 4, 100, 0.1, 1, True, torch.float),
            (4, 1024, 4, 4, 128, 0.1, 1, False, torch.float),
            (2, 512, 2, 4, 60, 1, 1, False, torch.float),
            (2, 1024, 2, 8, 128, 0.1, 1, True, torch.float),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    HV: int,
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
    v = torch.rand(B, T, HV, D, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, HV, D, dtype=torch.float)) / gate_logit_normalizer
    beta = torch.randn(B, T, HV, dtype=dtype).sigmoid()
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32)
    q, k, v, g, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, g, beta, h0))

    g_tri = g.clone()
    if triton.next_power_of_2(D) != D:
        # the kernel loads g in next_power_of_2(D) blocks;
        # poison the memory right after g so any read past D shows up as NaN instead of a silent no-op
        buf = torch.full((g.numel() + triton.next_power_of_2(D),), 1e30, dtype=torch.float, device=device)
        g_tri = buf[:g.numel()].view_as(g).copy_(g)

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
        g=g_tri,
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "allow_neg_eigval", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-allow_neg_eigval{}-{}".format(*test))
        for test in [
            (2, 256, 4, 64, False, torch.float),
            (2, 256, 4, 64, True, torch.float),
            (1, 512, 3, 60, True, torch.float),
        ]
    ],
)
def test_fused_recurrent_use_beta_sigmoid_in_kernel(
    B: int,
    T: int,
    H: int,
    D: int,
    allow_neg_eigval: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float))
    beta_post = torch.randn(B, T, H, dtype=dtype).sigmoid()
    beta_raw = torch.logit(beta_post.float().clamp_(1e-4, 1 - 1e-4)).to(dtype)
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, g, beta_post, beta_raw, h0 = map(lambda x: x.to(device), (q, k, v, g, beta_post, beta_raw, h0))

    ref, ref_ht = fused_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta_post.clone() * (2 if allow_neg_eigval else 1),
        scale=None,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    tri, tri_ht = fused_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta_raw.clone(),
        scale=None,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=allow_neg_eigval,
    )
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "scale", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 64, 1, 64, 1, 1, torch.float),
            (2, 512, 3, 60, 1, 1, torch.float),
            (3, 1000, 4, 100, 0.1, 1, torch.float),
            (4, 1024, 4, 128, 0.1, 1, torch.float),
        ]
    ],
)
def test_fused_recurrent_state_v_first(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float)) / gate_logit_normalizer
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    h0_kv = torch.randn(B, H, D, D, dtype=torch.float32)
    h0_vk = h0_kv.transpose(-1, -2).contiguous()
    q, k, v, g, beta, h0_kv, h0_vk = map(lambda x: x.to(device), (q, k, v, g, beta, h0_kv, h0_vk))

    ref, ref_ht = fused_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0_kv.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        state_v_first=False,
    )
    tri, tri_ht = fused_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0_vk.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        state_v_first=True,
    )
    assert_close("o", ref, tri, 1e-4)
    assert_close("ht", ref_ht, tri_ht.transpose(-1, -2), 1e-4)

    # the legacy `transpose_state_layout` kwarg maps to `state_v_first` with a warning,
    # and passing both names at once is rejected
    with pytest.warns(DeprecationWarning):
        fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, transpose_state_layout=True)
    with pytest.raises(ValueError):
        fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, state_v_first=True, transpose_state_layout=True)


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "scale", "has_a_log", "has_dt_bias", "safe_gate", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HV{}-D{}-scale{}-has_a_log{}-has_dt_bias{}-safe_gate{}-{}".format(*test))
        for test in [
            (1, 64, 1, 1, 64, 1, True, False, False, torch.float),
            (2, 256, 2, 2, 64, 1, True, True, False, torch.float),
            (2, 512, 2, 4, 64, 0.1, True, True, True, torch.float16),
            (3, 1000, 2, 8, 128, 1, True, False, False, torch.float16),
            (4, 1024, 4, 4, 128, 0.1, True, True, True, torch.float16),
            (1, 64, 1, 1, 64, 1, False, False, True, torch.float),
            (2, 256, 2, 4, 64, 1, False, True, True, torch.float),
        ]
    ],
)
def test_fused_recurrent_gate_in_kernel(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    scale: float,
    has_a_log: bool,
    has_dt_bias: bool,
    safe_gate: bool,
    dtype: torch.dtype,
):
    """fused_recurrent_kda with use_gate_in_kernel=True matches manual gate path."""
    torch.manual_seed(42)
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason="fused_recurrent_kda is not supported on alchemist for D>128")

    q = torch.rand(B, T, H, D, dtype=dtype, device=device)
    k = torch.rand(B, T, H, D, dtype=dtype, device=device)
    v = torch.rand(B, T, HV, D, dtype=dtype, device=device)
    beta = torch.rand(B, T, HV, dtype=dtype, device=device).sigmoid()
    g_raw = torch.randn(B, T, HV, D, dtype=torch.float32, device=device)
    A_log = torch.log(torch.empty(HV, dtype=torch.float32, device=device).uniform_(1, 16)) if has_a_log else None
    dt_bias = torch.randn(HV * D, dtype=torch.float32, device=device) if has_dt_bias else None
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32, device=device)

    lower_bound = -5.0 if safe_gate else None
    naive_gate_fn = naive_kda_lowerbound_gate if safe_gate else naive_kda_gate
    g_ref = naive_gate_fn(g_raw, A_log, dt_bias)

    ref, ref_ht = fused_recurrent_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g_ref,
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    tri, tri_ht = fused_recurrent_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g_raw.clone(),
        beta=beta.clone(),
        A_log=A_log.clone() if A_log is not None else None,
        dt_bias=dt_bias.clone() if dt_bias is not None else None,
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
    )
    assert_close("o", ref, tri, 0.002)
    assert_close("ht", ref_ht, tri_ht, 0.002)


@pytest.mark.parametrize(
    ("B", "H", "D", "scale", "gate_logit_normalizer", "use_qk_l2norm_in_kernel", "use_gate_in_kernel", "safe_gate", "dtype"),
    [
        pytest.param(*test, id="B{}-H{}-D{}-scale{}-norm{}-qk_l2{}-gate{}-safe_gate{}-dtype{}".format(*test))
        for test in [
            (16, 16, 128, 0.1, 1.0, True, False, False, torch.bfloat16),
            (32, 8, 64, 1.0, 1.0, False, False, False, torch.float16),
            (7, 32, 128, 0.5, 0.5, True, False, False, torch.bfloat16),
            (16, 16, 128, 0.1, 1.0, True, True, False, torch.bfloat16),
            (32, 8, 64, 1.0, 1.0, False, True, False, torch.float16),
            (7, 32, 128, 0.5, 0.5, True, True, True, torch.bfloat16),
        ]
    ],
)
@pytest.mark.skipif(not (IS_NVIDIA or IS_NPU), reason='test_fused_recurrent_vllm_decode requires CUDA or NPU')
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
    device = torch.device("npu" if IS_NPU else "cuda")

    max_cache_slots = B * 3
    state_pool = torch.randn(max_cache_slots, H, D, D, dtype=torch.float32, device=device)
    state_indices = torch.randperm(max_cache_slots, device=device)[:B].int()

    # fill unaccessed slots with a huge value to detect out-of-bounds access.
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
        lower_bound=lower_bound,
    )

    assert_close("o", ref_out, tri_out, 0.005)
    assert_close("ht", ref_state_pool[state_indices.long()], tri_state_pool[state_indices.long()], 0.005)

    mask = torch.ones(max_cache_slots, dtype=torch.bool, device=device)
    mask[state_indices.long()] = False
    assert_close("Untouched ht", ref_state_pool[mask], tri_state_pool[mask], 0.0)


@pytest.mark.parametrize('state_v_first', [False, True])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('index_stride', [1, 2])
@pytest.mark.parametrize('accepted_tokens', [1, 2])
@pytest.mark.parametrize('state_output', ['inplace', 'separate', 'none'])
@pytest.mark.skipif(not (IS_NVIDIA or IS_NPU), reason='Indexed KDA requires CUDA or NPU')
def test_fused_recurrent_indexed_state(state_v_first, packed, index_stride, accepted_tokens, state_output, monkeypatch):
    monkeypatch.setattr("fla.utils.testing.FLA_CI_ENV", False)
    torch.manual_seed(42)
    H, HV, K, V = 2, 4, 32, 24
    lengths = [2, 4] if packed else [3, 3]
    q, k = [F.normalize(torch.randn(1, 6, H, K, device=device), dim=-1) for _ in range(2)]
    v = torch.randn(1, 6, HV, V, device=device)
    g = F.logsigmoid(torch.randn(1, 6, HV, K, device=device))
    beta = torch.rand(1, 6, HV, device=device)
    original_state = torch.randn(16, HV, K, V, device=device)
    indices = torch.arange(16, dtype=torch.int32, device=device).reshape(2, 8)[:, ::2]
    if index_stride == 1:
        indices = indices.contiguous()
    accepted = torch.full((2,), accepted_tokens, dtype=torch.int32, device=device)
    expected_pool = original_state.clone()
    expected_outputs, expected_states = [], []
    offset = 0
    for sequence, length in enumerate(lengths):
        slot = indices[sequence, accepted_tokens - 1].item()
        state = original_state[slot:slot + 1].clone()
        for token in range(length):
            position = offset + token
            output, state = naive_recurrent_kda(
                q=q[:, position:position + 1],
                k=k[:, position:position + 1],
                v=v[:, position:position + 1],
                g=g[:, position:position + 1],
                beta=beta[:, position:position + 1],
                initial_state=state,
                output_final_state=True,
            )
            expected_outputs.append(output)
            expected_states.append(state.clone())
            expected_pool[indices[sequence, token].item()] = state[0]
        offset += length
    expected_output = torch.cat(expected_outputs, dim=1)
    expected_states = torch.cat(expected_states, dim=0)
    inputs = dict(q=q, k=k, v=v, g=g, beta=beta)
    if packed:
        inputs['cu_seqlens'] = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32, device=device)
    else:
        inputs = {name: tensor.reshape(2, 3, *tensor.shape[2:]) for name, tensor in inputs.items()}
    pool = original_state.transpose(-1, -2).contiguous() if state_v_first else original_state.clone()
    output, states = fused_recurrent_kda_fwd(
        **inputs,
        initial_state=pool,
        output_final_state=state_output == 'separate',
        inplace_final_state=state_output == 'inplace',
        state_v_first=state_v_first,
        ssm_state_indices=indices,
        num_accepted_tokens=accepted,
    )
    assert_close('indexed output', expected_output, output.reshape_as(expected_output), 0.002)
    pool_k_first = pool.transpose(-1, -2) if state_v_first else pool
    if state_output == 'inplace':
        untouched = torch.ones(16, dtype=torch.bool, device=device)
        for sequence, length in enumerate(lengths):
            untouched[indices[sequence, :length].long()] = False
        if not torch.equal(original_state[untouched], pool_k_first[untouched]):
            changed = (original_state != pool_k_first).flatten(1).any(1)
            unexpected_slots = torch.nonzero(untouched & changed).flatten().tolist()
            pytest.fail(f'Unaddressed state slots changed during indexed updates: {unexpected_slots}')
        try:
            assert_close('updated pool', expected_pool, pool_k_first, 0.002)
        except AssertionError:
            for sequence, length in enumerate(lengths):
                for token in range(length):
                    slot = indices[sequence, token].item()
                    error = (expected_pool[slot] - pool_k_first[slot]).abs()
                    head_errors = error.amax(dim=(-2, -1)).tolist()
                    head, offset = divmod(error.argmax().item(), K * V)
                    key, value = divmod(offset, V)
                    logging.getLogger(__name__).error(
                        f'Indexed state sequence={sequence} token={token} slot={slot} '
                        f'terminal={token == length - 1} head_max_abs={head_errors} '
                        f'head={head} key={key} value={value} '
                        f'expected={expected_pool[slot, head, key, value].item()} '
                        f'actual={pool_k_first[slot, head, key, value].item()}',
                    )
            raise
        terminal_states = pool_k_first[indices[torch.arange(2, device=device),
                                               torch.tensor(lengths, device=device) - 1].long()]
    else:
        if not torch.equal(original_state, pool_k_first):
            pytest.fail('The input state pool changed during out-of-place execution')
        if state_output == 'none':
            if states is not None:
                pytest.fail('State output must be None when state storage is disabled')
            return
        states = states.transpose(-1, -2) if state_v_first else states
        if states.shape != expected_states.shape:
            pytest.fail(f'Expected token states {expected_states.shape}, received {states.shape}')
        assert_close('token states', expected_states, states, 0.002)
        terminal_states = states[torch.tensor(lengths, device=device).cumsum(0) - 1]
    next_inputs = {name: tensor[:, :2].reshape(2, 1, *tensor.shape[2:])
                   for name, tensor in dict(q=q, k=k, v=v, g=g, beta=beta).items()}
    reference_terminal = expected_states[torch.tensor(lengths, device=device).cumsum(0) - 1]
    expected_next, expected_final = naive_recurrent_kda(
        **next_inputs,
        initial_state=reference_terminal,
        output_final_state=True,
    )
    next_output, next_state = fused_recurrent_kda_fwd(
        **next_inputs,
        initial_state=terminal_states.contiguous(),
        output_final_state=True,
        inplace_final_state=False,
    )
    assert_close('resumed output', expected_next, next_output, 0.002)
    assert_close('resumed state', expected_final, next_state, 0.002)


@pytest.mark.parametrize(
    (
        "B",
        "T",
        "H",
        "HV",
        "D",
        "scale",
        "gate_logit_normalizer",
        "mask_p",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "dtype",
        "safe_gate",
        "disable_recompute",
        "chunk_size",
    ),
    [
        pytest.param(
            *test,
            id=(
                "B{}-T{}-H{}-HV{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}"
                "-use_qk_l2norm{}-use_gate{}-{}-safe_gate{}-disable_recompute{}-chunk_size{}"
            ).format(*test),
        )
        for test in [
            (1, 63, 1, 1, 64, 1, 1, 0, False, False, torch.float16, True, False, 64),
            (2, 500, 3, 3, 60, 1, 1, 0, False, False, torch.float16, True, True, 64),
            (2, 1000, 3, 3, 64, 0.1, 1, 0.5, False, False, torch.float16, False, True, 64),
            (3, 1024, 4, 4, 100, 1, 0.1, 0, False, False, torch.float16, False, False, 64),
            (4, 1024, 4, 4, 128, 0.1, 1, 0, False, False, torch.float16, True, True, 64),
            (4, 1024, 4, 4, 128, 0.1, 1, 0, True, False, torch.float16, True, False, 64),
            (2, 1500, 4, 4, 128, 0.1, 10, 0, False, True, torch.float16, False, True, 64),
            (4, 2048, 8, 8, 64, 0.1, 1, 0, False, True, torch.float16, True, True, 64),

            (2, 1024, 2, 4, 64, 0.1, 1, 0, False, False, torch.float16, False, False, 64),
            (2, 1024, 2, 8, 64, 0.1, 1, 0, False, True, torch.float16, False, False, 64),
            (2, 1024, 4, 8, 128, 0.1, 1, 0, True, True, torch.float16, False, False, 64),
            (2, 160, 2, 4, 64, 0.1, 1, 0, False, True, torch.float16, True, True, 32),

            (2, 1024, 2, 4, 64, 0.1, 1, 0, True, True, torch.bfloat16, True, False, 64),
            (2, 1024, 2, 4, 64, 0.1, 1, 0, False, True, torch.bfloat16, False, False, 64),
            (2, 160, 2, 4, 64, 0.1, 1, 0, False, True, torch.bfloat16, True, True, 32),
            (2, 1024, 2, 8, 128, 0.1, 1, 0, True, True, torch.bfloat16, False, False, 64),
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
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    dtype: torch.dtype,
    safe_gate: bool,
    disable_recompute: bool,
    chunk_size: int,
):
    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, HV, D, dtype=dtype)
    g = torch.randn(B, T, HV, D, dtype=torch.float if not use_gate_in_kernel else dtype)
    if use_gate_in_kernel:
        A_log = torch.randn(HV, dtype=torch.float)
        dt_bias = torch.randn(HV * D, dtype=torch.float)
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

    beta = torch.randn(B, T, HV, dtype=dtype).sigmoid()
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32)
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
        disable_recompute=disable_recompute,
        chunk_size=chunk_size,
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
    ("B", "T", "H", "D", "scale", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, 1, torch.float16),
            (2, 500, 3, 60, 1, 1, torch.float16),
            (3, 1024, 4, 128, 0.1, 1, torch.float16),
            (4, 2048, 8, 64, 0.1, 1, torch.float16),
        ]
    ],
)
def test_chunk_state_v_first(B: int, T: int, H: int, D: int, scale: float, gate_logit_normalizer: float, dtype: torch.dtype):
    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float)) / gate_logit_normalizer
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    h0_kv = torch.randn(B, H, D, D, dtype=torch.float32)
    h0_vk = h0_kv.transpose(-1, -2).contiguous()
    q, k, v, g, beta, h0_kv, h0_vk = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, g, beta, h0_kv, h0_vk))

    do = torch.randn_like(v)
    dht_vk = torch.randn(B, H, D, D, dtype=torch.float32, device=device)
    dht_kv = dht_vk.transpose(-1, -2).contiguous()

    tri, tri_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0_vk.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        state_v_first=True,
    )
    ((tri * do).sum() + (tri_ht * dht_vk).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dg, tri_db, tri_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0_vk.grad
    q.grad = k.grad = v.grad = g.grad = beta.grad = h0_vk.grad = None

    ref, ref_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0_kv.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        state_v_first=False,
    )
    ((ref * do).sum() + (ref_ht * dht_kv).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0_kv.grad

    assert_close("o", ref, tri, 1e-4)
    assert_close("ht", ref_ht, tri_ht.transpose(-1, -2), 1e-4)
    assert_close("dq", ref_dq, tri_dq, 1e-4)
    assert_close("dk", ref_dk, tri_dk, 1e-4)
    assert_close("dv", ref_dv, tri_dv, 1e-4)
    assert_close("dg", ref_dg, tri_dg, 1e-4)
    assert_close("db", ref_db, tri_db, 1e-4)
    assert_close("dh0", ref_dh0, tri_dh0.transpose(-1, -2), 1e-4)


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "allow_neg_eigval", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-allow_neg_eigval{}-{}".format(*test))
        for test in [
            (1, 8192, 96, 128, False, torch.bfloat16),
            (1, 8192, 96, 128, True, torch.bfloat16),
            (2, 96, 2, 64, True, torch.float16),
        ]
    ],
)
def test_chunk_use_beta_sigmoid_in_kernel(B: int, T: int, H: int, D: int, allow_neg_eigval: bool, dtype: torch.dtype):
    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float))
    beta_raw = torch.randn(B, T, H, dtype=dtype)
    beta_post = beta_raw.float().sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)

    q, k, v, g, beta_post, beta_raw, h0 = map(
        lambda x: x.to(device).requires_grad_(True),
        (q, k, v, g, beta_post, beta_raw, h0),
    )

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    ref, ref_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta_post.clone() * (2 if allow_neg_eigval else 1),
        scale=None,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dh0 = (q.grad, k.grad, v.grad, g.grad, beta_post.grad, h0.grad)
    q.grad = k.grad = v.grad = g.grad = beta_post.grad = h0.grad = None

    tri, tri_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta_raw.clone(),
        scale=None,
        initial_state=h0.clone(),
        output_final_state=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=allow_neg_eigval,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dg, tri_db_raw, tri_dh0 = (q.grad, k.grad, v.grad, g.grad, beta_raw.grad, h0.grad)

    ref_db_raw = ref_db * beta_post.detach().float() * (1 - beta_post.detach().float())
    ref_db_raw = ref_db_raw.to(tri_db_raw.dtype)

    rtol = 1e-4
    assert_close("o", ref, tri, rtol)
    assert_close("ht", ref_ht, tri_ht, rtol)
    assert_close("dq", ref_dq, tri_dq, rtol)
    assert_close("dk", ref_dk, tri_dk, rtol)
    assert_close("dv", ref_dv, tri_dv, rtol)
    assert_close("dg", ref_dg, tri_dg, rtol)
    assert_close("db_raw", ref_db_raw, tri_db_raw, rtol)
    assert_close("dh0", ref_dh0, tri_dh0, rtol)


@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens", "dtype", "use_gate_in_kernel", "safe_gate", "disable_recompute", "chunk_size"),
    [
        pytest.param(
            *test,
            id=("H{}-D{}-mask_p{}-cu_seqlens{}-{}-gate{}" "-safe_gate{}-disable_recompute{}-chunk_size{}").format(*test),
        )
        for test in [
            (4, 60, 0.1, [0, 15], torch.float16, True, False, False, 64),
            (4, 64, 0.9, [0, 256, 500, 1000], torch.float16, True, False, False, 64),
            (4, 128, 0.5, [0, 256, 500, 1000], torch.float16, False, False, False, 64),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.float16, True, False, False, 64),
            (4, 256, 0, [0, 100, 300, 1200, 3000, 4096], torch.float16, False, True, True, 64),
            (4, 60, 0.1, [0, 31, 96, 160], torch.float16, True, False, True, 32),
        ]
    ],
)
@pytest.mark.smoke
def test_chunk_varlen(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    disable_recompute: bool,
    chunk_size: int,
):
    if FLA_CACHE_MODE.uses_default_config() and D in (64, 256):
        pytest.skip(reason="Skipping D=64/256 varlen KDA case with default_config")
    torch.manual_seed(42)
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
        # k is already normalized.
        k=k.clone(),
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
        disable_recompute=disable_recompute,
        chunk_size=chunk_size,
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
            # k is already normalized.
            k=k[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            v=v[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            g=(
                naive_kda_gate(
                    g=g[:, cu_seqlens[i]: cu_seqlens[i + 1]].to(torch.float),
                    A_log=A_log.to(torch.float),
                    dt_bias=dt_bias.to(torch.float),
                )
                if use_gate_in_kernel else g[:, cu_seqlens[i]: cu_seqlens[i + 1]]
            ),
            beta=beta[:, cu_seqlens[i]: cu_seqlens[i + 1]],
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
    ("H", "D", "mask_p", "cu_seqlens", "dtype", "use_gate_in_kernel", "safe_gate", "disable_recompute"),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}-gate{}-safe_gate{}-disable_recompute{}".format(*test))
        for test in [
            (4, 60, 0.1, [0, 8192], torch.float16, True, False, False),
            (4, 64, 0.9, [0, 256, 500, 1000], torch.float16, True, False, False),
            (4, 128, 0.5, [0, 256, 500, 1000], torch.float16, False, False, False),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.float16, True, False, False),
            (4, 256, 0, [0, 100, 300, 1200, 3000, 4096], torch.float16, False, True, True),
        ]
    ],
)
@torch.inference_mode()
def test_chunk_varlen_prefill(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    disable_recompute: bool,
):
    if FLA_CACHE_MODE.uses_default_config() and D == 256:
        pytest.skip(reason="Skipping D=256 varlen KDA prefill case with default_config")
    torch.manual_seed(42)
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    cu_seqlens_cpu = cu_seqlens.cpu()
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype).to(device)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype).to(device)
    v = torch.randn((1, T, H, D), dtype=dtype).to(device)
    g = torch.randn(1, T, H, D, dtype=torch.float if not use_gate_in_kernel else dtype).to(device)
    if use_gate_in_kernel:
        A_log = torch.log(torch.randn(1, 1, H, 1, dtype=torch.float32, device=device).uniform_(1, 16)).to(device)
        dt_bias = torch.randn(H * D, dtype=torch.float32, device=device).to(device)
    else:
        g = F.logsigmoid(g)
        g = g * (torch.rand_like(g) > mask_p)
    mask = torch.rand_like(g) > mask_p
    g = g * mask + (~mask) * (-1000)
    if safe_gate:
        assert use_gate_in_kernel is False
        g = g.clamp(-5, 0)

    beta = torch.rand(1, T, H, dtype=dtype).sigmoid().to(device)
    h0 = torch.randn((N, H, D, D), dtype=torch.float32).to(device)

    tri, tri_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        # k is already normalized.
        k=k.clone(),
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

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = naive_recurrent_kda(
            q=F.normalize(q[:, cu_seqlens[i]: cu_seqlens[i + 1]], p=2, dim=-1),
            # k is already normalized.
            k=k[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            v=v[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            g=(
                naive_kda_gate(
                    g=g[:, cu_seqlens[i]: cu_seqlens[i + 1]].to(torch.float),
                    A_log=A_log.to(torch.float),
                    dt_bias=dt_bias.to(torch.float),
                )
                if use_gate_in_kernel else g[:, cu_seqlens[i]: cu_seqlens[i + 1]]
            ),
            beta=beta[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "HAS_A_LOG", "HAS_BIAS", "LOWER_BOUND"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-a_log{}-bias{}-lowerbound{}".format(*test))
        for test in [
            (1, 2, 2, 12, True, False, -5.0),
            (1, 32, 2, 16, True, False, -5.0),
            (2, 64, 4, 32, True, False, -5.0),
            (4, 128, 8, 64, True, False, -5.0),
            (4, 128, 8, 128, True, False, None),
            (1, 2, 2, 12, True, True, None),
            (1, 32, 2, 16, True, True, None),
            (2, 64, 4, 32, True, True, None),
            (4, 128, 8, 64, True, True, None),
            (4, 128, 8, 128, True, True, None),
            (1, 2, 2, 12, False, False, -5.0),
            (2, 64, 4, 32, False, False, -5.0),
            (4, 128, 8, 64, False, True, -5.0),
            (4, 128, 8, 128, False, True, -5.0),
        ]
    ],
)
def test_gate(B: int, T: int, H: int, D: int, HAS_A_LOG: bool, HAS_BIAS: bool, LOWER_BOUND: float | None):
    torch.manual_seed(42)
    g = torch.randn(B, T, H, D, dtype=torch.float32) * 10
    A_log = torch.log(torch.randn(1, 1, H, 1, dtype=torch.float32).uniform_(1, 16)) if HAS_A_LOG else None
    dt_bias = torch.randn(H * D, dtype=torch.float32) if HAS_BIAS else None
    g = g.to(device).requires_grad_(True)
    if A_log is not None:
        A_log = A_log.to(device).requires_grad_(True)
    if dt_bias is not None:
        dt_bias = dt_bias.to(device).requires_grad_(True)
    do = torch.randn_like(g).view(B, T, H, D)

    if LOWER_BOUND is not None:
        ref = naive_kda_lowerbound_gate(
            g=g.clone(),
            A_log=A_log.clone() if A_log is not None else None,
            dt_bias=dt_bias.clone() if dt_bias is not None else None,
            lower_bound=LOWER_BOUND,
        )
    else:
        ref = naive_kda_gate(g=g.clone(), A_log=A_log.clone(), dt_bias=dt_bias.clone() if dt_bias is not None else None)
    tri = fused_kda_gate(
        g=g.clone(),
        A_log=A_log.clone() if A_log is not None else None,
        dt_bias=dt_bias.clone() if dt_bias is not None else None,
        lower_bound=LOWER_BOUND,
    )
    (ref * do).sum().backward(retain_graph=True)

    ref_dg = g.grad
    ref_dA = A_log.grad if A_log is not None else None
    ref_dbias = dt_bias.grad if dt_bias is not None else None
    g.grad = None
    if A_log is not None:
        A_log.grad = None
    if dt_bias is not None:
        dt_bias.grad = None

    ((tri * do).sum()).backward(retain_graph=True)
    tri_dg = g.grad
    tri_dA = A_log.grad if A_log is not None else None
    tri_dbias = dt_bias.grad if dt_bias is not None else None

    assert_close("o", ref, tri, 1e-4)
    assert_close("dg", ref_dg, tri_dg, 1e-4)
    if HAS_A_LOG:
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
        o, final_state, h = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=True,
            return_intermediate_states=True,
            disable_recompute=False,
        )

        assert o.shape == (B, T, H, D), f"Output shape mismatch: {o.shape}"
        assert final_state.shape == (B, H, D, D), f"Final state shape mismatch: {final_state.shape}"

        expected_nt = (T + chunk_size - 1) // chunk_size
        assert h.shape == (B, expected_nt, H, D, D), f"h shape mismatch: {h.shape}, expected: {(B, expected_nt, H, D, D)}"
        assert h.dtype == dtype, f"h dtype should be bfloat16, got: {h.dtype}"

        total_tokens = 1024
        N = 2
        seq_len = total_tokens // N
        cu_seqlens = torch.tensor([0, seq_len, total_tokens], dtype=torch.long, device=device)

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
            disable_recompute=False,
        )

        assert o_varlen.shape == (1, total_tokens, H, D), f"Varlen output shape mismatch: {o_varlen.shape}"
        assert final_state_varlen.shape == (N, H, D, D), f"Varlen final state shape mismatch: {final_state_varlen.shape}"

        assert h_varlen.shape[0] == 1, f"Varlen h batch dim should be 1, got: {h_varlen.shape[0]}"
        assert h_varlen.shape[2:] == (H, D, D), f"Varlen h dims mismatch: {h_varlen.shape[2:]}"
        assert h_varlen.dtype == dtype, f"Varlen h dtype should be {dtype}, got: {h_varlen.dtype}"


_FLASH_KDA_AVAILABLE = importlib.util.find_spec("flash_kda") is not None
_SKIP_FLASH_KDA = pytest.mark.skipif(
    device == "cpu" or not _FLASH_KDA_AVAILABLE or os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1",
    reason="FlashKDA tests require GPU, the flash_kda package, and backend dispatch",
)

_FLASH_KDA_REQUIRED_KWARGS = dict(
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    safe_gate=True,
    lower_bound=-5.0,
    state_v_first=True,
)

_FLASH_KDA_RTOL = 0.006


def _flash_kda_make_gate_params(H, D):
    A_log = torch.log(torch.empty(H, dtype=torch.float32, device=device).uniform_(1, 16))
    dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)
    return A_log, dt_bias


def _flash_kda_run(monkeypatch, positional=False, **kwargs):
    import flash_kda

    monkeypatch.setenv("FLA_FLASH_KDA", "1")
    calls = []
    fwd = flash_kda.fwd

    def tracked_fwd(*args, **kwargs):
        calls.append(True)
        return fwd(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(flash_kda, "fwd", tracked_fwd)
        with torch.inference_mode():
            if positional:
                bound = inspect.signature(chunk_kda).bind(**kwargs, **_FLASH_KDA_REQUIRED_KWARGS)
                bound.apply_defaults()
                result = chunk_kda(*bound.args, **bound.kwargs)
            else:
                result = chunk_kda(**kwargs, **_FLASH_KDA_REQUIRED_KWARGS)
    expected_flash_kda = (
        not kwargs.get('allow_neg_eigval', False)
        and kwargs.get('A_log') is not None
        and kwargs.get('dt_bias') is not None
    )
    assert bool(calls) == expected_flash_kda
    return result


def _flash_kda_gold(
    q,
    k,
    v,
    g,
    beta_raw,
    A_log,
    dt_bias,
    scale,
    initial_state,
    lower_bound=-5.0,
    cu_seqlens=None,
    allow_neg_eigval=False,
):
    kwargs = {}
    if cu_seqlens is not None:
        kwargs["cu_seqlens"] = cu_seqlens
    return fused_recurrent_kda(
        q=q.to(torch.float64),
        k=k.to(torch.float64),
        v=v.to(torch.float64),
        g=g.to(torch.float64),
        beta=torch.sigmoid(beta_raw.to(torch.float64)) * (2 if allow_neg_eigval else 1),
        A_log=A_log.to(torch.float64) if A_log is not None else None,
        dt_bias=dt_bias.to(torch.float64) if dt_bias is not None else None,
        scale=scale,
        initial_state=initial_state.to(torch.float64),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
        state_v_first=True,
        **kwargs,
    )


@_SKIP_FLASH_KDA
@pytest.mark.parametrize(
    ("B", "T", "H", "D", "allow_neg_eigval", "has_A", "has_bias", "positional"),
    [
        pytest.param(1, 1024, 4, 128, False, True, True, False, id="dense"),
        pytest.param(2, 2048, 8, 128, False, True, True, False, id="batched"),
        pytest.param(1, 4096, 16, 128, False, True, True, False, id="long"),
        pytest.param(1, 1024, 4, 128, True, True, True, False, id="negative-eigenvalues"),
        pytest.param(1, 1024, 4, 128, False, False, True, False, id="no-A"),
        pytest.param(1, 1024, 4, 128, False, True, False, False, id="no-bias"),
        pytest.param(1, 1024, 4, 128, False, True, True, True, id="positional"),
        pytest.param(1, 1024, 4, 128, True, True, True, True, id="positional-negative-eigenvalues"),
    ],
)
def test_flash_kda_chunk(B, T, H, D, allow_neg_eigval, has_A, has_bias, positional, monkeypatch):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    q = torch.rand(B, T, H, D, dtype=dtype, device=device)
    k = torch.rand(B, T, H, D, dtype=dtype, device=device)
    v = torch.rand(B, T, H, D, dtype=dtype, device=device)
    g = torch.randn(B, T, H, D, dtype=dtype, device=device)
    beta = torch.randn(B, T, H, dtype=dtype, device=device)
    A_log, dt_bias = _flash_kda_make_gate_params(H, D)
    A_log = A_log if has_A else None
    dt_bias = dt_bias if has_bias else None
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device)
    scale = D ** -0.5

    ref_o, ref_ht = _flash_kda_gold(q, k, v, g, beta, A_log, dt_bias, scale, h0.clone(), allow_neg_eigval=allow_neg_eigval)

    tri_o, tri_ht = _flash_kda_run(
        monkeypatch,
        positional=positional,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        allow_neg_eigval=allow_neg_eigval,
    )
    assert_close("o", ref_o, tri_o, _FLASH_KDA_RTOL)
    assert_close("ht", ref_ht, tri_ht.to(ref_ht.dtype), _FLASH_KDA_RTOL)


@_SKIP_FLASH_KDA
@pytest.mark.parametrize(
    ("H", "D", "cu_seqlens", "allow_neg_eigval", "has_A", "has_bias"),
    [
        pytest.param(4, 128, [0, 256, 500, 1000], False, True, True, id="varlen"),
        pytest.param(8, 128, [0, 100, 300, 1200, 2000], False, True, True, id="multi-sequence"),
        pytest.param(16, 128, [0, 101, 303, 1205, 3007, 4096], False, True, True, id="unaligned"),
        pytest.param(4, 128, [0, 256, 500, 1000], True, True, True, id="negative-eigenvalues"),
        pytest.param(4, 128, [0, 256, 500, 1000], False, False, True, id="no-A"),
        pytest.param(4, 128, [0, 256, 500, 1000], False, True, False, id="no-bias"),
    ],
)
def test_flash_kda_chunk_varlen(H, D, cu_seqlens, allow_neg_eigval, has_A, has_bias, monkeypatch):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    cu_seqlens_t = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.randn(1, T, H, D, dtype=dtype, device=device)
    k = torch.randn(1, T, H, D, dtype=dtype, device=device)
    v = torch.randn(1, T, H, D, dtype=dtype, device=device)
    g = torch.randn(1, T, H, D, dtype=dtype, device=device)
    beta = torch.randn(1, T, H, dtype=dtype, device=device)
    A_log, dt_bias = _flash_kda_make_gate_params(H, D)
    A_log = A_log if has_A else None
    dt_bias = dt_bias if has_bias else None
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=device)
    scale = D ** -0.5

    ref_o, ref_ht = _flash_kda_gold(
        q=q,
        k=k,
        v=v,
        g=g,
        beta_raw=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=h0.clone(),
        cu_seqlens=cu_seqlens_t,
        allow_neg_eigval=allow_neg_eigval,
    )
    tri_o, tri_ht = _flash_kda_run(
        monkeypatch,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
        allow_neg_eigval=allow_neg_eigval,
    )
    assert_close("o", ref_o, tri_o, _FLASH_KDA_RTOL)
    assert_close("ht", ref_ht, tri_ht.to(ref_ht.dtype), _FLASH_KDA_RTOL)


_TRITON_ASCEND_KDA_OPS = (
    'kda_gate_fwd',
    'kda_gate_bwd',
    'kda_gate_chunk_cumsum',
    'fused_kda_gate',
    'fused_recurrent_kda_fwd',
    'recompute_w_u_fwd',
    'chunk_kda_fwd_intra',
    'chunk_kda_fwd_intra_token_parallel',
    'chunk_kda_bwd_intra',
    'chunk_kda_bwd_dAv',
    'chunk_kda_bwd_wy_dqkg_fused',
)


def _spy_on_triton_ascend_kda_backend():
    """Patch every op of the Triton-Ascend KDA backend to record dispatched calls."""
    from fla import backends

    backend = backends._load_operation_registry('kda')._backends.get('triton_ascend')
    assert backend is not None, 'Triton-Ascend KDA backend is not registered'

    calls = []
    for name in _TRITON_ASCEND_KDA_OPS:
        original = getattr(backend, name)

        def make_spy(name, original):
            def spy(*args, **kwargs):
                calls.append(name)
                return original(*args, **kwargs)
            return spy

        setattr(backend, name, make_spy(name, original))
    return backend, calls


def _run_chunk_kda(safe_gate: bool, chunk_size: int = 64, use_gate_in_kernel: bool = True):
    B, T, H, HV, D = 2, 128, 2, 2, 64
    dtype = torch.float
    q = torch.rand(B, T, H, D, dtype=dtype, device=device)
    k = torch.rand(B, T, H, D, dtype=dtype, device=device)
    v = torch.rand(B, T, HV, D, dtype=dtype, device=device)
    g = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    beta = torch.randn(B, T, HV, dtype=dtype, device=device).sigmoid()
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32, device=device)
    tensors = [q, k, v, g, beta, h0]
    if use_gate_in_kernel:
        A_log = torch.randn(HV, dtype=torch.float, device=device)
        dt_bias = torch.randn(HV * D, dtype=torch.float, device=device)
        tensors += [A_log, dt_bias]
    else:
        A_log = dt_bias = None
    for x in tensors:
        x.requires_grad_(True)

    o, ht = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=1.,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=-5.0 if safe_gate else None,
        disable_recompute=False,
        chunk_size=chunk_size,
    )
    ((o * torch.randn_like(o)).sum() + (ht * torch.randn_like(ht)).sum()).backward()


def _run_fused_recurrent_kda_fwd():
    B, T, H, D = 2, 64, 2, 64
    dtype = torch.float
    q = torch.rand(B, T, H, D, dtype=dtype, device=device)
    k = torch.rand(B, T, H, D, dtype=dtype, device=device)
    v = torch.rand(B, T, H, D, dtype=dtype, device=device)
    g = torch.randn(B, T, H, D, dtype=dtype, device=device)
    beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device)
    fused_recurrent_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        scale=1.0,
        output_final_state=False,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )


@pytest.mark.skipif(not IS_NPU, reason='Triton-Ascend KDA backend routing is only exercised on NPU')
def test_triton_ascend_backend_routing():
    """Verify KDA operations dispatch to Triton-Ascend on NPU.

    Numerical parity can pass through the fallback, so record which implementations actually run.
    """
    backend, calls = _spy_on_triton_ascend_kda_backend()
    try:
        # safe_gate=True: forward and backward must hit the NPU kernels
        calls.clear()
        _run_chunk_kda(safe_gate=True)
        expected = {
            'kda_gate_chunk_cumsum',
            'chunk_kda_fwd_intra',
            'recompute_w_u_fwd',
            'chunk_kda_bwd_dAv',
            'chunk_kda_bwd_wy_dqkg_fused',
            'chunk_kda_bwd_intra',
            'kda_gate_bwd',
        }
        missing = expected - set(calls)
        assert not missing, f'ops not routed to the Triton-Ascend backend: {missing} (dispatched: {calls})'

        # safe_gate=False: the intra kernel delegates to the token-parallel variant
        calls.clear()
        _run_chunk_kda(safe_gate=False)
        assert 'chunk_kda_fwd_intra_token_parallel' in calls, (
            'chunk_kda_fwd_intra_token_parallel not routed to the Triton-Ascend backend '
            f'(dispatched: {calls})'
        )

        # chunk_size=16 must fall back to the default implementation and raise before any NPU kernel runs.
        calls.clear()
        with pytest.raises(ValueError, match=r"`chunk_size` must be either 32 or 64"):
            _run_chunk_kda(safe_gate=False, chunk_size=16, use_gate_in_kernel=False)
        rejected = {
            'chunk_kda_fwd_intra',
            'chunk_kda_fwd_intra_token_parallel',
            'chunk_kda_bwd_intra',
            'chunk_kda_bwd_wy_dqkg_fused',
        }
        assert rejected.isdisjoint(calls), (
            f'verifier-rejected ops were still routed to the Triton-Ascend backend: {rejected & set(calls)}'
        )

        calls.clear()
        _run_fused_recurrent_kda_fwd()
        assert 'fused_recurrent_kda_fwd' in calls, (
            'fused_recurrent_kda_fwd not routed to the Triton-Ascend backend '
            f'(dispatched: {calls})'
        )
    finally:
        for name in _TRITON_ASCEND_KDA_OPS:
            delattr(backend, name)
