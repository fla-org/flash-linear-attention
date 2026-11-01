# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""KDA fused recurrent forward kernel for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.backends import TritonAscendBackend, register
from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd
from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus
from fla.utils import ascend_compile_kwargs, input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    max_grid_axis_chunks,
)


@triton.heuristics({
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
    "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
    "HAS_A": lambda args: args["A_log"] is not None,
    "HAS_BIAS": lambda args: args["dt_bias"] is not None,
    "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
})
@triton.jit(do_not_specialize=["T", "task_offset"])
def fused_recurrent_kda_fwd_kernel_npu(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    h0,
    ht,
    o,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    lower_bound,
    T: tl.int64,
    task_offset,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    scale: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    ALLOW_NEG_EIGVAL: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    tl.static_assert(BK >= K)
    i_task = tl.cast(tl.program_id(0), tl.int64) + task_offset
    NV = tl.cdiv(V, BV)
    i_v = i_task % NV
    i_nh = i_task // NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = i_n * T

    if T > 0 or not IS_CONTINUOUS_BATCHING:
        o_k = tl.arange(0, BK).to(tl.int64)
        o_v = i_v * BV + tl.arange(0, BV)
        # block-pointer metadata requires int32 tile offsets on Ascend
        o_v_start = tl.cast(i_v * BV, tl.int32)

        o_qk = (bos * H + i_h) * K
        o_hv = bos * HV + i_hv

        p_q = q + o_qk + o_k
        p_k = k + o_qk + o_k
        p_v = v + o_hv * V + o_v
        p_g = g + o_hv * K + o_k
        p_o = o + o_hv * V + o_v
        if IS_BETA_HEADWISE:
            p_beta = beta + o_hv * V + o_v
        else:
            p_beta = beta + o_hv

        m_k = o_k < K
        m_v = o_v < V
        if STATE_V_FIRST:
            m_h = m_v[:, None] & m_k[None, :]
            b_h = tl.zeros([BV, BK], dtype=tl.float32)
        else:
            m_h = m_k[:, None] & m_v[None, :]
            b_h = tl.zeros([BK, BV], dtype=tl.float32)

        if USE_GATE_IN_KERNEL:
            b_A = tl.load(A_log + i_hv).to(tl.float32) if HAS_A else 1.0
            if HAS_BIAS:
                b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=m_k, other=0).to(tl.float32)
            else:
                b_bias = tl.zeros([BK], dtype=tl.float32)

        if USE_INITIAL_STATE:
            if IS_CONTINUOUS_BATCHING:
                if IS_SPEC_DECODING:
                    i_t0 = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
                else:
                    i_t0 = 0
                i_h0 = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t0).to(tl.int64)
                p_h0 = h0 + i_h0 * stride_init_state_token + i_hv * K * V
            else:
                p_h0 = h0 + i_nh * K * V
            if STATE_V_FIRST:
                p_h0 = p_h0 + o_v[:, None] * K + o_k[None, :]
            else:
                p_h0 = p_h0 + o_k[:, None] * V + o_v[None, :]
            b_h += tl.load(p_h0, mask=m_h, other=0).to(tl.float32)

        for i_t in tl.range(0, T):
            b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)

            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6) * scale)
                b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
            else:
                b_q = b_q * scale
            b_g = tl.load(p_g, mask=m_k, other=0).to(tl.float32)

            if USE_GATE_IN_KERNEL:
                b_g = b_g + b_bias
                if USE_LOWER_BOUND:
                    b_gk = lower_bound * tl.sigmoid((exp(b_A) if HAS_A else b_A) * b_g)
                else:
                    b_gk = -(exp(b_A) if HAS_A else b_A) * softplus(b_g)
            else:
                b_gk = b_g

            if STATE_V_FIRST:
                b_h *= exp(b_gk[None, :])
                b_v -= tl.sum(b_h * b_k[None, :], 1)
            else:
                b_h *= exp(b_gk[:, None])
                b_v -= tl.sum(b_h * b_k[:, None], 0)

            if IS_BETA_HEADWISE:
                b_beta = tl.load(p_beta, mask=m_v, other=0).to(tl.float32)
            else:
                b_beta = tl.load(p_beta).to(tl.float32)
            if APPLY_BETA_SIGMOID:
                b_beta = tl.sigmoid(b_beta)
                if ALLOW_NEG_EIGVAL:
                    b_beta = b_beta * 2
            b_v *= b_beta

            if STATE_V_FIRST:
                b_h += b_v[:, None] * b_k[None, :]
                b_o = tl.sum(b_h * b_q[None, :], 1)
            else:
                b_h += b_k[:, None] * b_v[None, :]
                b_o = tl.sum(b_h * b_q[:, None], 0)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)

            if IS_CONTINUOUS_BATCHING and STORE_FINAL_STATE:
                if INPLACE_FINAL_STATE:
                    i_ht = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64)
                    p_ht = ht + i_ht * stride_final_state_token + i_hv * K * V
                else:
                    p_ht = ht + (bos + i_t) * stride_final_state_token + i_hv * K * V
                if STATE_V_FIRST:
                    p_ht = tl.make_block_ptr(p_ht, (V, K), (K, 1), (o_v_start, 0), (BV, BK), (1, 0))
                else:
                    p_ht = tl.make_block_ptr(p_ht, (K, V), (V, 1), (0, o_v_start), (BK, BV), (1, 0))
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

            p_q += H * K
            p_k += H * K
            p_v += HV * V
            p_g += HV * K
            p_o += HV * V
            p_beta += HV * V if IS_BETA_HEADWISE else HV

        if STORE_FINAL_STATE and not IS_CONTINUOUS_BATCHING:
            p_ht = ht + i_nh * K * V
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(p_ht, (V, K), (K, 1), (o_v_start, 0), (BV, BK), (1, 0))
            else:
                p_ht = tl.make_block_ptr(p_ht, (K, V), (V, 1), (0, o_v_start), (BK, BV), (1, 0))
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@register(fused_recurrent_kda_fwd, backend=TritonAscendBackend)
@input_guard(no_guard_contiguous={'initial_state', 'out'})
def fused_recurrent_kda_fwd_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
    inplace_final_state: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    lower_bound: float | None = None,
    out: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    if scale is None:
        scale = K ** -0.5

    if initial_state is not None and not initial_state.is_contiguous():
        raise ValueError("`initial_state` must be contiguous")
    if out is None:
        o = torch.zeros_like(v)
    else:
        assert out.shape == v.shape
        if not out.is_contiguous():
            raise ValueError("`out` must be contiguous")
        o = out
    if inplace_final_state:
        assert initial_state is not None
        ht = initial_state
    elif output_final_state:
        num_states = B * T if ssm_state_indices is not None else N
        state_shape = (V, K) if state_v_first else (K, V)
        ht = q.new_empty(num_states, HV, *state_shape, dtype=torch.float32)
    else:
        ht = None

    stride_init_state_token = initial_state.stride(0) if initial_state is not None else 1
    stride_final_state_token = ht.stride(0) if ht is not None else 1
    stride_indices_seq = 1 if ssm_state_indices is None else ssm_state_indices.stride(0)

    BK = triton.next_power_of_2(K)
    # budget for the fp32 state [BK, BV] and live query, key, gate, value, output and beta vectors
    BV = compute_row_tile_block_size(
        row_dim=BK,
        fixed_dim=V,
        memory_multiplier=3.0,
        tiling_row=False,
        safety_margin=0.80,
        dtype_size=4,
        fallback=32,
        min_block=16,
        max_block=min(256, triton.next_power_of_2(V)),
    )
    num_tasks = triton.cdiv(V, BV) * N * HV
    max_tasks = max_grid_axis_chunks(num_tasks, 1, max_grid=ASCEND_MAX_GRID_DIM)
    for task_offset in range(0, num_tasks, max_tasks):
        task_count = min(max_tasks, num_tasks - task_offset)
        fused_recurrent_kda_fwd_kernel_npu[(task_count,)](
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            h0=initial_state,
            ht=ht,
            o=o,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            lower_bound=lower_bound,
            T=T,
            task_offset=task_offset,
            H=H,
            HV=HV,
            K=K,
            V=V,
            stride_init_state_token=stride_init_state_token,
            stride_final_state_token=stride_final_state_token,
            stride_indices_seq=stride_indices_seq,
            scale=scale,
            BK=BK,
            BV=BV,
            INPLACE_FINAL_STATE=inplace_final_state,
            IS_BETA_HEADWISE=beta.ndim == v.ndim,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            USE_GATE_IN_KERNEL=use_gate_in_kernel,
            APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
            ALLOW_NEG_EIGVAL=allow_neg_eigval,
            STATE_V_FIRST=state_v_first,
            **ascend_compile_kwargs(),
        )

    return o, ht
