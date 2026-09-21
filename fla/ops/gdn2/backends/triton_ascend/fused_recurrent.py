# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GDN2 fused recurrent forward for Ascend NPUs."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus
from fla.utils import ascend_compile_kwargs, get_multiprocessor_count, input_guard
from fla.utils.ascend_ub_manager import compute_row_tile_block_size

# includes compiler temporaries for the recurrent state tile, calibrated at BK=256.
_RECUR_MEM_MULT = 4.1
_SAFETY_MARGIN = 0.80
_FALLBACK_BV = 32
_MAX_BV = 256
# smaller states favor K-first decay broadcasts over V-first reductions.
_VFIRST_MIN_DIM = 256


def _get_bv(K: int, V: int) -> int:
    bk = triton.next_power_of_2(K)
    return compute_row_tile_block_size(
        bk,
        V,
        _RECUR_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=_FALLBACK_BV,
        min_block=1,
        max_block=min(_MAX_BV, triton.next_power_of_2(V)),
    )


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
        "HAS_A": lambda args: args["A_log"] is not None,
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "task_num", "num_core"])
def fused_recurrent_gdn2_fwd_kernel_npu(
    q,
    k,
    v,
    g,
    b,
    w,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    lower_bound,
    scale: tl.constexpr,
    T: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    task_num,
    num_core,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    core_id = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    for task_idx in tl.range(core_id, task_num, num_core):
        task_id = tl.cast(task_idx, tl.int64)
        i_v = task_id % NV
        i_nh = task_id // NV
        i_n, i_hv = i_nh // HV, i_nh % HV
        i_n = tl.cast(i_n, tl.int64)
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            bos, eos = (
                tl.load(cu_seqlens + i_n).to(tl.int64),
                tl.load(cu_seqlens + i_n + 1).to(tl.int64),
            )
            T_cur = (eos - bos).to(tl.int32)
        else:
            bos = i_n * T
            T_cur = T

        if T_cur > 0:
            o_k = tl.arange(0, BK).to(tl.int64)
            o_v = i_v * BV + tl.arange(0, BV)

            base_qk = (bos * H + i_h) * K
            base_hv = bos * HV + i_hv

            p_q = q + base_qk + o_k
            p_k = k + base_qk + o_k
            p_v = v + base_hv * V + o_v
            p_b = b + base_hv * K + o_k
            p_w = w + base_hv * V + o_v
            p_g = g + base_hv * K + o_k
            p_o = o + base_hv * V + o_v

            mask_k = o_k < K
            mask_v = o_v < V
            if STATE_V_FIRST:
                mask_h = mask_v[:, None] & mask_k[None, :]
                b_h = tl.zeros([BV, BK], dtype=tl.float32)
            else:
                mask_h = mask_k[:, None] & mask_v[None, :]
                b_h = tl.zeros([BK, BV], dtype=tl.float32)

            if USE_GATE_IN_KERNEL:
                if HAS_A:
                    b_A = tl.load(A_log + i_hv).to(tl.float32)
                else:
                    b_A = 1.0
                if HAS_BIAS:
                    b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0).to(tl.float32)
                else:
                    b_bias = tl.zeros([BK], dtype=tl.float32)

            if USE_INITIAL_STATE:
                if IS_CONTINUOUS_BATCHING:
                    if IS_SPEC_DECODING:
                        i_t0 = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
                    else:
                        i_t0 = 0
                    state_base = (
                        tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t0).to(tl.int64) * stride_init_state_token
                    )
                    p_h0 = h0 + state_base + i_hv * K * V
                else:
                    p_h0 = h0 + (i_n * HV + i_hv) * K * V
                if STATE_V_FIRST:
                    p_h0 = p_h0 + o_v[:, None] * K + o_k[None, :]
                else:
                    p_h0 = p_h0 + o_k[:, None] * V + o_v[None, :]
                b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

            stride_qk = H * K
            stride_hv_v = HV * V
            stride_hv_k = HV * K

            for i_t in tl.range(0, T_cur):
                i_t64 = tl.cast(i_t, tl.int64)
                b_q = tl.load(p_q + i_t64 * stride_qk, mask=mask_k, other=0).to(tl.float32)
                b_k = tl.load(p_k + i_t64 * stride_qk, mask=mask_k, other=0).to(tl.float32)
                b_g = tl.load(p_g + i_t64 * stride_hv_k, mask=mask_k, other=0).to(tl.float32)
                b_b = tl.load(p_b + i_t64 * stride_hv_k, mask=mask_k, other=0).to(tl.float32)
                b_v = tl.load(p_v + i_t64 * stride_hv_v, mask=mask_v, other=0).to(tl.float32)
                b_w = tl.load(p_w + i_t64 * stride_hv_v, mask=mask_v, other=0).to(tl.float32)

                if USE_QK_L2NORM_IN_KERNEL:
                    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
                    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
                b_q = b_q * scale

                if USE_GATE_IN_KERNEL:
                    b_g = b_g + b_bias
                    if USE_LOWER_BOUND:
                        if HAS_A:
                            b_gk = lower_bound * tl.sigmoid(exp(b_A) * b_g)
                        else:
                            b_gk = lower_bound * tl.sigmoid(b_g)
                    else:
                        b_gk = -exp(b_A) * softplus(b_g)
                else:
                    b_gk = b_g

                b_h *= exp(b_gk[:, None] if not STATE_V_FIRST else b_gk[None, :])
                b_bk = b_b * b_k
                if STATE_V_FIRST:
                    erase_d = tl.sum(b_h * b_bk[None, :], 1)
                else:
                    erase_d = tl.sum(b_h * b_bk[:, None], 0)
                b_v_new = b_w * b_v - erase_d
                if STATE_V_FIRST:
                    b_h += b_v_new[:, None] * b_k[None, :]
                    b_o = tl.sum(b_h * b_q[None, :], 1)
                else:
                    b_h += b_k[:, None] * b_v_new[None, :]
                    b_o = tl.sum(b_h * b_q[:, None], 0)
                tl.store(p_o + i_t64 * stride_hv_v, b_o.to(p_o.dtype.element_ty), mask=mask_v)

                if IS_CONTINUOUS_BATCHING:
                    if INPLACE_FINAL_STATE:
                        state_base = (
                            tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t64).to(tl.int64)
                            * stride_final_state_token
                        )
                        p_ht = ht + state_base + i_hv * K * V
                    else:
                        p_ht = ht + (bos + i_t64) * stride_final_state_token + i_hv * K * V
                    if STATE_V_FIRST:
                        p_ht = p_ht + o_v[:, None] * K + o_k[None, :]
                    else:
                        p_ht = p_ht + o_k[:, None] * V + o_v[None, :]
                    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

            if not IS_CONTINUOUS_BATCHING:
                if STORE_FINAL_STATE:
                    p_ht = ht + (i_n * HV + i_hv) * K * V
                    if STATE_V_FIRST:
                        p_ht = p_ht + o_v[:, None] * K + o_k[None, :]
                    else:
                        p_ht = p_ht + o_k[:, None] * V + o_v[None, :]
                    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@input_guard(no_guard_contiguous={'initial_state', 'out'})
def fused_recurrent_gdn2_fwd_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
    inplace_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    out: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = k.shape[-1] ** -0.5

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)

    if initial_state is not None and not initial_state.is_contiguous():
        raise ValueError("`initial_state` must be contiguous")
    user_out = out
    if user_out is not None:
        assert user_out.shape == v.shape
        if not user_out.is_contiguous():
            raise ValueError("`out` must be contiguous")

    BV = _get_bv(K, V)
    h0 = initial_state

    # V-first speeds up large-state reductions; only transpose buffers allocated here.
    internal_v_first = (
        not state_v_first
        and h0 is None
        and not inplace_final_state
        and cu_seqlens is None
        and ssm_state_indices is None
        and K >= _VFIRST_MIN_DIM
        and V >= _VFIRST_MIN_DIM
    )
    kernel_v_first = state_v_first or internal_v_first

    out = torch.zeros_like(v) if user_out is None else user_out

    if inplace_final_state:
        assert initial_state is not None, "inplace_final_state=True requires an initial_state"
        final_state = h0
    elif output_final_state:
        state_shape = (N, HV, V, K) if kernel_v_first else (N, HV, K, V)
        final_state = q.new_empty(*state_shape, dtype=torch.float32)
    else:
        final_state = None

    stride_init_state_token = h0.stride(0) if h0 is not None else 1
    stride_final_state_token = final_state.stride(0) if final_state is not None else 1

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    task_num = triton.cdiv(V, BV) * N * HV
    num_core = get_multiprocessor_count(q.device.index)

    kernel_kwargs = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=h0,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        lower_bound=lower_bound,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        task_num=task_num,
        num_core=num_core,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_GATE_IN_KERNEL=use_gate_in_kernel,
        STATE_V_FIRST=kernel_v_first,
    )
    # Each program owns a core and processes independent state tiles in a loop.
    # Persistent state liveness plus automatic multibuffering exceeds UB.
    fused_recurrent_gdn2_fwd_kernel_npu[(num_core,)](**kernel_kwargs, **ascend_compile_kwargs())

    if internal_v_first and final_state is not None:
        final_state = final_state.transpose(-1, -2).contiguous()
    return out, final_state
