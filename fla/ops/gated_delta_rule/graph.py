# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Restricted NPU graph validation and forward stages used by GDN autograd."""

import math

import torch


def _validate_graph_inputs(q, k, v, g, beta, scale, initial_state, cu_seqlens, options):
    # only inspect host metadata: offset values belong to the caller's replay contract.
    if q.device.type != 'npu':
        raise ValueError("use_graph=True is supported only on NPU")
    allowed = {
        'use_qk_l2norm_in_kernel': False, 'use_beta_sigmoid_in_kernel': False,
        'allow_neg_eigval': False, 'state_v_first': False, 'cu_seqlens_cpu': None,
        'chunk_indices': None, 'cp_context': None, 'use_gate_in_kernel': False,
        'A_log': None, 'dt_bias': None, 'chunk_size': 64,
    }
    for name, value in options.items():
        expected = allowed.get(name)
        if name not in allowed or type(value) is not type(expected) or value != expected:
            raise ValueError(f"use_graph=True does not support {name}")
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("graph q must have shape [1, T_max>0, H>0, D]")
    if q.shape[-1] not in (64, 128) or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("graph requires equal q/k/v shapes and K=V in {64, 128}")
    if g.shape != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("graph g and beta must have shape [1, T_max, H]")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("graph q/k/v must have the same fp16 or bf16 dtype")
    if g.dtype != torch.float32 or beta.dtype != q.dtype:
        raise ValueError("graph requires fp32 g and beta with q dtype")
    if cu_seqlens is None or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("graph requires cu_seqlens with shape [N_max+1], N_max>0")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("graph cu_seqlens must have int32 or int64 dtype")
    if initial_state is not None:
        shape = (cu_seqlens.numel() - 1, q.shape[2], q.shape[3], q.shape[3])
        if initial_state.shape != shape or initial_state.dtype != torch.float32:
            raise ValueError("graph initial_state must be fp32 [N_max, H, K, V]")
    for tensor in (q, k, v, g, beta, cu_seqlens, initial_state):
        if tensor is not None:
            if tensor.device != q.device or not tensor.is_contiguous():
                raise ValueError("graph inputs must be contiguous and on the same NPU")
    if scale is not None and (not isinstance(scale, int | float) or not math.isfinite(scale)):
        raise ValueError("graph scale must be a finite Python number")


def chunk_gated_delta_rule_graph_fwd(
    q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, chunk_indices, chunk_offsets,
):
    # import only after validation, so unsupported platforms never load Ascend kernels.
    from fla.ops.common.backends.triton_ascend.chunk_delta_h import chunk_gated_delta_rule_fwd_h_npu
    from fla.ops.common.backends.triton_ascend.chunk_o import chunk_fwd_o_npu
    from fla.ops.common.backends.triton_ascend.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd_npu
    from fla.ops.gated_delta_rule.backends.triton_ascend.wy_fast import recompute_w_u_fwd_npu
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.triton_ascend.cumsum import chunk_local_cumsum_npu
    from fla.ops.utils.triton_ascend.solve_tril import _solve_tril_npu

    common = dict(cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, use_graph=True)
    g = chunk_local_cumsum_npu(g, chunk_size=64, scale=RCP_LN2, **common)
    A = chunk_scaled_dot_kkt_fwd_npu(k=k, g=g, beta=beta, chunk_size=64, **common)
    A = _solve_tril_npu(A, output_dtype=k.dtype, **common)
    w, u = recompute_w_u_fwd_npu(k=k, v=v, beta=beta, A=A, g=g, **common)
    h, v_new, ht = chunk_gated_delta_rule_fwd_h_npu(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_offsets=chunk_offsets,
        **common,
    )
    o = chunk_fwd_o_npu(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=q.shape[-1] ** -0.5 if scale is None else scale,
        chunk_offsets=chunk_offsets,
        **common,
    )
    return g, o, A, ht
