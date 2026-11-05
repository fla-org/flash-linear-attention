# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Ascend GDN graph forward orchestration."""

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
    from fla.ops.utils.triton_ascend.solve_tril import solve_tril_npu

    common = dict(cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, use_graph=True)
    g = chunk_local_cumsum_npu(g, chunk_size=64, scale=RCP_LN2, **common)
    A = chunk_scaled_dot_kkt_fwd_npu(k=k, g=g, beta=beta, chunk_size=64, **common)
    A = solve_tril_npu(A, output_dtype=k.dtype, **common)
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
