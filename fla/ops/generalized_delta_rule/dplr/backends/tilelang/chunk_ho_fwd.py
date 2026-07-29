# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Default fused DPLR H/O forward.

Computes the chunk start state, per-token ``v_new``, and final output in one
forward TileLang stage.  The public DPLR recompute forward returns only
``(o, final_state)``, so this kernel avoids the global forward write/read of
``h`` and ``v_new``.  Backward recompute still uses the split H/O kernels
because its gradients consume those intermediates.
"""

import tilelang
import tilelang.language as T
import torch

from .utils import ChunkLayout, build_rect_chunk_layout, build_varlen_chunk_layout

_HO_FWD_CONFIGS = [
    {"BV": 64, "threads": 256},
]


def _dtype_bytes(dtype: str) -> int:
    return 4 if dtype in {"float32", "float"} else 2


def _ho_smem_bytes(BT: int, K: int, BV: int, in_dtype: str) -> int:
    elem = _dtype_bytes(in_dtype)
    # Keep this in sync with the shared buffers in _chunk_dplr_fwd_ho_kernel.
    return (
        elem * (K * BV + BT * K + BT * K + BT * BV + BT * BV + BT * K + BT * BT + BT * BT)
        + 4 * (K * BV + K * BV + BT * BV + BT * K + K)
    )


def _ho_fwd_configs(
    H, K, V, BT,
    in_dtype, state_dtype, USE_INITIAL_STATE, STORE_FINAL_STATE,
    BV: int = 64, threads: int = 128,
):
    if not torch.cuda.is_available():
        return [{"BV": 16, "threads": 64}]
    major = torch.cuda.get_device_capability()[0]
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    smem_limit = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)

    def pick(candidates: list[int]) -> int:
        for candidate in candidates:
            if candidate <= V and _ho_smem_bytes(BT, K, candidate, in_dtype) <= smem_limit:
                return candidate
        return 16

    if BT <= 16:
        # 16-row GEMMs only partition across <= 2 warps (m_warp x n_warp must
        # equal num_warps with >= 16 rows and >= 8 cols per warp).
        config = {"BV": pick([64, 32, 16]), "threads": 64}
    elif major >= 9 and K <= 128:
        config = {"BV": pick([64, 32, 16]), "threads": 256}
    elif major == 8:
        config = {"BV": pick([32, 16]), "threads": 128}
    else:
        config = {"BV": 16, "threads": 64}
    return [config]


def _ho_fragment_merge_flags(
    K: int,
    V: int,
    BT: int,
    in_dtype: str,
    store_context: bool,
    config: dict[str, int],
) -> dict[str, bool]:
    capability = torch.cuda.get_device_capability()
    common_shape = K == 128 and V == 128 and BT == 32 and in_dtype == "bfloat16"
    if (
        common_shape
        and capability[0] == 12
        and config == {"BV": 32, "threads": 256}
    ):
        return {"DIRECT_KG_FRAGMENT": True, "DIRECT_BG_FRAGMENT": True}
    if (
        common_shape
        and capability == (9, 0)
        and not store_context
        and config == {"BV": 64, "threads": 256}
    ):
        return {"DIRECT_KG_FRAGMENT": True, "DIRECT_BG_FRAGMENT": False}
    return {"DIRECT_KG_FRAGMENT": False, "DIRECT_BG_FRAGMENT": False}


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    },
)
def _chunk_dplr_fwd_ho_kernel(
    H, K, V, BT,
    in_dtype, state_dtype,
    USE_INITIAL_STATE: bool,
    STORE_FINAL_STATE: bool,
    BV: int = 64,
    threads: int = 128,
    num_stages: int = 0,
    DIRECT_KG_FRAGMENT: bool = False,
    DIRECT_BG_FRAGMENT: bool = False,
):
    acc_dtype = "float32"
    n_tokens, n_seq_plus_one, n_h0, n_ht = T.dynamic(
        "n_tokens, n_seq_plus_one, n_h0, n_ht"
    )
    n_seqs = n_seq_plus_one - 1

    @T.prim_func
    def chunk_dplr_fwd_ho_tl(
        qg: T.Tensor((n_tokens, H, K), in_dtype),
        kg: T.Tensor((n_tokens, H, K), in_dtype),
        v: T.Tensor((n_tokens, H, V), in_dtype),
        w: T.Tensor((n_tokens, H, K), in_dtype),
        u: T.Tensor((n_tokens, H, V), in_dtype),
        bg: T.Tensor((n_tokens, H, K), in_dtype),
        gk: T.Tensor((n_tokens, H, K), acc_dtype),
        A_qk: T.Tensor((n_tokens, H, BT), in_dtype),
        A_qb: T.Tensor((n_tokens, H, BT), in_dtype),
        h0: T.Tensor((n_h0, H, K, V), acc_dtype),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_offsets: T.Tensor((n_seq_plus_one,), "int32"),
        o: T.Tensor((n_tokens, H, V), in_dtype),
        ht: T.Tensor((n_ht, H, K, V), state_dtype),
    ):
        with T.Kernel(T.ceildiv(V, BV), n_seqs, H, threads=threads) as (i_v, i_n, i_h):
            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            n_chunks = chunk_offsets[i_n + 1] - chunk_offsets[i_n]

            b_h = T.alloc_fragment((K, BV), acc_dtype)
            b_h_shared = T.alloc_shared((K, BV), in_dtype)
            b_hc = T.alloc_fragment((K, BV), acc_dtype)
            b_hc_kg = T.alloc_fragment((K, BV), acc_dtype)
            b_hc_bg = T.alloc_fragment((K, BV), acc_dtype)
            if not DIRECT_KG_FRAGMENT:
                b_hc_kg_shared = T.alloc_shared((K, BV), acc_dtype)
            if not DIRECT_BG_FRAGMENT:
                b_hc_bg_shared = T.alloc_shared((K, BV), acc_dtype)
            v2_frag = T.alloc_fragment((BT, BV), acc_dtype)
            v2_acc_shared = T.alloc_shared((BT, BV), acc_dtype)
            v2_shared = T.alloc_shared((BT, BV), in_dtype)
            kg_shared = T.alloc_shared((BT, K), in_dtype)
            bg_shared = T.alloc_shared((BT, K), acc_dtype)
            w_shared = T.alloc_shared((BT, K), in_dtype)
            v_shared = T.alloc_shared((BT, BV), in_dtype)
            u_shared = T.alloc_shared((BT, BV), in_dtype)
            qg_shared = T.alloc_shared((BT, K), in_dtype)
            A_qk_shared = T.alloc_shared((BT, BT), in_dtype)
            A_qb_shared = T.alloc_shared((BT, BT), in_dtype)
            o_frag = T.alloc_fragment((BT, BV), acc_dtype)
            gk_last_shared = T.alloc_shared((K,), acc_dtype)

            if USE_INITIAL_STATE:
                for k_idx, vv in T.Parallel(K, BV):
                    g_v = i_v * BV + vv
                    if g_v < V:
                        b_h[k_idx, vv] = h0[i_n, i_h, k_idx, g_v]
                    else:
                        b_h[k_idx, vv] = 0.0
            else:
                for k_idx, vv in T.Parallel(K, BV):
                    b_h[k_idx, vv] = 0.0

            for i_t in T.Pipelined(n_chunks, num_stages=num_stages):
                chunk_bos = bos + i_t * BT
                T.copy(b_h, b_h_shared)
                T.clear(b_hc)

                full_tile = chunk_bos + BT <= eos
                if full_tile:
                    # bulk vectorized copies for interior chunks
                    T.copy(kg[chunk_bos: chunk_bos + BT, i_h, 0:K], kg_shared)
                    T.copy(w[chunk_bos: chunk_bos + BT, i_h, 0:K], w_shared)
                    T.copy(qg[chunk_bos: chunk_bos + BT, i_h, 0:K], qg_shared)
                    T.copy(v[chunk_bos: chunk_bos + BT, i_h, i_v * BV: i_v * BV + BV], v_shared)
                    T.copy(u[chunk_bos: chunk_bos + BT, i_h, i_v * BV: i_v * BV + BV], u_shared)
                    T.copy(A_qk[chunk_bos: chunk_bos + BT, i_h, 0:BT], A_qk_shared)
                    T.copy(A_qb[chunk_bos: chunk_bos + BT, i_h, 0:BT], A_qb_shared)
                else:
                    for c, k_idx in T.Parallel(BT, K):
                        t = chunk_bos + c
                        if t < eos:
                            kg_shared[c, k_idx] = kg[t, i_h, k_idx]
                            w_shared[c, k_idx] = w[t, i_h, k_idx]
                            qg_shared[c, k_idx] = qg[t, i_h, k_idx]
                        else:
                            kg_shared[c, k_idx] = T.Cast(in_dtype, 0.0)
                            w_shared[c, k_idx] = T.Cast(in_dtype, 0.0)
                            qg_shared[c, k_idx] = T.Cast(in_dtype, 0.0)

                    for c, vv in T.Parallel(BT, BV):
                        t = chunk_bos + c
                        g_v = i_v * BV + vv
                        if (t < eos) and (g_v < V):
                            v_shared[c, vv] = v[t, i_h, g_v]
                            u_shared[c, vv] = u[t, i_h, g_v]
                        else:
                            v_shared[c, vv] = T.Cast(in_dtype, 0.0)
                            u_shared[c, vv] = T.Cast(in_dtype, 0.0)

                    for r, c in T.Parallel(BT, BT):
                        t = chunk_bos + r
                        if (t < eos) and (r >= c):
                            A_qk_shared[r, c] = A_qk[t, i_h, c]
                            A_qb_shared[r, c] = A_qb[t, i_h, c]
                        else:
                            A_qk_shared[r, c] = T.Cast(in_dtype, 0.0)
                            A_qb_shared[r, c] = T.Cast(in_dtype, 0.0)

                for c, k_idx in T.Parallel(BT, K):
                    t = chunk_bos + c
                    if t < eos:
                        bg_shared[c, k_idx] = T.Cast(acc_dtype, bg[t, i_h, k_idx])
                    else:
                        bg_shared[c, k_idx] = 0.0

                T.gemm(w_shared, b_h_shared, v2_frag, clear_accum=True)
                for c, vv in T.Parallel(BT, BV):
                    t = chunk_bos + c
                    g_v = i_v * BV + vv
                    v2 = T.ieee_add(v2_frag[c, vv], T.Cast(acc_dtype, u_shared[c, vv]))
                    if (t < eos) and (g_v < V):
                        v2_acc_shared[c, vv] = v2
                        v2_shared[c, vv] = T.Cast(in_dtype, v2)
                    else:
                        v2_acc_shared[c, vv] = T.Cast(acc_dtype, 0.0)
                        v2_shared[c, vv] = T.Cast(in_dtype, 0.0)

                T.gemm(
                    kg_shared,
                    v_shared,
                    b_hc_kg,
                    transpose_A=True,
                    clear_accum=True,
                )
                if not DIRECT_KG_FRAGMENT:
                    T.copy(b_hc_kg, b_hc_kg_shared)
                T.gemm(
                    bg_shared,
                    v2_acc_shared,
                    b_hc_bg,
                    transpose_A=True,
                    clear_accum=True,
                )
                if not DIRECT_BG_FRAGMENT:
                    T.copy(b_hc_bg, b_hc_bg_shared)
                if DIRECT_KG_FRAGMENT and DIRECT_BG_FRAGMENT:
                    for k_idx, vv in T.Parallel(K, BV):
                        b_hc[k_idx, vv] = T.ieee_add(
                            b_hc_kg[k_idx, vv],
                            b_hc_bg[k_idx, vv],
                        )
                elif DIRECT_KG_FRAGMENT:
                    for k_idx, vv in T.Parallel(K, BV):
                        b_hc[k_idx, vv] = T.ieee_add(
                            b_hc_kg[k_idx, vv],
                            b_hc_bg_shared[k_idx, vv],
                        )
                elif DIRECT_BG_FRAGMENT:
                    for k_idx, vv in T.Parallel(K, BV):
                        b_hc[k_idx, vv] = T.ieee_add(
                            b_hc_kg_shared[k_idx, vv],
                            b_hc_bg[k_idx, vv],
                        )
                else:
                    for k_idx, vv in T.Parallel(K, BV):
                        b_hc[k_idx, vv] = T.ieee_add(
                            b_hc_kg_shared[k_idx, vv],
                            b_hc_bg_shared[k_idx, vv],
                        )

                T.gemm(qg_shared, b_h_shared, o_frag, clear_accum=True)

                T.gemm(A_qk_shared, v_shared, o_frag)
                T.gemm(A_qb_shared, v2_shared, o_frag)

                for c, vv in T.Parallel(BT, BV):
                    t = chunk_bos + c
                    g_v = i_v * BV + vv
                    if (t < eos) and (g_v < V):
                        o[t, i_h, g_v] = T.Cast(in_dtype, o_frag[c, vv])

                last_idx = T.min(chunk_bos + BT - 1, eos - 1)
                for k_idx in T.Parallel(K):
                    gk_last_shared[k_idx] = gk[last_idx, i_h, k_idx]
                for k_idx, vv in T.Parallel(K, BV):
                    b_h[k_idx, vv] = T.ieee_mul(T.exp2(gk_last_shared[k_idx]), b_h[k_idx, vv])
                for k_idx, vv in T.Parallel(K, BV):
                    b_h[k_idx, vv] = T.ieee_add(b_h[k_idx, vv], b_hc[k_idx, vv])

            if STORE_FINAL_STATE:
                for k_idx, vv in T.Parallel(K, BV):
                    g_v = i_v * BV + vv
                    if g_v < V:
                        ht[i_n, i_h, k_idx, g_v] = T.Cast(state_dtype, b_h[k_idx, vv])

    return chunk_dplr_fwd_ho_tl


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    },
)
def _chunk_dplr_fwd_ho_ctx_kernel(
    H, K, V, BT,
    in_dtype, state_dtype,
    USE_INITIAL_STATE: bool,
    STORE_FINAL_STATE: bool,
    BV: int = 64,
    threads: int = 128,
    STORE_H_CTX: bool = True,
    STORE_V_NEW_CTX: bool = True,
    DIRECT_KG_FRAGMENT: bool = False,
    DIRECT_BG_FRAGMENT: bool = False,
):
    acc_dtype = "float32"
    n_tokens, n_seq_plus_one, n_h0, n_ht, n_h_ctx, n_v_ctx = T.dynamic(
        "n_tokens, n_seq_plus_one, n_h0, n_ht, n_h_ctx, n_v_ctx"
    )
    n_seqs = n_seq_plus_one - 1

    @T.prim_func
    def chunk_dplr_fwd_ho_ctx_tl(
        qg: T.Tensor((n_tokens, H, K), in_dtype),
        kg: T.Tensor((n_tokens, H, K), in_dtype),
        v: T.Tensor((n_tokens, H, V), in_dtype),
        w: T.Tensor((n_tokens, H, K), in_dtype),
        u: T.Tensor((n_tokens, H, V), in_dtype),
        bg: T.Tensor((n_tokens, H, K), in_dtype),
        gk: T.Tensor((n_tokens, H, K), acc_dtype),
        A_qk: T.Tensor((n_tokens, H, BT), in_dtype),
        A_qb: T.Tensor((n_tokens, H, BT), in_dtype),
        h0: T.Tensor((n_h0, H, K, V), acc_dtype),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_offsets: T.Tensor((n_seq_plus_one,), "int32"),
        o: T.Tensor((n_tokens, H, V), in_dtype),
        ht: T.Tensor((n_ht, H, K, V), state_dtype),
        h_ctx: T.Tensor((n_h_ctx, H, K, V), in_dtype),
        v_new_ctx: T.Tensor((n_v_ctx, H, V), in_dtype),
    ):
        with T.Kernel(T.ceildiv(V, BV), n_seqs, H, threads=threads) as (i_v, i_n, i_h):
            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            boh = chunk_offsets[i_n]
            n_chunks = chunk_offsets[i_n + 1] - boh

            b_h = T.alloc_fragment((K, BV), acc_dtype)
            b_h_shared = T.alloc_shared((K, BV), in_dtype)
            b_hc = T.alloc_fragment((K, BV), acc_dtype)
            b_hc_kg = T.alloc_fragment((K, BV), acc_dtype)
            b_hc_bg = T.alloc_fragment((K, BV), acc_dtype)
            if not DIRECT_KG_FRAGMENT:
                b_hc_kg_shared = T.alloc_shared((K, BV), acc_dtype)
            if not DIRECT_BG_FRAGMENT:
                b_hc_bg_shared = T.alloc_shared((K, BV), acc_dtype)
            v2_frag = T.alloc_fragment((BT, BV), acc_dtype)
            v2_acc_shared = T.alloc_shared((BT, BV), acc_dtype)
            v2_shared = T.alloc_shared((BT, BV), in_dtype)
            kg_shared = T.alloc_shared((BT, K), in_dtype)
            bg_shared = T.alloc_shared((BT, K), acc_dtype)
            w_shared = T.alloc_shared((BT, K), in_dtype)
            v_shared = T.alloc_shared((BT, BV), in_dtype)
            u_shared = T.alloc_shared((BT, BV), in_dtype)
            qg_shared = T.alloc_shared((BT, K), in_dtype)
            A_qk_shared = T.alloc_shared((BT, BT), in_dtype)
            A_qb_shared = T.alloc_shared((BT, BT), in_dtype)
            o_frag = T.alloc_fragment((BT, BV), acc_dtype)
            gk_last_shared = T.alloc_shared((K,), acc_dtype)

            if USE_INITIAL_STATE:
                for k_idx, vv in T.Parallel(K, BV):
                    g_v = i_v * BV + vv
                    if g_v < V:
                        b_h[k_idx, vv] = h0[i_n, i_h, k_idx, g_v]
                    else:
                        b_h[k_idx, vv] = 0.0
            else:
                for k_idx, vv in T.Parallel(K, BV):
                    b_h[k_idx, vv] = 0.0

            for i_t in T.serial(n_chunks):
                chunk_bos = bos + i_t * BT
                chunk_row = boh + i_t
                T.copy(b_h, b_h_shared)
                if STORE_H_CTX:
                    for k_idx, vv in T.Parallel(K, BV):
                        g_v = i_v * BV + vv
                        if g_v < V:
                            h_ctx[chunk_row, i_h, k_idx, g_v] = T.Cast(
                                in_dtype, b_h[k_idx, vv]
                            )
                T.clear(b_hc)

                full_tile = chunk_bos + BT <= eos
                if full_tile:
                    # bulk vectorized copies for interior chunks
                    T.copy(kg[chunk_bos: chunk_bos + BT, i_h, 0:K], kg_shared)
                    T.copy(w[chunk_bos: chunk_bos + BT, i_h, 0:K], w_shared)
                    T.copy(qg[chunk_bos: chunk_bos + BT, i_h, 0:K], qg_shared)
                    T.copy(v[chunk_bos: chunk_bos + BT, i_h, i_v * BV: i_v * BV + BV], v_shared)
                    T.copy(u[chunk_bos: chunk_bos + BT, i_h, i_v * BV: i_v * BV + BV], u_shared)
                    T.copy(A_qk[chunk_bos: chunk_bos + BT, i_h, 0:BT], A_qk_shared)
                    T.copy(A_qb[chunk_bos: chunk_bos + BT, i_h, 0:BT], A_qb_shared)
                else:
                    for c, k_idx in T.Parallel(BT, K):
                        t = chunk_bos + c
                        if t < eos:
                            kg_shared[c, k_idx] = kg[t, i_h, k_idx]
                            w_shared[c, k_idx] = w[t, i_h, k_idx]
                            qg_shared[c, k_idx] = qg[t, i_h, k_idx]
                        else:
                            kg_shared[c, k_idx] = T.Cast(in_dtype, 0.0)
                            w_shared[c, k_idx] = T.Cast(in_dtype, 0.0)
                            qg_shared[c, k_idx] = T.Cast(in_dtype, 0.0)

                    for c, vv in T.Parallel(BT, BV):
                        t = chunk_bos + c
                        g_v = i_v * BV + vv
                        if (t < eos) and (g_v < V):
                            v_shared[c, vv] = v[t, i_h, g_v]
                            u_shared[c, vv] = u[t, i_h, g_v]
                        else:
                            v_shared[c, vv] = T.Cast(in_dtype, 0.0)
                            u_shared[c, vv] = T.Cast(in_dtype, 0.0)

                    for r, c in T.Parallel(BT, BT):
                        t = chunk_bos + r
                        if (t < eos) and (r >= c):
                            A_qk_shared[r, c] = A_qk[t, i_h, c]
                            A_qb_shared[r, c] = A_qb[t, i_h, c]
                        else:
                            A_qk_shared[r, c] = T.Cast(in_dtype, 0.0)
                            A_qb_shared[r, c] = T.Cast(in_dtype, 0.0)

                for c, k_idx in T.Parallel(BT, K):
                    t = chunk_bos + c
                    if t < eos:
                        bg_shared[c, k_idx] = T.Cast(acc_dtype, bg[t, i_h, k_idx])
                    else:
                        bg_shared[c, k_idx] = 0.0

                T.gemm(w_shared, b_h_shared, v2_frag, clear_accum=True)
                for c, vv in T.Parallel(BT, BV):
                    t = chunk_bos + c
                    g_v = i_v * BV + vv
                    v2 = T.ieee_add(v2_frag[c, vv], T.Cast(acc_dtype, u_shared[c, vv]))
                    if (t < eos) and (g_v < V):
                        v2_acc_shared[c, vv] = v2
                        v2_shared[c, vv] = T.Cast(in_dtype, v2)
                        if STORE_V_NEW_CTX:
                            v_new_ctx[t, i_h, g_v] = v2_shared[c, vv]
                    else:
                        v2_acc_shared[c, vv] = T.Cast(acc_dtype, 0.0)
                        v2_shared[c, vv] = T.Cast(in_dtype, 0.0)

                T.gemm(
                    kg_shared,
                    v_shared,
                    b_hc_kg,
                    transpose_A=True,
                    clear_accum=True,
                )
                if not DIRECT_KG_FRAGMENT:
                    T.copy(b_hc_kg, b_hc_kg_shared)
                T.gemm(
                    bg_shared,
                    v2_acc_shared,
                    b_hc_bg,
                    transpose_A=True,
                    clear_accum=True,
                )
                if not DIRECT_BG_FRAGMENT:
                    T.copy(b_hc_bg, b_hc_bg_shared)
                if DIRECT_KG_FRAGMENT and DIRECT_BG_FRAGMENT:
                    for k_idx, vv in T.Parallel(K, BV):
                        b_hc[k_idx, vv] = T.ieee_add(
                            b_hc_kg[k_idx, vv],
                            b_hc_bg[k_idx, vv],
                        )
                elif DIRECT_KG_FRAGMENT:
                    for k_idx, vv in T.Parallel(K, BV):
                        b_hc[k_idx, vv] = T.ieee_add(
                            b_hc_kg[k_idx, vv],
                            b_hc_bg_shared[k_idx, vv],
                        )
                elif DIRECT_BG_FRAGMENT:
                    for k_idx, vv in T.Parallel(K, BV):
                        b_hc[k_idx, vv] = T.ieee_add(
                            b_hc_kg_shared[k_idx, vv],
                            b_hc_bg[k_idx, vv],
                        )
                else:
                    for k_idx, vv in T.Parallel(K, BV):
                        b_hc[k_idx, vv] = T.ieee_add(
                            b_hc_kg_shared[k_idx, vv],
                            b_hc_bg_shared[k_idx, vv],
                        )

                T.gemm(qg_shared, b_h_shared, o_frag, clear_accum=True)

                T.gemm(A_qk_shared, v_shared, o_frag)
                T.gemm(A_qb_shared, v2_shared, o_frag)

                for c, vv in T.Parallel(BT, BV):
                    t = chunk_bos + c
                    g_v = i_v * BV + vv
                    if (t < eos) and (g_v < V):
                        o[t, i_h, g_v] = T.Cast(in_dtype, o_frag[c, vv])

                last_idx = T.min(chunk_bos + BT - 1, eos - 1)
                for k_idx in T.Parallel(K):
                    gk_last_shared[k_idx] = gk[last_idx, i_h, k_idx]
                for k_idx, vv in T.Parallel(K, BV):
                    b_h[k_idx, vv] = T.ieee_mul(T.exp2(gk_last_shared[k_idx]), b_h[k_idx, vv])
                for k_idx, vv in T.Parallel(K, BV):
                    b_h[k_idx, vv] = T.ieee_add(b_h[k_idx, vv], b_hc[k_idx, vv])

            if STORE_FINAL_STATE:
                for k_idx, vv in T.Parallel(K, BV):
                    g_v = i_v * BV + vv
                    if g_v < V:
                        ht[i_n, i_h, k_idx, g_v] = T.Cast(state_dtype, b_h[k_idx, vv])

    return chunk_dplr_fwd_ho_ctx_tl


def chunk_dplr_fwd_ho(
    qg: torch.Tensor,
    kg: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    bg: torch.Tensor,
    gk: torch.Tensor,
    A_qk: torch.Tensor,
    A_qb: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_layout: ChunkLayout | None = None,
    allocate_state_cache: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T_, H, K = kg.shape
    V = v.shape[-1]

    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert B == 1
        layout = chunk_layout if chunk_layout is not None else build_varlen_chunk_layout(cu_seqlens, chunk_size, T_)
    else:
        layout = build_rect_chunk_layout(B, T_, chunk_size, kg.device)
    active_nseq = layout.cu_seqlens.shape[0] - 1
    token_rows = B * T_
    in_dtype = str(kg.dtype).split(".")[-1]
    state_dtype = "float32"
    use_h0 = initial_state is not None
    allocate_state_cache = bool(allocate_state_cache) if allocate_state_cache is not None else False
    n_ht = active_nseq if (output_final_state or allocate_state_cache) else 1
    if use_h0:
        h0 = initial_state
    elif allocate_state_cache:
        h0 = torch.zeros((active_nseq, H, K, V), dtype=torch.float32, device=kg.device)
    else:
        h0 = torch.empty((1, H, K, V), dtype=torch.float32, device=kg.device)
    if h0.dtype != torch.float32:
        h0 = h0.to(torch.float32)

    qg_f = qg.reshape(token_rows, H, K).contiguous()
    kg_f = kg.reshape(token_rows, H, K).contiguous()
    v_f = v.reshape(token_rows, H, V).contiguous()
    w_f = w.reshape(token_rows, H, K).contiguous()
    u_f = u.reshape(token_rows, H, V).contiguous()
    bg_f = bg.reshape(token_rows, H, K).contiguous()
    gk_f = gk.reshape(token_rows, H, K).contiguous()
    A_qk_f = A_qk.reshape(token_rows, H, chunk_size).contiguous()
    A_qb_f = A_qb.reshape(token_rows, H, chunk_size).contiguous()

    config = _ho_fwd_configs(
        H, K, V, chunk_size,
        in_dtype, state_dtype, use_h0, output_final_state,
    )[0]
    merge_flags = _ho_fragment_merge_flags(
        K, V, chunk_size, in_dtype, False, config
    )
    kernel = _chunk_dplr_fwd_ho_kernel(
        H, K, V, chunk_size,
        in_dtype, state_dtype, use_h0, output_final_state,
        **merge_flags,
        **config,
    )
    o_f = torch.empty((token_rows, H, V), dtype=v.dtype, device=v.device)
    ht = torch.empty((n_ht, H, K, V), dtype=torch.float32, device=kg.device)
    kernel(
        qg_f, kg_f, v_f, w_f, u_f, bg_f, gk_f, A_qk_f, A_qb_f,
        h0, layout.cu_seqlens, layout.chunk_offsets, o_f, ht,
    )
    final_state = ht if output_final_state else None
    return o_f.view(B, T_, H, V), final_state


def _chunk_dplr_fwd_ho_context_schedule(
    qg: torch.Tensor,
    kg: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    bg: torch.Tensor,
    gk: torch.Tensor,
    A_qk: torch.Tensor,
    A_qb: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_layout: ChunkLayout | None = None,
    store_context: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    B, T_, H, K = kg.shape
    V = v.shape[-1]

    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert B == 1
        layout = chunk_layout if chunk_layout is not None else build_varlen_chunk_layout(cu_seqlens, chunk_size, T_)
    else:
        layout = build_rect_chunk_layout(B, T_, chunk_size, kg.device)
    active_nseq = layout.cu_seqlens.shape[0] - 1
    chunk_rows = layout.chunk_indices.shape[0]
    token_rows = B * T_
    in_dtype = str(kg.dtype).split(".")[-1]
    state_dtype = "float32"
    use_h0 = initial_state is not None
    n_ht = active_nseq if output_final_state else 1
    if use_h0:
        h0 = initial_state
    else:
        h0 = torch.empty((1, H, K, V), dtype=torch.float32, device=kg.device)
    if h0.dtype != torch.float32:
        h0 = h0.to(torch.float32)

    qg_f = qg.reshape(token_rows, H, K).contiguous()
    kg_f = kg.reshape(token_rows, H, K).contiguous()
    v_f = v.reshape(token_rows, H, V).contiguous()
    w_f = w.reshape(token_rows, H, K).contiguous()
    u_f = u.reshape(token_rows, H, V).contiguous()
    bg_f = bg.reshape(token_rows, H, K).contiguous()
    gk_f = gk.reshape(token_rows, H, K).contiguous()
    A_qk_f = A_qk.reshape(token_rows, H, chunk_size).contiguous()
    A_qb_f = A_qb.reshape(token_rows, H, chunk_size).contiguous()

    config = _ho_fwd_configs(
        H, K, V, chunk_size,
        in_dtype, state_dtype, use_h0, output_final_state,
    )[0]
    merge_flags = _ho_fragment_merge_flags(
        K, V, chunk_size, in_dtype, store_context, config
    )
    h_ctx_rows = chunk_rows
    kernel = _chunk_dplr_fwd_ho_ctx_kernel(
        H, K, V, chunk_size,
        in_dtype, state_dtype, use_h0, output_final_state,
        STORE_H_CTX=store_context,
        STORE_V_NEW_CTX=store_context,
        **merge_flags,
        **config,
    )
    o_f = torch.empty((token_rows, H, V), dtype=v.dtype, device=v.device)
    ht = torch.empty((n_ht, H, K, V), dtype=torch.float32, device=kg.device)
    if store_context:
        h_ctx = torch.empty((h_ctx_rows, H, K, V), dtype=kg.dtype, device=kg.device)
        v_new_ctx = torch.empty((token_rows, H, V), dtype=v.dtype, device=v.device)
        h_ctx_arg = h_ctx
        v_new_ctx_arg = v_new_ctx
    else:
        h_ctx_arg = torch.empty((1, H, K, V), dtype=kg.dtype, device=kg.device)
        v_new_ctx_arg = torch.empty((1, H, V), dtype=v.dtype, device=v.device)
        h_ctx = kg.new_empty((1,)).expand(h_ctx_rows, H, K, V)
        v_new_ctx = v.new_empty((1,)).expand(token_rows, H, V)
    kernel(
        qg_f, kg_f, v_f, w_f, u_f, bg_f, gk_f, A_qk_f, A_qb_f,
        h0, layout.cu_seqlens, layout.chunk_offsets, o_f, ht, h_ctx_arg, v_new_ctx_arg,
    )
    final_state = ht if output_final_state else None
    return o_f.view(B, T_, H, V), final_state, h_ctx, v_new_ctx.view(B, T_, H, V)


def chunk_dplr_fwd_ho_with_context(
    qg: torch.Tensor,
    kg: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    bg: torch.Tensor,
    gk: torch.Tensor,
    A_qk: torch.Tensor,
    A_qb: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_layout: ChunkLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    return _chunk_dplr_fwd_ho_context_schedule(
        qg=qg,
        kg=kg,
        v=v,
        w=w,
        u=u,
        bg=bg,
        gk=gk,
        A_qk=A_qk,
        A_qb=A_qb,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_layout=chunk_layout,
        store_context=True,
    )


def chunk_dplr_fwd_ho_context_elided(
    qg: torch.Tensor,
    kg: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    bg: torch.Tensor,
    gk: torch.Tensor,
    A_qk: torch.Tensor,
    A_qb: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_layout: ChunkLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    return _chunk_dplr_fwd_ho_context_schedule(
        qg=qg,
        kg=kg,
        v=v,
        w=w,
        u=u,
        bg=bg,
        gk=gk,
        A_qk=A_qk,
        A_qb=A_qb,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_layout=chunk_layout,
        store_context=False,
    )
