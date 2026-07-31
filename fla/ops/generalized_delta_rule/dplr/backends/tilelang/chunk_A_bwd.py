# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""DPLR intra-chunk backward (dq, dk, da, db, dgk).

Inputs:
    q, k, a, b: (B, T, H, K) — raw activations
    gi, ge: (B, T, H, K) fp32 — cumsum gates
    dAqk, dAqb: (B, T, H, BT) in_dtype — gradients of the 2 q-side A-matrices
    dAak, dAab: (B, T, H, BT) fp32 — gradients of the 2 a-side A-matrices
    dqg, dkg, dag, dbg: (B, T, H, K) — gradients of pre-gated outputs from chunk_A_fwd
    dgk_last: (chunk_rows, H, K) fp32 — from chunk_dplr_bwd_o

Outputs:
    dq, dk, da, db: (B, T, H, K)
    dgk: (B, T, H, K) — cumulative final gate gradient, cast to original gk dtype

Math: a per-row, per-j loop combining contributions from each j-th token
in the chunk into the 4 gradient accumulators, then a final mixing with
the chunk-pregated gradients (dqg/dkg/dag/dbg), then a reverse-cumsum
to compose dgk_last back into the per-token dgk.
"""

import tilelang
import tilelang.language as T
import torch

from fla.ops.utils.constant import RCP_LN2

from .schedules import device_cc
from .utils import ChunkLayout, build_rect_chunk_layout, build_varlen_chunk_layout


def _a_bwd_config(K: int, BT: int, in_dtype: str, device: torch.device) -> dict[str, int]:
    # BK=64 with 256 threads on cc90 (fits 228KB smem, 1 CTA/SM there so the
    # extra warps hide unpipelined load latency); BK=32 elsewhere (cc120's
    # 99KB cap).  Bulk vectorized copies only pay at BT=64 (measured flat to
    # -4% at BT<=32, where occupancy already hides latency).
    cc = device_cc(device)
    return {
        "BK": 64 if cc == 90 else 32,
        "threads": 256 if cc == 90 else 128,
        "num_stages": 1,
        "bulk_copy": cc == 90 and BT >= 64,
    }


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
                  tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
                  },
)
def _chunk_dplr_bwd_kernel_intra(
    H, K, BT, in_dtype, out_dtype,
    scale_value: float,
    FUSE_QSIDE_DA: bool = False,
    V: int = 64,
    BV: int = 32,
    BK: int = 32,
    threads: int = 128,
    num_stages: int = 0,
    USE_SWIZZLE: bool = False,
    DERIVE_GE: bool = False,
    bulk_copy: bool = False,
    cumsum_scale_value: float = RCP_LN2,
    gk_dtype: str | None = None,
):
    acc_dtype = "float32"
    # DERIVE_GE loads raw gk, which may stay in its own dtype (e.g. fp32)
    ge_in_dtype = (gk_dtype or in_dtype) if DERIVE_GE else acc_dtype
    n_tokens, n_seq_plus_one, n_chunks, n_tokens_d = T.dynamic(
        "n_tokens, n_seq_plus_one, n_chunks, n_tokens_d"
    )

    @T.prim_func
    def chunk_dplr_bwd_intra_tl(
        q: T.Tensor((n_tokens, H, K), in_dtype),
        k: T.Tensor((n_tokens, H, K), in_dtype),
        a: T.Tensor((n_tokens, H, K), in_dtype),
        b: T.Tensor((n_tokens, H, K), in_dtype),
        gi: T.Tensor((n_tokens, H, K), acc_dtype),
        ge: T.Tensor((n_tokens, H, K), ge_in_dtype),
        # FUSE_QSIDE_DA never reads dAqk/dAqb globals (the q-side dA tiles are
        # recomputed in-CTA), so their leading extent gets its own symbol and
        # callers may pass size-1 dummies instead of full [n_tokens, H, BT]
        # workspaces.
        dAqk: T.Tensor((n_tokens_d, H, BT), in_dtype),
        dAqb: T.Tensor((n_tokens_d, H, BT), in_dtype),
        dAak: T.Tensor((n_tokens, H, BT), in_dtype),
        dAab: T.Tensor((n_tokens, H, BT), in_dtype),
        do: T.Tensor((n_tokens, H, V), in_dtype),
        v: T.Tensor((n_tokens, H, V), in_dtype),
        v_new: T.Tensor((n_tokens, H, V), in_dtype),
        dqg: T.Tensor((n_tokens, H, K), in_dtype),
        dkg: T.Tensor((n_tokens, H, K), in_dtype),
        dag: T.Tensor((n_tokens, H, K), in_dtype),
        dbg: T.Tensor((n_tokens, H, K), in_dtype),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_indices: T.Tensor((n_chunks, 2), "int32"),
        dgk_last: T.Tensor((n_chunks, H, K), acc_dtype),
        dq: T.Tensor((n_tokens, H, K), in_dtype),
        dk: T.Tensor((n_tokens, H, K), in_dtype),
        da: T.Tensor((n_tokens, H, K), in_dtype),
        db: T.Tensor((n_tokens, H, K), in_dtype),
        dgk_output: T.Tensor((n_tokens, H, K), out_dtype),
    ):
        with T.Kernel(T.ceildiv(K, BK), n_chunks, H, threads=threads) as (i_k, i_c, i_h):
            T.use_swizzle(10, enable=USE_SWIZZLE)
            i_n = chunk_indices[i_c, 0]
            i_t = chunk_indices[i_c, 1]
            safe_i_n = T.max(i_n, 0)
            seq_bos = cu_seqlens[safe_i_n]
            seq_eos = cu_seqlens[safe_i_n + 1]
            bos_raw = seq_bos + i_t * BT
            eos_raw = T.min(bos_raw + BT, seq_eos)
            is_valid_row = i_n >= 0
            bos = T.if_then_else(is_valid_row, bos_raw, T.int32(0))
            eos = T.if_then_else(is_valid_row, eos_raw, T.int32(0))
            last_idx = T.max(eos - 1, 0)
            is_valid_chunk = is_valid_row and bos < eos

            q_shared = T.alloc_shared((BT, BK), in_dtype)
            k_shared = T.alloc_shared((BT, BK), in_dtype)
            a_shared = T.alloc_shared((BT, BK), in_dtype)
            b_shared = T.alloc_shared((BT, BK), in_dtype)
            gi_shared = T.alloc_shared((BT, BK), acc_dtype)
            ge_shared = T.alloc_shared((BT, BK), acc_dtype)

            dAqk_shared = T.alloc_shared((BT, BT), acc_dtype)
            dAqb_shared = T.alloc_shared((BT, BT), acc_dtype)
            dAak_shared = T.alloc_shared((BT, BT), acc_dtype)
            dAab_shared = T.alloc_shared((BT, BT), acc_dtype)
            if FUSE_QSIDE_DA:
                do_shared = T.alloc_shared((BT, BV), in_dtype)
                v_shared = T.alloc_shared((BT, BV), in_dtype)
                dAqk_frag = T.alloc_fragment((BT, BT), acc_dtype)
                v_new_shared = T.alloc_shared((BT, BV), in_dtype)
                dAqb_frag = T.alloc_fragment((BT, BT), acc_dtype)

            dqg_shared = T.alloc_shared((BT, BK), in_dtype)
            dkg_shared = T.alloc_shared((BT, BK), in_dtype)
            dag_shared = T.alloc_shared((BT, BK), in_dtype)
            dbg_shared = T.alloc_shared((BT, BK), in_dtype)
            g_last = T.alloc_shared((BK,), acc_dtype)

            # Load Q, K, A, B, gi, ge tiles.  On cc90 (1 CTA/SM at BT=64) the
            # kernel is latency-bound, so interior chunks take bulk vectorized
            # copies and the scalar predicated path (~1.5TB/s cap) is kept for
            # boundary chunks only.  Off cc90 occupancy already hides the load
            # latency and bulk copies measured ~4% slower, so the fast path is
            # compiled out there.  (cp_async/sync alternates measured slower
            # than the scalar path on both devices.)
            full_tile = (is_valid_chunk and (bos + BT <= eos)) if bulk_copy else False
            if full_tile:
                T.copy(q[bos: bos + BT, i_h, i_k * BK: i_k * BK + BK], q_shared)
                T.copy(k[bos: bos + BT, i_h, i_k * BK: i_k * BK + BK], k_shared)
                T.copy(a[bos: bos + BT, i_h, i_k * BK: i_k * BK + BK], a_shared)
                T.copy(b[bos: bos + BT, i_h, i_k * BK: i_k * BK + BK], b_shared)
                T.copy(gi[bos: bos + BT, i_h, i_k * BK: i_k * BK + BK], gi_shared)
                if DERIVE_GE:
                    # ge_shared first holds raw gk, then derives in place.
                    T.copy(ge[bos: bos + BT, i_h, i_k * BK: i_k * BK + BK], ge_shared)
                    for r, c in T.Parallel(BT, BK):
                        ge_shared[r, c] = (
                            gi_shared[r, c] - ge_shared[r, c] * T.Cast(acc_dtype, cumsum_scale_value)
                        )
                else:
                    T.copy(ge[bos: bos + BT, i_h, i_k * BK: i_k * BK + BK], ge_shared)
            else:
                for r, c in T.Parallel(BT, BK):
                    t = bos + r
                    k_idx = i_k * BK + c
                    if t < eos and k_idx < K:
                        q_shared[r, c] = q[t, i_h, k_idx]
                        k_shared[r, c] = k[t, i_h, k_idx]
                        a_shared[r, c] = a[t, i_h, k_idx]
                        b_shared[r, c] = b[t, i_h, k_idx]
                        if DERIVE_GE:
                            giv = gi[t, i_h, k_idx]
                            gi_shared[r, c] = giv
                            ge_shared[r, c] = giv - T.Cast(acc_dtype, ge[t, i_h, k_idx]) * \
                                T.Cast(acc_dtype, cumsum_scale_value)
                        else:
                            gi_shared[r, c] = gi[t, i_h, k_idx]
                            ge_shared[r, c] = ge[t, i_h, k_idx]
                    else:
                        q_shared[r, c] = T.Cast(in_dtype, 0.0)
                        k_shared[r, c] = T.Cast(in_dtype, 0.0)
                        a_shared[r, c] = T.Cast(in_dtype, 0.0)
                        b_shared[r, c] = T.Cast(in_dtype, 0.0)
                        gi_shared[r, c] = T.float32(0.0)
                        ge_shared[r, c] = T.float32(0.0)

            if FUSE_QSIDE_DA:
                T.clear(dAqk_frag)
                T.clear(dAqb_frag)
                for i_v in T.serial(T.ceildiv(V, BV)):
                    if full_tile:
                        T.copy(do[bos: bos + BT, i_h, i_v * BV: i_v * BV + BV], do_shared)
                        T.copy(v[bos: bos + BT, i_h, i_v * BV: i_v * BV + BV], v_shared)
                        T.copy(v_new[bos: bos + BT, i_h, i_v * BV: i_v * BV + BV], v_new_shared)
                    else:
                        for r, c in T.Parallel(BT, BV):
                            t = bos + r
                            g_v = i_v * BV + c
                            if (t < eos) and (g_v < V):
                                do_shared[r, c] = do[t, i_h, g_v]
                                v_shared[r, c] = v[t, i_h, g_v]
                                v_new_shared[r, c] = v_new[t, i_h, g_v]
                            else:
                                do_shared[r, c] = T.Cast(in_dtype, 0.0)
                                v_shared[r, c] = T.Cast(in_dtype, 0.0)
                                v_new_shared[r, c] = T.Cast(in_dtype, 0.0)
                    T.gemm(do_shared, v_shared, dAqk_frag, transpose_B=True)
                    T.gemm(do_shared, v_new_shared, dAqb_frag, transpose_B=True)
                for r, c in T.Parallel(BT, BT):
                    if r >= c:
                        # Match the saved path's materialized bf16 boundary.
                        # Without this explicit round, fused q-side recompute
                        # keeps dAqk/dAqb in fp32 and can create one-ULP
                        # dq/dk spikes versus the default FLA/saved envelope.
                        dAqk_shared[r, c] = T.Cast(
                            acc_dtype,
                            T.Cast(in_dtype, dAqk_frag[r, c] * T.Cast(acc_dtype, scale_value)),
                        )
                        dAqb_shared[r, c] = T.Cast(
                            acc_dtype,
                            T.Cast(in_dtype, dAqb_frag[r, c] * T.Cast(acc_dtype, scale_value)),
                        )
                    else:
                        dAqk_shared[r, c] = T.float32(0.0)
                        dAqb_shared[r, c] = T.float32(0.0)
            else:
                for r, c in T.Parallel(BT, BT):
                    t = bos + r
                    if t < eos:
                        dAqk_shared[r, c] = T.Cast(acc_dtype, dAqk[t, i_h, c])
                        dAqb_shared[r, c] = T.Cast(acc_dtype, dAqb[t, i_h, c])
                    else:
                        dAqk_shared[r, c] = T.float32(0.0)
                        dAqb_shared[r, c] = T.float32(0.0)
            if full_tile:
                T.copy(dAak[bos: bos + BT, i_h, 0:BT], dAak_shared)
                T.copy(dAab[bos: bos + BT, i_h, 0:BT], dAab_shared)
            else:
                for r, c in T.Parallel(BT, BT):
                    t = bos + r
                    if t < eos:
                        dAak_shared[r, c] = T.Cast(acc_dtype, dAak[t, i_h, c])
                        dAab_shared[r, c] = T.Cast(acc_dtype, dAab[t, i_h, c])
                    else:
                        dAak_shared[r, c] = T.float32(0.0)
                        dAab_shared[r, c] = T.float32(0.0)

            # Apply causal masks to dA matrices
            for r, c in T.Parallel(BT, BT):
                if r < c:
                    dAqk_shared[r, c] = T.float32(0.0)
                    dAqb_shared[r, c] = T.float32(0.0)
                if r <= c:
                    dAak_shared[r, c] = T.float32(0.0)
                    dAab_shared[r, c] = T.float32(0.0)

            # Compute stabilization offset from gi at mid-point of valid tokens
            valid_len = T.min(eos - bos, BT)
            mid = valid_len // 2
            offset = T.alloc_shared((BK,), acc_dtype)
            for c in T.Parallel(BK):
                offset[c] = gi_shared[mid, c]

            # Compute stabilized ops
            q_ops = T.alloc_shared((BT, BK), acc_dtype)
            k_ops = T.alloc_shared((BT, BK), acc_dtype)
            a_ops = T.alloc_shared((BT, BK), acc_dtype)
            b_ops = T.alloc_shared((BT, BK), acc_dtype)
            for r, c in T.Parallel(BT, BK):
                q_ops[r, c] = T.Cast(acc_dtype, q_shared[r, c]) * T.exp2(gi_shared[r, c] - offset[c])
                k_ops[r, c] = T.Cast(acc_dtype, k_shared[r, c]) * T.exp2(-(gi_shared[r, c] - offset[c]))
                b_ops[r, c] = T.Cast(acc_dtype, b_shared[r, c]) * T.exp2(-(gi_shared[r, c] - offset[c]))
                a_ops[r, c] = T.Cast(acc_dtype, a_shared[r, c]) * T.exp2(ge_shared[r, c] - offset[c])

            # Intra-chunk gradients via GEMMs
            dq_intra = T.alloc_fragment((BT, BK), acc_dtype)
            da_intra = T.alloc_fragment((BT, BK), acc_dtype)
            dk_intra = T.alloc_fragment((BT, BK), acc_dtype)
            db_intra = T.alloc_fragment((BT, BK), acc_dtype)
            T.clear(dq_intra)
            T.clear(da_intra)
            T.clear(dk_intra)
            T.clear(db_intra)

            # dq += dAqk @ k_ops + dAqb @ b_ops
            T.gemm(dAqk_shared, k_ops, dq_intra)
            T.gemm(dAqb_shared, b_ops, dq_intra)
            # da += dAak @ k_ops + dAab @ b_ops
            T.gemm(dAak_shared, k_ops, da_intra)
            T.gemm(dAab_shared, b_ops, da_intra)
            # dk += dAqk^T @ q_ops + dAak^T @ a_ops
            T.gemm(dAqk_shared, q_ops, dk_intra, transpose_A=True)
            T.gemm(dAak_shared, a_ops, dk_intra, transpose_A=True)
            # db += dAqb^T @ q_ops + dAab^T @ a_ops
            T.gemm(dAqb_shared, q_ops, db_intra, transpose_A=True)
            T.gemm(dAab_shared, a_ops, db_intra, transpose_A=True)

            # Load inter-chunk gradients and g_last
            if full_tile:
                T.copy(dqg[bos: bos + BT, i_h, i_k * BK: i_k * BK + BK], dqg_shared)
                T.copy(dkg[bos: bos + BT, i_h, i_k * BK: i_k * BK + BK], dkg_shared)
                T.copy(dag[bos: bos + BT, i_h, i_k * BK: i_k * BK + BK], dag_shared)
                T.copy(dbg[bos: bos + BT, i_h, i_k * BK: i_k * BK + BK], dbg_shared)
            else:
                for r, c in T.Parallel(BT, BK):
                    t = bos + r
                    k_idx = i_k * BK + c
                    if t < eos and k_idx < K:
                        dqg_shared[r, c] = dqg[t, i_h, k_idx]
                        dkg_shared[r, c] = dkg[t, i_h, k_idx]
                        dag_shared[r, c] = dag[t, i_h, k_idx]
                        dbg_shared[r, c] = dbg[t, i_h, k_idx]
                    else:
                        dqg_shared[r, c] = T.Cast(in_dtype, 0.0)
                        dkg_shared[r, c] = T.Cast(in_dtype, 0.0)
                        dag_shared[r, c] = T.Cast(in_dtype, 0.0)
                        dbg_shared[r, c] = T.Cast(in_dtype, 0.0)
            for c in T.Parallel(BK):
                k_idx = i_k * BK + c
                if is_valid_chunk and k_idx < K:
                    g_last[c] = gi[last_idx, i_h, k_idx]
                else:
                    g_last[c] = T.float32(0.0)

            # Combine intra + inter, un-stabilize, and store
            scale_v = T.Cast(acc_dtype, scale_value)
            if full_tile:
                for r, c in T.Parallel(BT, BK):
                    t = bos + r
                    k_idx = i_k * BK + c
                    dq_val = dq_intra[r, c] * T.exp2(gi_shared[r, c] - offset[c]) + \
                        T.Cast(acc_dtype, dqg_shared[r, c]) * T.exp2(gi_shared[r, c]) * scale_v
                    da_val = da_intra[r, c] * T.exp2(ge_shared[r, c] - offset[c]) + \
                        T.Cast(acc_dtype, dag_shared[r, c]) * T.exp2(ge_shared[r, c])
                    dk_val = dk_intra[r, c] * T.exp2(-(gi_shared[r, c] - offset[c])) + \
                        T.Cast(acc_dtype, dkg_shared[r, c]) * T.exp2(g_last[c] - gi_shared[r, c])
                    db_val = db_intra[r, c] * T.exp2(-(gi_shared[r, c] - offset[c])) + \
                        T.Cast(acc_dtype, dbg_shared[r, c]) * T.exp2(g_last[c] - gi_shared[r, c])

                    dq[t, i_h, k_idx] = T.Cast(in_dtype, dq_val)
                    dk[t, i_h, k_idx] = T.Cast(in_dtype, dk_val)
                    da[t, i_h, k_idx] = T.Cast(in_dtype, da_val)
                    db[t, i_h, k_idx] = T.Cast(in_dtype, db_val)
                    q_ops[r, c] = dq_val * T.Cast(acc_dtype, q_shared[r, c]) + da_val * T.Cast(acc_dtype, a_shared[r, c]) - \
                        dk_val * T.Cast(acc_dtype, k_shared[r, c]) - db_val * T.Cast(acc_dtype, b_shared[r, c])
                    k_ops[r, c] = da_val * T.Cast(acc_dtype, a_shared[r, c])
            else:
                for r, c in T.Parallel(BT, BK):
                    t = bos + r
                    k_idx = i_k * BK + c
                    if t < eos and k_idx < K:
                        dq_val = dq_intra[r, c] * T.exp2(gi_shared[r, c] - offset[c]) + \
                            T.Cast(acc_dtype, dqg_shared[r, c]) * T.exp2(gi_shared[r, c]) * scale_v
                        da_val = da_intra[r, c] * T.exp2(ge_shared[r, c] - offset[c]) + \
                            T.Cast(acc_dtype, dag_shared[r, c]) * T.exp2(ge_shared[r, c])
                        dk_val = dk_intra[r, c] * T.exp2(-(gi_shared[r, c] - offset[c])) + \
                            T.Cast(acc_dtype, dkg_shared[r, c]) * T.exp2(g_last[c] - gi_shared[r, c])
                        db_val = db_intra[r, c] * T.exp2(-(gi_shared[r, c] - offset[c])) + \
                            T.Cast(acc_dtype, dbg_shared[r, c]) * T.exp2(g_last[c] - gi_shared[r, c])

                        dq[t, i_h, k_idx] = T.Cast(in_dtype, dq_val)
                        dk[t, i_h, k_idx] = T.Cast(in_dtype, dk_val)
                        da[t, i_h, k_idx] = T.Cast(in_dtype, da_val)
                        db[t, i_h, k_idx] = T.Cast(in_dtype, db_val)
                        q_ops[r, c] = dq_val * T.Cast(acc_dtype, q_shared[r, c]) + da_val * T.Cast(acc_dtype, a_shared[r, c]) - \
                            dk_val * T.Cast(acc_dtype, k_shared[r, c]) - db_val * T.Cast(acc_dtype, b_shared[r, c])
                        k_ops[r, c] = da_val * T.Cast(acc_dtype, a_shared[r, c])
                    else:
                        q_ops[r, c] = T.float32(0.0)
                        k_ops[r, c] = T.float32(0.0)

            # Reuse q_ops/k_ops as dgk_raw/dgk_offset scratch and finish the
            # reverse cumsum inside this kernel, avoiding two fp32 full-tensor outputs.
            # The suffix sum must not read-modify-write q_ops: when BK < threads
            # the T.Parallel lowering shares one column across several threads
            # (address masked to tid % BK), and a fast sibling can overwrite a
            # row before the others read it (hd128 BK=32 dgk corruption).  q_ops
            # stays read-only here; suffixes land in dgk_cum, where duplicate
            # same-value writes are benign.
            T.sync_threads()
            dgk_suffix = T.alloc_fragment((BK,), acc_dtype)
            dgk_cum = T.alloc_shared((BT, BK), acc_dtype)
            for c in T.Parallel(BK):
                dgk_suffix[c] = T.float32(0.0)
            for r_rev in T.serial(BT):
                r = BT - 1 - r_rev
                for c in T.Parallel(BK):
                    dgk_suffix[c] += q_ops[r, c]
                    dgk_cum[r, c] = dgk_suffix[c]
            T.sync_threads()
            if full_tile:
                for r, c in T.Parallel(BT, BK):
                    k_idx = i_k * BK + c
                    dgk_output[bos + r, i_h, k_idx] = T.Cast(
                        out_dtype, dgk_cum[r, c] + dgk_last[i_c, i_h, k_idx] - k_ops[r, c])
            else:
                for r, c in T.Parallel(BT, BK):
                    t = bos + r
                    k_idx = i_k * BK + c
                    if t < eos and k_idx < K:
                        dgk_output[t, i_h, k_idx] = T.Cast(
                            out_dtype, dgk_cum[r, c] + T.if_then_else(is_valid_chunk, dgk_last[i_c, i_h, k_idx], T.float32(0.0)) - k_ops[r, c])

    return chunk_dplr_bwd_intra_tl


def chunk_dplr_bwd_dqk_intra_fused_qside_into(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor | None,
    do: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    dAak: torch.Tensor,
    dAab: torch.Tensor,
    dqg: torch.Tensor,
    dkg: torch.Tensor,
    dag: torch.Tensor,
    dbg: torch.Tensor,
    dgk_last: torch.Tensor,
    dq_out: torch.Tensor,
    dk_out: torch.Tensor,
    da_out: torch.Tensor,
    db_out: torch.Tensor,
    dgk_out: torch.Tensor,
    gk: torch.Tensor | None = None,
    scale: float = 1.0,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 16,
    chunk_layout: ChunkLayout | None = None,
    dgk_dtype: torch.dtype | None = None,
):
    """Recompute q-side dA tiles on-chip and consume them immediately.

    When ``gk`` is passed (``ge`` may then be None), the kernel runs its
    DERIVE_GE specialization: it loads raw bf16 ``gk`` instead of fp32 ``ge``
    and derives ``ge = gi - gk*RCP_LN2`` in-CTA, removing the fp32 ``ge``
    tensor from the training backward.
    """
    if ge is None and gk is None:
        raise ValueError("chunk_dplr_bwd_dqk_intra_fused_qside_into needs ge or gk")
    for out in (dq_out, dk_out, da_out, db_out, dgk_out):
        assert out.is_contiguous(), "chunk_dplr_bwd_dqk_intra_fused_qside_into requires contiguous outputs"
    derive_ge = gk is not None
    B, T_, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    is_varlen = cu_seqlens is not None

    if is_varlen:
        assert B == 1
        layout = chunk_layout if chunk_layout is not None else build_varlen_chunk_layout(cu_seqlens, BT, T_)
    else:
        layout = build_rect_chunk_layout(B, T_, BT, q.device)
    n_chunks = layout.chunk_indices.shape[0]
    N_tokens = B * T_

    in_dtype = str(q.dtype).split(".")[-1]
    dgk_out_dtype = str(dgk_dtype or dgk_out.dtype).split(".")[-1]

    q_f = q.reshape(N_tokens, H, K).contiguous()
    k_f = k.reshape(N_tokens, H, K).contiguous()
    a_f = a.reshape(N_tokens, H, K).contiguous()
    b_f = b.reshape(N_tokens, H, K).contiguous()
    gi_f = gi.reshape(N_tokens, H, K).contiguous()
    ge_f = (gk if derive_ge else ge).reshape(N_tokens, H, K).contiguous()
    do_f = do.reshape(N_tokens, H, V).contiguous()
    v_f = v.reshape(N_tokens, H, V).contiguous()
    v_new_f = v_new.reshape(N_tokens, H, V).contiguous()
    dAak_f = dAak.reshape(N_tokens, H, BT).contiguous()
    dAab_f = dAab.reshape(N_tokens, H, BT).contiguous()
    dqg_f = dqg.reshape(N_tokens, H, K).contiguous()
    dkg_f = dkg.reshape(N_tokens, H, K).contiguous()
    dag_f = dag.reshape(N_tokens, H, K).contiguous()
    dbg_f = dbg.reshape(N_tokens, H, K).contiguous()
    dgk_last_f = dgk_last.reshape(n_chunks, H, K).contiguous()
    dq_f = dq_out.reshape(N_tokens, H, K).contiguous()
    dk_f = dk_out.reshape(N_tokens, H, K).contiguous()
    da_f = da_out.reshape(N_tokens, H, K).contiguous()
    db_f = db_out.reshape(N_tokens, H, K).contiguous()
    dgk_out_f = dgk_out.reshape(N_tokens, H, K).contiguous()
    # the fused kernel specialization ignores dAqk/dAqb (their leading extent
    # is a separate JIT symbol), so size-1 dummies satisfy the signature
    dummy_dA = q.new_empty((1, H, BT))
    # Deterministic BV selection (env knob removed): full V when possible,
    # else the largest power-of-two divisor of V not exceeding 64.
    bv = min(64, V)
    while V % bv != 0 and bv > 1:
        bv //= 2
    config = _a_bwd_config(K, BT, in_dtype, q.device)
    if BT < 32:
        config["threads"] = min(config["threads"], 32)

    kernel = _chunk_dplr_bwd_kernel_intra(
        H, K, BT, in_dtype, dgk_out_dtype, float(scale),
        FUSE_QSIDE_DA=True, V=V, BV=bv,
        USE_SWIZZLE=False,
        DERIVE_GE=derive_ge,
        gk_dtype=str(gk.dtype).split(".")[-1] if derive_ge else None,
        **config,
    )
    kernel(
        q_f, k_f, a_f, b_f, gi_f, ge_f, dummy_dA, dummy_dA, dAak_f, dAab_f,
        do_f, v_f, v_new_f,
        dqg_f, dkg_f, dag_f, dbg_f, layout.cu_seqlens, layout.chunk_indices, dgk_last_f,
        dq_f, dk_f, da_f, db_f, dgk_out_f,
    )
    return dq_out, dk_out, da_out, db_out, dgk_out
