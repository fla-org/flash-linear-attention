# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Streaming DPLR backward.

The reverse `dhu` scan is fused with the q/o-side backward consumer so the
per-chunk `dh` state stays inside one sequence/head program instead of being
materialized as a global `(n_chunks, H, K, V)` tensor.
"""

import tilelang
import tilelang.language as T
import torch

from fla.utils import get_device_capability, get_device_smem_optin

from .schedules import (
    stream_bwd_num_stages,
    stream_bwd_schedule_or_none,
    stream_default_threads,
    stream_high_smem_bytes,
    stream_low_bwd_config,
    stream_low_smem_bytes,
    stream_mid_smem_bytes,
)
from .utils import ChunkLayout, build_rect_chunk_layout, build_varlen_chunk_layout


def _stream_bwd_config(BT: int, cc: int) -> dict[str, int]:
    # Micro-autotune on H800 (cc90) shows the high-SMEM schedule is faster with
    # 256 threads for all BT>=32 training shapes, not just K=V=128.
    if cc >= 90 and BT >= 32:
        threads = 256
    else:
        threads = stream_default_threads(BT)
    return {"threads": threads}


def _select_stream_bwd_schedule(
    *,
    K: int,
    V: int,
    BT: int,
    in_dtype: str,
    device: torch.device,
) -> tuple[str, dict[str, int]]:
    index = device.index or 0
    smem_cap = get_device_smem_optin(index)
    major, minor = get_device_capability(index)
    cc = major * 10 + minor
    selected = stream_bwd_schedule_or_none(K=K, V=V, BT=BT, in_dtype=in_dtype, smem_cap=smem_cap)
    if selected == "high":
        config = _stream_bwd_config(BT, cc)
        config["num_stages"] = stream_bwd_num_stages(
            selected, K=K, V=V, BT=BT, in_dtype=in_dtype, smem_cap=smem_cap,
        )
        return "high", config
    if selected == "mid":
        config = _stream_bwd_config(BT, cc)
        config["num_stages"] = stream_bwd_num_stages(
            selected, K=K, V=V, BT=BT, in_dtype=in_dtype, smem_cap=smem_cap,
        )
        return "mid", config
    if selected == "low":
        return "low", stream_low_bwd_config(BT, V)
    name = torch.cuda.get_device_properties(index).name
    raise RuntimeError(
        f"No launchable DPLR stream backward schedule for K={K}, V={V}, BT={BT}, "
        f"dtype={in_dtype} on {name} cc{cc}: high={stream_high_smem_bytes(K, V, BT, in_dtype)}B, "
        f"mid={stream_mid_smem_bytes(K, V, BT, in_dtype)}B, "
        f"low={stream_low_smem_bytes(K, V, BT, in_dtype)}B, device cap={smem_cap}B"
    )


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    },
)
def _chunk_dplr_bwd_stream_kernel(
    H, K, V, BT,
    in_dtype, state_dtype,
    USE_FINAL_STATE_GRADIENT: bool,
    USE_INITIAL_STATE: bool,
    threads: int = 128,
    alias_kv: bool = False,
    num_stages: int = 0,
):
    acc_dtype = "float32"
    # num_stages >= 2 double-buffers the nine operand tiles: the next chunk is
    # prefetched with cp.async while the current one is consumed.  cp.async
    # cannot mask rows by `eos`, so the last (only possibly ragged) chunk of
    # each sequence is staged with predicated scalar loads before the loop and
    # the loop itself only ever prefetches full interior chunks.
    PIPELINED = num_stages >= 2
    n_bufs = 2 if PIPELINED else 1
    n_tokens, n_seq_plus_one, n_chunks, n_dht, n_dh0 = T.dynamic(
        "n_tokens, n_seq_plus_one, n_chunks, n_dht, n_dh0"
    )
    n_seqs = n_seq_plus_one - 1

    @T.prim_func
    def chunk_dplr_bwd_stream_tl(
        qg: T.Tensor((n_tokens, H, K), in_dtype),
        bg: T.Tensor((n_tokens, H, K), in_dtype),
        w: T.Tensor((n_tokens, H, K), in_dtype),
        kg: T.Tensor((n_tokens, H, K), in_dtype),
        v: T.Tensor((n_tokens, H, V), in_dtype),
        v_new: T.Tensor((n_tokens, H, V), in_dtype),
        gk: T.Tensor((n_tokens, H, K), acc_dtype),
        do: T.Tensor((n_tokens, H, V), in_dtype),
        h: T.Tensor((n_chunks, H, K, V), in_dtype),
        A_qb: T.Tensor((n_tokens, H, BT), in_dtype),
        A_qk: T.Tensor((n_tokens, H, BT), in_dtype),
        dht: T.Tensor((n_dht, H, K, V), state_dtype),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_offsets: T.Tensor((n_seq_plus_one,), "int32"),
        dq_out: T.Tensor((n_tokens, H, K), in_dtype),
        dk_out: T.Tensor((n_tokens, H, K), in_dtype),
        dw_out: T.Tensor((n_tokens, H, K), in_dtype),
        db_out: T.Tensor((n_tokens, H, K), in_dtype),
        dgk_last: T.Tensor((n_chunks, H, K), acc_dtype),
        dv2: T.Tensor((n_tokens, H, V), in_dtype),
        dv_full: T.Tensor((n_tokens, H, V), in_dtype),
        dh0: T.Tensor((n_dh0, H, K, V), state_dtype),
    ):
        with T.Kernel(n_seqs, H, threads=threads) as (i_n, i_h):
            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            boh = chunk_offsets[i_n]
            n_chunks = chunk_offsets[i_n + 1] - boh

            b_dh = T.alloc_fragment((K, V), acc_dtype)
            b_dh_tmp = T.alloc_fragment((K, V), acc_dtype)
            if alias_kv:
                # One (K, V) tile holds dh first (dv2/dk/db/dv_full GEMMs),
                # then h (dgk_h/dq/dw GEMMs); one (BT, K) tile stages the four
                # output rows sequentially.  Fits the 227KB cc90 cap at
                # K=V=128, BT=64 where the high schedule needs 291KB.
                kv_shared = T.alloc_shared((K, V), in_dtype)
                out_shared = T.alloc_shared((BT, K), in_dtype)
            else:
                b_dh_shared = T.alloc_shared((K, V), in_dtype)
                # h rows are defined for every chunk (no ragged edge), so the
                # (K, V) h tile joins the cp.async prefetch set under
                # pipelining instead of stalling each iteration on a
                # synchronous global->shared copy.
                h_shared = T.alloc_shared((n_bufs, K, V), in_dtype)

            qg_shared = T.alloc_shared((n_bufs, BT, K), in_dtype)
            bg_shared = T.alloc_shared((n_bufs, BT, K), in_dtype)
            w_shared = T.alloc_shared((n_bufs, BT, K), in_dtype)
            kg_shared = T.alloc_shared((n_bufs, BT, K), in_dtype)
            v_shared = T.alloc_shared((n_bufs, BT, V), in_dtype)
            v_new_shared = T.alloc_shared((n_bufs, BT, V), in_dtype)
            do_shared = T.alloc_shared((n_bufs, BT, V), in_dtype)
            A_qb_shared = T.alloc_shared((n_bufs, BT, BT), in_dtype)
            A_qk_shared = T.alloc_shared((n_bufs, BT, BT), in_dtype)

            dv_intra_frag = T.alloc_fragment((BT, V), acc_dtype)
            dv2_frag = T.alloc_fragment((BT, V), acc_dtype)
            dv2_shared = T.alloc_shared((BT, V), in_dtype)
            dv_full_frag = T.alloc_fragment((BT, V), acc_dtype)
            dv_full_shared = T.alloc_shared((BT, V), in_dtype)

            dq_frag = T.alloc_fragment((BT, K), acc_dtype)
            dk_frag = T.alloc_fragment((BT, K), acc_dtype)
            dw_frag = T.alloc_fragment((BT, K), acc_dtype)
            db_frag = T.alloc_fragment((BT, K), acc_dtype)
            if not alias_kv:
                dq_shared = T.alloc_shared((BT, K), in_dtype)
                dw_shared = T.alloc_shared((BT, K), in_dtype)
                dk_shared = T.alloc_shared((BT, K), in_dtype)
                db_shared = T.alloc_shared((BT, K), in_dtype)
            dgk_last_frag = T.alloc_fragment((K,), acc_dtype)
            hdh_frag = T.alloc_fragment((K, V), acc_dtype)
            dgk_part = T.alloc_fragment((4, K), acc_dtype)
            dgk_part_shared = T.alloc_shared((4, K), acc_dtype)
            gk_last_shared = T.alloc_shared((K,), acc_dtype)

            if USE_FINAL_STATE_GRADIENT:
                for k_idx, vv in T.Parallel(K, V):
                    b_dh[k_idx, vv] = dht[i_n, i_h, k_idx, vv]
            else:
                for k_idx, vv in T.Parallel(K, V):
                    b_dh[k_idx, vv] = T.float32(0.0)

            if PIPELINED:
                t_pro = bos + (n_chunks - 1) * BT
                for r, c in T.Parallel(BT, K):
                    t = t_pro + r
                    if t < eos:
                        qg_shared[0, r, c] = qg[t, i_h, c]
                        bg_shared[0, r, c] = bg[t, i_h, c]
                        w_shared[0, r, c] = w[t, i_h, c]
                        kg_shared[0, r, c] = kg[t, i_h, c]
                    else:
                        qg_shared[0, r, c] = T.Cast(in_dtype, 0.0)
                        bg_shared[0, r, c] = T.Cast(in_dtype, 0.0)
                        w_shared[0, r, c] = T.Cast(in_dtype, 0.0)
                        kg_shared[0, r, c] = T.Cast(in_dtype, 0.0)

                for r, c in T.Parallel(BT, V):
                    t = t_pro + r
                    if t < eos:
                        v_shared[0, r, c] = v[t, i_h, c]
                        v_new_shared[0, r, c] = v_new[t, i_h, c]
                        do_shared[0, r, c] = do[t, i_h, c]
                    else:
                        v_shared[0, r, c] = T.Cast(in_dtype, 0.0)
                        v_new_shared[0, r, c] = T.Cast(in_dtype, 0.0)
                        do_shared[0, r, c] = T.Cast(in_dtype, 0.0)

                for r, c in T.Parallel(BT, BT):
                    t = t_pro + r
                    if (t < eos) and (r >= c):
                        A_qb_shared[0, r, c] = A_qb[t, i_h, c]
                        A_qk_shared[0, r, c] = A_qk[t, i_h, c]
                    else:
                        A_qb_shared[0, r, c] = T.Cast(in_dtype, 0.0)
                        A_qk_shared[0, r, c] = T.Cast(in_dtype, 0.0)

                if not alias_kv:
                    T.copy(h[boh + n_chunks - 1, i_h, 0:K, 0:V], h_shared[0, :, :])

            for i_t_rev in T.serial(n_chunks):
                i_t = n_chunks - 1 - i_t_rev
                t_off = bos + i_t * BT
                chunk_row = boh + i_t
                cur = i_t_rev % 2 if PIPELINED else 0

                T.clear(dq_frag)
                T.clear(dk_frag)
                T.clear(dw_frag)
                T.clear(db_frag)
                T.clear(b_dh_tmp)
                for k_idx in T.Parallel(K):
                    dgk_last_frag[k_idx] = T.float32(0.0)
                for gg, c in T.Parallel(4, K):
                    dgk_part[gg, c] = T.float32(0.0)

                if PIPELINED:
                    # The current buffer must be fully consumed before the
                    # cp.async below overwrites the idle one.
                    T.sync_threads()
                    if i_t_rev + 1 < n_chunks:
                        nxt = (i_t_rev + 1) % 2
                        t_nxt = t_off - BT
                        T.async_copy(qg[t_nxt: t_nxt + BT, i_h, 0:K], qg_shared[nxt, :, :])
                        T.async_copy(bg[t_nxt: t_nxt + BT, i_h, 0:K], bg_shared[nxt, :, :])
                        T.async_copy(w[t_nxt: t_nxt + BT, i_h, 0:K], w_shared[nxt, :, :])
                        T.async_copy(kg[t_nxt: t_nxt + BT, i_h, 0:K], kg_shared[nxt, :, :])
                        T.async_copy(v[t_nxt: t_nxt + BT, i_h, 0:V], v_shared[nxt, :, :])
                        T.async_copy(v_new[t_nxt: t_nxt + BT, i_h, 0:V], v_new_shared[nxt, :, :])
                        T.async_copy(do[t_nxt: t_nxt + BT, i_h, 0:V], do_shared[nxt, :, :])
                        # Stored A matrices are already causally masked.
                        T.async_copy(A_qb[t_nxt: t_nxt + BT, i_h, 0:BT], A_qb_shared[nxt, :, :])
                        T.async_copy(A_qk[t_nxt: t_nxt + BT, i_h, 0:BT], A_qk_shared[nxt, :, :])
                        if not alias_kv:
                            T.async_copy(h[chunk_row - 1, i_h, 0:K, 0:V], h_shared[nxt, :, :])
                        # One commit group per async_copy: leave the just
                        # issued groups pending, wait for the ones from the
                        # previous iteration (the current buffer).
                        T.ptx_wait_group(9 if alias_kv else 10)
                    else:
                        T.ptx_wait_group(0)
                else:
                    full_tile = t_off + BT <= eos
                    if full_tile:
                        # Bulk vectorized copies for interior chunks (TIRx
                        # showed the scalar predicated loads cap at ~1.5TB/s).
                        T.copy(qg[t_off: t_off + BT, i_h, 0:K], qg_shared[0, :, :])
                        T.copy(bg[t_off: t_off + BT, i_h, 0:K], bg_shared[0, :, :])
                        T.copy(w[t_off: t_off + BT, i_h, 0:K], w_shared[0, :, :])
                        T.copy(kg[t_off: t_off + BT, i_h, 0:K], kg_shared[0, :, :])
                        T.copy(v[t_off: t_off + BT, i_h, 0:V], v_shared[0, :, :])
                        T.copy(v_new[t_off: t_off + BT, i_h, 0:V], v_new_shared[0, :, :])
                        T.copy(do[t_off: t_off + BT, i_h, 0:V], do_shared[0, :, :])
                        # Stored A matrices are already causally masked.
                        T.copy(A_qb[t_off: t_off + BT, i_h, 0:BT], A_qb_shared[0, :, :])
                        T.copy(A_qk[t_off: t_off + BT, i_h, 0:BT], A_qk_shared[0, :, :])
                    else:
                        for r, c in T.Parallel(BT, K):
                            t = t_off + r
                            if t < eos:
                                qg_shared[0, r, c] = qg[t, i_h, c]
                                bg_shared[0, r, c] = bg[t, i_h, c]
                                w_shared[0, r, c] = w[t, i_h, c]
                                kg_shared[0, r, c] = kg[t, i_h, c]
                            else:
                                qg_shared[0, r, c] = T.Cast(in_dtype, 0.0)
                                bg_shared[0, r, c] = T.Cast(in_dtype, 0.0)
                                w_shared[0, r, c] = T.Cast(in_dtype, 0.0)
                                kg_shared[0, r, c] = T.Cast(in_dtype, 0.0)

                        for r, c in T.Parallel(BT, V):
                            t = t_off + r
                            if t < eos:
                                v_shared[0, r, c] = v[t, i_h, c]
                                v_new_shared[0, r, c] = v_new[t, i_h, c]
                                do_shared[0, r, c] = do[t, i_h, c]
                            else:
                                v_shared[0, r, c] = T.Cast(in_dtype, 0.0)
                                v_new_shared[0, r, c] = T.Cast(in_dtype, 0.0)
                                do_shared[0, r, c] = T.Cast(in_dtype, 0.0)

                        for r, c in T.Parallel(BT, BT):
                            t = t_off + r
                            if (t < eos) and (r >= c):
                                A_qb_shared[0, r, c] = A_qb[t, i_h, c]
                                A_qk_shared[0, r, c] = A_qk[t, i_h, c]
                            else:
                                A_qb_shared[0, r, c] = T.Cast(in_dtype, 0.0)
                                A_qk_shared[0, r, c] = T.Cast(in_dtype, 0.0)

                if alias_kv:
                    T.copy(b_dh, kv_shared)
                else:
                    if not PIPELINED:
                        T.copy(h[chunk_row, i_h, 0:K, 0:V], h_shared[0, :, :])
                    T.copy(b_dh, b_dh_shared)

                # dv2 = A_qb^T @ do + bg @ dh
                T.gemm(A_qb_shared[cur, :, :], do_shared[cur, :, :],
                       dv_intra_frag, transpose_A=True, clear_accum=True)
                T.gemm(bg_shared[cur, :, :], kv_shared if alias_kv else b_dh_shared,
                       dv2_frag, clear_accum=True)
                for r, vv in T.Parallel(BT, V):
                    t = t_off + r
                    dv2_frag[r, vv] = dv2_frag[r, vv] + dv_intra_frag[r, vv]
                    if t < eos:
                        dv2[t, i_h, vv] = T.Cast(in_dtype, dv2_frag[r, vv])
                T.copy(dv2_frag, dv2_shared)

                if alias_kv:
                    # dh-side reductions/GEMMs while kv_shared holds dh.  The
                    # dgk_h term reads h straight from global memory; fragment
                    # reads in a cross-layout reduction miscompile (wrong
                    # thread map), so dh must come from the shared tile.  The
                    # product goes through a dedicated fragment so the loads
                    # vectorize and the row reduction stays lowerable.
                    for k_idx, vv in T.Parallel(K, V):
                        hdh_frag[k_idx, vv] = (
                            T.Cast(acc_dtype, h[chunk_row, i_h, k_idx, vv])
                            * T.Cast(acc_dtype, kv_shared[k_idx, vv])
                        )
                    T.reduce_sum(hdh_frag, dgk_last_frag, dim=1, clear=False)
                    T.gemm(v_shared[cur, :, :], kv_shared, dk_frag, transpose_B=True)
                    T.gemm(v_new_shared[cur, :, :], kv_shared, db_frag, transpose_B=True)
                    T.gemm(kg_shared[cur, :, :], kv_shared, dv_full_frag, clear_accum=True)
                    T.gemm(A_qk_shared[cur, :, :], do_shared[cur, :, :],
                           dv_full_frag, transpose_A=True)
                    T.copy(dv_full_frag, dv_full_shared)
                    for r, vv in T.Parallel(BT, V):
                        t = t_off + r
                        if t < eos:
                            dv_full[t, i_h, vv] = T.Cast(in_dtype, dv_full_shared[r, vv])

                    # Stage each output row through the single (BT, K) tile,
                    # folding the dgk reduction into the dk/db passes.
                    T.copy(dk_frag, out_shared)
                    for r, c in T.Parallel(BT, K):
                        t = t_off + r
                        if t < eos:
                            dk_out[t, i_h, c] = out_shared[r, c]
                    for r_local in T.serial(BT // 4):
                        for gg, c in T.Parallel(4, K):
                            r = gg * (BT // 4) + r_local
                            dgk_part[gg, c] = (
                                dgk_part[gg, c]
                                + T.Cast(acc_dtype, kg_shared[cur, r, c])
                                * T.Cast(acc_dtype, out_shared[r, c])
                            )
                    T.copy(db_frag, out_shared)
                    for r, c in T.Parallel(BT, K):
                        t = t_off + r
                        if t < eos:
                            db_out[t, i_h, c] = out_shared[r, c]
                    for r_local in T.serial(BT // 4):
                        for gg, c in T.Parallel(4, K):
                            r = gg * (BT // 4) + r_local
                            dgk_part[gg, c] = (
                                dgk_part[gg, c]
                                + T.Cast(acc_dtype, bg_shared[cur, r, c])
                                * T.Cast(acc_dtype, out_shared[r, c])
                            )

                    # h-side GEMMs after overwriting the state tile with h.
                    T.copy(h[chunk_row, i_h, 0:K, 0:V], kv_shared)
                    T.gemm(do_shared[cur, :, :], kv_shared, dq_frag, transpose_B=True)
                    T.gemm(dv2_shared, kv_shared, dw_frag, transpose_B=True)
                    T.copy(dq_frag, out_shared)
                    for r, c in T.Parallel(BT, K):
                        t = t_off + r
                        if t < eos:
                            dq_out[t, i_h, c] = out_shared[r, c]
                    T.copy(dw_frag, out_shared)
                    for r, c in T.Parallel(BT, K):
                        t = t_off + r
                        if t < eos:
                            dw_out[t, i_h, c] = out_shared[r, c]
                else:
                    # q/o-side consumer of current dh.  The h*dh product goes
                    # through a dedicated fragment so the smem loads vectorize
                    # and the row reduction stays lowerable.
                    for k_idx, vv in T.Parallel(K, V):
                        hdh_frag[k_idx, vv] = (
                            T.Cast(acc_dtype, h_shared[cur, k_idx, vv])
                            * T.Cast(acc_dtype, b_dh_shared[k_idx, vv])
                        )
                    T.reduce_sum(hdh_frag, dgk_last_frag, dim=1, clear=False)
                    T.gemm(do_shared[cur, :, :], h_shared[cur, :, :], dq_frag, transpose_B=True)
                    T.gemm(v_shared[cur, :, :], b_dh_shared, dk_frag, transpose_B=True)
                    T.gemm(v_new_shared[cur, :, :], b_dh_shared, db_frag, transpose_B=True)
                    T.gemm(dv2_shared, h_shared[cur, :, :], dw_frag, transpose_B=True)

                    T.gemm(kg_shared[cur, :, :], b_dh_shared, dv_full_frag, clear_accum=True)
                    T.gemm(A_qk_shared[cur, :, :], do_shared[cur, :, :],
                           dv_full_frag, transpose_A=True)
                    T.copy(dv_full_frag, dv_full_shared)
                    for r, vv in T.Parallel(BT, V):
                        t = t_off + r
                        if t < eos:
                            dv_full[t, i_h, vv] = T.Cast(in_dtype, dv_full_shared[r, vv])

                    T.copy(dk_frag, dk_shared)
                    T.copy(db_frag, db_shared)
                    T.copy(dq_frag, dq_shared)
                    T.copy(dw_frag, dw_shared)
                    for r, c in T.Parallel(BT, K):
                        t = t_off + r
                        if t < eos:
                            dq_out[t, i_h, c] = dq_shared[r, c]
                            dk_out[t, i_h, c] = dk_shared[r, c]
                            dw_out[t, i_h, c] = dw_shared[r, c]
                            db_out[t, i_h, c] = db_shared[r, c]
                    # Split the serial over-BT dgk reduction across 4 lane
                    # groups so every thread participates (was 64/256 lanes,
                    # 21.5% of the kernel in the TIRx profile).
                    for r_local in T.serial(BT // 4):
                        for gg, c in T.Parallel(4, K):
                            r = gg * (BT // 4) + r_local
                            dgk_part[gg, c] = (
                                dgk_part[gg, c]
                                + T.Cast(acc_dtype, kg_shared[cur, r, c])
                                * T.Cast(acc_dtype, dk_shared[r, c])
                                + T.Cast(acc_dtype, bg_shared[cur, r, c])
                                * T.Cast(acc_dtype, db_shared[r, c])
                            )

                last_idx = T.min(t_off + BT - 1, eos - 1)
                for c in T.Parallel(K):
                    gk_last_shared[c] = gk[last_idx, i_h, c]
                    dgk_last_frag[c] = dgk_last_frag[c] * T.exp2(gk_last_shared[c])
                for gg, c in T.Parallel(4, K):
                    dgk_part_shared[gg, c] = dgk_part[gg, c]
                T.sync_threads()
                for c in T.Parallel(K):
                    dgk_last[chunk_row, i_h, c] = (
                        dgk_last_frag[c]
                        + dgk_part_shared[0, c]
                        + dgk_part_shared[1, c]
                        + dgk_part_shared[2, c]
                        + dgk_part_shared[3, c]
                    )

                # Update dh for the previous chunk.
                T.gemm(qg_shared[cur, :, :], do_shared[cur, :, :], b_dh_tmp, transpose_A=True)
                T.gemm(w_shared[cur, :, :], dv2_shared, b_dh_tmp, transpose_A=True)
                for k_idx, vv in T.Parallel(K, V):
                    b_dh[k_idx, vv] = T.exp2(gk_last_shared[k_idx]) * b_dh[k_idx, vv] + b_dh_tmp[k_idx, vv]

            if USE_INITIAL_STATE:
                for k_idx, vv in T.Parallel(K, V):
                    dh0[i_n, i_h, k_idx, vv] = T.Cast(state_dtype, b_dh[k_idx, vv])

    return chunk_dplr_bwd_stream_tl


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    },
)
def _chunk_dplr_bwd_stream_low_smem_kernel(
    H, K, V, BT,
    in_dtype, state_dtype,
    USE_FINAL_STATE_GRADIENT: bool,
    USE_INITIAL_STATE: bool,
    threads: int = 128,
    qside_bv: int = 16,
):
    acc_dtype = "float32"
    # Exact qside tiling lets the h slices bulk-load; ragged Vs keep the
    # scalar predicated staging.
    qside_exact = V % qside_bv == 0
    n_tokens, n_seq_plus_one, n_chunks, n_dht, n_dh0 = T.dynamic(
        "n_tokens, n_seq_plus_one, n_chunks, n_dht, n_dh0"
    )
    n_seqs = n_seq_plus_one - 1

    @T.prim_func
    def chunk_dplr_bwd_stream_low_smem_tl(
        qg: T.Tensor((n_tokens, H, K), in_dtype),
        bg: T.Tensor((n_tokens, H, K), in_dtype),
        w: T.Tensor((n_tokens, H, K), in_dtype),
        kg: T.Tensor((n_tokens, H, K), in_dtype),
        v: T.Tensor((n_tokens, H, V), in_dtype),
        v_new: T.Tensor((n_tokens, H, V), in_dtype),
        gk: T.Tensor((n_tokens, H, K), acc_dtype),
        do: T.Tensor((n_tokens, H, V), in_dtype),
        h: T.Tensor((n_chunks, H, K, V), in_dtype),
        A_qb: T.Tensor((n_tokens, H, BT), in_dtype),
        A_qk: T.Tensor((n_tokens, H, BT), in_dtype),
        dht: T.Tensor((n_dht, H, K, V), state_dtype),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_offsets: T.Tensor((n_seq_plus_one,), "int32"),
        dq_out: T.Tensor((n_tokens, H, K), in_dtype),
        dk_out: T.Tensor((n_tokens, H, K), in_dtype),
        dw_out: T.Tensor((n_tokens, H, K), in_dtype),
        db_out: T.Tensor((n_tokens, H, K), in_dtype),
        dgk_last: T.Tensor((n_chunks, H, K), acc_dtype),
        dv2: T.Tensor((n_tokens, H, V), in_dtype),
        dv_full: T.Tensor((n_tokens, H, V), in_dtype),
        dh0: T.Tensor((n_dh0, H, K, V), state_dtype),
    ):
        with T.Kernel(n_seqs, H, threads=threads) as (i_n, i_h):
            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            boh = chunk_offsets[i_n]
            n_chunks = chunk_offsets[i_n + 1] - boh

            b_dh = T.alloc_fragment((K, V), acc_dtype)
            state_shared = T.alloc_shared((K, V), in_dtype)

            qg_shared = T.alloc_shared((BT, K), in_dtype)
            bg_shared = T.alloc_shared((BT, K), in_dtype)
            w_shared = T.alloc_shared((BT, K), in_dtype)
            kg_shared = T.alloc_shared((BT, K), in_dtype)
            do_shared = T.alloc_shared((BT, V), in_dtype)
            v_like_shared = T.alloc_shared((BT, V), in_dtype)
            qside_do_shared = T.alloc_shared((BT, qside_bv), in_dtype)
            qside_value_shared = T.alloc_shared((BT, qside_bv), in_dtype)
            qside_state_shared = T.alloc_shared((K, qside_bv), in_dtype)
            A_shared = T.alloc_shared((BT, BT), in_dtype)

            dv_intra_frag = T.alloc_fragment((BT, V), acc_dtype)
            dv2_frag = T.alloc_fragment((BT, V), acc_dtype)
            dv2_shared = T.alloc_shared((BT, V), in_dtype)
            dv_full_frag = T.alloc_fragment((BT, V), acc_dtype)

            dq_frag = T.alloc_fragment((BT, K), acc_dtype)
            dk_frag = T.alloc_fragment((BT, K), acc_dtype)
            dw_frag = T.alloc_fragment((BT, K), acc_dtype)
            db_frag = T.alloc_fragment((BT, K), acc_dtype)
            dgk_part = T.alloc_fragment((4, K), acc_dtype)
            dgk_part_shared = T.alloc_shared((4, K), acc_dtype)
            dgk_h_frag = T.alloc_fragment((K,), acc_dtype)
            gk_last_frag = T.alloc_fragment((K,), acc_dtype)

            if USE_FINAL_STATE_GRADIENT:
                for k_idx, vv in T.Parallel(K, V):
                    b_dh[k_idx, vv] = dht[i_n, i_h, k_idx, vv]
            else:
                for k_idx, vv in T.Parallel(K, V):
                    b_dh[k_idx, vv] = T.float32(0.0)

            for i_t_rev in T.serial(n_chunks):
                i_t = n_chunks - 1 - i_t_rev
                t_off = bos + i_t * BT
                chunk_row = boh + i_t

                T.clear(dq_frag)
                T.clear(dk_frag)
                T.clear(dw_frag)
                T.clear(db_frag)
                for gg, c in T.Parallel(4, K):
                    dgk_part[gg, c] = T.float32(0.0)
                for k_idx in T.Parallel(K):
                    dgk_h_frag[k_idx] = T.float32(0.0)
                    gk_last_frag[k_idx] = T.float32(0.0)

                full_tile = t_off + BT <= eos
                if full_tile:
                    # Bulk vectorized copies for interior chunks (TIRx showed
                    # the scalar predicated loads cap at ~1.5TB/s).
                    T.copy(qg[t_off: t_off + BT, i_h, 0:K], qg_shared)
                    T.copy(bg[t_off: t_off + BT, i_h, 0:K], bg_shared)
                    T.copy(w[t_off: t_off + BT, i_h, 0:K], w_shared)
                    T.copy(kg[t_off: t_off + BT, i_h, 0:K], kg_shared)
                    T.copy(do[t_off: t_off + BT, i_h, 0:V], do_shared)
                    # Stored A matrices are already causally masked.
                    T.copy(A_qb[t_off: t_off + BT, i_h, 0:BT], A_shared)
                else:
                    for r, c in T.Parallel(BT, K):
                        t = t_off + r
                        if t < eos:
                            qg_shared[r, c] = qg[t, i_h, c]
                            bg_shared[r, c] = bg[t, i_h, c]
                            w_shared[r, c] = w[t, i_h, c]
                            kg_shared[r, c] = kg[t, i_h, c]
                        else:
                            qg_shared[r, c] = T.Cast(in_dtype, 0.0)
                            bg_shared[r, c] = T.Cast(in_dtype, 0.0)
                            w_shared[r, c] = T.Cast(in_dtype, 0.0)
                            kg_shared[r, c] = T.Cast(in_dtype, 0.0)

                    for r, c in T.Parallel(BT, V):
                        t = t_off + r
                        if t < eos:
                            do_shared[r, c] = do[t, i_h, c]
                        else:
                            do_shared[r, c] = T.Cast(in_dtype, 0.0)

                    for r, c in T.Parallel(BT, BT):
                        t = t_off + r
                        if (t < eos) and (r >= c):
                            A_shared[r, c] = A_qb[t, i_h, c]
                        else:
                            A_shared[r, c] = T.Cast(in_dtype, 0.0)

                # state_shared first holds the current reverse state dH.
                T.copy(b_dh, state_shared)

                # dv2 = A_qb^T @ do + bg @ dh
                T.gemm(A_shared, do_shared, dv_intra_frag, transpose_A=True, clear_accum=True)
                T.gemm(bg_shared, state_shared, dv2_frag, clear_accum=True)
                for r, vv in T.Parallel(BT, V):
                    t = t_off + r
                    dv2_frag[r, vv] = dv2_frag[r, vv] + dv_intra_frag[r, vv]
                    if t < eos:
                        dv2[t, i_h, vv] = T.Cast(in_dtype, dv2_frag[r, vv])
                T.copy(dv2_frag, dv2_shared)

                # Reuse A_shared for A_qk; write dv_full directly from registers.
                if full_tile:
                    T.copy(A_qk[t_off: t_off + BT, i_h, 0:BT], A_shared)
                else:
                    for r, c in T.Parallel(BT, BT):
                        t = t_off + r
                        if (t < eos) and (r >= c):
                            A_shared[r, c] = A_qk[t, i_h, c]
                        else:
                            A_shared[r, c] = T.Cast(in_dtype, 0.0)
                T.gemm(kg_shared, state_shared, dv_full_frag, clear_accum=True)
                T.gemm(A_shared, do_shared, dv_full_frag, transpose_A=True)
                for r, vv in T.Parallel(BT, V):
                    t = t_off + r
                    if t < eos:
                        dv_full[t, i_h, vv] = T.Cast(in_dtype, dv_full_frag[r, vv])

                last_idx = T.min(t_off + BT - 1, eos - 1)
                for c in T.Parallel(K):
                    gk_last_frag[c] = gk[last_idx, i_h, c]

                # q/o-side consumers.  Tile over V to match the saved path's
                # accumulation order and avoid long-bf16 spike drift in dq/dk.
                # v_new and v are staged like do so the slices below never
                # touch global memory; boundary rows stay zero-padded.
                if full_tile:
                    T.copy(v_new[t_off: t_off + BT, i_h, 0:V], v_like_shared)
                else:
                    for r, c in T.Parallel(BT, V):
                        t = t_off + r
                        if t < eos:
                            v_like_shared[r, c] = v_new[t, i_h, c]
                        else:
                            v_like_shared[r, c] = T.Cast(in_dtype, 0.0)
                for i_v in T.serial(T.ceildiv(V, qside_bv)):
                    # h tile: dgk_h, dq, and dw consume this tile.  h rows are
                    # defined for every chunk, so exact slices bulk-load.
                    if qside_exact:
                        T.copy(h[chunk_row, i_h, 0:K, i_v * qside_bv: (i_v + 1) * qside_bv],
                               qside_state_shared)
                        for k_idx, vv in T.Parallel(K, qside_bv):
                            dgk_h_frag[k_idx] = (
                                dgk_h_frag[k_idx]
                                + T.Cast(acc_dtype, qside_state_shared[k_idx, vv])
                                * T.Cast(acc_dtype, state_shared[k_idx, i_v * qside_bv + vv])
                            )
                    else:
                        for k_idx, vv in T.Parallel(K, qside_bv):
                            g_v = i_v * qside_bv + vv
                            if g_v < V:
                                qside_state_shared[k_idx, vv] = h[chunk_row, i_h, k_idx, g_v]
                                dgk_h_frag[k_idx] = (
                                    dgk_h_frag[k_idx]
                                    + T.Cast(acc_dtype, qside_state_shared[k_idx, vv])
                                    * T.Cast(acc_dtype, state_shared[k_idx, g_v])
                                )
                            else:
                                qside_state_shared[k_idx, vv] = T.Cast(in_dtype, 0.0)
                    for r, vv in T.Parallel(BT, qside_bv):
                        t = t_off + r
                        g_v = i_v * qside_bv + vv
                        if (t < eos) and (g_v < V):
                            qside_do_shared[r, vv] = do_shared[r, g_v]
                        else:
                            qside_do_shared[r, vv] = T.Cast(in_dtype, 0.0)
                    T.gemm(qside_do_shared, qside_state_shared, dq_frag, transpose_B=True)
                    for r, vv in T.Parallel(BT, qside_bv):
                        g_v = i_v * qside_bv + vv
                        if g_v < V:
                            qside_value_shared[r, vv] = dv2_shared[r, g_v]
                        else:
                            qside_value_shared[r, vv] = T.Cast(in_dtype, 0.0)
                    T.gemm(qside_value_shared, qside_state_shared, dw_frag, transpose_B=True)

                    # dh tile: db consumes it with the staged v_new, dk with v
                    # (staged below).  Reloading dh into the same K x BV
                    # scratch avoids the K x V shared-memory conflict that
                    # broke the previous low-smem schedule.
                    for k_idx, vv in T.Parallel(K, qside_bv):
                        g_v = i_v * qside_bv + vv
                        if g_v < V:
                            qside_state_shared[k_idx, vv] = state_shared[k_idx, g_v]
                        else:
                            qside_state_shared[k_idx, vv] = T.Cast(in_dtype, 0.0)
                    for r, vv in T.Parallel(BT, qside_bv):
                        g_v = i_v * qside_bv + vv
                        if g_v < V:
                            qside_value_shared[r, vv] = v_like_shared[r, g_v]
                        else:
                            qside_value_shared[r, vv] = T.Cast(in_dtype, 0.0)
                    T.gemm(qside_value_shared, qside_state_shared, db_frag, transpose_B=True)

                if full_tile:
                    T.copy(v[t_off: t_off + BT, i_h, 0:V], v_like_shared)
                else:
                    for r, c in T.Parallel(BT, V):
                        t = t_off + r
                        if t < eos:
                            v_like_shared[r, c] = v[t, i_h, c]
                        else:
                            v_like_shared[r, c] = T.Cast(in_dtype, 0.0)
                for i_v in T.serial(T.ceildiv(V, qside_bv)):
                    for k_idx, vv in T.Parallel(K, qside_bv):
                        g_v = i_v * qside_bv + vv
                        if g_v < V:
                            qside_state_shared[k_idx, vv] = state_shared[k_idx, g_v]
                        else:
                            qside_state_shared[k_idx, vv] = T.Cast(in_dtype, 0.0)
                    for r, vv in T.Parallel(BT, qside_bv):
                        g_v = i_v * qside_bv + vv
                        if g_v < V:
                            qside_value_shared[r, vv] = v_like_shared[r, g_v]
                        else:
                            qside_value_shared[r, vv] = T.Cast(in_dtype, 0.0)
                    T.gemm(qside_value_shared, qside_state_shared, dk_frag, transpose_B=True)

                # Reuse v_like_shared as a single K-shaped scratch for the
                # dgk_last terms, with the over-BT reduction split across 4
                # lane groups so every thread participates (21.5% of the
                # kernel in the high schedule's TIRx profile when serial).
                T.copy(dk_frag, v_like_shared)
                for r_local in T.serial(BT // 4):
                    for gg, c in T.Parallel(4, K):
                        r = gg * (BT // 4) + r_local
                        dgk_part[gg, c] = (
                            dgk_part[gg, c]
                            + T.Cast(acc_dtype, kg_shared[r, c]) * T.Cast(acc_dtype, v_like_shared[r, c])
                        )
                T.copy(db_frag, v_like_shared)
                for r_local in T.serial(BT // 4):
                    for gg, c in T.Parallel(4, K):
                        r = gg * (BT // 4) + r_local
                        dgk_part[gg, c] = (
                            dgk_part[gg, c]
                            + T.Cast(acc_dtype, bg_shared[r, c]) * T.Cast(acc_dtype, v_like_shared[r, c])
                        )

                for gg, c in T.Parallel(4, K):
                    dgk_part_shared[gg, c] = dgk_part[gg, c]
                T.sync_threads()
                for c in T.Parallel(K):
                    dgk_last[chunk_row, i_h, c] = (
                        dgk_h_frag[c] * T.exp2(gk_last_frag[c])
                        + dgk_part_shared[0, c]
                        + dgk_part_shared[1, c]
                        + dgk_part_shared[2, c]
                        + dgk_part_shared[3, c]
                    )

                for r, c in T.Parallel(BT, K):
                    t = t_off + r
                    if t < eos:
                        dq_out[t, i_h, c] = T.Cast(in_dtype, dq_frag[r, c])
                        dk_out[t, i_h, c] = T.Cast(in_dtype, dk_frag[r, c])
                        dw_out[t, i_h, c] = T.Cast(in_dtype, dw_frag[r, c])
                        db_out[t, i_h, c] = T.Cast(in_dtype, db_frag[r, c])

                # Update dh for the previous chunk: scale the carried state
                # in place so both GEMMs accumulate straight into it (mirrors
                # the Triton dhu kernel; one less (K, V) fp32 fragment).
                for k_idx, vv in T.Parallel(K, V):
                    b_dh[k_idx, vv] = b_dh[k_idx, vv] * T.exp2(gk_last_frag[k_idx])
                T.gemm(qg_shared, do_shared, b_dh, transpose_A=True)
                T.gemm(w_shared, dv2_shared, b_dh, transpose_A=True)

            if USE_INITIAL_STATE:
                for k_idx, vv in T.Parallel(K, V):
                    dh0[i_n, i_h, k_idx, vv] = T.Cast(state_dtype, b_dh[k_idx, vv])

    return chunk_dplr_bwd_stream_low_smem_tl


def chunk_dplr_bwd_stream_into(
    qg: torch.Tensor,
    bg: torch.Tensor,
    w: torch.Tensor,
    kg: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    gk: torch.Tensor,
    h: torch.Tensor,
    h0: torch.Tensor | None,
    dht: torch.Tensor | None,
    do: torch.Tensor,
    A_qb_for_dv: torch.Tensor,
    A_qk: torch.Tensor,
    dq_out: torch.Tensor,
    dk_out: torch.Tensor,
    dw_out: torch.Tensor,
    db_out: torch.Tensor,
    dgk_last_out: torch.Tensor,
    dv2_out: torch.Tensor,
    dv_full_out: torch.Tensor,
    dh0_out: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 16,
    chunk_layout: ChunkLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T_, H, K = qg.shape
    V = do.shape[-1]
    BT = int(chunk_size)
    is_varlen = cu_seqlens is not None
    for out in (dq_out, dk_out, dw_out, db_out, dgk_last_out, dv2_out, dv_full_out, dh0_out):
        assert out.is_contiguous(), "chunk_dplr_bwd_stream_into requires contiguous outputs"
    if V != K:
        raise NotImplementedError("The fused DPLR stream backward requires K == V.")
    if is_varlen:
        assert B == 1
        layout = chunk_layout if chunk_layout is not None else build_varlen_chunk_layout(cu_seqlens, BT, T_)
    else:
        layout = build_rect_chunk_layout(B, T_, BT, qg.device)
    n_chunks = layout.chunk_indices.shape[0]
    n_seqs = layout.cu_seqlens.shape[0] - 1
    n_tokens = B * T_
    in_dtype = str(qg.dtype).split(".")[-1]
    state_dtype = "float32"
    use_dht = dht is not None
    use_dh0 = h0 is not None
    n_dh0 = n_seqs if use_dh0 else 1

    qg_f = qg.reshape(n_tokens, H, K).contiguous()
    bg_f = bg.reshape(n_tokens, H, K).contiguous()
    w_f = w.reshape(n_tokens, H, K).contiguous()
    kg_f = kg.reshape(n_tokens, H, K).contiguous()
    v_f = v.reshape(n_tokens, H, V).contiguous()
    v_new_f = v_new.reshape(n_tokens, H, V).contiguous()
    gk_f = gk.reshape(n_tokens, H, K).contiguous()
    do_f = do.reshape(n_tokens, H, V).contiguous()
    h_f = h.reshape(n_chunks, H, K, V).contiguous()
    A_qb_f = A_qb_for_dv.reshape(n_tokens, H, BT).contiguous()
    A_qk_f = A_qk.reshape(n_tokens, H, BT).contiguous()

    if use_dht:
        dht_f = dht.reshape(n_seqs, H, K, V).contiguous().to(torch.float32)
    else:
        dht_f = torch.empty((1, H, K, V), dtype=torch.float32, device=qg.device)

    dq_f = dq_out.reshape(n_tokens, H, K).contiguous()
    dk_f = dk_out.reshape(n_tokens, H, K).contiguous()
    dw_f = dw_out.reshape(n_tokens, H, K).contiguous()
    db_f = db_out.reshape(n_tokens, H, K).contiguous()
    dgk_last_f = dgk_last_out.reshape(n_chunks, H, K).contiguous()
    dv2_f = dv2_out.reshape(n_tokens, H, V).contiguous()
    dv_full_f = dv_full_out.reshape(n_tokens, H, V).contiguous()
    dh0_f = dh0_out.reshape(n_dh0, H, K, V).contiguous()

    schedule, config = _select_stream_bwd_schedule(
        K=K,
        V=V,
        BT=BT,
        in_dtype=in_dtype,
        device=qg.device,
    )
    if schedule == "low":
        kernel = _chunk_dplr_bwd_stream_low_smem_kernel(
            H, K, V, BT,
            in_dtype, state_dtype, use_dht, use_dh0,
            **config,
        )
    else:
        kernel = _chunk_dplr_bwd_stream_kernel(
            H, K, V, BT,
            in_dtype, state_dtype, use_dht, use_dh0,
            alias_kv=(schedule == "mid"),
            **config,
        )
    kernel(
        qg_f, bg_f, w_f, kg_f, v_f, v_new_f, gk_f, do_f, h_f,
        A_qb_f, A_qk_f, dht_f, layout.cu_seqlens, layout.chunk_offsets,
        dq_f, dk_f, dw_f, db_f, dgk_last_f, dv2_f, dv_full_f, dh0_f,
    )
    dh0 = dh0_out if use_dh0 else None
    return dq_out, dk_out, dw_out, db_out, dgk_last_out, dv2_out, dv_full_out, dh0
