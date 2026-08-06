# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Streaming DPLR backward diagnostic kernels.

V1 fuses the reverse `dhu` scan with the q/o-side backward consumer so the
per-chunk `dh` state stays inside one sequence/head program instead of being
materialized as a global `(n_chunks, H, K, V)` tensor.  It deliberately keeps
the existing global `dv_full` workspace so correctness is isolated to the
`dh` lifetime change.
"""

import tilelang
import tilelang.language as T
import torch

from .utils import ChunkLayout, build_rect_chunk_layout, build_varlen_chunk_layout


def _stream_default_threads(BT: int) -> int:
    return 128 if BT >= 32 else 64


def _stream_bwd_config(
    BT: int,
    *,
    K: int,
    V: int,
    in_dtype: str,
    cc: int,
):
    # Micro-autotune on H800 (cc90) shows the high-SMEM schedule is faster with
    # 256 threads for all BT>=32 training shapes, not just K=V=128.
    if cc >= 90 and BT >= 32:
        threads = 256
    else:
        threads = 128 if BT >= 32 else 64
    return {"threads": threads}


def _stream_low_default_qside_bv(BT: int) -> int:
    return 32 if BT >= 64 else 16


def _stream_low_bwd_config(BT: int, V: int, *, qside_bv: int | None = None):
    config = {"threads": _stream_default_threads(BT)}
    if qside_bv is None:
        qside_bv = _stream_low_default_qside_bv(BT)
    if qside_bv > V:
        raise ValueError(f"DPLR low-SMEM qside_bv={qside_bv} exceeds V={V}")
    config["qside_bv"] = qside_bv
    return config


def _dtype_nbytes(dtype: str) -> int:
    if dtype in {"float32", "float"}:
        return 4
    if dtype in {"bfloat16", "float16", "half"}:
        return 2
    raise ValueError(f"unsupported DPLR stream backward dtype {dtype!r}")


def _stream_high_smem_bytes(K: int, V: int, BT: int, in_dtype: str) -> int:
    elem = _dtype_nbytes(in_dtype)
    return elem * (2 * K * V + 8 * BT * K + 5 * BT * V + 2 * BT * BT) + 4 * K


def _stream_low_smem_bytes(K: int, V: int, BT: int, in_dtype: str) -> int:
    qside_bv = _stream_low_default_qside_bv(BT)
    return _stream_reuse_smem_bytes(K, V, BT, in_dtype, qside_bv)


def _stream_reuse_smem_bytes(K: int, V: int, BT: int, in_dtype: str, qside_bv: int) -> int:
    elem = _dtype_nbytes(in_dtype)
    return elem * (
        K * V
        + 4 * BT * K
        + 3 * BT * V
        + BT * BT
        + BT * qside_bv
        + K * qside_bv
    )


def _device_shared_memory_cap(device: torch.device) -> tuple[int, int, str]:
    props = torch.cuda.get_device_properties(device)
    cap = int(getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block))
    cc = int(props.major) * 10 + int(props.minor)
    return cap, cc, str(props.name)


def _select_stream_bwd_schedule(
    *,
    K: int,
    V: int,
    BT: int,
    in_dtype: str,
    device: torch.device,
) -> tuple[str, dict[str, int]]:
    high_smem = _stream_high_smem_bytes(K, V, BT, in_dtype)
    low_config = _stream_low_bwd_config(BT, V)
    low_smem = _stream_reuse_smem_bytes(K, V, BT, in_dtype, low_config["qside_bv"])
    smem_cap, cc, name = _device_shared_memory_cap(device)
    low_dtype_supported = in_dtype in {"bfloat16", "float16", "half"}

    def _check(selected: str, required: int) -> str:
        if required > smem_cap:
            raise RuntimeError(
                f"DPLR stream backward schedule {selected} needs {required}B shared memory "
                f"for K={K}, V={V}, BT={BT}, dtype={in_dtype}, but device {name} "
                f"allows {smem_cap}B"
            )
        return selected

    # PRO6000 cc120's 99KB cap does not fit the head_dim128 high-SMEM schedule;
    # route it directly to the low-SMEM compatibility schedule instead of
    # probing the over-cap high-SMEM kernel.
    if cc == 120 and K == V == 128:
        if not low_dtype_supported:
            raise RuntimeError(
                "DPLR stream backward low-smem schedule is only enabled for "
                f"bf16/fp16 training dtypes; got {in_dtype!r}"
            )
        return _check("low_v2", low_smem), low_config

    if high_smem <= smem_cap:
        return "high", _stream_bwd_config(BT, K=K, V=V, in_dtype=in_dtype, cc=cc)
    if low_dtype_supported and low_smem <= smem_cap:
        return "low_v2", low_config
    raise RuntimeError(
        f"No launchable DPLR stream backward schedule for K={K}, V={V}, BT={BT}, "
        f"dtype={in_dtype} on {name} cc{cc}: high={high_smem}B, "
        f"low={low_smem}B, device cap={smem_cap}B"
    )


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    },
)
def _chunk_dplr_bwd_stream_dhu_o_kernel(
    H, K, V, BT,
    in_dtype, state_dtype,
    scale_value: float,
    USE_FINAL_STATE_GRADIENT: bool,
    USE_INITIAL_STATE: bool,
    threads: int = 128,
):
    acc_dtype = "float32"
    n_tokens, n_seq_plus_one, n_chunks, n_dht, n_dh0 = T.dynamic(
        "n_tokens, n_seq_plus_one, n_chunks, n_dht, n_dh0"
    )
    n_seqs = n_seq_plus_one - 1

    @T.prim_func
    def chunk_dplr_bwd_stream_dhu_o_tl(
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
            b_dh_shared = T.alloc_shared((K, V), in_dtype)
            h_shared = T.alloc_shared((K, V), in_dtype)

            qg_shared = T.alloc_shared((BT, K), in_dtype)
            bg_shared = T.alloc_shared((BT, K), in_dtype)
            w_shared = T.alloc_shared((BT, K), in_dtype)
            kg_shared = T.alloc_shared((BT, K), in_dtype)
            v_shared = T.alloc_shared((BT, V), in_dtype)
            v_new_shared = T.alloc_shared((BT, V), in_dtype)
            do_shared = T.alloc_shared((BT, V), in_dtype)
            A_qb_shared = T.alloc_shared((BT, BT), in_dtype)
            A_qk_shared = T.alloc_shared((BT, BT), in_dtype)

            dv_intra_frag = T.alloc_fragment((BT, V), acc_dtype)
            dv2_frag = T.alloc_fragment((BT, V), acc_dtype)
            dv2_shared = T.alloc_shared((BT, V), in_dtype)
            dv_full_frag = T.alloc_fragment((BT, V), acc_dtype)
            dv_full_shared = T.alloc_shared((BT, V), in_dtype)

            dq_frag = T.alloc_fragment((BT, K), acc_dtype)
            dk_frag = T.alloc_fragment((BT, K), acc_dtype)
            dw_frag = T.alloc_fragment((BT, K), acc_dtype)
            db_frag = T.alloc_fragment((BT, K), acc_dtype)
            dq_shared = T.alloc_shared((BT, K), in_dtype)
            dw_shared = T.alloc_shared((BT, K), in_dtype)
            dk_shared = T.alloc_shared((BT, K), in_dtype)
            db_shared = T.alloc_shared((BT, K), in_dtype)
            dgk_last_frag = T.alloc_fragment((K,), acc_dtype)
            gk_last_shared = T.alloc_shared((K,), acc_dtype)

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
                T.clear(b_dh_tmp)
                for k_idx in T.Parallel(K):
                    dgk_last_frag[k_idx] = T.float32(0.0)

                full_tile = t_off + BT <= eos
                if full_tile:
                    # Bulk vectorized copies for interior chunks (TIRx showed
                    # the scalar predicated loads cap at ~1.5TB/s).
                    T.copy(qg[t_off: t_off + BT, i_h, 0:K], qg_shared)
                    T.copy(bg[t_off: t_off + BT, i_h, 0:K], bg_shared)
                    T.copy(w[t_off: t_off + BT, i_h, 0:K], w_shared)
                    T.copy(kg[t_off: t_off + BT, i_h, 0:K], kg_shared)
                    T.copy(v[t_off: t_off + BT, i_h, 0:V], v_shared)
                    T.copy(v_new[t_off: t_off + BT, i_h, 0:V], v_new_shared)
                    T.copy(do[t_off: t_off + BT, i_h, 0:V], do_shared)
                    # Stored A matrices are already causally masked.
                    T.copy(A_qb[t_off: t_off + BT, i_h, 0:BT], A_qb_shared)
                    T.copy(A_qk[t_off: t_off + BT, i_h, 0:BT], A_qk_shared)
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
                            v_shared[r, c] = v[t, i_h, c]
                            v_new_shared[r, c] = v_new[t, i_h, c]
                            do_shared[r, c] = do[t, i_h, c]
                        else:
                            v_shared[r, c] = T.Cast(in_dtype, 0.0)
                            v_new_shared[r, c] = T.Cast(in_dtype, 0.0)
                            do_shared[r, c] = T.Cast(in_dtype, 0.0)

                    for r, c in T.Parallel(BT, BT):
                        t = t_off + r
                        if (t < eos) and (r >= c):
                            A_qb_shared[r, c] = A_qb[t, i_h, c]
                            A_qk_shared[r, c] = A_qk[t, i_h, c]
                        else:
                            A_qb_shared[r, c] = T.Cast(in_dtype, 0.0)
                            A_qk_shared[r, c] = T.Cast(in_dtype, 0.0)

                T.copy(h[chunk_row, i_h, 0:K, 0:V], h_shared)
                T.copy(b_dh, b_dh_shared)

                # dv2 = A_qb^T @ do + bg @ dh
                T.gemm(A_qb_shared, do_shared, dv_intra_frag, transpose_A=True, clear_accum=True)
                T.gemm(bg_shared, b_dh_shared, dv2_frag, clear_accum=True)
                for r, vv in T.Parallel(BT, V):
                    t = t_off + r
                    dv2_frag[r, vv] = dv2_frag[r, vv] + dv_intra_frag[r, vv]
                    if t < eos:
                        dv2[t, i_h, vv] = T.Cast(in_dtype, dv2_frag[r, vv])
                T.copy(dv2_frag, dv2_shared)

                # q/o-side consumer of current dh.
                for k_idx, vv in T.Parallel(K, V):
                    dgk_last_frag[k_idx] = (
                        dgk_last_frag[k_idx]
                        + T.Cast(acc_dtype, h_shared[k_idx, vv])
                        * T.Cast(acc_dtype, b_dh_shared[k_idx, vv])
                    )
                T.gemm(do_shared, h_shared, dq_frag, transpose_B=True)
                T.gemm(v_shared, b_dh_shared, dk_frag, transpose_B=True)
                T.gemm(v_new_shared, b_dh_shared, db_frag, transpose_B=True)
                T.gemm(dv2_shared, h_shared, dw_frag, transpose_B=True)

                T.gemm(kg_shared, b_dh_shared, dv_full_frag, clear_accum=True)
                T.gemm(A_qk_shared, do_shared, dv_full_frag, transpose_A=True)
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

                last_idx = T.min(t_off + BT - 1, eos - 1)
                for c in T.Parallel(K):
                    gk_last_shared[c] = gk[last_idx, i_h, c]
                    dgk_last_frag[c] = dgk_last_frag[c] * T.exp2(gk_last_shared[c])
                # Split the serial over-BT reduction across 4 lane groups so
                # every thread participates (was 64/256 lanes, 21.5% of the
                # kernel in the TIRx profile).
                dgk_part = T.alloc_fragment((4, K), acc_dtype)
                dgk_part_shared = T.alloc_shared((4, K), acc_dtype)
                for gg, c in T.Parallel(4, K):
                    dgk_part[gg, c] = T.float32(0.0)
                for r_local in T.serial(BT // 4):
                    for gg, c in T.Parallel(4, K):
                        r = gg * (BT // 4) + r_local
                        dgk_part[gg, c] = (
                            dgk_part[gg, c]
                            + T.Cast(acc_dtype, kg_shared[r, c]) * T.Cast(acc_dtype, dk_shared[r, c])
                            + T.Cast(acc_dtype, bg_shared[r, c]) * T.Cast(acc_dtype, db_shared[r, c])
                        )
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
                T.gemm(qg_shared, do_shared, b_dh_tmp, transpose_A=True)
                T.gemm(w_shared, dv2_shared, b_dh_tmp, transpose_A=True)
                for k_idx, vv in T.Parallel(K, V):
                    b_dh[k_idx, vv] = T.exp2(gk_last_shared[k_idx]) * b_dh[k_idx, vv] + b_dh_tmp[k_idx, vv]

            if USE_INITIAL_STATE:
                for k_idx, vv in T.Parallel(K, V):
                    dh0[i_n, i_h, k_idx, vv] = T.Cast(state_dtype, b_dh[k_idx, vv])

    return chunk_dplr_bwd_stream_dhu_o_tl


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    },
)
def _chunk_dplr_bwd_stream_dhu_o_low_smem_kernel(
    H, K, V, BT,
    in_dtype, state_dtype,
    scale_value: float,
    USE_FINAL_STATE_GRADIENT: bool,
    USE_INITIAL_STATE: bool,
    threads: int = 128,
    qside_bv: int = 16,
):
    acc_dtype = "float32"
    n_tokens, n_seq_plus_one, n_chunks, n_dht, n_dh0 = T.dynamic(
        "n_tokens, n_seq_plus_one, n_chunks, n_dht, n_dh0"
    )
    n_seqs = n_seq_plus_one - 1

    @T.prim_func
    def chunk_dplr_bwd_stream_dhu_o_low_smem_tl(
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
            dgk_last_frag = T.alloc_fragment((K,), acc_dtype)
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
                T.clear(dgk_last_frag)
                T.clear(b_dh_tmp)
                for k_idx in T.Parallel(K):
                    dgk_h_frag[k_idx] = T.float32(0.0)
                    gk_last_frag[k_idx] = T.float32(0.0)

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

                for c in T.Parallel(K):
                    dgk_last_frag[c] = T.float32(0.0)

                # q/o-side consumers.  Tile over V to match the saved path's
                # accumulation order and avoid long-bf16 spike drift in dq/dk.
                for i_v in T.serial(T.ceildiv(V, qside_bv)):
                    # h tile: dgk_h, dq, and dw consume this tile.
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

                    # dh tile: dk and db consume this tile.  Reloading it into
                    # the same K x BV scratch avoids the K x V shared-memory
                    # conflict that broke the previous low-smem schedule.
                    for k_idx, vv in T.Parallel(K, qside_bv):
                        g_v = i_v * qside_bv + vv
                        if g_v < V:
                            qside_state_shared[k_idx, vv] = state_shared[k_idx, g_v]
                        else:
                            qside_state_shared[k_idx, vv] = T.Cast(in_dtype, 0.0)
                    for r, vv in T.Parallel(BT, qside_bv):
                        t = t_off + r
                        g_v = i_v * qside_bv + vv
                        if (t < eos) and (g_v < V):
                            qside_value_shared[r, vv] = v_new[t, i_h, g_v]
                        else:
                            qside_value_shared[r, vv] = T.Cast(in_dtype, 0.0)
                    T.gemm(qside_value_shared, qside_state_shared, db_frag, transpose_B=True)
                    for r, vv in T.Parallel(BT, qside_bv):
                        t = t_off + r
                        g_v = i_v * qside_bv + vv
                        if (t < eos) and (g_v < V):
                            qside_value_shared[r, vv] = v[t, i_h, g_v]
                        else:
                            qside_value_shared[r, vv] = T.Cast(in_dtype, 0.0)
                    T.gemm(qside_value_shared, qside_state_shared, dk_frag, transpose_B=True)

                # Reuse v_like_shared as a single K-shaped scratch for the
                # dgk_last terms.  This keeps the low-smem schedule under the
                # PRO6000 cap while avoiding unsupported fragment reduction
                # layouts.
                T.copy(dk_frag, v_like_shared)
                for r in T.serial(BT):
                    for c in T.Parallel(K):
                        dgk_last_frag[c] = (
                            dgk_last_frag[c]
                            + T.Cast(acc_dtype, kg_shared[r, c]) * T.Cast(acc_dtype, v_like_shared[r, c])
                        )
                T.copy(db_frag, v_like_shared)
                for r in T.serial(BT):
                    for c in T.Parallel(K):
                        dgk_last_frag[c] = (
                            dgk_last_frag[c]
                            + T.Cast(acc_dtype, bg_shared[r, c]) * T.Cast(acc_dtype, v_like_shared[r, c])
                        )

                for c in T.Parallel(K):
                    dgk_last[chunk_row, i_h, c] = (
                        dgk_h_frag[c] * T.exp2(gk_last_frag[c])
                        + dgk_last_frag[c]
                    )

                for r, c in T.Parallel(BT, K):
                    t = t_off + r
                    if t < eos:
                        dq_out[t, i_h, c] = T.Cast(in_dtype, dq_frag[r, c])
                        dk_out[t, i_h, c] = T.Cast(in_dtype, dk_frag[r, c])
                        dw_out[t, i_h, c] = T.Cast(in_dtype, dw_frag[r, c])
                        db_out[t, i_h, c] = T.Cast(in_dtype, db_frag[r, c])

                # Update dh for the previous chunk.
                T.gemm(qg_shared, do_shared, b_dh_tmp, transpose_A=True)
                T.gemm(w_shared, dv2_shared, b_dh_tmp, transpose_A=True)
                for k_idx, vv in T.Parallel(K, V):
                    b_dh[k_idx, vv] = T.exp2(gk_last_frag[k_idx]) * b_dh[k_idx, vv] + b_dh_tmp[k_idx, vv]

            if USE_INITIAL_STATE:
                for k_idx, vv in T.Parallel(K, V):
                    dh0[i_n, i_h, k_idx, vv] = T.Cast(state_dtype, b_dh[k_idx, vv])

    return chunk_dplr_bwd_stream_dhu_o_low_smem_tl


def chunk_dplr_bwd_stream_dhu_o_into(
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
    scale: float = 1.0,
    chunk_layout: ChunkLayout | None = None,
    allocate_state_cache: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T_, H, K = qg.shape
    V = do.shape[-1]
    BT = int(chunk_size)
    is_varlen = cu_seqlens is not None
    if V != K:
        raise NotImplementedError("stream_bwd_v1 currently targets K == V training shapes.")
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
    allocate_state_cache = bool(allocate_state_cache) if allocate_state_cache is not None else False
    n_dh0 = n_seqs if (use_dh0 or allocate_state_cache) else 1

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
    elif allocate_state_cache:
        dht_f = torch.zeros((n_seqs, H, K, V), dtype=torch.float32, device=qg.device)
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
    kernel_factory = (
        _chunk_dplr_bwd_stream_dhu_o_low_smem_kernel
        if schedule == "low_v2"
        else _chunk_dplr_bwd_stream_dhu_o_kernel
    )
    kernel = kernel_factory(
        H, K, V, BT,
        in_dtype, state_dtype, float(scale), use_dht, use_dh0,
        **config,
    )
    kernel(
        qg_f, bg_f, w_f, kg_f, v_f, v_new_f, gk_f, do_f, h_f,
        A_qb_f, A_qk_f, dht_f, layout.cu_seqlens, layout.chunk_offsets,
        dq_f, dk_f, dw_f, db_f, dgk_last_f, dv2_f, dv_full_f, dh0_f,
    )
    dh0 = dh0_out if use_dh0 else None
    return dq_out, dk_out, dw_out, db_out, dgk_last_out, dv2_out, dv_full_out, dh0
