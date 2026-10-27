# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang causal attention with GQA, gating, sliding windows, sink bias, and variable-length sequences.

Forward returns log2 LSE, matching the Triton backward interface.
"""

import tilelang
import tilelang.language as T
import torch

from fla.ops.attn.parallel import parallel_attn_bwd_preprocess
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def parallel_attn_fwd_kernel(
    B: int,
    H: int,
    HQ: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype,
    scale: float,
    BT: int,
    BS: int,
    USE_G: bool,
    USE_SINK: bool,
    USE_WINDOW: bool,
    W: int,
    IS_VARLEN: bool,
    num_warps: int,
    num_stages: int,
):
    accum_dtype = T.float32
    scale_log2 = scale * RCP_LN2
    G = HQ // H
    # pad shared buffers to satisfy WGMMA/MMA inner-dimension alignment
    BK = tilelang.cdiv(K, 16) * 16
    BV = tilelang.cdiv(V, 16) * 16
    seq_len = T.dynamic("seq_len")
    num_chunks = T.dynamic("num_chunks") if IS_VARLEN else 0
    num_offsets = T.dynamic("num_offsets") if IS_VARLEN else 0

    @T.prim_func
    def main(
        q: T.Tensor((B, seq_len, HQ, K), dtype),
        k: T.Tensor((B, seq_len, H, K), dtype),
        v: T.Tensor((B, seq_len, H, V), dtype),
        g: T.Tensor((B, seq_len, HQ), accum_dtype),
        sink_bias: T.Tensor((HQ,), accum_dtype),
        o: T.Tensor((B, seq_len, HQ, V), output_dtype),
        lse: T.Tensor((B, seq_len, HQ), accum_dtype),
        cu_seqlens: T.Tensor((num_offsets,), T.int32),
        chunk_indices: T.Tensor((num_chunks, 2), T.int32),
    ):
        with T.Kernel(HQ, num_chunks if IS_VARLEN else T.ceildiv(seq_len, BT), B, threads=num_warps * 32) as (i_hq, i_t, i_b):
            if IS_VARLEN:
                i_n = chunk_indices[i_t, 0]
                i_t_local = chunk_indices[i_t, 1]
                bos = cu_seqlens[i_n]
                eos = cu_seqlens[i_n + 1]
            else:
                i_t_local = i_t
                bos = 0
                eos = seq_len
            length = eos - bos
            q_start = bos + i_t_local * BT
            i_h = i_hq // G

            s_q = T.alloc_shared((BT, BK), dtype)
            s_k = T.alloc_shared((BS, BK), dtype)
            s_v = T.alloc_shared((BS, BV), dtype)

            b_s = T.alloc_fragment((BT, BS), accum_dtype)
            b_p = T.alloc_fragment((BT, BS), dtype)
            b_o = T.alloc_fragment((BT, BV), accum_dtype)

            b_m = T.alloc_fragment((BT,), accum_dtype)
            b_m_prev = T.alloc_fragment((BT,), accum_dtype)
            b_l = T.alloc_fragment((BT,), accum_dtype)
            b_p_sum = T.alloc_fragment((BT,), accum_dtype)
            b_scale = T.alloc_fragment((BT,), accum_dtype)

            if USE_G:
                s_gq = T.alloc_shared((BT,), accum_dtype)
                s_gk = T.alloc_shared((BS,), accum_dtype)
                T.copy(g[i_b, q_start:q_start + BT, i_hq], s_gq)

            T.copy(q[i_b, q_start:q_start + BT, i_hq, :], s_q)
            T.fill(b_o, 0)
            T.fill(b_l, 0)
            T.fill(b_m, -T.infinity(accum_dtype))

            if USE_WINDOW:
                # include every key visible to the first query in this tile
                loop_start = T.max(i_t_local * BT - W + 1, 0) // BS
            else:
                loop_start = 0
            loop_limit = T.ceildiv((i_t_local + 1) * BT, BS)
            loop_end = T.min(loop_limit, T.ceildiv(length, BS))

            for i_s in T.Pipelined(loop_start, loop_end, num_stages=num_stages):
                k_start = bos + i_s * BS

                T.copy(k[i_b, k_start:k_start + BS, i_h, :], s_k)
                T.copy(v[i_b, k_start:k_start + BS, i_h, :], s_v)
                if USE_G:
                    T.copy(g[i_b, k_start:k_start + BS, i_hq], s_gk)

                T.clear(b_s)
                T.gemm(s_q, s_k, b_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(BT, BS):
                    q_pos = q_start + i
                    k_pos = k_start + j
                    causal = (k_pos <= q_pos) & (k_pos < length + bos) & (q_pos < length + bos)
                    if USE_WINDOW:
                        mask = causal & (q_pos - k_pos < W)
                    else:
                        mask = causal
                    if USE_G:
                        s = b_s[i, j] * scale_log2 + s_gq[i] - s_gk[j]
                    else:
                        s = b_s[i, j] * scale_log2
                    b_s[i, j] = T.if_then_else(mask, s, -T.infinity(accum_dtype))

                T.copy(b_m, b_m_prev)
                T.reduce_max(b_s, b_m, dim=1, clear=False)
                for i in T.Parallel(BT):
                    b_m[i] = T.max(b_m[i], b_m_prev[i])
                # use a finite pivot for all-masked rows to avoid (-inf) - (-inf)
                b_m_stable = T.alloc_fragment((BT,), accum_dtype)
                for i in T.Parallel(BT):
                    b_m_stable[i] = T.if_then_else(b_m[i] == -T.infinity(accum_dtype), 0., b_m[i])
                for i in T.Parallel(BT):
                    b_scale[i] = T.if_then_else(
                        b_m_prev[i] == -T.infinity(accum_dtype),
                        0.,
                        T.exp2(b_m_prev[i] - b_m_stable[i]),
                    )
                for i, j in T.Parallel(BT, BV):
                    b_o[i, j] *= b_scale[i]
                for i, j in T.Parallel(BT, BS):
                    b_s[i, j] = T.exp2(b_s[i, j] - b_m_stable[i])
                T.reduce_sum(b_s, b_p_sum, dim=1)
                for i in T.Parallel(BT):
                    b_l[i] = b_l[i] * b_scale[i] + b_p_sum[i]

                T.copy(b_s, b_p)
                T.gemm(b_p, s_v, b_o, policy=T.GemmWarpPolicy.FullRow)

            # sink-bias contributes to the normalizer but not to the value matmul
            if USE_SINK:
                # keep sink_bias - b_m finite when the row has no valid keys
                for i in T.Parallel(BT):
                    b_m[i] = T.if_then_else(b_m[i] == -T.infinity(accum_dtype), 0., b_m[i])
                for i in T.Parallel(BT):
                    b_l[i] += T.exp2(sink_bias[i_hq] - b_m[i])

            s_o = T.alloc_shared((BT, BV), output_dtype)
            s_lse = T.alloc_shared((BT,), accum_dtype)
            for i, d in T.Parallel(BT, BV):
                b_o[i, d] = b_o[i, d] / b_l[i]
            T.copy(b_o, s_o)
            for i in T.Parallel(BT):
                s_lse[i] = b_m[i] + T.log2(b_l[i])
            for i, d in T.Parallel(BT, V):
                if q_start + i < length + bos:
                    o[i_b, q_start + i, i_hq, d] = s_o[i, d]
            for i in T.Parallel(BT):
                if q_start + i < length + bos:
                    lse[i_b, q_start + i, i_hq] = s_lse[i]

    return main


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def parallel_attn_bwd_kernel(
    B: int,
    H: int,
    HQ: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    do_dtype: torch.dtype,
    scale: float,
    BT: int,
    USE_G: bool,
    USE_WINDOW: bool,
    W: int,
    IS_VARLEN: bool,
    num_warps: int,
):
    accum_dtype = T.float32
    scale_log2 = scale * RCP_LN2
    G = HQ // H
    # pad shared buffers to satisfy WGMMA/MMA inner-dimension alignment
    BK = tilelang.cdiv(K, 16) * 16
    BV = tilelang.cdiv(V, 16) * 16
    seq_len = T.dynamic("seq_len")
    num_chunks = T.dynamic("num_chunks") if IS_VARLEN else 0
    num_offsets = T.dynamic("num_offsets") if IS_VARLEN else 0

    @T.prim_func
    def main(
        q: T.Tensor((B, seq_len, HQ, K), dtype),
        k: T.Tensor((B, seq_len, H, K), dtype),
        v: T.Tensor((B, seq_len, H, V), dtype),
        g: T.Tensor((B, seq_len, HQ), accum_dtype),
        lse: T.Tensor((B, seq_len, HQ), accum_dtype),
        delta: T.Tensor((B, seq_len, HQ), accum_dtype),
        do: T.Tensor((B, seq_len, HQ, V), do_dtype),
        dq: T.Tensor((B, seq_len, HQ, K), accum_dtype),
        dk: T.Tensor((B, seq_len, H, K), accum_dtype),
        dv: T.Tensor((B, seq_len, H, V), accum_dtype),
        dg: T.Tensor((B, seq_len, HQ), accum_dtype),
        cu_seqlens: T.Tensor((num_offsets,), T.int32),
        chunk_indices: T.Tensor((num_chunks, 2), T.int32),
    ):
        with T.Kernel(HQ, num_chunks if IS_VARLEN else T.ceildiv(seq_len, BT), B, threads=num_warps * 32) as (i_hq, i_t, i_b):
            if IS_VARLEN:
                i_n = chunk_indices[i_t, 0]
                bos = cu_seqlens[i_n]
                length = cu_seqlens[i_n + 1] - bos
                k_start = bos + chunk_indices[i_t, 1] * BT
            else:
                bos = 0
                length = seq_len
                k_start = i_t * BT
            i_h = i_hq // G

            s_k = T.alloc_shared((BT, BK), dtype)
            s_v = T.alloc_shared((BT, BV), dtype)
            T.copy(k[i_b, k_start:k_start + BT, i_h, :], s_k)
            T.copy(v[i_b, k_start:k_start + BT, i_h, :], s_v)

            if USE_G:
                s_gk = T.alloc_shared((BT,), accum_dtype)
                T.copy(g[i_b, k_start:k_start + BT, i_hq], s_gk)

            b_p = T.alloc_fragment((BT, BT), accum_dtype)
            b_p_cast = T.alloc_fragment((BT, BT), dtype)
            b_ds = T.alloc_fragment((BT, BT), accum_dtype)
            b_ds_cast = T.alloc_fragment((BT, BT), dtype)

            b_dv = T.alloc_fragment((BT, BV), accum_dtype)
            b_dk = T.alloc_fragment((BT, BK), accum_dtype)
            T.clear(b_dv)
            T.clear(b_dk)

            if USE_G:
                b_dgk = T.alloc_fragment((BT,), accum_dtype)
                T.clear(b_dgk)

            s_q = T.alloc_shared((BT, BK), dtype)
            s_do = T.alloc_shared((BT, BV), dtype)
            s_lse = T.alloc_shared((BT,), accum_dtype)
            s_delta = T.alloc_shared((BT,), accum_dtype)
            s_ds = T.alloc_shared((BT, BT), dtype)
            b_dq = T.alloc_fragment((BT, BK), accum_dtype)
            if USE_G:
                s_gq = T.alloc_shared((BT,), accum_dtype)
                b_dgq = T.alloc_fragment((BT,), accum_dtype)
                b_dgk_tile = T.alloc_fragment((BT,), accum_dtype)

            i_t_local = (k_start - bos) // BT
            loop_start = i_t_local
            if USE_WINDOW:
                # include the last query that can attend to this key tile
                window_end = i_t_local + 1 + (W + BT - 2) // BT
                loop_end = T.min(T.ceildiv(length, BT), window_end)
            else:
                loop_end = T.ceildiv(length, BT)

            # pipelining with atomic dq updates corrupts results for non-power-of-two head dimensions
            for i_s in T.serial(loop_start, loop_end):
                q_start = bos + i_s * BT

                T.copy(q[i_b, q_start:q_start + BT, i_hq, :], s_q)
                T.copy(do[i_b, q_start:q_start + BT, i_hq, :], s_do)
                T.copy(lse[i_b, q_start:q_start + BT, i_hq], s_lse)
                T.copy(delta[i_b, q_start:q_start + BT, i_hq], s_delta)

                if USE_G:
                    T.copy(g[i_b, q_start:q_start + BT, i_hq], s_gq)

                T.clear(b_p)
                T.gemm(s_k, s_q, b_p, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(BT, BT):
                    if USE_G:
                        b_p[i, j] = T.exp2(b_p[i, j] * scale_log2 + s_gq[j] - s_gk[i] - s_lse[j])
                    else:
                        b_p[i, j] = T.exp2(b_p[i, j] * scale_log2 - s_lse[j])

                for i, j in T.Parallel(BT, BT):
                    k_pos = k_start + i
                    q_pos = q_start + j
                    causal = (k_pos <= q_pos) & (k_pos < length + bos) & (q_pos < length + bos)
                    if USE_WINDOW:
                        mask = causal & (q_pos - k_pos < W)
                    else:
                        mask = causal
                    b_p[i, j] = T.if_then_else(mask, b_p[i, j], T.cast(0, accum_dtype))

                T.copy(b_p, b_p_cast)
                T.gemm(b_p_cast, s_do, b_dv, policy=T.GemmWarpPolicy.FullRow)

                T.clear(b_ds)
                T.gemm(s_v, s_do, b_ds, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(BT, BT):
                    b_ds[i, j] = b_p[i, j] * (b_ds[i, j] - s_delta[j])
                for i, j in T.Parallel(BT, BT):
                    b_ds_cast[i, j] = b_ds[i, j] * scale

                T.gemm(b_ds_cast, s_q, b_dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(b_ds_cast, s_ds)
                T.clear(b_dq)
                T.gemm(s_ds, s_k, b_dq, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                for i, d in T.Parallel(BT, K):
                    if q_start + i < length + bos:
                        T.atomic_add(dq[i_b, q_start + i, i_hq, d], b_dq[i, d])

                if USE_G:
                    T.reduce_sum(b_ds, b_dgq, dim=0)
                    for j in T.Parallel(BT):
                        if q_start + j < length + bos:
                            T.atomic_add(dg[i_b, q_start + j, i_hq], b_dgq[j])

                    T.reduce_sum(b_ds, b_dgk_tile, dim=1)
                    for i in T.Parallel(BT):
                        b_dgk[i] += b_dgk_tile[i]

            # grouped query heads share KV heads, so their gradients require atomic updates
            for i, d in T.Parallel(BT, V):
                if k_start + i < length + bos:
                    T.atomic_add(dv[i_b, k_start + i, i_h, d], b_dv[i, d])
            for i, d in T.Parallel(BT, K):
                if k_start + i < length + bos:
                    T.atomic_add(dk[i_b, k_start + i, i_h, d], b_dk[i, d])

            if USE_G:
                for i in T.Parallel(BT):
                    if k_start + i < length + bos:
                        T.atomic_add(dg[i_b, k_start + i, i_hq], -b_dgk[i])

    return main


def parallel_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor | None,
    sink_bias: torch.Tensor | None,
    scale: float,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[-1]

    USE_G = g_cumsum is not None
    USE_WINDOW = window_size is not None
    USE_SINK = sink_bias is not None
    IS_VARLEN = cu_seqlens is not None

    is_hopper_plus = torch.cuda.get_device_capability()[0] >= 9
    if is_hopper_plus:
        BT, BS, num_warps = (128, 64, 8) if max(K, V) <= 64 else (64, 64, 4)
    else:
        BT, BS, num_warps = (128, 64, 4) if max(K, V) <= 128 else (64, 32, 4)

    if chunk_indices is None and IS_VARLEN:
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT)

    scale = scale if scale is not None else K ** -0.5

    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported dtype {q.dtype} for TileLang backend")

    o = torch.empty(B, T, HQ, V, dtype=output_dtype or q.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float32, device=q.device)

    g = g_cumsum.float() if USE_G else None
    sink_bias = sink_bias.float() if USE_SINK else None

    kernel = parallel_attn_fwd_kernel(
        B=B,
        H=H,
        HQ=HQ,
        K=K,
        V=V,
        dtype=q.dtype,
        output_dtype=o.dtype,
        scale=scale,
        BT=BT,
        BS=BS,
        USE_G=USE_G,
        USE_SINK=USE_SINK,
        USE_WINDOW=USE_WINDOW,
        W=window_size or 0,
        IS_VARLEN=IS_VARLEN,
        num_warps=num_warps,
        num_stages=0 if USE_WINDOW else 1,
    )

    kernel(
        q, k, v, g, sink_bias, o, lse,
        cu_seqlens.int() if IS_VARLEN else None,
        chunk_indices.int() if IS_VARLEN else None,
    )

    return o, lse


def parallel_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    g_cumsum: torch.Tensor | None,
    lse: torch.Tensor,
    do: torch.Tensor,
    sink_bias: torch.Tensor | None = None,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[-1]
    BT = 64
    scale = scale if scale is not None else K ** -0.5

    USE_G = g_cumsum is not None
    USE_WINDOW = window_size is not None
    IS_VARLEN = cu_seqlens is not None

    if chunk_indices is None and IS_VARLEN:
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT)

    delta = parallel_attn_bwd_preprocess(o=o, do=do)

    # zero-initialized fp32 outputs accumulate atomic updates across heads and key tiles
    dq = torch.zeros(B, T, HQ, K, dtype=torch.float32, device=q.device)
    dk = torch.zeros(B, T, H, K, dtype=torch.float32, device=k.device)
    dv = torch.zeros(B, T, H, V, dtype=torch.float32, device=v.device)
    dg = torch.zeros(B, T, HQ, dtype=torch.float32, device=q.device) if USE_G else None

    g = g_cumsum.float() if USE_G else None

    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported dtype {q.dtype} for TileLang backend")

    kernel = parallel_attn_bwd_kernel(
        B=B,
        H=H,
        HQ=HQ,
        K=K,
        V=V,
        dtype=q.dtype,
        do_dtype=do.dtype,
        scale=scale,
        BT=BT,
        USE_G=USE_G,
        USE_WINDOW=USE_WINDOW,
        W=window_size or 0,
        IS_VARLEN=IS_VARLEN,
        num_warps=4,
    )

    kernel(
        q, k, v, g, lse, delta, do, dq, dk, dv, dg,
        cu_seqlens.int() if IS_VARLEN else None,
        chunk_indices.int() if IS_VARLEN else None,
    )

    # saved LSE includes the sink mass in the softmax normalizer
    dsink_bias = None
    if sink_bias is not None:
        p_sink = torch.exp2(sink_bias.float()[None, None, :] - lse.float())
        dsink_bias = -(p_sink * delta.float()).sum((0, 1))

    dq = dq.to(q.dtype)
    dk = dk.to(k.dtype)
    dv = dv.to(v.dtype)
    return dq, dk, dv, dg, dsink_bias
