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

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2


def _pick_tile(K: int, V: int, is_hopper_plus: bool) -> tuple[int, int, int]:
    """Return (BT, BS, num_warps) tuned for head dim."""
    if is_hopper_plus:
        if max(K, V) <= 64:
            return 128, 64, 8
        if max(K, V) <= 128:
            return 64, 64, 4
        return 64, 64, 4
    # tune pre-Hopper devices separately
    if max(K, V) <= 64:
        return 128, 64, 4
    if max(K, V) <= 128:
        return 128, 64, 4
    return 64, 32, 4


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def _build_parallel_attn_fwd_kernel(
    B,
    HQ,
    H,
    K,
    V,
    BT,
    BS,
    sm_scale,
    log2e_scale,
    dtype_str,
    USE_G=False,
    USE_WINDOW=False,
    WINDOW_SIZE=0,
    USE_SINK=False,
    IS_VARLEN=False,
    num_warps=8,
    num_stages=1,
):
    """Build a forward kernel specialized to the batch size and head dimensions."""
    dtype_map = {'float16': T.float16, 'bfloat16': T.bfloat16, 'float32': T.float32}
    dtype = dtype_map[dtype_str]
    accum_dtype = T.float32
    threads = num_warps * 32
    # pad shared buffers to satisfy WGMMA/MMA inner-dimension alignment
    BK = (K + 15) // 16 * 16
    BV = (V + 15) // 16 * 16
    G = HQ // H

    T_d, NT_d, Ncu_d = T.dynamic("T, NT, Ncu")

    q_s = (B, T_d, HQ, K)
    kv_s = (B, T_d, H, K)
    vv_s = (B, T_d, H, V)
    o_s = (B, T_d, HQ, V)
    lse_s = (B, T_d, HQ)
    sink_s = (HQ,)

    @T.macro
    def kernel_body(q, k, v, g, sink_bias, o, lse, i_b, i_h, i_t, t_s, T_seq, bos):
        i_hkv = i_h // G

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
            T.copy(g[i_b, t_s:t_s + BT, i_h], s_gq)

        T.copy(q[i_b, t_s:t_s + BT, i_h, :], s_q)
        T.fill(b_o, 0)
        T.fill(b_l, 0)
        T.fill(b_m, -T.infinity(accum_dtype))

        i_t_local = (t_s - bos) // BT
        if USE_WINDOW:
            # include every key visible to the first query in this tile
            s0_raw = i_t_local * BT - WINDOW_SIZE + 1
            s0 = T.max(s0_raw, 0)
            loop_st = (s0 // BS)
        else:
            loop_st = 0
        loop_ed_raw = T.ceildiv((i_t_local + 1) * BT, BS)
        loop_ed = T.min(loop_ed_raw, T.ceildiv(T_seq, BS))

        for k_idx in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
            k_t_s = bos + k_idx * BS

            T.copy(k[i_b, k_t_s:k_t_s + BS, i_hkv, :], s_k)
            T.copy(v[i_b, k_t_s:k_t_s + BS, i_hkv, :], s_v)
            if USE_G:
                T.copy(g[i_b, k_t_s:k_t_s + BS, i_h], s_gk)

            T.clear(b_s)
            T.gemm(s_q, s_k, b_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(BT, BS):
                q_pos = t_s + i
                key_pos = k_t_s + j
                causal = (key_pos <= q_pos) & (key_pos < T_seq + bos) & (q_pos < T_seq + bos)
                if USE_WINDOW:
                    mask = causal & (q_pos - key_pos < WINDOW_SIZE)
                else:
                    mask = causal
                if USE_G:
                    s = b_s[i, j] * log2e_scale + s_gq[i] - s_gk[j]
                else:
                    s = b_s[i, j] * log2e_scale
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
                b_scale[i] = T.if_then_else(b_m_prev[i] == -T.infinity(accum_dtype), 0., T.exp2(b_m_prev[i] - b_m_stable[i]))
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
                b_l[i] += T.exp2(sink_bias[i_h] - b_m[i])

        s_o = T.alloc_shared((BT, BV), dtype)
        s_lse = T.alloc_shared((BT,), accum_dtype)
        for i, d in T.Parallel(BT, BV):
            b_o[i, d] = b_o[i, d] / b_l[i]
        T.copy(b_o, s_o)
        for i in T.Parallel(BT):
            s_lse[i] = b_m[i] + T.log2(b_l[i])
        for i, d in T.Parallel(BT, V):
            if t_s + i < T_seq + bos:
                o[i_b, t_s + i, i_h, d] = s_o[i, d]
        for i in T.Parallel(BT):
            if t_s + i < T_seq + bos:
                lse[i_b, t_s + i, i_h] = s_lse[i]

    if IS_VARLEN:
        @T.prim_func
        def kernel(
            q: T.Tensor(q_s, dtype),
            k: T.Tensor(kv_s, dtype),
            v: T.Tensor(vv_s, dtype),
            g: T.Tensor(lse_s, accum_dtype),
            sink_bias: T.Tensor(sink_s, accum_dtype),
            o: T.Tensor(o_s, dtype),
            lse: T.Tensor(lse_s, accum_dtype),
            cu_seqlens: T.Tensor((Ncu_d,), T.int32),
            chunk_indices: T.Tensor((NT_d, 2), T.int32),
        ):
            with T.Kernel(HQ, NT_d, 1, threads=threads) as (i_h, i_t, _):
                i_n = chunk_indices[i_t, 0]
                i_t_local = chunk_indices[i_t, 1]
                bos = cu_seqlens[i_n]
                eos = cu_seqlens[i_n + 1]
                T_seq = eos - bos
                t_s = bos + i_t_local * BT
                kernel_body(q, k, v, g, sink_bias, o, lse, 0, i_h, i_t, t_s, T_seq, bos)
    else:
        @T.prim_func
        def kernel(
            q: T.Tensor(q_s, dtype),
            k: T.Tensor(kv_s, dtype),
            v: T.Tensor(vv_s, dtype),
            g: T.Tensor(lse_s, accum_dtype),
            sink_bias: T.Tensor(sink_s, accum_dtype),
            o: T.Tensor(o_s, dtype),
            lse: T.Tensor(lse_s, accum_dtype),
        ):
            with T.Kernel(HQ, T.ceildiv(T_d, BT), B, threads=threads) as (i_h, i_t, i_b):
                t_s = i_t * BT
                kernel_body(q, k, v, g, sink_bias, o, lse, i_b, i_h, i_t, t_s, T_d, 0)

    return kernel


def parallel_attn_fwd_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor | None,
    sink_bias: torch.Tensor | None,
    scale: float,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[-1]

    USE_G = g_cumsum is not None
    USE_WINDOW = window_size is not None
    USE_SINK = sink_bias is not None
    IS_VARLEN = cu_seqlens is not None

    is_hopper_plus = torch.cuda.get_device_capability()[0] >= 9
    BT, BS, num_warps = _pick_tile(K=K, V=V, is_hopper_plus=is_hopper_plus)

    if chunk_indices is None and IS_VARLEN:
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT)

    sm_scale = scale if scale is not None else K ** -0.5
    log2e_scale = sm_scale * RCP_LN2

    dtype_str = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}.get(q.dtype)
    if dtype_str is None:
        raise ValueError(f"Unsupported dtype {q.dtype} for TileLang backend")

    o = torch.empty(B, T, HQ, V, dtype=q.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float32, device=q.device)

    g_kern = g_cumsum.float() if USE_G else torch.zeros(B, T, HQ, dtype=torch.float32, device=q.device)
    sink_kern = sink_bias.float() if USE_SINK else torch.zeros(HQ, dtype=torch.float32, device=q.device)

    kernel = _build_parallel_attn_fwd_kernel(
        B=B,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        sm_scale=sm_scale,
        log2e_scale=log2e_scale,
        dtype_str=dtype_str,
        USE_G=USE_G,
        USE_WINDOW=USE_WINDOW,
        WINDOW_SIZE=window_size or 0,
        USE_SINK=USE_SINK,
        IS_VARLEN=IS_VARLEN,
        num_warps=num_warps,
        num_stages=1 if USE_WINDOW else 2,
    )

    if IS_VARLEN:
        kernel(q, k, v, g_kern, sink_kern, o, lse, cu_seqlens.int(), chunk_indices.int())
    else:
        kernel(q, k, v, g_kern, sink_kern, o, lse)

    return o, lse


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def _build_parallel_attn_bwd_kernel(
    B,
    HQ,
    H,
    K,
    V,
    BT,
    sm_scale,
    log2e_scale,
    dtype_str,
    USE_G=False,
    USE_WINDOW=False,
    WINDOW_SIZE=0,
    IS_VARLEN=False,
    num_warps=4,
):
    """Build a backward kernel specialized to the batch size and head dimensions."""
    dtype_map = {'float16': T.float16, 'bfloat16': T.bfloat16, 'float32': T.float32}
    dtype = dtype_map[dtype_str]
    accum_dtype = T.float32
    threads = num_warps * 32
    # pad shared buffers to satisfy WGMMA/MMA inner-dimension alignment
    BK = (K + 15) // 16 * 16
    BV = (V + 15) // 16 * 16
    G = HQ // H

    T_d, NT_d, Ncu_d = T.dynamic("T, NT, Ncu")

    q_s = (B, T_d, HQ, K)
    kv_s = (B, T_d, H, K)
    vv_s = (B, T_d, H, V)
    lse_s = (B, T_d, HQ)
    dq_s = (B, T_d, HQ, K)
    dkv_s = (B, T_d, H, K)
    dvv_s = (B, T_d, H, V)

    @T.macro
    def kernel_body(q, k, v, g, lse, delta, do, dq_out, dk_out, dv_out, dg, i_b, i_h, i_t, t_s, T_seq, bos):
        i_hkv = i_h // G

        s_k = T.alloc_shared((BT, BK), dtype)
        s_v = T.alloc_shared((BT, BV), dtype)
        T.copy(k[i_b, t_s:t_s + BT, i_hkv, :], s_k)
        T.copy(v[i_b, t_s:t_s + BT, i_hkv, :], s_v)

        if USE_G:
            s_gk = T.alloc_shared((BT,), accum_dtype)
            T.copy(g[i_b, t_s:t_s + BT, i_h], s_gk)

        b_p_t = T.alloc_fragment((BT, BT), accum_dtype)
        b_p_t_cast = T.alloc_fragment((BT, BT), dtype)
        b_ds_t = T.alloc_fragment((BT, BT), accum_dtype)
        b_ds_t_cast = T.alloc_fragment((BT, BT), dtype)
        b_ds = T.alloc_fragment((BT, BT), dtype)

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
        s_ds_t = T.alloc_shared((BT, BT), dtype)
        b_dq = T.alloc_fragment((BT, BK), accum_dtype)
        if USE_G:
            s_gq = T.alloc_shared((BT,), accum_dtype)
            b_dgq = T.alloc_fragment((BT,), accum_dtype)
            b_dgk_tile = T.alloc_fragment((BT,), accum_dtype)

        i_t_local = (t_s - bos) // BT
        loop_st = i_t_local
        if USE_WINDOW:
            # include the last query that can attend to this key tile
            loop_ed_swa = i_t_local + 1 + (WINDOW_SIZE + BT - 2) // BT
            loop_ed = T.min(T.ceildiv(T_seq, BT), loop_ed_swa)
        else:
            loop_ed = T.ceildiv(T_seq, BT)

        # pipelining with atomic dq updates corrupts results for non-power-of-two head dimensions
        for k_local in T.serial(loop_st, loop_ed):
            q_t_s = bos + k_local * BT

            T.copy(q[i_b, q_t_s:q_t_s + BT, i_h, :], s_q)
            T.copy(do[i_b, q_t_s:q_t_s + BT, i_h, :], s_do)
            T.copy(lse[i_b, q_t_s:q_t_s + BT, i_h], s_lse)
            T.copy(delta[i_b, q_t_s:q_t_s + BT, i_h], s_delta)

            if USE_G:
                T.copy(g[i_b, q_t_s:q_t_s + BT, i_h], s_gq)

            T.clear(b_p_t)
            T.gemm(s_k, s_q, b_p_t, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(BT, BT):
                if USE_G:
                    b_p_t[i, j] = T.exp2(b_p_t[i, j] * log2e_scale + s_gq[j] - s_gk[i] - s_lse[j])
                else:
                    b_p_t[i, j] = T.exp2(b_p_t[i, j] * log2e_scale - s_lse[j])

            for i, j in T.Parallel(BT, BT):
                key_pos = t_s + i
                q_pos = q_t_s + j
                causal = (key_pos <= q_pos) & (key_pos < T_seq + bos) & (q_pos < T_seq + bos)
                if USE_WINDOW:
                    in_window = (q_pos - key_pos < WINDOW_SIZE)
                    mask = causal & in_window
                else:
                    mask = causal
                b_p_t[i, j] = T.if_then_else(mask, b_p_t[i, j], T.cast(0, accum_dtype))

            T.copy(b_p_t, b_p_t_cast)
            T.gemm(b_p_t_cast, s_do, b_dv, policy=T.GemmWarpPolicy.FullRow)

            T.clear(b_ds_t)
            T.gemm(s_v, s_do, b_ds_t, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(BT, BT):
                b_ds_t[i, j] = b_p_t[i, j] * (b_ds_t[i, j] - s_delta[j])
            for i, j in T.Parallel(BT, BT):
                b_ds_t_cast[i, j] = b_ds_t[i, j] * sm_scale

            T.copy(b_ds_t_cast, b_ds)
            T.gemm(b_ds, s_q, b_dk, policy=T.GemmWarpPolicy.FullRow)

            T.copy(b_ds_t_cast, s_ds_t)
            T.clear(b_dq)
            T.gemm(s_ds_t, s_k, b_dq, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
            for i, d in T.Parallel(BT, K):
                if q_t_s + i < T_seq + bos:
                    T.atomic_add(dq_out[i_b, q_t_s + i, i_h, d], b_dq[i, d])

            if USE_G:
                T.reduce_sum(b_ds_t, b_dgq, dim=0)
                for j in T.Parallel(BT):
                    if q_t_s + j < T_seq + bos:
                        T.atomic_add(dg[i_b, q_t_s + j, i_h], b_dgq[j])

                T.reduce_sum(b_ds_t, b_dgk_tile, dim=1)
                for i in T.Parallel(BT):
                    b_dgk[i] += b_dgk_tile[i]

        # grouped query heads share KV heads, so their gradients require atomic updates
        for i, d in T.Parallel(BT, V):
            if t_s + i < T_seq + bos:
                T.atomic_add(dv_out[i_b, t_s + i, i_hkv, d], b_dv[i, d])
        for i, d in T.Parallel(BT, K):
            if t_s + i < T_seq + bos:
                T.atomic_add(dk_out[i_b, t_s + i, i_hkv, d], b_dk[i, d])

        if USE_G:
            for i in T.Parallel(BT):
                if t_s + i < T_seq + bos:
                    T.atomic_add(dg[i_b, t_s + i, i_h], -b_dgk[i])

    if IS_VARLEN:
        @T.prim_func
        def kernel(
            q: T.Tensor(q_s, dtype),
            k: T.Tensor(kv_s, dtype),
            v: T.Tensor(vv_s, dtype),
            g: T.Tensor(lse_s, accum_dtype),
            lse: T.Tensor(lse_s, accum_dtype),
            delta: T.Tensor(lse_s, accum_dtype),
            do: T.Tensor(q_s, dtype),
            dq_out: T.Tensor(dq_s, accum_dtype),
            dk_out: T.Tensor(dkv_s, accum_dtype),
            dv_out: T.Tensor(dvv_s, accum_dtype),
            dg: T.Tensor(lse_s, accum_dtype),
            cu_seqlens: T.Tensor((Ncu_d,), T.int32),
            chunk_indices: T.Tensor((NT_d, 2), T.int32),
        ):
            with T.Kernel(HQ, NT_d, 1, threads=threads) as (i_h, i_t, _):
                i_n = chunk_indices[i_t, 0]
                i_t_local = chunk_indices[i_t, 1]
                bos = cu_seqlens[i_n]
                T_seq = cu_seqlens[i_n + 1] - bos
                t_s = bos + i_t_local * BT
                kernel_body(q, k, v, g, lse, delta, do, dq_out, dk_out, dv_out, dg, 0, i_h, i_t, t_s, T_seq, bos)
    else:
        @T.prim_func
        def kernel(
            q: T.Tensor(q_s, dtype),
            k: T.Tensor(kv_s, dtype),
            v: T.Tensor(vv_s, dtype),
            g: T.Tensor(lse_s, accum_dtype),
            lse: T.Tensor(lse_s, accum_dtype),
            delta: T.Tensor(lse_s, accum_dtype),
            do: T.Tensor(q_s, dtype),
            dq_out: T.Tensor(dq_s, accum_dtype),
            dk_out: T.Tensor(dkv_s, accum_dtype),
            dv_out: T.Tensor(dvv_s, accum_dtype),
            dg: T.Tensor(lse_s, accum_dtype),
        ):
            with T.Kernel(HQ, T.ceildiv(T_d, BT), B, threads=threads) as (i_h, i_t, i_b):
                t_s = i_t * BT
                kernel_body(q, k, v, g, lse, delta, do, dq_out, dk_out, dv_out, dg, i_b, i_h, i_t, t_s, T_d, 0)

    return kernel


def parallel_attn_bwd_tilelang(
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
    chunk_size: int = 128,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[-1]
    BT = 64
    sm_scale = scale if scale is not None else K ** -0.5
    log2e_scale = sm_scale * RCP_LN2

    USE_G = g_cumsum is not None
    USE_WINDOW = window_size is not None
    IS_VARLEN = cu_seqlens is not None

    if chunk_indices is None and IS_VARLEN:
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT)

    # reuse the Triton delta calculation for unpadded tensors
    from fla.ops.attn.parallel import parallel_attn_bwd_preprocess
    delta = parallel_attn_bwd_preprocess(o=o, do=do)

    # zero-initialized fp32 outputs accumulate atomic updates across heads and key tiles
    dq = torch.zeros(B, T, HQ, K, dtype=torch.float32, device=q.device)
    dk = torch.zeros(B, T, H, K, dtype=torch.float32, device=k.device)
    dv = torch.zeros(B, T, H, V, dtype=torch.float32, device=v.device)
    dg = torch.zeros(B, T, HQ, dtype=torch.float32, device=q.device) if USE_G else None

    g_kern = g_cumsum.float() if USE_G else torch.zeros(B, T, HQ, dtype=torch.float32, device=q.device)
    dg_kern = dg if USE_G else torch.zeros(B, T, HQ, dtype=torch.float32, device=q.device)

    dtype_str = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}.get(q.dtype)
    if dtype_str is None:
        raise ValueError(f"Unsupported dtype {q.dtype} for TileLang backend")

    kernel = _build_parallel_attn_bwd_kernel(
        B=B,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        sm_scale=sm_scale,
        log2e_scale=log2e_scale,
        dtype_str=dtype_str,
        USE_G=USE_G,
        USE_WINDOW=USE_WINDOW,
        WINDOW_SIZE=window_size or 0,
        IS_VARLEN=IS_VARLEN,
    )

    if IS_VARLEN:
        kernel(q, k, v, g_kern, lse, delta, do, dq, dk, dv, dg_kern, cu_seqlens.int(), chunk_indices.int())
    else:
        kernel(q, k, v, g_kern, lse, delta, do, dq, dk, dv, dg_kern)

    # saved LSE includes the sink mass in the softmax normalizer
    dsink_bias = None
    if sink_bias is not None:
        p_sink = torch.exp2(sink_bias.float()[None, None, :] - lse.float())
        dsink_bias = -(p_sink * delta.float()).sum((0, 1))

    dq = dq.to(q.dtype)
    dk = dk.to(k.dtype)
    dv = dv.to(v.dtype)
    return dq, dk, dv, dg, dsink_bias
