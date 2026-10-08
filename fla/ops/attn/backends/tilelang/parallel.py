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
    _dtype = dtype_map[dtype_str]
    accum_dtype = T.float32
    threads = num_warps * 32
    _BT, _BS = BT, BS
    _B, _HQ, _H = B, HQ, H
    _K_orig, _V_orig = K, V
    # pad shared buffers to satisfy WGMMA/MMA inner-dimension alignment
    _K = (K + 15) // 16 * 16
    _V = (V + 15) // 16 * 16
    _G = HQ // H
    _USE_G = USE_G
    _USE_WINDOW = USE_WINDOW
    _USE_SINK = USE_SINK
    _W = WINDOW_SIZE
    _sm_scale = sm_scale
    _log2e_scale = log2e_scale

    T_d, NT_d, Ncu_d = T.dynamic("T, NT, Ncu")

    q_s = (_B, T_d, _HQ, _K_orig)
    kv_s = (_B, T_d, _H, _K_orig)
    vv_s = (_B, T_d, _H, _V_orig)
    o_s = (_B, T_d, _HQ, _V_orig)
    lse_s = (_B, T_d, _HQ)
    sink_s = (_HQ,)

    @T.macro
    def kernel_body(q, k, v, g, sink_bias, o, lse, i_b, i_h, i_t, t_s, T_seq, bos):
        i_hkv = i_h // _G

        Q_shared = T.alloc_shared((_BT, _K), _dtype)
        K_shared = T.alloc_shared((_BS, _K), _dtype)
        V_shared = T.alloc_shared((_BS, _V), _dtype)

        acc_s = T.alloc_fragment((_BT, _BS), accum_dtype)
        acc_s_cast = T.alloc_fragment((_BT, _BS), _dtype)
        acc_o = T.alloc_fragment((_BT, _V), accum_dtype)

        m_f = T.alloc_fragment((_BT,), accum_dtype)
        m_prev = T.alloc_fragment((_BT,), accum_dtype)
        l_f = T.alloc_fragment((_BT,), accum_dtype)
        p_sum = T.alloc_fragment((_BT,), accum_dtype)
        scale_row = T.alloc_fragment((_BT,), accum_dtype)

        if _USE_G:
            g_q_shared = T.alloc_shared((_BT,), accum_dtype)
            g_k_shared = T.alloc_shared((_BS,), accum_dtype)
            T.copy(g[i_b, t_s:t_s + _BT, i_h], g_q_shared)

        T.copy(q[i_b, t_s:t_s + _BT, i_h, :], Q_shared)
        T.fill(acc_o, 0)
        T.fill(l_f, 0)
        T.fill(m_f, -T.infinity(accum_dtype))

        i_t_local = (t_s - bos) // _BT
        if _USE_WINDOW:
            # include every key visible to the first query in this tile
            s0_raw = i_t_local * _BT - _W + 1
            s0 = T.max(s0_raw, 0)
            loop_st = (s0 // _BS)
        else:
            loop_st = 0
        loop_ed_raw = T.ceildiv((i_t_local + 1) * _BT, _BS)
        loop_ed = T.min(loop_ed_raw, T.ceildiv(T_seq, _BS))

        for k_idx in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
            k_t_s = bos + k_idx * _BS

            T.copy(k[i_b, k_t_s:k_t_s + _BS, i_hkv, :], K_shared)
            T.copy(v[i_b, k_t_s:k_t_s + _BS, i_hkv, :], V_shared)
            if _USE_G:
                T.copy(g[i_b, k_t_s:k_t_s + _BS, i_h], g_k_shared)

            T.clear(acc_s)
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(_BT, _BS):
                q_pos = t_s + i
                key_pos = k_t_s + j
                causal = (key_pos <= q_pos) & (key_pos < T_seq + bos) & (q_pos < T_seq + bos)
                if _USE_WINDOW:
                    mask = causal & (q_pos - key_pos < _W)
                else:
                    mask = causal
                if _USE_G:
                    s = acc_s[i, j] * _log2e_scale + g_q_shared[i] - g_k_shared[j]
                else:
                    s = acc_s[i, j] * _log2e_scale
                acc_s[i, j] = T.if_then_else(mask, s, -T.infinity(accum_dtype))

            T.copy(m_f, m_prev)
            T.reduce_max(acc_s, m_f, dim=1, clear=False)
            for i in T.Parallel(_BT):
                m_f[i] = T.max(m_f[i], m_prev[i])
            # use a finite pivot for all-masked rows to avoid (-inf) - (-inf)
            m_stable = T.alloc_fragment((_BT,), accum_dtype)
            for i in T.Parallel(_BT):
                m_stable[i] = T.if_then_else(m_f[i] == -T.infinity(accum_dtype), 0., m_f[i])
            for i in T.Parallel(_BT):
                scale_row[i] = T.if_then_else(m_prev[i] == -T.infinity(accum_dtype), 0., T.exp2(m_prev[i] - m_stable[i]))
            for i, j in T.Parallel(_BT, _V):
                acc_o[i, j] *= scale_row[i]
            for i, j in T.Parallel(_BT, _BS):
                acc_s[i, j] = T.exp2(acc_s[i, j] - m_stable[i])
            T.reduce_sum(acc_s, p_sum, dim=1)
            for i in T.Parallel(_BT):
                l_f[i] = l_f[i] * scale_row[i] + p_sum[i]

            T.copy(acc_s, acc_s_cast)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        # sink-bias contributes to the normalizer but not to the value matmul
        if _USE_SINK:
            # keep sink_bias - m_f finite when the row has no valid keys
            for i in T.Parallel(_BT):
                m_f[i] = T.if_then_else(m_f[i] == -T.infinity(accum_dtype), 0., m_f[i])
            for i in T.Parallel(_BT):
                l_f[i] += T.exp2(sink_bias[i_h] - m_f[i])

        O_shared = T.alloc_shared((_BT, _V), _dtype)
        lse_shared = T.alloc_shared((_BT,), accum_dtype)
        for i, d in T.Parallel(_BT, _V):
            acc_o[i, d] = acc_o[i, d] / l_f[i]
        T.copy(acc_o, O_shared)
        for i in T.Parallel(_BT):
            lse_shared[i] = m_f[i] + T.log2(l_f[i])
        for i, d in T.Parallel(_BT, _V_orig):
            if t_s + i < T_seq + bos:
                o[i_b, t_s + i, i_h, d] = O_shared[i, d]
        for i in T.Parallel(_BT):
            if t_s + i < T_seq + bos:
                lse[i_b, t_s + i, i_h] = lse_shared[i]

    if IS_VARLEN:
        @T.prim_func
        def kernel(
            q: T.Tensor(q_s, _dtype),
            k: T.Tensor(kv_s, _dtype),
            v: T.Tensor(vv_s, _dtype),
            g: T.Tensor(lse_s, accum_dtype),
            sink_bias: T.Tensor(sink_s, accum_dtype),
            o: T.Tensor(o_s, _dtype),
            lse: T.Tensor(lse_s, accum_dtype),
            cu_seqlens: T.Tensor((Ncu_d,), T.int32),
            chunk_indices: T.Tensor((NT_d, 2), T.int32),
        ):
            with T.Kernel(_HQ, NT_d, 1, threads=threads) as (i_h, i_t, _):
                i_n = chunk_indices[i_t, 0]
                i_t_local = chunk_indices[i_t, 1]
                bos = cu_seqlens[i_n]
                eos = cu_seqlens[i_n + 1]
                T_seq = eos - bos
                t_s = bos + i_t_local * _BT
                kernel_body(q, k, v, g, sink_bias, o, lse, 0, i_h, i_t, t_s, T_seq, bos)
    else:
        @T.prim_func
        def kernel(
            q: T.Tensor(q_s, _dtype),
            k: T.Tensor(kv_s, _dtype),
            v: T.Tensor(vv_s, _dtype),
            g: T.Tensor(lse_s, accum_dtype),
            sink_bias: T.Tensor(sink_s, accum_dtype),
            o: T.Tensor(o_s, _dtype),
            lse: T.Tensor(lse_s, accum_dtype),
        ):
            with T.Kernel(_HQ, T.ceildiv(T_d, _BT), _B, threads=threads) as (i_h, i_t, i_b):
                t_s = i_t * _BT
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
):
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
    _dtype = dtype_map[dtype_str]
    accum_dtype = T.float32
    threads = num_warps * 32
    _BT = BT
    _B, _HQ, _H = B, HQ, H
    _K_orig, _V_orig = K, V
    # pad shared buffers to satisfy WGMMA/MMA inner-dimension alignment
    _K = (K + 15) // 16 * 16
    _V = (V + 15) // 16 * 16
    _G = HQ // H
    _USE_G = USE_G
    _USE_WINDOW = USE_WINDOW
    _W = WINDOW_SIZE
    _sm_scale = sm_scale
    _log2e_scale = log2e_scale

    T_d, NT_d, Ncu_d = T.dynamic("T, NT, Ncu")

    q_s = (_B, T_d, _HQ, _K_orig)
    kv_s = (_B, T_d, _H, _K_orig)
    vv_s = (_B, T_d, _H, _V_orig)
    lse_s = (_B, T_d, _HQ)
    dq_s = (_B, T_d, _HQ, _K_orig)
    dkv_s = (_B, T_d, _H, _K_orig)
    dvv_s = (_B, T_d, _H, _V_orig)

    @T.macro
    def kernel_body(q, k, v, g, lse, delta, do, dq_out, dk_out, dv_out, dg, i_b, i_h, i_t, t_s, T_seq, bos):
        i_hkv = i_h // _G

        K_shared = T.alloc_shared((_BT, _K), _dtype)
        V_shared = T.alloc_shared((_BT, _V), _dtype)
        T.copy(k[i_b, t_s:t_s + _BT, i_hkv, :], K_shared)
        T.copy(v[i_b, t_s:t_s + _BT, i_hkv, :], V_shared)

        if _USE_G:
            g_k_shared = T.alloc_shared((_BT,), accum_dtype)
            T.copy(g[i_b, t_s:t_s + _BT, i_h], g_k_shared)

        qkT = T.alloc_fragment((_BT, _BT), accum_dtype)
        qkT_cast = T.alloc_fragment((_BT, _BT), _dtype)
        dsT = T.alloc_fragment((_BT, _BT), accum_dtype)
        dsT_cast = T.alloc_fragment((_BT, _BT), _dtype)
        ds = T.alloc_fragment((_BT, _BT), _dtype)

        dv_local = T.alloc_fragment((_BT, _V), accum_dtype)
        dk_local = T.alloc_fragment((_BT, _K), accum_dtype)
        T.clear(dv_local)
        T.clear(dk_local)

        if _USE_G:
            dg_k_acc = T.alloc_fragment((_BT,), accum_dtype)
            T.clear(dg_k_acc)

        # reuse these buffers across query tiles
        q_shared = T.alloc_shared((_BT, _K), _dtype)
        do_shared = T.alloc_shared((_BT, _V), _dtype)
        lse_shared = T.alloc_shared((_BT,), accum_dtype)
        delta_shared = T.alloc_shared((_BT,), accum_dtype)
        dsT_shared = T.alloc_shared((_BT, _BT), _dtype)
        dq_local = T.alloc_fragment((_BT, _K), accum_dtype)
        if _USE_G:
            g_q_shared = T.alloc_shared((_BT,), accum_dtype)
            f_dg_q = T.alloc_fragment((_BT,), accum_dtype)
            f_dg_k = T.alloc_fragment((_BT,), accum_dtype)

        i_t_local = (t_s - bos) // _BT
        loop_st = i_t_local
        if _USE_WINDOW:
            # include the last query that can attend to this key tile
            loop_ed_swa = i_t_local + 1 + (_W + _BT - 2) // _BT
            loop_ed = T.min(T.ceildiv(T_seq, _BT), loop_ed_swa)
        else:
            loop_ed = T.ceildiv(T_seq, _BT)

        # pipelining with atomic dq updates corrupts results for non-power-of-two head dimensions
        for k_local in T.serial(loop_st, loop_ed):
            q_t_s = bos + k_local * _BT

            T.copy(q[i_b, q_t_s:q_t_s + _BT, i_h, :], q_shared)
            T.copy(do[i_b, q_t_s:q_t_s + _BT, i_h, :], do_shared)
            T.copy(lse[i_b, q_t_s:q_t_s + _BT, i_h], lse_shared)
            T.copy(delta[i_b, q_t_s:q_t_s + _BT, i_h], delta_shared)

            if _USE_G:
                T.copy(g[i_b, q_t_s:q_t_s + _BT, i_h], g_q_shared)

            T.clear(qkT)
            T.gemm(K_shared, q_shared, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(_BT, _BT):
                if _USE_G:
                    qkT[i, j] = T.exp2(qkT[i, j] * _log2e_scale + g_q_shared[j] - g_k_shared[i] - lse_shared[j])
                else:
                    qkT[i, j] = T.exp2(qkT[i, j] * _log2e_scale - lse_shared[j])

            # causal + boundary (+ optional sliding window) mask
            for i, j in T.Parallel(_BT, _BT):
                key_pos = t_s + i
                q_pos = q_t_s + j
                causal = (key_pos <= q_pos) & (key_pos < T_seq + bos) & (q_pos < T_seq + bos)
                if _USE_WINDOW:
                    in_window = (q_pos - key_pos < _W)
                    mask = causal & in_window
                else:
                    mask = causal
                qkT[i, j] = T.if_then_else(mask, qkT[i, j], T.cast(0, accum_dtype))

            # dv += p @ do
            T.copy(qkT, qkT_cast)
            T.gemm(qkT_cast, do_shared, dv_local, policy=T.GemmWarpPolicy.FullRow)

            # ds = p * (dp - delta)
            T.clear(dsT)
            T.gemm(V_shared, do_shared, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(_BT, _BT):
                dsT[i, j] = qkT[i, j] * (dsT[i, j] - delta_shared[j])
            for i, j in T.Parallel(_BT, _BT):
                dsT_cast[i, j] = dsT[i, j] * _sm_scale

            # dk += ds^T @ q
            T.copy(dsT_cast, ds)
            T.gemm(ds, q_shared, dk_local, policy=T.GemmWarpPolicy.FullRow)

            # dq += ds @ k (atomic_add into standard-layout output)
            T.copy(dsT_cast, dsT_shared)
            T.clear(dq_local)
            T.gemm(dsT_shared, K_shared, dq_local, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
            for i, d in T.Parallel(_BT, _K_orig):
                if q_t_s + i < T_seq + bos:
                    T.atomic_add(dq_out[i_b, q_t_s + i, i_h, d], dq_local[i, d])

            if _USE_G:
                # dg for query positions (positive)
                T.reduce_sum(dsT, f_dg_q, dim=0)
                for j in T.Parallel(_BT):
                    if q_t_s + j < T_seq + bos:
                        T.atomic_add(dg[i_b, q_t_s + j, i_h], f_dg_q[j])

                # accumulate dg for key positions (negative)
                T.reduce_sum(dsT, f_dg_k, dim=1)
                for i in T.Parallel(_BT):
                    dg_k_acc[i] += f_dg_k[i]

        # GQA query heads accumulate into the same KV head, requiring atomic updates
        for i, d in T.Parallel(_BT, _V_orig):
            if t_s + i < T_seq + bos:
                T.atomic_add(dv_out[i_b, t_s + i, i_hkv, d], dv_local[i, d])
        for i, d in T.Parallel(_BT, _K_orig):
            if t_s + i < T_seq + bos:
                T.atomic_add(dk_out[i_b, t_s + i, i_hkv, d], dk_local[i, d])

        if _USE_G:
            for i in T.Parallel(_BT):
                if t_s + i < T_seq + bos:
                    T.atomic_add(dg[i_b, t_s + i, i_h], -dg_k_acc[i])

    if IS_VARLEN:
        @T.prim_func
        def kernel(
            q: T.Tensor(q_s, _dtype),
            k: T.Tensor(kv_s, _dtype),
            v: T.Tensor(vv_s, _dtype),
            g: T.Tensor(lse_s, accum_dtype),
            lse: T.Tensor(lse_s, accum_dtype),
            delta: T.Tensor(lse_s, accum_dtype),
            do: T.Tensor(q_s, _dtype),
            dq_out: T.Tensor(dq_s, accum_dtype),
            dk_out: T.Tensor(dkv_s, accum_dtype),
            dv_out: T.Tensor(dvv_s, accum_dtype),
            dg: T.Tensor(lse_s, accum_dtype),
            cu_seqlens: T.Tensor((Ncu_d,), T.int32),
            chunk_indices: T.Tensor((NT_d, 2), T.int32),
        ):
            with T.Kernel(_HQ, NT_d, 1, threads=threads) as (i_h, i_t, _):
                i_n = chunk_indices[i_t, 0]
                i_t_local = chunk_indices[i_t, 1]
                bos = cu_seqlens[i_n]
                T_seq = cu_seqlens[i_n + 1] - bos
                t_s = bos + i_t_local * _BT
                kernel_body(q, k, v, g, lse, delta, do, dq_out, dk_out, dv_out, dg, 0, i_h, i_t, t_s, T_seq, bos)
    else:
        @T.prim_func
        def kernel(
            q: T.Tensor(q_s, _dtype),
            k: T.Tensor(kv_s, _dtype),
            v: T.Tensor(vv_s, _dtype),
            g: T.Tensor(lse_s, accum_dtype),
            lse: T.Tensor(lse_s, accum_dtype),
            delta: T.Tensor(lse_s, accum_dtype),
            do: T.Tensor(q_s, _dtype),
            dq_out: T.Tensor(dq_s, accum_dtype),
            dk_out: T.Tensor(dkv_s, accum_dtype),
            dv_out: T.Tensor(dvv_s, accum_dtype),
            dg: T.Tensor(lse_s, accum_dtype),
        ):
            with T.Kernel(_HQ, T.ceildiv(T_d, _BT), _B, threads=threads) as (i_h, i_t, i_b):
                t_s = i_t * _BT
                kernel_body(q, k, v, g, lse, delta, do, dq_out, dk_out, dv_out, dg, i_b, i_h, i_t, t_s, T_d, 0)

    return kernel


def parallel_attn_bwd_tilelang(
    q,
    k,
    v,
    o,
    g_cumsum,
    lse,
    do,
    sink_bias=None,
    scale=None,
    window_size=None,
    chunk_size=128,
    cu_seqlens=None,
    chunk_indices=None,
):
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

    # padded copies handle boundaries; float32 buffers accumulate atomic updates across heads and key tiles
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
