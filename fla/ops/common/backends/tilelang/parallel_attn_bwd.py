# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import tilelang
import tilelang.language as T
import torch

from fla.ops.utils import prepare_chunk_indices


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def _build_parallel_attn_bwd_kernel(
    B, HQ, H, K, V, BT, sm_scale, log2e_scale, dtype_str,
    USE_G=False, IS_VARLEN=False, num_warps=4,
):
    """Build the TileLang JIT kernel for parallel attention backward.

    Note: B (batch size) is baked into the compiled kernel as a compile-time
    constant because it appears in static tensor shape annotations. Varying B
    at runtime will trigger a recompilation. Callers should either keep B fixed
    per process or pad batches to a fixed size.
    """

    dtype_map = {'float16': T.float16, 'bfloat16': T.bfloat16, 'float32': T.float32}
    _dtype = dtype_map[dtype_str]
    accum_dtype = T.float32
    threads = num_warps * 32
    _BT = BT
    _B, _HQ, _H = B, HQ, H
    _K_orig, _V_orig = K, V
    # Pad to next multiple of 16 for WGMMA inner-dim constraints (shared mem only)
    _K = (K + 15) // 16 * 16
    _V = (V + 15) // 16 * 16
    _G = HQ // H
    _USE_G = USE_G
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
    def kernel_body(q, k, v, g, lse, delta, do, dq_out, dk_out, dv_out, dg,
                    i_b, i_h, i_t, t_s, T_seq, bos):
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

        # Allocate shared/fragment buffers outside the loop so they can be reused
        # across query-tile iterations.
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
        loop_ed = T.ceildiv(T_seq, _BT)

        # Note: T.Pipelined is intentionally avoided here because combining it
        # with standard-layout T.atomic_add into dq_out produces garbage values
        # (observed as inf/575+ diffs) for non-power-of-2 dimensions.
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

            # causal + boundary mask
            for i, j in T.Parallel(_BT, _BT):
                key_pos = t_s + i
                q_pos = q_t_s + j
                causal = (key_pos <= q_pos) & (key_pos < T_seq + bos) & (q_pos < T_seq + bos)
                qkT[i, j] = T.if_then_else(causal, qkT[i, j], T.cast(0, accum_dtype))

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
            for i, d in T.Parallel(_BT, _K):
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

        # store dv and dk via element-wise atomic_add with explicit boundary checks.
        # atomic_add is required because in GQA multiple query-head blocks map to
        # the same KV head and accumulate into the same dk/dv rows.
        for i, d in T.Parallel(_BT, _V):
            if t_s + i < T_seq + bos:
                T.atomic_add(dv_out[i_b, t_s + i, i_hkv, d], dv_local[i, d])
        for i, d in T.Parallel(_BT, _K):
            if t_s + i < T_seq + bos:
                T.atomic_add(dk_out[i_b, t_s + i, i_hkv, d], dk_local[i, d])

        if _USE_G:
            for i in T.Parallel(_BT):
                if t_s + i < T_seq + bos:
                    T.atomic_add(dg[i_b, t_s + i, i_h], -dg_k_acc[i])

    if IS_VARLEN:
        @T.prim_func
        def kernel(
            q: T.Tensor(q_s, _dtype), k: T.Tensor(kv_s, _dtype),
            v: T.Tensor(vv_s, _dtype), g: T.Tensor(lse_s, accum_dtype),
            lse: T.Tensor(lse_s, accum_dtype), delta: T.Tensor(lse_s, accum_dtype),
            do: T.Tensor(q_s, _dtype), dq_out: T.Tensor(dq_s, accum_dtype),
            dk_out: T.Tensor(dkv_s, accum_dtype), dv_out: T.Tensor(dvv_s, accum_dtype),
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
                kernel_body(q, k, v, g, lse, delta, do, dq_out, dk_out, dv_out, dg,
                            0, i_h, i_t, t_s, T_seq, bos)
    else:
        @T.prim_func
        def kernel(
            q: T.Tensor(q_s, _dtype), k: T.Tensor(kv_s, _dtype),
            v: T.Tensor(vv_s, _dtype), g: T.Tensor(lse_s, accum_dtype),
            lse: T.Tensor(lse_s, accum_dtype), delta: T.Tensor(lse_s, accum_dtype),
            do: T.Tensor(q_s, _dtype), dq_out: T.Tensor(dq_s, accum_dtype),
            dk_out: T.Tensor(dkv_s, accum_dtype), dv_out: T.Tensor(dvv_s, accum_dtype),
            dg: T.Tensor(lse_s, accum_dtype),
        ):
            with T.Kernel(_HQ, T.ceildiv(T_d, _BT), _B, threads=threads) as (i_h, i_t, i_b):
                t_s = i_t * _BT
                kernel_body(q, k, v, g, lse, delta, do, dq_out, dk_out, dv_out, dg,
                            i_b, i_h, i_t, t_s, T_d, 0)

    return kernel


def parallel_attn_bwd_tilelang(
    q, k, v, o, g_cumsum, lse, do,
    sink_bias=None, scale=None, window_size=None,
    chunk_size=128, cu_seqlens=None, chunk_indices=None,
):
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[-1]
    BT = 64
    sm_scale = scale if scale is not None else K ** -0.5
    log2e_scale = sm_scale * 1.44269504

    USE_G = g_cumsum is not None
    IS_VARLEN = cu_seqlens is not None

    if chunk_indices is None and IS_VARLEN:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    # reuse Triton preprocess for delta (works with unpadded tensors)
    from fla.ops.attn.parallel import parallel_attn_bwd_preprocess
    delta = parallel_attn_bwd_preprocess(o, do)

    # No padding on the last dim: TileLang T.copy auto-zero-fills OOB columns
    # when loading to shared memory, and auto-skips OOB writes. dq/dk/dv all
    # need float32 buffers because GQA has multiple query-head blocks writing
    # to the same KV head (dk/dv) and query rows receive contributions from
    # multiple key tiles (dq).
    dq = torch.zeros(B, T, HQ, K, dtype=torch.float32, device=q.device)
    dk = torch.zeros(B, T, H, K, dtype=torch.float32, device=k.device)
    dv = torch.zeros(B, T, H, V, dtype=torch.float32, device=v.device)
    dg = torch.zeros(B, T, HQ, dtype=torch.float32, device=q.device) if USE_G else None

    g_kern = g_cumsum.float() if USE_G else torch.zeros(B, T, HQ, dtype=torch.float32, device=q.device)
    dg_kern = dg if USE_G else torch.zeros(B, T, HQ, dtype=torch.float32, device=q.device)

    dtype_str = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}[q.dtype]

    kernel = _build_parallel_attn_bwd_kernel(
        B, HQ, H, K, V, BT, sm_scale, log2e_scale, dtype_str,
        USE_G=USE_G, IS_VARLEN=IS_VARLEN,
    )

    if IS_VARLEN:
        kernel(q, k, v, g_kern, lse, delta, do, dq, dk, dv, dg_kern,
               cu_seqlens.int(), chunk_indices.int())
    else:
        kernel(q, k, v, g_kern, lse, delta, do, dq, dk, dv, dg_kern)

    dq = dq.to(q.dtype)
    dk = dk.to(k.dtype)
    dv = dv.to(v.dtype)
    return dq, dk, dv, dg, None
