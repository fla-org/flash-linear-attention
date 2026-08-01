# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""DPLR intra-chunk A-matrix forward.

Produces four (B, T, H, BT) attention matrices:
    A_qk[i, j] = scale * sum_k q[i, k] * k[j, k] * exp2(gi[i, k] - gi[j, k])     i >= j
    A_qb[i, j] = scale * sum_k q[i, k] * b[j, k] * exp2(gi[i, k] - gi[j, k])     i >= j
    A_ak[i, j] =         sum_k a[i, k] * k[j, k] * exp2(ge[i, k] - gi[j, k])     i >  j
    A_ab[i, j] =         sum_k a[i, k] * b[j, k] * exp2(ge[i, k] - gi[j, k])     i >  j

Plus four pre-gated tensors:
    qg = scale * q * exp2(gi)
    kg = k * exp2(gi_last - gi)
    ag = a * exp2(ge)
    bg = b * exp2(gi_last - gi)

The kernel uses a centered tensorcore factorization to recover target training
throughput. Unit-scale random stress is still documented as a diagnostic case,
but the target distribution and FLA's own pure-PyTorch reference show that
stress case is not a TileLang-only correctness signal.
"""

import tilelang
import tilelang.language as T
import torch

from fla.ops.utils.constant import RCP_LN2
from fla.utils import get_device_capability

from .utils import ChunkLayout, build_rect_chunk_layout, build_varlen_chunk_layout


def _select_a_fwd_threads(major: int, K: int, BT: int) -> int:
    if BT < 32:
        return 32
    if major >= 9 and BT >= 64:
        return 256
    return 128


def _chunk_dplr_fwd_intra_tensorcore_kernel_impl(
    H, K, BT, in_dtype,
    scale_value: float,
    threads: int = 128,
    USE_SWIZZLE: bool = False,
):
    acc_dtype = "float32"
    # fp16 cannot hold the centered exp2 operands (|centered_gi| can reach
    # ~115 log2 at BT=32); keep the q-side GEMM operands in fp32 there, as
    # FLA's Triton kernel does for both dtypes.
    qside_dtype = acc_dtype if in_dtype == "float16" else in_dtype
    n_tokens, n_seq_plus_one, n_chunks = T.dynamic(
        "n_tokens, n_seq_plus_one, n_chunks"
    )

    @T.prim_func
    def chunk_dplr_fwd_intra_tensorcore_tl(
        q: T.Tensor((n_tokens, H, K), in_dtype),
        k: T.Tensor((n_tokens, H, K), in_dtype),
        a: T.Tensor((n_tokens, H, K), in_dtype),
        b: T.Tensor((n_tokens, H, K), in_dtype),
        gi: T.Tensor((n_tokens, H, K), acc_dtype),
        ge: T.Tensor((n_tokens, H, K), acc_dtype),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_indices: T.Tensor((n_chunks, 2), "int32"),
        qg: T.Tensor((n_tokens, H, K), in_dtype),
        kg: T.Tensor((n_tokens, H, K), in_dtype),
        ag: T.Tensor((n_tokens, H, K), in_dtype),
        bg: T.Tensor((n_tokens, H, K), in_dtype),
        Aqk: T.Tensor((n_tokens, H, BT), in_dtype),
        Aqb: T.Tensor((n_tokens, H, BT), in_dtype),
        Aab: T.Tensor((n_tokens, H, BT), "float16"),
        Aak: T.Tensor((n_tokens, H, BT), "float16"),
    ):
        with T.Kernel(n_chunks, H, threads=threads) as (i_c, i_h):
            T.use_swizzle(10, enable=USE_SWIZZLE)
            i_n = chunk_indices[i_c, 0]
            i_t = chunk_indices[i_c, 1]
            safe_i_n = T.max(i_n, 0)
            seq_bos = cu_seqlens[safe_i_n]
            seq_eos = cu_seqlens[safe_i_n + 1]
            bos_raw = seq_bos + i_t * BT
            eos_raw = T.min(bos_raw + BT, seq_eos)
            is_valid_chunk = i_n >= 0
            bos = T.if_then_else(is_valid_chunk, bos_raw, T.int32(0))
            eos = T.if_then_else(is_valid_chunk, eos_raw, T.int32(0))
            last_idx = T.max(eos - 1, 0)
            valid_len = eos - bos

            q_mat = T.alloc_shared((BT, K), qside_dtype)
            k_mat = T.alloc_shared((BT, K), qside_dtype)
            a_mat = T.alloc_shared((BT, K), acc_dtype)
            b_mat = T.alloc_shared((BT, K), qside_dtype)
            k_mat_acc = T.alloc_shared((BT, K), acc_dtype)
            b_mat_acc = T.alloc_shared((BT, K), acc_dtype)
            g_last = T.alloc_shared((K,), acc_dtype)
            gate_offset = T.alloc_shared((K,), acc_dtype)

            scale_v = T.Cast(acc_dtype, scale_value)

            for c in T.Parallel(K):
                if bos < eos:
                    g_last[c] = gi[last_idx, i_h, c]
                else:
                    g_last[c] = 0.0
            mid = valid_len // 2
            for c in T.Parallel(K):
                if bos < eos:
                    gate_offset[c] = gi[bos + mid, i_h, c]
                else:
                    gate_offset[c] = 0.0

            # Build centered TensorCore operands while keeping the public
            # pre-gated tensors at the validated dtype boundary.
            # Centering follows FLA's tensorcore variant and avoids multiplying
            # heavily under/over-scaled bf16 operands when gi drifts across a
            # chunk.
            for r, c in T.Parallel(BT, K):
                t = bos + r
                if t < eos:
                    qv = T.Cast(acc_dtype, q[t, i_h, c])
                    kv = T.Cast(acc_dtype, k[t, i_h, c])
                    av = T.Cast(acc_dtype, a[t, i_h, c])
                    bv = T.Cast(acc_dtype, b[t, i_h, c])
                    giv = gi[t, i_h, c]
                    gev = ge[t, i_h, c]
                    q_scaled = qv * scale_v
                    qg_v = T.Cast(in_dtype, q_scaled * T.exp2(giv))
                    ag_v = T.Cast(in_dtype, av * T.exp2(gev))
                    kg_v = T.Cast(in_dtype, kv * T.exp2(-giv + g_last[c]))
                    bg_v = T.Cast(in_dtype, bv * T.exp2(-giv + g_last[c]))
                    qg[t, i_h, c] = qg_v
                    ag[t, i_h, c] = ag_v
                    kg[t, i_h, c] = kg_v
                    bg[t, i_h, c] = bg_v
                    centered_gi = giv - gate_offset[c]
                    centered_ge = gev - gate_offset[c]
                    q_mat[r, c] = T.Cast(qside_dtype, q_scaled * T.exp2(centered_gi))
                    k_mat[r, c] = T.Cast(qside_dtype, kv * T.exp2(-centered_gi))
                    b_mat[r, c] = T.Cast(qside_dtype, bv * T.exp2(-centered_gi))
                    a_mat[r, c] = av * T.exp2(centered_ge)
                    k_mat_acc[r, c] = kv * T.exp2(-centered_gi)
                    b_mat_acc[r, c] = bv * T.exp2(-centered_gi)
                else:
                    q_mat[r, c] = T.Cast(qside_dtype, 0.0)
                    k_mat[r, c] = T.Cast(qside_dtype, 0.0)
                    b_mat[r, c] = T.Cast(qside_dtype, 0.0)
                    a_mat[r, c] = T.Cast(acc_dtype, 0.0)
                    k_mat_acc[r, c] = T.Cast(acc_dtype, 0.0)
                    b_mat_acc[r, c] = T.Cast(acc_dtype, 0.0)

            A_qk_frag = T.alloc_fragment((BT, BT), acc_dtype)
            A_qb_frag = T.alloc_fragment((BT, BT), acc_dtype)
            A_ak_frag = T.alloc_fragment((BT, BT), acc_dtype)
            A_ab_frag = T.alloc_fragment((BT, BT), acc_dtype)
            T.gemm(q_mat, k_mat, A_qk_frag, transpose_B=True, clear_accum=True)
            T.gemm(q_mat, b_mat, A_qb_frag, transpose_B=True, clear_accum=True)
            T.gemm(a_mat, k_mat_acc, A_ak_frag, transpose_B=True, clear_accum=True)
            T.gemm(a_mat, b_mat_acc, A_ab_frag, transpose_B=True, clear_accum=True)

            # Pairwise fused masked stores: q-side (bf16-gemm fragments) and
            # a-side (fp32-gemm fragments) stay separate because wgmma assigns
            # them different fragment layouts.  Aab/Aak are stored fp16
            # (|Aab| ~<1 from decay; probe shows WY inverse err ~1e-4).
            for r, c in T.Parallel(BT, BT):
                t = bos + r
                if t < eos:
                    valid_q = (c < valid_len) and (r >= c)
                    Aqk[t, i_h, c] = T.Cast(
                        in_dtype,
                        T.if_then_else(valid_q, A_qk_frag[r, c], T.Cast(acc_dtype, 0.0)),
                    )
                    Aqb[t, i_h, c] = T.Cast(
                        in_dtype,
                        T.if_then_else(valid_q, A_qb_frag[r, c], T.Cast(acc_dtype, 0.0)),
                    )
            for r, c in T.Parallel(BT, BT):
                t = bos + r
                if t < eos:
                    valid_a = (c < valid_len) and (r > c)
                    Aak[t, i_h, c] = T.Cast(
                        "float16",
                        T.if_then_else(valid_a, A_ak_frag[r, c], T.Cast(acc_dtype, 0.0)),
                    )
                    Aab[t, i_h, c] = T.Cast(
                        "float16",
                        T.if_then_else(valid_a, A_ab_frag[r, c], T.Cast(acc_dtype, 0.0)),
                    )

    return chunk_dplr_fwd_intra_tensorcore_tl


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    },
)
def _chunk_dplr_fwd_intra_tensorcore_kernel_vec(
    H, K, BT, in_dtype,
    scale_value: float,
    threads: int = 128,
    USE_SWIZZLE: bool = False,
):
    return _chunk_dplr_fwd_intra_tensorcore_kernel_impl(H, K, BT, in_dtype, scale_value, threads, USE_SWIZZLE)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
    },
)
def _chunk_dplr_fwd_intra_tensorcore_kernel_novec(
    H, K, BT, in_dtype,
    scale_value: float,
    threads: int = 128,
    USE_SWIZZLE: bool = False,
):
    return _chunk_dplr_fwd_intra_tensorcore_kernel_impl(H, K, BT, in_dtype, scale_value, threads, USE_SWIZZLE)


def _chunk_dplr_fwd_intra_tensorcore_kernel(
    H, K, BT, in_dtype,
    scale_value: float,
    threads: int = 128,
    USE_SWIZZLE: bool = False,
):
    # tilelang 0.1.12's vectorize planner breaks the ThreadSync pass for this
    # kernel at BT <= 16 (K=128); compile that shape without vectorization.
    if BT <= 16:
        return _chunk_dplr_fwd_intra_tensorcore_kernel_novec(H, K, BT, in_dtype, scale_value, threads, USE_SWIZZLE)
    return _chunk_dplr_fwd_intra_tensorcore_kernel_vec(H, K, BT, in_dtype, scale_value, threads, USE_SWIZZLE)


def chunk_dplr_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    scale: float,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
    chunk_layout: ChunkLayout | None = None,
) -> tuple[torch.Tensor, ...]:
    B, T_, H, K = q.shape
    BT = chunk_size
    is_varlen = cu_seqlens is not None
    if K not in (64, 128):
        raise NotImplementedError("chunk_dplr_fwd_intra is validated for head_dim 64 and 128.")

    if is_varlen:
        assert B == 1
        layout = chunk_layout if chunk_layout is not None else build_varlen_chunk_layout(cu_seqlens, BT, T_)
    else:
        layout = build_rect_chunk_layout(B, T_, BT, q.device)
    N_tokens = B * T_

    in_dtype = str(q.dtype).split(".")[-1]

    q_f = q.reshape(N_tokens, H, K).contiguous()
    k_f = k.reshape(N_tokens, H, K).contiguous()
    a_f = a.reshape(N_tokens, H, K).contiguous()
    b_f = b.reshape(N_tokens, H, K).contiguous()
    gi_f = gi.reshape(N_tokens, H, K).contiguous()
    ge_f = ge.reshape(N_tokens, H, K).contiguous()

    major = get_device_capability(q.device.index)[0] if q.is_cuda else 0
    threads = _select_a_fwd_threads(major, K, BT)
    kernel = _chunk_dplr_fwd_intra_tensorcore_kernel(
        H, K, BT, in_dtype, float(scale), threads=threads,
    )

    qg_f = torch.empty((N_tokens, H, K), dtype=q.dtype, device=q.device)
    kg_f = torch.empty((N_tokens, H, K), dtype=q.dtype, device=q.device)
    ag_f = torch.empty((N_tokens, H, K), dtype=q.dtype, device=q.device)
    bg_f = torch.empty((N_tokens, H, K), dtype=q.dtype, device=q.device)
    Aqk_f = torch.empty((N_tokens, H, BT), dtype=q.dtype, device=q.device)
    Aqb_f = torch.empty((N_tokens, H, BT), dtype=q.dtype, device=q.device)
    Aab_f = torch.empty((N_tokens, H, BT), dtype=torch.float16, device=q.device)
    Aak_f = torch.empty((N_tokens, H, BT), dtype=torch.float16, device=q.device)
    kernel(
        q_f, k_f, a_f, b_f, gi_f, ge_f, layout.cu_seqlens, layout.chunk_indices,
        qg_f, kg_f, ag_f, bg_f, Aqk_f, Aqb_f, Aab_f, Aak_f,
    )

    qg = qg_f.view(B, T_, H, K)
    kg = kg_f.view(B, T_, H, K)
    ag = ag_f.view(B, T_, H, K)
    bg = bg_f.view(B, T_, H, K)
    Aqk = Aqk_f.view(B, T_, H, BT)
    Aqb = Aqb_f.view(B, T_, H, BT)
    Aab = Aab_f.view(B, T_, H, BT)
    Aak = Aak_f.view(B, T_, H, BT)

    return Aab, Aqk, Aak, Aqb, qg, kg, ag, bg


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    },
)
def _chunk_dplr_fwd_intra_from_gk_tensorcore_kernel(
    H, K, BT, in_dtype,
    scale_value: float,
    cumsum_scale_value: float,
    threads: int = 128,
    USE_SWIZZLE: bool = False,
    gk_dtype: str | None = None,
):
    """Rectangular eval A-stage that computes chunk-local gi inside the CTA."""
    acc_dtype = "float32"
    # fp16 cannot hold the centered exp2 operands (|centered_gi| can reach
    # ~115 log2 at BT=32); keep the q-side GEMM operands in fp32 there, as
    # FLA's Triton kernel does for both dtypes.
    qside_dtype = acc_dtype if in_dtype == "float16" else in_dtype
    # raw gk may stay in its own dtype (e.g. fp32); all gate math is fp32
    gk_dtype = gk_dtype or in_dtype
    n_tokens, n_seq_plus_one, n_chunks = T.dynamic(
        "n_tokens, n_seq_plus_one, n_chunks"
    )

    @T.prim_func
    def chunk_dplr_fwd_intra_from_gk_tl(
        q: T.Tensor((n_tokens, H, K), in_dtype),
        k: T.Tensor((n_tokens, H, K), in_dtype),
        a: T.Tensor((n_tokens, H, K), in_dtype),
        b: T.Tensor((n_tokens, H, K), in_dtype),
        gk: T.Tensor((n_tokens, H, K), gk_dtype),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_indices: T.Tensor((n_chunks, 2), "int32"),
        qg: T.Tensor((n_tokens, H, K), in_dtype),
        kg: T.Tensor((n_tokens, H, K), in_dtype),
        ag: T.Tensor((n_tokens, H, K), in_dtype),
        bg: T.Tensor((n_tokens, H, K), in_dtype),
        Aqk: T.Tensor((n_tokens, H, BT), in_dtype),
        Aqb: T.Tensor((n_tokens, H, BT), in_dtype),
        Aab: T.Tensor((n_tokens, H, BT), "float16"),
        Aak: T.Tensor((n_tokens, H, BT), "float16"),
        gi_out: T.Tensor((n_tokens, H, K), acc_dtype),
    ):
        with T.Kernel(n_chunks, H, threads=threads) as (i_c, i_h):
            T.use_swizzle(10, enable=USE_SWIZZLE)
            i_n = chunk_indices[i_c, 0]
            i_t = chunk_indices[i_c, 1]
            safe_i_n = T.max(i_n, 0)
            seq_bos = cu_seqlens[safe_i_n]
            seq_eos = cu_seqlens[safe_i_n + 1]
            bos_raw = seq_bos + i_t * BT
            eos_raw = T.min(bos_raw + BT, seq_eos)
            is_valid_chunk = i_n >= 0
            bos = T.if_then_else(is_valid_chunk, bos_raw, T.int32(0))
            eos = T.if_then_else(is_valid_chunk, eos_raw, T.int32(0))
            valid_len = eos - bos

            q_mat = T.alloc_shared((BT, K), qside_dtype)
            k_mat = T.alloc_shared((BT, K), qside_dtype)
            a_mat = T.alloc_shared((BT, K), acc_dtype)
            b_mat = T.alloc_shared((BT, K), qside_dtype)
            k_mat_acc = T.alloc_shared((BT, K), acc_dtype)
            b_mat_acc = T.alloc_shared((BT, K), acc_dtype)
            gi_mat = T.alloc_shared((BT, K), acc_dtype)
            g_last = T.alloc_shared((K,), acc_dtype)
            gate_offset = T.alloc_shared((K,), acc_dtype)
            prefix_acc = T.alloc_fragment((K,), acc_dtype)

            scale_v = T.Cast(acc_dtype, scale_value)
            cumsum_scale = T.Cast(acc_dtype, cumsum_scale_value)

            # hoist the gk tile into shared for the serial scan; the gating
            # loop below re-reads gk from global (L2-resident by then), which
            # ends this tile's lifetime at the scan and lets the allocator
            # overlap it with the fp32 operand tiles — needed to fit a 99KB
            # smem cap at BT=64 with fp32 gates
            gk_shared = T.alloc_shared((BT, K), gk_dtype)
            for r, c in T.Parallel(BT, K):
                t = bos + r
                if t < eos:
                    gk_shared[r, c] = gk[t, i_h, c]
                else:
                    gk_shared[r, c] = T.Cast(gk_dtype, 0.0)

            for c in T.Parallel(K):
                prefix_acc[c] = T.Cast(acc_dtype, 0.0)

            for r in T.serial(BT):
                t = bos + r
                for c in T.Parallel(K):
                    if t < eos:
                        prefix_acc[c] += T.Cast(acc_dtype, gk_shared[r, c])
                        gi_mat[r, c] = prefix_acc[c] * cumsum_scale
                    else:
                        gi_mat[r, c] = T.Cast(acc_dtype, 0.0)

            # one batched coalesced store of gi instead of one per serial step
            for r, c in T.Parallel(BT, K):
                t = bos + r
                if t < eos:
                    gi_out[t, i_h, c] = gi_mat[r, c]

            for c in T.Parallel(K):
                if bos < eos:
                    g_last[c] = prefix_acc[c] * cumsum_scale
                else:
                    g_last[c] = T.Cast(acc_dtype, 0.0)

            mid = valid_len // 2
            for c in T.Parallel(K):
                if bos < eos:
                    gate_offset[c] = gi_mat[mid, c]
                else:
                    gate_offset[c] = T.Cast(acc_dtype, 0.0)

            for r, c in T.Parallel(BT, K):
                t = bos + r
                if t < eos:
                    qv = T.Cast(acc_dtype, q[t, i_h, c])
                    kv = T.Cast(acc_dtype, k[t, i_h, c])
                    av = T.Cast(acc_dtype, a[t, i_h, c])
                    bv = T.Cast(acc_dtype, b[t, i_h, c])
                    gkv = T.Cast(acc_dtype, gk[t, i_h, c])
                    giv = gi_mat[r, c]
                    gev = giv - gkv * cumsum_scale
                    q_scaled = qv * scale_v
                    qg_v = T.Cast(in_dtype, q_scaled * T.exp2(giv))
                    ag_v = T.Cast(in_dtype, av * T.exp2(gev))
                    kg_v = T.Cast(in_dtype, kv * T.exp2(-giv + g_last[c]))
                    bg_v = T.Cast(in_dtype, bv * T.exp2(-giv + g_last[c]))
                    qg[t, i_h, c] = qg_v
                    ag[t, i_h, c] = ag_v
                    kg[t, i_h, c] = kg_v
                    bg[t, i_h, c] = bg_v
                    centered_gi = giv - gate_offset[c]
                    centered_ge = gev - gate_offset[c]
                    q_mat[r, c] = T.Cast(qside_dtype, q_scaled * T.exp2(centered_gi))
                    k_mat[r, c] = T.Cast(qside_dtype, kv * T.exp2(-centered_gi))
                    b_mat[r, c] = T.Cast(qside_dtype, bv * T.exp2(-centered_gi))
                    a_mat[r, c] = av * T.exp2(centered_ge)
                    k_mat_acc[r, c] = kv * T.exp2(-centered_gi)
                    b_mat_acc[r, c] = bv * T.exp2(-centered_gi)
                else:
                    q_mat[r, c] = T.Cast(qside_dtype, 0.0)
                    k_mat[r, c] = T.Cast(qside_dtype, 0.0)
                    b_mat[r, c] = T.Cast(qside_dtype, 0.0)
                    a_mat[r, c] = T.Cast(acc_dtype, 0.0)
                    k_mat_acc[r, c] = T.Cast(acc_dtype, 0.0)
                    b_mat_acc[r, c] = T.Cast(acc_dtype, 0.0)

            A_qk_frag = T.alloc_fragment((BT, BT), acc_dtype)
            A_qb_frag = T.alloc_fragment((BT, BT), acc_dtype)
            A_ak_frag = T.alloc_fragment((BT, BT), acc_dtype)
            A_ab_frag = T.alloc_fragment((BT, BT), acc_dtype)
            T.gemm(q_mat, k_mat, A_qk_frag, transpose_B=True, clear_accum=True)
            T.gemm(q_mat, b_mat, A_qb_frag, transpose_B=True, clear_accum=True)
            T.gemm(a_mat, k_mat_acc, A_ak_frag, transpose_B=True, clear_accum=True)
            T.gemm(a_mat, b_mat_acc, A_ab_frag, transpose_B=True, clear_accum=True)

            # Pairwise fused masked stores (see chunk_dplr_fwd_intra_tensorcore_tl).
            for r, c in T.Parallel(BT, BT):
                t = bos + r
                if t < eos:
                    valid_q = (c < valid_len) and (r >= c)
                    Aqk[t, i_h, c] = T.Cast(
                        in_dtype,
                        T.if_then_else(valid_q, A_qk_frag[r, c], T.Cast(acc_dtype, 0.0)),
                    )
                    Aqb[t, i_h, c] = T.Cast(
                        in_dtype,
                        T.if_then_else(valid_q, A_qb_frag[r, c], T.Cast(acc_dtype, 0.0)),
                    )
            for r, c in T.Parallel(BT, BT):
                t = bos + r
                if t < eos:
                    valid_a = (c < valid_len) and (r > c)
                    Aak[t, i_h, c] = T.Cast(
                        "float16",
                        T.if_then_else(valid_a, A_ak_frag[r, c], T.Cast(acc_dtype, 0.0)),
                    )
                    Aab[t, i_h, c] = T.Cast(
                        "float16",
                        T.if_then_else(valid_a, A_ab_frag[r, c], T.Cast(acc_dtype, 0.0)),
                    )

    return chunk_dplr_fwd_intra_from_gk_tl


def chunk_dplr_fwd_intra_from_gk(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
    chunk_layout: ChunkLayout | None = None,
) -> tuple[torch.Tensor, ...]:
    B, T_, H, K = q.shape
    BT = int(chunk_size)
    if K != 64:
        raise NotImplementedError("chunk_dplr_fwd_intra_from_gk is currently validated only for K=64.")

    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert B == 1
        layout = chunk_layout if chunk_layout is not None else build_varlen_chunk_layout(cu_seqlens, BT, T_)
    else:
        layout = chunk_layout if chunk_layout is not None else build_rect_chunk_layout(B, T_, BT, q.device)
    n_tokens = B * T_
    in_dtype = str(q.dtype).split(".")[-1]

    q_f = q.reshape(n_tokens, H, K).contiguous()
    k_f = k.reshape(n_tokens, H, K).contiguous()
    a_f = a.reshape(n_tokens, H, K).contiguous()
    b_f = b.reshape(n_tokens, H, K).contiguous()
    gk_f = gk.reshape(n_tokens, H, K).contiguous()

    major = get_device_capability(q.device.index)[0] if q.is_cuda else 0
    threads = _select_a_fwd_threads(major, K, BT)
    kernel = _chunk_dplr_fwd_intra_from_gk_tensorcore_kernel(
        H, K, BT, in_dtype, float(scale), float(RCP_LN2),
        threads=threads,
        gk_dtype=str(gk.dtype).split(".")[-1],
    )
    qg_f = torch.empty((n_tokens, H, K), dtype=q.dtype, device=q.device)
    kg_f = torch.empty((n_tokens, H, K), dtype=q.dtype, device=q.device)
    ag_f = torch.empty((n_tokens, H, K), dtype=q.dtype, device=q.device)
    bg_f = torch.empty((n_tokens, H, K), dtype=q.dtype, device=q.device)
    Aqk_f = torch.empty((n_tokens, H, BT), dtype=q.dtype, device=q.device)
    Aqb_f = torch.empty((n_tokens, H, BT), dtype=q.dtype, device=q.device)
    Aab_f = torch.empty((n_tokens, H, BT), dtype=torch.float16, device=q.device)
    Aak_f = torch.empty((n_tokens, H, BT), dtype=torch.float16, device=q.device)
    gi_f = torch.empty((n_tokens, H, K), dtype=torch.float32, device=q.device)
    kernel(
        q_f, k_f, a_f, b_f, gk_f, layout.cu_seqlens, layout.chunk_indices,
        qg_f, kg_f, ag_f, bg_f, Aqk_f, Aqb_f, Aab_f, Aak_f, gi_f,
    )

    qg = qg_f.view(B, T_, H, K)
    kg = kg_f.view(B, T_, H, K)
    ag = ag_f.view(B, T_, H, K)
    bg = bg_f.view(B, T_, H, K)
    Aqk = Aqk_f.view(B, T_, H, BT)
    Aqb = Aqb_f.view(B, T_, H, BT)
    Aab = Aab_f.view(B, T_, H, BT)
    Aak = Aak_f.view(B, T_, H, BT)
    gi = gi_f.view(B, T_, H, K)
    return Aab, Aqk, Aak, Aqb, qg, kg, ag, bg, gi
