# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Chunk-local cumulative sum for log-decay gates.

Produces two outputs:
    gi[bos + t, h, k] = scale * sum_{j=0..t}     g[bos + j, h, k]   (inclusive)
    ge[bos + t, h, k] = scale * sum_{j=0..t-1}   g[bos + j, h, k]   (exclusive)

where each chunk of size BT cumsums independently. `scale = RCP_LN2` lets
downstream kernels use `T.exp2` directly.

Rectangular batches compute the prefix as a matmul against constant
(strict-)lower-triangular ones matrices (FLA's chunk_rwkv6_fwd_cumsum
formulation): the gate tile is loaded once, two tensor-core GEMMs produce
the inclusive and exclusive prefixes, and results are scaled and stored.
Same global traffic as a scan with no serial phases; the GEMMs accumulate
in fp32 from tf32-rounded inputs, matching the Triton reference's numerics.
This replaces the earlier vectorized PyTorch chain (fp32 cast + pad +
cumsum + scale + shifted zeros), which cost ~2.5 ms of elementwise/scan
glue kernels per call at h4096, and a segmented serial-scan kernel that
measured 2.2x behind Triton on sm_90.

Varlen batches keep the irregular-boundary kernel keyed by chunk_indices.
"""

import tilelang
import tilelang.language as T
import torch

from .layout import ChunkLayout, build_varlen_chunk_layout

# ---------------------------------------------------------------------------
# TileLang segmented scan kernel for rectangular batches.
# ---------------------------------------------------------------------------


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    },
)
def _chunk_local_cumsum_rect_kernel_tl(
    H, K, BT, BS, in_dtype, scale_value: float, OUTPUT_GE: bool, threads: int = 128
):
    acc_dtype = "float32"
    B, Tt = T.dynamic("B, Tt")

    @T.prim_func
    def chunk_local_cumsum_rect_tl(
        g: T.Tensor((B, Tt, H, K), in_dtype),
        gi: T.Tensor((B, Tt, H, K), acc_dtype),
        ge: T.Tensor((B, Tt, H, K), acc_dtype),
    ):
        tpc = T.ceildiv(Tt, BT)
        with T.Kernel(T.ceildiv(K, BS), B * tpc, H, threads=threads) as (i_sb, i_c, i_h):
            i_n = i_c // tpc
            bos = (i_c % tpc) * BT

            g_shared = T.alloc_shared((BT, BS), acc_dtype)
            mask_i = T.alloc_shared((BT, BT), acc_dtype)
            if OUTPUT_GE:
                mask_e = T.alloc_shared((BT, BT), acc_dtype)
            gi_frag = T.alloc_fragment((BT, BS), acc_dtype)
            if OUTPUT_GE:
                ge_frag = T.alloc_fragment((BT, BS), acc_dtype)

            # Interior chunks bulk-copy; the last chunk of a batch element
            # (T % BT != 0) takes the predicated scalar path.
            if bos + BT <= Tt:
                T.copy(g[i_n, bos: bos + BT, i_h, i_sb * BS: i_sb * BS + BS], g_shared)
            else:
                for r, c in T.Parallel(BT, BS):
                    t = bos + r
                    k_idx = i_sb * BS + c
                    if (t < Tt) and (k_idx < K):
                        g_shared[r, c] = T.Cast(acc_dtype, g[i_n, t, i_h, k_idx])
                    else:
                        g_shared[r, c] = T.Cast(acc_dtype, 0.0)

            # Mask entries are exact 0/1 values.
            for r, c in T.Parallel(BT, BT):
                mask_i[r, c] = T.if_then_else(r >= c, T.Cast(acc_dtype, 1.0), T.Cast(acc_dtype, 0.0))
                if OUTPUT_GE:
                    mask_e[r, c] = T.if_then_else(r > c, T.Cast(acc_dtype, 1.0), T.Cast(acc_dtype, 0.0))

            T.gemm(mask_i, g_shared, gi_frag, clear_accum=True)
            if OUTPUT_GE:
                T.gemm(mask_e, g_shared, ge_frag, clear_accum=True)

            for r, c in T.Parallel(BT, BS):
                t = bos + r
                k_idx = i_sb * BS + c
                if (t < Tt) and (k_idx < K):
                    gi[i_n, t, i_h, k_idx] = gi_frag[r, c] * scale_value
                    if OUTPUT_GE:
                        ge[i_n, t, i_h, k_idx] = ge_frag[r, c] * scale_value

    return chunk_local_cumsum_rect_tl


# ---------------------------------------------------------------------------
# TileLang chunk-local scan kernel — kept for varlen batches.
# ---------------------------------------------------------------------------


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    },
)
def _chunk_local_cumsum_kernel_tl(
    H, K, BT, BS, in_dtype, scale_value: float, OUTPUT_GE: bool
):
    acc_dtype = "float32"
    n_tokens, n_seq_plus_one, n_chunks = T.dynamic(
        "n_tokens, n_seq_plus_one, n_chunks"
    )

    @T.prim_func
    def chunk_local_cumsum_tl(
        g: T.Tensor((n_tokens, H, K), in_dtype),
        gi: T.Tensor((n_tokens, H, K), acc_dtype),
        ge: T.Tensor((n_tokens, H, K), acc_dtype),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_indices: T.Tensor((n_chunks, 2), "int32"),
    ):
        with T.Kernel(T.ceildiv(K, BS), n_chunks, H, threads=128) as (i_sb, i_c, i_h):
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

            s_scalar = T.Cast(acc_dtype, scale_value)
            acc = T.alloc_fragment((BS,), acc_dtype)

            for s_local in T.Parallel(BS):
                acc[s_local] = T.Cast(acc_dtype, 0.0)

            # For the target BT=64,K=64 path, one thread lane owns one channel
            # and scans the chunk time dimension in registers. This avoids the
            # heavier generic T.cumsum shared-memory scan while keeping each
            # timestep's K-lane loads/stores coalesced.
            for t_local in T.serial(BT):
                t = bos + t_local
                for s_local in T.Parallel(BS):
                    s = i_sb * BS + s_local
                    if (t < eos) and (s < K):
                        if OUTPUT_GE:
                            ge[t, i_h, s] = acc[s_local] * s_scalar
                        acc[s_local] += T.Cast(acc_dtype, g[t, i_h, s])
                        gi[t, i_h, s] = acc[s_local] * s_scalar

    return chunk_local_cumsum_tl


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_layout: ChunkLayout | None = None,
    output_ge: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunk-local inclusive (gi) and exclusive (ge) cumsum over the time axis.

    With ``output_ge=False`` the kernels never touch the ge buffer, which is
    then aliased to gi to skip the unused fp32 allocation, and None is
    returned in its place.
    """
    is_varlen = cu_seqlens is not None
    scale_f = float(scale) if scale is not None else 1.0
    in_dtype = str(g.dtype).split(".")[-1]

    if not is_varlen:
        B, T_, H, K = g.shape
        kernel = _chunk_local_cumsum_rect_kernel_tl(
            H, K, chunk_size, 64, in_dtype, scale_f, output_ge
        )
        gi = torch.empty((B, T_, H, K), dtype=torch.float32, device=g.device)
        ge = torch.empty((B, T_, H, K), dtype=torch.float32, device=g.device) if output_ge else gi
        kernel(g.contiguous(), gi, ge)
        return gi, ge if output_ge else None

    # Varlen path: the chunk_indices-keyed kernel handles irregular boundaries.
    B, T_, H, K = g.shape
    assert B == 1, "Varlen expects B==1"
    BT = chunk_size

    N_tokens = B * T_
    layout = chunk_layout if chunk_layout is not None else build_varlen_chunk_layout(cu_seqlens, BT, N_tokens)
    g_flat = g.reshape(N_tokens, H, K).contiguous()

    BS = K if K <= 64 else 64
    while BS > 16 and K % BS != 0:
        BS //= 2
    if K < BS:
        BS = K

    kernel = _chunk_local_cumsum_kernel_tl(H, K, BT, BS, in_dtype, scale_f, output_ge)
    gi_flat = torch.empty((N_tokens, H, K), dtype=torch.float32, device=g.device)
    ge_flat = torch.empty((N_tokens, H, K), dtype=torch.float32, device=g.device) if output_ge else gi_flat
    kernel(g_flat, gi_flat, ge_flat, layout.cu_seqlens, layout.chunk_indices)
    return gi_flat.view(B, T_, H, K), ge_flat.view(B, T_, H, K) if output_ge else None


__all__ = ["chunk_local_cumsum"]
