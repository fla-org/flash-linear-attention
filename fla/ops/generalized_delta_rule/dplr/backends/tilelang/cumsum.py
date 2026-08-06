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

Strategy:
- **Rectangular batches** (cu_seqlens is None): vectorized PyTorch fp32 cumsum.
  This is ~2.3x faster than TileLang T.cumsum for rectangular inputs
  (0.24 ms vs 0.55 ms for B=1,T=8192,H=32,K=64,BT=64).
- **Varlen batches** (cu_seqlens is not None): TileLang T.cumsum kernel,
  which handles irregular chunk boundaries without Python loops.

The earlier T.cumsum-only approach (commit e99c20a) was based on a flawed
microbenchmark that claimed parity (0.37 ms vs 0.39 ms). End-to-end profiling
revealed T.cumsum is actually significantly slower for rectangular batches.
"""

import tilelang
import tilelang.language as T
import torch

from .utils import ChunkLayout, build_varlen_chunk_layout

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
    H, K, BT, BS, in_dtype, scale_value: float
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
                        ge[t, i_h, s] = acc[s_local] * s_scalar
                        acc[s_local] += T.Cast(acc_dtype, g[t, i_h, s])
                        gi[t, i_h, s] = acc[s_local] * s_scalar

    return chunk_local_cumsum_tl


def _chunk_local_cumsum_pytorch(
    g: torch.Tensor,
    chunk_size: int,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized PyTorch fp32 cumsum for rectangular batches.

    Steps:
    1. Cast to fp32 (critical for precision — bf16 cumsum causes 43% error).
    2. Pad time dim to multiple of chunk_size.
    3. Reshape to (B, n_chunks, BT, H, K).
    4. Cumsum along chunk time axis (dim=2).
    5. Derive exclusive by shifting inclusive result.
    6. Scale, reshape back, trim padding.
    """
    B, T_, H, K = g.shape
    BT = chunk_size
    scale_f = float(scale) if scale is not None else 1.0

    # Cast to fp32 before cumsum for numerical stability.
    g_fp32 = g.to(torch.float32)

    # Pad to multiple of BT.
    pad_len = (BT - T_ % BT) % BT
    if pad_len > 0:
        g_pad = torch.cat(
            [g_fp32, torch.zeros(B, pad_len, H, K, dtype=torch.float32, device=g.device)],
            dim=1,
        )
    else:
        g_pad = g_fp32

    # Reshape: (B, n_chunks, BT, H, K)
    n_chunks = g_pad.shape[1] // BT
    g_chunks = g_pad.view(B, n_chunks, BT, H, K)

    # Inclusive cumsum along time axis within each chunk.
    gi_chunks = g_chunks.cumsum(dim=2) * scale_f

    # Exclusive: shift inclusive down by 1, pad 0 at top of each chunk.
    ge_chunks = torch.zeros_like(gi_chunks)
    ge_chunks[:, :, 1:, :, :] = gi_chunks[:, :, :-1, :, :]

    # Flatten back and trim padding.
    gi = gi_chunks.view(B, -1, H, K)[:, :T_]
    ge = ge_chunks.view(B, -1, H, K)[:, :T_]

    return gi, ge


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_layout: ChunkLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk-local inclusive (gi) and exclusive (ge) cumsum over the time axis."""
    is_varlen = cu_seqlens is not None

    if not is_varlen:
        # Fast path: vectorized PyTorch for rectangular batches.
        return _chunk_local_cumsum_pytorch(g, chunk_size, scale=scale)

    # Varlen path: TileLang T.cumsum kernel handles irregular chunk boundaries.
    B, T_, H, K = g.shape
    assert B == 1, "Varlen expects B==1"
    BT = chunk_size

    N_tokens = B * T_
    layout = chunk_layout if chunk_layout is not None else build_varlen_chunk_layout(cu_seqlens, BT, N_tokens)
    g_flat = g.reshape(N_tokens, H, K).contiguous()
    scale_f = float(scale) if scale is not None else 1.0

    BS = K if K <= 64 else 64
    while BS > 16 and K % BS != 0:
        BS //= 2
    if K < BS:
        BS = K

    in_dtype = str(g.dtype).split(".")[-1]
    kernel = _chunk_local_cumsum_kernel_tl(H, K, BT, BS, in_dtype, scale_f)
    gi_flat = torch.empty((N_tokens, H, K), dtype=torch.float32, device=g.device)
    ge_flat = torch.empty((N_tokens, H, K), dtype=torch.float32, device=g.device)
    kernel(g_flat, gi_flat, ge_flat, layout.cu_seqlens, layout.chunk_indices)
    return gi_flat.view(B, T_, H, K), ge_flat.view(B, T_, H, K)


__all__ = ["chunk_local_cumsum"]
