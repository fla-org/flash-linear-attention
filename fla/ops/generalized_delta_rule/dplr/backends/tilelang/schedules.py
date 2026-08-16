# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backward schedule arithmetic for the TileLang DPLR backend.

Pure-python helpers shared by the kernel launcher (which selects a schedule
for the current device) and the backend verifier (which must reject configs
no schedule can launch, without importing tilelang).
"""

from __future__ import annotations

import torch

from fla.utils import get_device_capability


def device_cc(device: torch.device) -> int:
    major, minor = get_device_capability(device.index or 0)
    return major * 10 + minor


# Compiler-reported dynamic shared memory of the fused A-backward kernel
# (chunk_dplr_bwd_intra_tl, FUSE_QSIDE_DA=True) at BT=64 with the BK=32
# config used off cc90. All of its shared tiles are (BT, BK), (BT, BT) or
# (BT, BV) with BV=64, so the footprint is independent of K and identical
# for bf16/fp16 (accumulator tiles are fp32 either way). The two aliased
# dA staging pairs keep it under cc120's 101376B optin; the A-forward
# from_gk stage sits 128B below it, so this stays the binding footprint.
A_BWD_FUSED_BT64_SMEM_BYTES = 98432


def stream_default_threads(BT: int) -> int:
    return 128 if BT >= 32 else 64


def stream_low_default_qside_bv(BT: int) -> int:
    return 32 if BT >= 64 else 16


def stream_low_bwd_config(BT: int, V: int) -> dict[str, int]:
    # At K=V=128 the persistent (K, V) fp32 dh fragment alone is 128 regs per
    # thread at 128 threads, so the serial chunk loop spills to local memory
    # every iteration; 256 threads halves that to a spill-free 64.
    return {
        "threads": 256 if V >= 128 else stream_default_threads(BT),
        "qside_bv": stream_low_default_qside_bv(BT),
    }


def dtype_nbytes(dtype: str) -> int:
    if dtype in {"float32", "float"}:
        return 4
    if dtype in {"bfloat16", "float16", "half"}:
        return 2
    raise ValueError(f"unsupported DPLR stream backward dtype {dtype!r}")


def stream_pipeline_extra_smem_bytes(K: int, V: int, BT: int, in_dtype: str) -> int:
    """Extra smem of the double-buffered chunk loop: a second version of
    every global->shared operand tile (4x (BT, K), 3x (BT, V), 2x (BT, BT)).
    The fp32 gk_last row stays single-buffered (scalar per-chunk loads)."""
    elem = dtype_nbytes(in_dtype)
    return elem * (4 * BT * K + 3 * BT * V + 2 * BT * BT)


def stream_high_smem_bytes(K: int, V: int, BT: int, in_dtype: str, num_stages: int = 1) -> int:
    elem = dtype_nbytes(in_dtype)
    # fp32 tiles: gk_last (K,) plus the (4, K) dgk_part staging buffer.
    base = elem * (2 * K * V + 8 * BT * K + 5 * BT * V + 2 * BT * BT) + 20 * K
    if num_stages < 2:
        return base
    # Double-buffered: a second version of the nine operand tiles plus the
    # (K, V) h tile.
    return base + stream_pipeline_extra_smem_bytes(K, V, BT, in_dtype) + elem * K * V


def stream_mid_smem_bytes(K: int, V: int, BT: int, in_dtype: str, num_stages: int = 1) -> int:
    """High schedule with the two (K, V) state tiles aliased into one and the
    four (BT, K) output staging tiles merged into one sequential buffer.  The
    alias_kv path keeps the h tile single-buffered when pipelined."""
    elem = dtype_nbytes(in_dtype)
    base = stream_high_smem_bytes(K, V, BT, in_dtype, 1) - elem * (K * V + 3 * BT * K)
    if num_stages < 2:
        return base
    return base + stream_pipeline_extra_smem_bytes(K, V, BT, in_dtype)


def stream_reuse_smem_bytes(K: int, V: int, BT: int, in_dtype: str, qside_bv: int) -> int:
    elem = dtype_nbytes(in_dtype)
    # The trailing fp32 term is the (4, K) dgk_part lane-split staging buffer.
    return (
        elem * (
            K * V
            + 4 * BT * K
            + 3 * BT * V
            + BT * BT
            + BT * qside_bv
            + K * qside_bv
        )
        + 16 * K
    )


def stream_low_smem_bytes(K: int, V: int, BT: int, in_dtype: str) -> int:
    return stream_reuse_smem_bytes(K, V, BT, in_dtype, stream_low_default_qside_bv(BT))


def stream_bwd_schedule_or_none(
    *,
    K: int,
    V: int,
    BT: int,
    in_dtype: str,
    smem_cap: int,
) -> str | None:
    """Return the stream-backward schedule that fits `smem_cap`, or None.

    Mirrors `_select_stream_bwd_schedule`; keep the two in sync so the
    verifier never accepts a config the launcher cannot schedule.
    """
    high_smem = stream_high_smem_bytes(K, V, BT, in_dtype)
    mid_smem = stream_mid_smem_bytes(K, V, BT, in_dtype)
    low_smem = stream_low_smem_bytes(K, V, BT, in_dtype)
    low_dtype_supported = in_dtype in {"bfloat16", "float16", "half"}

    if high_smem <= smem_cap:
        return "high"
    if mid_smem <= smem_cap:
        return "mid"
    if low_dtype_supported and low_smem <= smem_cap:
        return "low"
    return None


def stream_bwd_num_stages(
    schedule: str,
    *,
    K: int,
    V: int,
    BT: int,
    in_dtype: str,
    smem_cap: int,
) -> int:
    """Software-pipeline depth for the high/mid chunk loop.

    Double-buffering the operand tiles pays off wherever the versioned
    footprint still fits the optin cap; the low schedule stays serial.
    Returns 0 (a plain serial loop) or 2.
    """
    if schedule == "high":
        smem = stream_high_smem_bytes(K, V, BT, in_dtype, num_stages=2)
    elif schedule == "mid":
        smem = stream_mid_smem_bytes(K, V, BT, in_dtype, num_stages=2)
    else:
        return 0
    return 2 if smem <= smem_cap else 0


def chunk64_schedule_or_none(
    *,
    K: int,
    V: int,
    in_dtype: str,
    smem_cap: int,
    cc: int,
) -> str | None:
    """Stream-backward schedule for BT=64, or None if any stage overflows.

    Acceptance must imply launchability of every BT=64 kernel: on cc90 the
    fused A-backward stage runs its BK=64 config and fits the 228KB cap, and
    elsewhere it runs BK=32 at A_BWD_FUSED_BT64_SMEM_BYTES, which a 99KB
    optin (e.g. cc120) just fits — there the A-forward from_gk stage is the
    next-tightest at 98304B. At K=V=128 the stream backward still exceeds
    every sub-228KB cap, so this returns None regardless of the A stages.
    """
    if cc != 90 and smem_cap < A_BWD_FUSED_BT64_SMEM_BYTES:
        return None
    return stream_bwd_schedule_or_none(
        K=K, V=V, BT=64, in_dtype=in_dtype, smem_cap=smem_cap,
    )
