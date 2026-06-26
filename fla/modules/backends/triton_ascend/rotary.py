# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Rotary embedding kernels adapted for triton-ascend on Huawei NPU."""

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.utils import autotune_cache_kwargs, get_multiprocessor_count

# Ascend vector UB is small; large num_warps / stages explodes compile-time UB (see bishengir ub overflow).
NUM_WARPS_AUTOTUNE = [2, 4]
NUM_STAGES_AUTOTUNE = [1, 2]


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in NUM_STAGES_AUTOTUNE
    ],
    key=['B', 'H', 'D', 'INTERLEAVED'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def rotary_embedding_kernel(
    x,
    cos,
    sin,
    y,
    cu_seqlens,
    chunk_indices,
    seq_offsets,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    TR: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
):
    i_t, i_b, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n), tl.load(cu_seqlens + i_n + 1)
        T = eos - bos
        x = x + bos * H*D + i_h * D
        y = y + bos * H*D + i_h * D
    else:
        i_n = i_b
        x = x + i_n * T*H*D + i_h * D
        y = y + i_n * T*H*D + i_h * D

    if i_t * BT >= T:
        return

    o_t = i_t * BT + tl.arange(0, BT)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        o_cs = o_t + seq_offsets
    else:
        o_cs = o_t + tl.load(seq_offsets + i_n)
    m_t = (o_t >= 0) & (o_t < T) & (o_cs >= 0) & (o_cs < TR)

    if not INTERLEAVED:
        o_r = tl.arange(0, BD // 2)
        p_x = x + o_t[:, None] * H*D + o_r[None, :]
        p_cos = cos + (o_cs[:, None] * R + o_r[None, :])
        p_sin = sin + (o_cs[:, None] * R + o_r[None, :])
        mask = m_t[:, None] & (o_r < R)[None, :]

        b_cos = tl.load(p_cos, mask=mask, other=1.0).to(tl.float32)
        b_sin = tl.load(p_sin, mask=mask, other=0.0).to(tl.float32)
        b_x0 = tl.load(p_x, mask=mask, other=0.0).to(tl.float32)
        b_x1 = tl.load(p_x + R, mask=mask, other=0.0).to(tl.float32)
        if CONJUGATE:
            b_sin = -b_sin
        b_o0 = b_x0 * b_cos - b_x1 * b_sin
        b_o1 = b_x0 * b_sin + b_x1 * b_cos
        p_y = y + (o_t[:, None] * H*D + o_r[None, :])
        tl.store(p_y, b_o0, mask=mask)
        tl.store(p_y + R, b_o1, mask=mask)
    else:
        o_d = tl.arange(0, BD)
        o_d_swap = o_d + ((o_d + 1) % 2) * 2 - 1
        o_d_repeat = tl.arange(0, BD) // 2
        p_x0 = x + o_t[:, None] * H*D + o_d[None, :]
        p_x1 = x + o_t[:, None] * H*D + o_d_swap[None, :]
        p_cos = cos + (o_cs[:, None] * R + o_d_repeat[None, :])
        p_sin = sin + (o_cs[:, None] * R + o_d_repeat[None, :])
        mask = m_t[:, None] & (o_d_repeat < R)[None, :]

        b_cos = tl.load(p_cos, mask=mask, other=1.0).to(tl.float32)
        b_sin = tl.load(p_sin, mask=mask, other=0.0).to(tl.float32)
        b_x0 = tl.load(p_x0, mask=mask, other=0.0).to(tl.float32)
        b_x1 = tl.load(p_x1, mask=mask, other=0.0).to(tl.float32)
        if CONJUGATE:
            b_sin = -b_sin
        b_o0 = b_x0 * b_cos
        b_o1 = b_x1 * b_sin
        b_y = tl.where(o_d[None, :] % 2 == 0, b_o0 - b_o1, b_o0 + b_o1)
        p_y = y + (o_t[:, None] * H*D + o_d[None, :])
        tl.store(p_y, b_y, mask=mask)


def rotary_embedding_fwdbwd_npu(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: int | torch.Tensor = 0,
    cu_seqlens: torch.Tensor | None = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    is_varlen = cu_seqlens is not None

    B, T, H, D = x.shape
    N = B if not is_varlen else cu_seqlens.shape[0] - 1
    TR, R = cos.shape
    R2 = R * 2

    assert D <= 256, "Only support D <= 256"
    assert TR >= T, f"TR must be >= T, got {TR} and {T}"

    assert cos.dtype == sin.dtype, f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    assert x.dtype == cos.dtype, f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (N,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
    else:
        assert seqlen_offsets + T <= TR

    y = torch.zeros_like(x) if not inplace else x
    if R2 < D and not inplace:
        y[..., R2:].copy_(x[..., R2:])

    BD = triton.next_power_of_2(R2)
    # GPU: one program loads a [BT, R] tile of cos/sin/x in fp32; Ascend UB (~192KiB) overflows for BT=128, R=128.
    if R >= 128:
        bt_cap = 16
    elif R >= 64:
        bt_cap = 32
    else:
        bt_cap = 64
    BT = min(bt_cap, triton.next_power_of_2(triton.cdiv(T, get_multiprocessor_count(x.device.index))))
    if chunk_indices is None and is_varlen:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if is_varlen else triton.cdiv(T, BT)

    grid = (NT, B, H)
    rotary_embedding_kernel[grid](
        x,
        cos,
        sin,
        y,
        cu_seqlens,
        chunk_indices,
        seqlen_offsets,
        B=B,
        T=T,
        H=H,
        D=D,
        R=R,
        TR=TR,
        BT=BT,
        BD=BD,
        IS_SEQLEN_OFFSETS_TENSOR=isinstance(seqlen_offsets, torch.Tensor),
        IS_VARLEN=is_varlen,
        INTERLEAVED=interleaved,
        CONJUGATE=conjugate,
    )
    return y
