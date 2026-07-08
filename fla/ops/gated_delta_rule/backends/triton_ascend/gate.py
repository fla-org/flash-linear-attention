# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GDN gate kernels adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_ub_block_size,
    max_grid_axis_chunks,
)

_NUM_WARPS = 4
# Peak live fp32 vectors: input + output (+ bias path).
_GATE_FWD_MEM_MULT = 3.0
_GATE_BWD_MEM_MULT = 5.0
_SAFETY_MARGIN = 0.85
_FALLBACK_BT = 32
_FALLBACK_BT_FWD = 64


def _get_gate_fwd_bt(T: int) -> int:
    return compute_ub_block_size(
        T,
        _GATE_FWD_MEM_MULT,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=_FALLBACK_BT_FWD,
        desired=min(triton.next_power_of_2(T), _FALLBACK_BT_FWD),
    )


def _get_gate_bwd_bt(T: int) -> int:
    return compute_ub_block_size(
        T,
        _GATE_BWD_MEM_MULT,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=_FALLBACK_BT,
        desired=min(triton.next_power_of_2(T), _FALLBACK_BT),
    )


@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def gdn_gate_fwd_kernel_npu(
    g,
    A_log,
    dt_bias,
    yg,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    H_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_h = tl.program_id(1) + H_OFFSET

    b_A = tl.load(A_log + i_h).to(tl.float32)

    p_g = tl.make_block_ptr(g + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_yg = tl.make_block_ptr(yg + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + i_h).to(tl.float32)
    b_yg = -exp(b_A) * softplus(b_g)
    tl.store(p_yg, b_yg.to(p_yg.dtype.element_ty), boundary_check=(0,))


def _launch_gate_fwd(
    *,
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    yg: torch.Tensor,
    T: int,
    H: int,
    BT: int,
) -> None:
    NT = triton.cdiv(T, BT)
    kernel_kwargs = dict(
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        yg=yg,
        T=T,
        H=H,
        BT=BT,
        num_warps=_NUM_WARPS,
    )
    max_nt = max_grid_axis_chunks(NT, H, max_grid=ASCEND_MAX_GRID_DIM)
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        max_h = max_grid_axis_chunks(H, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for h_off in range(0, H, max_h):
            h_len = min(max_h, H - h_off)
            gdn_gate_fwd_kernel_npu[(nt_len, h_len)](
                **kernel_kwargs,
                NT_OFFSET=nt_off,
                H_OFFSET=h_off,
            )


@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def gdn_gate_chunk_cumsum_scalar_kernel_npu(
    g,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + i_h).to(tl.float32)
    b_A = tl.load(A_log + i_h).to(tl.float32)
    b_gate = -exp(b_A) * softplus(b_g)

    b_o = tl.cumsum(b_gate, axis=0)
    if REVERSE:
        b_z = tl.sum(b_gate, axis=0)
        b_o = -b_o + b_z[None] + b_gate
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def _launch_gate_chunk_cumsum(
    *,
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    o: torch.Tensor,
    scale: float | None,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None,
    T: int,
    B: int,
    H: int,
    BT: int,
    NT: int,
    reverse: bool,
) -> None:
    bh_total = B * H
    kernel_kwargs = dict(
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        REVERSE=reverse,
        num_warps=_NUM_WARPS,
    )
    max_nt = max_grid_axis_chunks(NT, bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        max_bh = max_grid_axis_chunks(bh_total, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            gdn_gate_chunk_cumsum_scalar_kernel_npu[(nt_len, bh_len)](
                **kernel_kwargs,
                NT_OFFSET=nt_off,
                BH_OFFSET=bh_off,
            )


@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def gdn_gate_bwd_kernel_npu(
    g,
    A_log,
    dt_bias,
    dyg,
    dg,
    dA,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    H_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_h = tl.program_id(1) + H_OFFSET

    b_A = tl.load(A_log + i_h).to(tl.float32)

    p_g = tl.make_block_ptr(g + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dg = tl.make_block_ptr(dg + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dyg = tl.make_block_ptr(dyg + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    b_dyg = tl.load(p_dyg, boundary_check=(0,)).to(tl.float32)

    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + i_h).to(tl.float32)

    b_neg_expA = -exp(b_A)
    b_yg = b_neg_expA * softplus(b_g)
    b_dg = b_neg_expA * (b_dyg * tl.sigmoid(b_g))
    b_dA = tl.sum(b_dyg * b_yg, 0)

    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
    tl.store(dA + i_t * H + i_h, b_dA)


def _launch_gate_bwd(
    *,
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    dyg: torch.Tensor,
    dg: torch.Tensor,
    dA: torch.Tensor,
    T: int,
    H: int,
    BT: int,
) -> None:
    NT = triton.cdiv(T, BT)
    kernel_kwargs = dict(
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        dyg=dyg,
        dg=dg,
        dA=dA,
        T=T,
        H=H,
        BT=BT,
        num_warps=_NUM_WARPS,
    )
    max_nt = max_grid_axis_chunks(NT, H, max_grid=ASCEND_MAX_GRID_DIM)
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        max_h = max_grid_axis_chunks(H, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for h_off in range(0, H, max_h):
            h_len = min(max_h, H - h_off)
            gdn_gate_bwd_kernel_npu[(nt_len, h_len)](
                **kernel_kwargs,
                NT_OFFSET=nt_off,
                H_OFFSET=h_off,
            )


@input_guard
def gdn_gate_fwd_npu(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    H = g.shape[-1]
    T = g.numel() // H
    BT = _get_gate_fwd_bt(T)
    yg = torch.empty_like(g, dtype=output_dtype)
    _launch_gate_fwd(g=g, A_log=A_log, dt_bias=dt_bias, yg=yg, T=T, H=H, BT=BT)
    return yg


@input_guard
def gdn_gate_chunk_cumsum_npu(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    B, T, H = g.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    o = torch.empty_like(g, dtype=output_dtype or g.dtype)
    _launch_gate_chunk_cumsum(
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        NT=NT,
        reverse=False,
    )
    return o


def gdn_gate_bwd_npu(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    dyg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    H = g.shape[-1]
    T = g.numel() // H
    BT = _get_gate_bwd_bt(T)
    dg = torch.empty_like(g, dtype=torch.float32)
    NT = triton.cdiv(T, BT)
    dA = A_log.new_empty(NT, H, dtype=torch.float32)
    _launch_gate_bwd(
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        dyg=dyg,
        dg=dg,
        dA=dA,
        T=T,
        H=H,
        BT=BT,
    )
    dg = dg.view_as(g).type_as(g)
    dA = dA.sum(0).view_as(A_log).type_as(A_log)
    dbias = dg.view(-1, H).sum(0).to(dt_bias) if dt_bias is not None else None
    return dg, dA, dbias
