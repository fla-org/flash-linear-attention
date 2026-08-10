# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""KDA gate kernels adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    max_grid_axis_chunks,
)

_NUM_WARPS = 4
# Peak live fp32 tiles: input + output (+ bias path).
_GATE_FWD_MEM_MULT = 3.0
_GATE_BWD_MEM_MULT = 5.0
_CHUNK_CUMSUM_MEM_MULT = 8.0
_SAFETY_MARGIN = 0.85
_FALLBACK_BT = 32
_FALLBACK_BT_FWD = 64
_FALLBACK_BS = 32


def _get_gate_fwd_bt(T: int, K: int) -> int:
    return compute_row_tile_block_size(
        T,
        K,
        _GATE_FWD_MEM_MULT,
        tiling_row=True,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=_FALLBACK_BT_FWD,
        max_block=_FALLBACK_BT_FWD,
    )


def _get_gate_bwd_bt(T: int, K: int) -> int:
    return compute_row_tile_block_size(
        T,
        K,
        _GATE_BWD_MEM_MULT,
        tiling_row=True,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=_FALLBACK_BT,
        max_block=_FALLBACK_BT,
    )


def _get_chunk_cumsum_bs(BT: int, S: int) -> int:
    return compute_row_tile_block_size(
        BT,
        S,
        _CHUNK_CUMSUM_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=_FALLBACK_BS,
        min_block=1,
        max_block=triton.next_power_of_2(S),
    )


@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
    'USE_LOWER_BOUND': lambda args: args['lower_bound'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def kda_gate_fwd_kernel_npu(
    g,
    A_log,
    dt_bias,
    yg,
    lower_bound,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    H_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_h = tl.program_id(1) + H_OFFSET

    b_A = tl.load(A_log + i_h).to(tl.float32)

    p_g = tl.make_block_ptr(g + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_yg = tl.make_block_ptr(yg + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        p_b = tl.make_block_ptr(dt_bias, (H * D,), (1,), (i_h * D,), (BD,), (0,))
        b_g = b_g + tl.load(p_b, boundary_check=(0,)).to(tl.float32)
    if not USE_LOWER_BOUND:
        b_yg = -exp(b_A) * softplus(b_g)
    else:
        b_yg = lower_bound * tl.sigmoid(exp(b_A) * b_g)
    tl.store(p_yg, b_yg.to(p_yg.dtype.element_ty), boundary_check=(0, 1))


def _launch_gate_fwd(
    *,
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    yg: torch.Tensor,
    lower_bound: float | None,
    T: int,
    H: int,
    K: int,
    BT: int,
) -> None:
    NT = triton.cdiv(T, BT)
    BD = triton.next_power_of_2(K)
    kernel_kwargs = dict(
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        yg=yg,
        lower_bound=lower_bound,
        T=T,
        H=H,
        D=K,
        BT=BT,
        BD=BD,
        num_warps=_NUM_WARPS,
    )
    max_nt = max_grid_axis_chunks(NT, H, max_grid=ASCEND_MAX_GRID_DIM)
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        max_h = max_grid_axis_chunks(H, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for h_off in range(0, H, max_h):
            h_len = min(max_h, H - h_off)
            kda_gate_fwd_kernel_npu[(nt_len, h_len)](
                **kernel_kwargs,
                NT_OFFSET=nt_off,
                H_OFFSET=h_off,
            )


@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
    'USE_LOWER_BOUND': lambda args: args['lower_bound'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def kda_gate_bwd_kernel_npu(
    g,
    A_log,
    dt_bias,
    dyg,
    dg,
    dA,
    lower_bound,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    H_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_h = tl.program_id(1) + H_OFFSET

    b_A = tl.load(A_log + i_h).to(tl.float32)

    p_g = tl.make_block_ptr(g + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_dg = tl.make_block_ptr(dg + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_dyg = tl.make_block_ptr(dyg + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))

    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_dyg = tl.load(p_dyg, boundary_check=(0, 1)).to(tl.float32)

    if HAS_BIAS:
        p_b = tl.make_block_ptr(dt_bias, (H * D,), (1,), (i_h * D,), (BD,), (0,))
        b_g = b_g + tl.load(p_b, boundary_check=(0,)).to(tl.float32)

    if not USE_LOWER_BOUND:
        b_A = -exp(b_A)
        b_yg = b_A * softplus(b_g)
        b_dg = b_A * (b_dyg * tl.sigmoid(b_g))
        b_dA = tl.sum(tl.sum(b_dyg * b_yg, 1), 0)
    else:
        b_A = exp(b_A)
        b_inner = b_A * b_g
        b_sig = tl.sigmoid(b_inner)
        b_dsig = b_sig * (1.0 - b_sig)
        b_d_inner_term = b_dyg * (lower_bound * b_dsig)
        b_dg = b_d_inner_term * b_A
        b_dA = tl.sum(tl.sum(b_dg * b_g, 1), 0)

    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dA + i_t * H + i_h, b_dA)


def _launch_gate_bwd(
    *,
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    dyg: torch.Tensor,
    dg: torch.Tensor,
    dA: torch.Tensor,
    lower_bound: float | None,
    T: int,
    H: int,
    K: int,
    BT: int,
) -> None:
    NT = triton.cdiv(T, BT)
    BD = triton.next_power_of_2(K)
    kernel_kwargs = dict(
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        dyg=dyg,
        dg=dg,
        dA=dA,
        lower_bound=lower_bound,
        T=T,
        H=H,
        D=K,
        BT=BT,
        BD=BD,
        num_warps=_NUM_WARPS,
    )
    max_nt = max_grid_axis_chunks(NT, H, max_grid=ASCEND_MAX_GRID_DIM)
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        max_h = max_grid_axis_chunks(H, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for h_off in range(0, H, max_h):
            h_len = min(max_h, H - h_off)
            kda_gate_bwd_kernel_npu[(nt_len, h_len)](
                **kernel_kwargs,
                NT_OFFSET=nt_off,
                H_OFFSET=h_off,
            )


@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_LOWER_BOUND': lambda args: args['lower_bound'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def kda_gate_chunk_cumsum_vector_kernel_npu(
    s,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    lower_bound,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t += NT_OFFSET
    i_bh += BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H * S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H * S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)

    if HAS_BIAS:
        p_b = tl.make_block_ptr(dt_bias + i_h * S, (S,), (1,), (i_s * BS,), (BS,), (0,))
        b_bias = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
        b_s = b_s + b_bias[None, :]

    b_A = tl.load(A_log + i_h).to(tl.float32)
    if not USE_LOWER_BOUND:
        b_gate = -exp(b_A) * softplus(b_s)
    else:
        b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)

    if REVERSE:
        b_o = tl.cumsum(b_gate, axis=0, reverse=True)
    else:
        b_o = tl.cumsum(b_gate, axis=0)

    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def _launch_gate_chunk_cumsum(
    *,
    s: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    o: torch.Tensor,
    scale: float | None,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None,
    lower_bound: float | None,
    T: int,
    B: int,
    H: int,
    S: int,
    BT: int,
    BS: int,
    NT: int,
    reverse: bool,
) -> None:
    bh_total = B * H
    ns = triton.cdiv(S, BS)
    kernel_kwargs = dict(
        s=s,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
        T=T,
        H=H,
        S=S,
        BT=BT,
        BS=BS,
        REVERSE=reverse,
        num_warps=_NUM_WARPS,
    )
    max_nt = max_grid_axis_chunks(NT, ns * bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        if cu_seqlens is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['chunk_indices'] = chunk_indices
            kernel_kwargs['NT_OFFSET'] = nt_off
        max_bh = max_grid_axis_chunks(bh_total, ns * nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            kda_gate_chunk_cumsum_vector_kernel_npu[(ns, nt_len, bh_len)](**kernel_kwargs)


@input_guard
def kda_gate_fwd_npu(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    H, K = g.shape[-2:]
    T = g.numel() // (H * K)
    BT = _get_gate_fwd_bt(T, K)
    yg = torch.empty_like(g, dtype=output_dtype)
    _launch_gate_fwd(
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        yg=yg,
        lower_bound=lower_bound,
        T=T,
        H=H,
        K=K,
        BT=BT,
    )
    return yg


@input_guard
def kda_gate_bwd_npu(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    dyg: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    H, K = g.shape[-2:]
    T = g.numel() // (H * K)
    BT = _get_gate_bwd_bt(T, K)
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
        lower_bound=lower_bound,
        T=T,
        H=H,
        K=K,
        BT=BT,
    )
    dg = dg.view_as(g).type_as(g)
    dA = dA.sum(0).view_as(A_log).type_as(A_log)
    dbias = dg.view(-1, H * K).sum(0).to(dt_bias) if dt_bias is not None else None
    return dg, dA, dbias


@input_guard
def kda_gate_chunk_cumsum_npu(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
    lower_bound: float | None = None,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    assert len(g.shape) == 4
    B, T, H, S = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"

    BS = _get_chunk_cumsum_bs(BT, S)
    g_org, o = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    _launch_gate_chunk_cumsum(
        s=g_org,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=BT,
        BS=BS,
        NT=NT,
        reverse=False,
    )
    return o


class KDAGateFunctionNPU(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        g: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor | None = None,
        lower_bound: float | None = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        yg = kda_gate_fwd_npu(
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            output_dtype=output_dtype,
        )
        ctx.save_for_backward(g, A_log, dt_bias)
        ctx.lower_bound = lower_bound
        return yg

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dyg: torch.Tensor):
        g, A_log, dt_bias = ctx.saved_tensors
        dg, dA, dbias = kda_gate_bwd_npu(
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            dyg=dyg,
            lower_bound=ctx.lower_bound,
        )
        return dg, dA, dbias, None, None


def fused_kda_gate_npu(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return KDAGateFunctionNPU.apply(g, A_log, dt_bias, lower_bound, output_dtype)
