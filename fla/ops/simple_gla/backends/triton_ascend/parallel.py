# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Ascend NPU realization of `parallel_simple_gla`.

Triton-Ascend cannot compile `parallel_simple_gla_fwd_kernel`:
program-dependent sub-block loop bounds trigger a bishengir SIGSEGV in `ConvertLinalgRToBinary`.
The NPU path uses chunk decomposition and materializes `output_attentions=True` scores with a dedicated kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.backends import TritonAscendBackend, register
from fla.ops.simple_gla.backends.triton_ascend.utils import simple_gla_verifier
from fla.ops.simple_gla.parallel import parallel_simple_gla
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.cumsum import chunk_global_cumsum
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import ASCEND_LAUNCH_BLOCK_BUDGET, launch_grid_chunked


@triton.jit(do_not_specialize=['T', 'TQ_OFFSET', 'TS_OFFSET', 'BH_OFFSET'])
def parallel_attn_npu_kernel(
    q,
    k,
    gc,
    attn,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    NT: tl.constexpr,
    NK: tl.constexpr,
    USE_G: tl.constexpr,
    TQ_OFFSET,
    TS_OFFSET,
    BH_OFFSET,
):
    """Materialize the parallel attention scores on Ascend.

    One program owns a single (query block, key block) tile of the score matrix,
    avoiding program-dependent loop bounds that Ascend cannot lower (see `parallel_simple_gla_fwd_kernel`).
    The upper triangle returns immediately, halving the work on a full NT x NT grid.
    """
    i_t = tl.program_id(0) + TQ_OFFSET
    i_s = tl.program_id(1) + TS_OFFSET
    i_bh = tl.program_id(2).to(tl.int64) + BH_OFFSET
    if i_s > i_t:
        return
    i_b, i_h = i_bh // H, i_bh % H
    o_q = i_t * BT + tl.arange(0, BT)
    m_q = o_q < T
    o_k = i_s * BT + tl.arange(0, BT)
    m_k = o_k < T
    qk_base = (i_b * T * H + i_h) * K
    g_base = i_b * T * H + i_h
    if USE_G:
        b_gq = tl.load(gc + g_base + o_q * H, mask=m_q, other=0.0).to(tl.float32)
        b_gk = tl.load(gc + g_base + o_k * H, mask=m_k, other=0.0).to(tl.float32)

    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_kk in tl.static_range(NK):
        o_kk = i_kk * BK + tl.arange(0, BK)
        m_kk = (o_kk < K)[:, None]
        b_q = tl.load(q + qk_base + o_q[:, None] * (H * K) + o_kk[None, :], mask=m_q[:, None] & (o_kk < K)[None, :], other=0.0)
        # load K in [BK, BT] layout: no in-register transpose needed
        b_k = tl.load(k + qk_base + o_kk[:, None] + o_k[None, :] * (H * K), mask=m_kk & m_k[None, :], other=0.0)
        b_s += tl.dot(b_q, b_k)
    if USE_G:
        b_s = b_s * exp2(b_gq[:, None] - b_gk[None, :])
    b_s = b_s * scale
    m_blk = m_q[:, None] & m_k[None, :] & (o_q[:, None] >= o_k[None, :])
    p_a = attn + (i_bh * T + o_q[:, None]) * T + o_k[None, :]
    tl.store(p_a, tl.where(m_blk, b_s, 0.0).to(p_a.dtype.element_ty), mask=m_q[:, None] & m_k[None, :])


@input_guard
def _parallel_attn_npu(q: torch.Tensor, k: torch.Tensor, g: torch.Tensor | None, scale: float) -> torch.Tensor:
    """Ascend implementation of `parallel_simple_gla(output_attentions=True)`.

    The returned score matrix carries no `grad_fn`;
    the CUDA path instead produces it inside `ParallelSimpleGLAFunction`.
    """
    B, T, H, K = q.shape
    BT = 64
    BK = min(64, triton.next_power_of_2(K))
    NT, NK = triton.cdiv(T, BT), triton.cdiv(K, BK)
    gc = chunk_global_cumsum(s=g, scale=RCP_LN2) if g is not None else None
    # zero-init: the upper triangle is skipped by the causal early-exit above
    attn = torch.zeros(B, H, T, T, dtype=q.dtype, device=q.device)
    grid = (NT, NT, B * H)
    kwargs = dict(
        q=q,
        k=k,
        gc=gc,
        attn=attn,
        scale=scale,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BK=BK,
        NT=NT,
        NK=NK,
        USE_G=g is not None,
        TQ_OFFSET=0,
        TS_OFFSET=0,
        BH_OFFSET=0,
    )
    if grid[0] * grid[1] * grid[2] > ASCEND_LAUNCH_BLOCK_BUDGET:
        launch_grid_chunked(
            kernel=parallel_attn_npu_kernel,
            grid=grid,
            offset_keys=('TQ_OFFSET', 'TS_OFFSET', 'BH_OFFSET'),
            kernel_kwargs=kwargs,
        )
    else:
        parallel_attn_npu_kernel[grid](**kwargs)
    return attn


@register(parallel_simple_gla, backend=TritonAscendBackend, verifier=simple_gla_verifier)
def parallel_simple_gla_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    output_attentions: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from fla.ops.simple_gla.chunk import chunk_simple_gla

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
    if output_attentions:
        assert cu_seqlens is None, "output_attentions=True is not supported with variable-length sequences"

    if output_attentions:
        attn = _parallel_attn_npu(q=q, k=k, g=g, scale=k.shape[-1] ** -0.5 if scale is None else scale)
    else:
        attn = None
    o, _ = chunk_simple_gla(q=q, k=k, v=v, g=g, scale=scale, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu)
    return o, attn
