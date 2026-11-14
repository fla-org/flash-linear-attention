# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Ascend fused GKA. Used only when the stock `[BK, BK]` state does not fit in UB.

`H_t` is symmetric, so the matvec is `sum_i H[i, j] x[i]` (the same reduction as the stock
fused kernel). Panels are row slices of width ``BS``; only the first ``K`` rows and columns
are live.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.backends import TritonAscendBackend, register
from fla.ops.gka.chunk import check_gka_inputs
from fla.ops.gka.fused_recurrent import FusedRecurrentGKAFunction, fused_recurrent_gka
from fla.ops.simple_gla import fused_recurrent_simple_gla
from fla.ops.utils.op import exp
from fla.utils import ascend_compile_kwargs, input_guard
from fla.utils.ascend_ub_manager import ASCEND_MAX_GRID_DIM, get_ub_manager

# Four resident [BS, BK] fp32 panels plus the rank-1 update. A full [BK, BK] tile at BK=128
# is 64KiB and the broadcasts overflow the 192KiB UB.
_BK = 128
_BS = 32
assert _BK // _BS == 4
_peak_bytes = 5 * _BS * _BK * 4


@triton.jit
def _load_panels(ptr, K, BK: tl.constexpr, BS: tl.constexpr):
    cols = tl.arange(0, BK)
    col_mask = cols < K
    b0 = tl.zeros([BS, BK], dtype=tl.float32)
    b1 = tl.zeros([BS, BK], dtype=tl.float32)
    b2 = tl.zeros([BS, BK], dtype=tl.float32)
    b3 = tl.zeros([BS, BK], dtype=tl.float32)
    for s0 in tl.static_range(4):
        rows = s0 * BS + tl.arange(0, BS)
        panel = tl.load(
            ptr + rows[:, None] * K + cols[None, :],
            mask=(rows < K)[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if s0 == 0:
            b0 = panel
        elif s0 == 1:
            b1 = panel
        elif s0 == 2:
            b2 = panel
        else:
            b3 = panel
    return b0, b1, b2, b3


@triton.jit
def _store_panels(ptr, b0, b1, b2, b3, K, BK: tl.constexpr, BS: tl.constexpr):
    cols = tl.arange(0, BK)
    col_mask = cols < K
    for s0 in tl.static_range(4):
        rows = s0 * BS + tl.arange(0, BS)
        mask = (rows < K)[:, None] & col_mask[None, :]
        if s0 == 0:
            panel = b0
        elif s0 == 1:
            panel = b1
        elif s0 == 2:
            panel = b2
        else:
            panel = b3
        tl.store(ptr + rows[:, None] * K + cols[None, :], panel, mask=mask)


@triton.jit
def _zeros_panels(BK: tl.constexpr, BS: tl.constexpr):
    return (
        tl.zeros([BS, BK], dtype=tl.float32),
        tl.zeros([BS, BK], dtype=tl.float32),
        tl.zeros([BS, BK], dtype=tl.float32),
        tl.zeros([BS, BK], dtype=tl.float32),
    )


@triton.jit
def _scale_panels(b0, b1, b2, b3, gamma):
    return b0 * gamma, b1 * gamma, b2 * gamma, b3 * gamma


@triton.jit
def _rank1_panels(b0, b1, b2, b3, b_k, p_k, K, BK: tl.constexpr, BS: tl.constexpr):
    for s0 in tl.static_range(4):
        rows = s0 * BS + tl.arange(0, BS)
        b_kr = tl.load(p_k + rows, mask=rows < K, other=0.0).to(tl.float32)
        update = b_kr[:, None] * b_k[None, :]
        if s0 == 0:
            b0 = b0 + update
        elif s0 == 1:
            b1 = b1 + update
        elif s0 == 2:
            b2 = b2 + update
        else:
            b3 = b3 + update
    return b0, b1, b2, b3


@triton.jit
def _panel_fro(b0, b1, b2, b3):
    sq = tl.sum(b0 * b0) + tl.sum(b1 * b1) + tl.sum(b2 * b2) + tl.sum(b3 * b3)
    return tl.sqrt(sq)


@triton.jit
def _panel_matvec(b0, b1, b2, b3, b_x, K, BK: tl.constexpr, BS: tl.constexpr):
    cols = tl.arange(0, BK)
    b_r = tl.zeros([BK], dtype=tl.float32)
    for s0 in tl.static_range(4):
        rows = s0 * BS + tl.arange(0, BS)
        # Keep the iterate in UB: a global scratch round-trip can read stale data
        # across successive iterations on Ascend.
        b_xr = tl.gather(b_x, rows, axis=0)
        if s0 == 0:
            panel = b0
        elif s0 == 1:
            panel = b1
        elif s0 == 2:
            panel = b2
        else:
            panel = b3
        b_r += tl.sum(panel * b_xr[:, None], axis=0)
    return tl.where(cols < K, b_r, 0.0)


@triton.jit
def _chebyshev(b0, b1, b2, b3, b_q, b_fro, ridge_ratio, num_iter, K, BK: tl.constexpr, BS: tl.constexpr):
    # Spectrum of H + lamb I lies in [lamb, lamb + ||H||_F], lamb = ridge_ratio * ||H||_F.
    b_lamb = ridge_ratio * b_fro
    b_step = 2.0 / (2.0 * b_lamb + b_fro)
    b_rho = b_fro / (2.0 * b_lamb + b_fro) / 2.0
    b_rho = b_rho * b_rho
    cols = tl.arange(0, BK)
    b_x_prev = tl.zeros([BK], dtype=tl.float32)
    b_x = tl.where(cols < K, b_step * b_q, 0.0)
    b_w = 2.0
    for _ in range(num_iter):
        b_w = 1.0 / (1.0 - b_rho * b_w)
        b_r = _panel_matvec(b0, b1, b2, b3, b_x, K, BK, BS) + b_lamb * b_x - b_q
        b_d = b_step * b_w * b_r + (b_w - 1.0) * b_x_prev
        b_x_prev = b_x
        b_x = tl.where(cols < K, b_w * b_x - b_d, 0.0)
    return b_x


@triton.jit
def _absorb_token(
    b0, b1, b2, b3, b_q, b_k, p_k, b_fro, LOAD_FRO: tl.constexpr,
    ridge_ratio, num_iter, K, BK: tl.constexpr, BS: tl.constexpr,
):
    b0, b1, b2, b3 = _rank1_panels(b0, b1, b2, b3, b_k, p_k, K, BK, BS)
    if not LOAD_FRO:
        b_fro = _panel_fro(b0, b1, b2, b3)
    b_x = _chebyshev(b0, b1, b2, b3, b_q, b_fro, ridge_ratio, num_iter, K, BK, BS)
    return b0, b1, b2, b3, b_x, b_fro


@triton.heuristics({
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'task_offset', 'num_iter'])
def fused_recurrent_gka_solve_fwd_kernel_npu(
    q, k, gk, x, h0, ht, cu_seqlens,
    ridge_ratio, T, task_offset, num_iter,
    H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BS: tl.constexpr,
    USE_GK: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_task = tl.cast(tl.program_id(0), tl.int64) + task_offset
    i_n = i_task // H
    i_h = i_task % H
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        n_tokens = eos - bos
    else:
        bos = i_n * T
        n_tokens = T

    cols = tl.arange(0, BK)
    stride = tl.cast(H, tl.int64) * K
    head = tl.cast(i_h, tl.int64) * K
    state = i_task * K * K
    if USE_INITIAL_STATE:
        b0, b1, b2, b3 = _load_panels(h0 + state, K, BK, BS)
    else:
        b0, b1, b2, b3 = _zeros_panels(BK, BS)

    for t in range(n_tokens):
        token = bos + tl.cast(t, tl.int64)
        p_qk = token * stride + head
        b_q = tl.load(q + p_qk + cols, mask=cols < K, other=0.0).to(tl.float32)
        b_k = tl.load(k + p_qk + cols, mask=cols < K, other=0.0).to(tl.float32)
        if USE_GK:
            # gk is [B, H, T] (packed: [1, H, T]); time is contiguous.
            if IS_VARLEN:
                g_off = tl.cast(i_h, tl.int64) * T + token
            else:
                g_off = (i_n * H + i_h) * T + tl.cast(t, tl.int64)
            b0, b1, b2, b3 = _scale_panels(b0, b1, b2, b3, exp(tl.load(gk + g_off).to(tl.float32)))
        b0, b1, b2, b3, b_x, _ = _absorb_token(
            b0, b1, b2, b3, b_q, b_k, k + p_qk, 0.0, False,
            ridge_ratio, num_iter, K, BK, BS,
        )
        tl.store(x + p_qk + cols, b_x.to(x.dtype.element_ty, fp_downcast_rounding='rtne'), mask=cols < K)

    if STORE_FINAL_STATE:
        _store_panels(ht + state, b0, b1, b2, b3, K, BK, BS)


def _run_fused_solve(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor | None,
    ridge_ratio: float,
    num_iter: int,
    h0: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, K = q.shape
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    x = torch.empty_like(q)
    ht = q.new_empty(N, H, K, K, dtype=torch.float) if output_final_state else None
    # [B, H, T] so repeated loads along time are contiguous. H == 1 does not copy.
    gk_c = None if gk is None else gk.transpose(1, 2)
    if gk_c is not None and not gk_c.is_contiguous():
        gk_c = gk_c.contiguous()
    task_num = N * H
    if task_num > 0:
        # workspace multibuffer adds a CV slot that pushes the 4-panel state over the 192KiB UB.
        compile_kwargs = ascend_compile_kwargs(workspace_multibuffer=0)
        kwargs = dict(
            q=q, k=k, gk=gk_c, x=x, h0=h0, ht=ht, cu_seqlens=cu_seqlens,
            ridge_ratio=ridge_ratio, T=T, num_iter=num_iter,
            H=H, K=K, BK=_BK, BS=_BS,
        )
        for offset in range(0, task_num, ASCEND_MAX_GRID_DIM):
            count = min(ASCEND_MAX_GRID_DIM, task_num - offset)
            fused_recurrent_gka_solve_fwd_kernel_npu[(count,)](
                task_offset=offset, **kwargs, **compile_kwargs,
            )
    return x, ht


def fused_recurrent_gka_verifier(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    scale: float | None = None,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[bool, str | None]:
    del v, g, gk, alpha, scale, ridge_ratio, num_iter, initial_state, output_final_state, cu_seqlens
    if q.device.type != 'npu' or k.device.type != 'npu':
        return False, 'GKA Ascend solve requires NPU tensors'
    budget = int(get_ub_manager().ub_capacity_bytes * 0.85)
    if _peak_bytes > budget:
        return False, f'GKA state panels need {_peak_bytes} bytes but the UB safety budget is {budget} bytes'
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32) or k.dtype not in (
        torch.float16, torch.bfloat16, torch.float32,
    ):
        return False, 'GKA Ascend solve received an unsupported dtype'
    return True, None


class FusedRecurrentGKANPUFunction(FusedRecurrentGKAFunction):

    @staticmethod
    @input_guard
    def forward(ctx, q, k, v, g, gk, alpha, scale, ridge_ratio, num_iter, h_kk0, h_kv0, output_final_state, cu_seqlens):
        x, h_kk = _run_fused_solve(q, k, gk, ridge_ratio, num_iter, h_kk0, output_final_state, cu_seqlens)
        q_mix = x if alpha is None else (q + alpha[..., None] * (x - q)).to(q.dtype)
        o, h_kv = fused_recurrent_simple_gla(
            q=q_mix,
            k=k,
            v=v,
            g=g,
            scale=scale,
            initial_state=h_kv0,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        return o, h_kk, h_kv


@register(fused_recurrent_gka, backend=TritonAscendBackend, verifier=fused_recurrent_gka_verifier)
def fused_recurrent_gka_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    scale: float | None = None,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
):
    # The solve kernel addresses tokens as packed `[B, T, ...]`. A decode step passes
    # `x[:, t:t+1]`, whose batch stride is still the full sequence, so batch > 0 would
    # read the wrong token. The stock path gets this from `input_guard`; match it here.
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    g = g.contiguous() if g is not None else None
    gk = gk.contiguous() if gk is not None else None
    alpha = alpha.contiguous() if alpha is not None else None
    if initial_state is not None:
        initial_state = tuple(s.contiguous() if s is not None else None for s in initial_state)
    h_kk0, h_kv0, scale = check_gka_inputs(q, k, v, g, gk, alpha, scale, num_iter, initial_state, cu_seqlens)
    o, h_kk, h_kv = FusedRecurrentGKANPUFunction.apply(
        q, k, v, g, gk, alpha, scale, ridge_ratio, num_iter, h_kk0, h_kv0, output_final_state, cu_seqlens,
    )
    final_state = (h_kk, h_kv) if output_final_state else None
    return o, final_state
