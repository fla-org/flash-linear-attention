# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from fla.backends import BaseBackend, register
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd_dqkvbg
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import IS_NVIDIA

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext


@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_bwd_kernel_rewy(
    q, k, v, g, beta, A, dr, r, h, dh, do, dq, dk, dv, db, dg, cu_seqlens, chunk_indices,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    BV: tl.constexpr, scale: tl.constexpr, IS_VARLEN: tl.constexpr, STATE_V_FIRST: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_ht = i_t
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int64)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        T = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    else:
        bos = i_b * T
        i_ht = i_b * tl.cdiv(T, BT) + i_t
    o_c = tl.arange(0, BT)
    o_t = i_t * BT + o_c
    o_v = tl.arange(0, BV)
    o_th = (bos + o_t) * H + i_h
    m_t = o_t < T
    m_pair = m_t[:, None] & m_t[None, :]
    m_v = m_t[:, None] & (o_v[None, :] < V)
    b_at = tl.load(A + o_th[None, :] * BT + o_c[:, None], mask=m_pair, other=0)
    b_dr = tl.load(dr + o_th[:, None] * V + o_v[None, :], mask=m_v, other=0)
    b_p = tl.dot(b_at, b_dr, allow_tf32=False)
    b_r = tl.load(r + o_th[:, None] * V + o_v[None, :], mask=m_v, other=0)
    b_do = tl.load(do + o_th[:, None] * V + o_v[None, :], mask=m_v, other=0)
    # the residual pullback eliminates the inverse adjoint; https://arxiv.org/abs/2611.01940, section 3.1.
    b_dm = -tl.dot(b_p.to(b_r.dtype), tl.trans(b_r), allow_tf32=False)
    b_ds = tl.dot(b_do, tl.trans(b_r), allow_tf32=False)
    b_v = tl.load(v + o_th[:, None] * V + o_v[None, :], mask=m_v, other=0)
    b_beta = tl.load(beta + o_th, mask=m_t, other=0).to(tl.float32)
    b_db = tl.sum(b_p * b_v, axis=1)
    tl.store(dv + o_th[:, None] * V + o_v[None, :], b_p * b_beta[:, None], mask=m_v)

    b_g = tl.load(g + o_th, mask=m_t, other=0).to(tl.float32)
    i_last = tl.minimum((i_t + 1) * BT, T) - 1
    b_g_last = tl.load(g + (bos + i_last) * H + i_h).to(tl.float32)
    b_exp_g = exp2(b_g)
    b_exp_last = exp2(b_g_last)
    b_exp_tail = tl.where(m_t, exp2(b_g_last - b_g), 0)
    b_decay = exp2(tl.where(m_pair & (o_c[:, None] >= o_c[None, :]), b_g[:, None] - b_g[None, :], 0))
    b_dm = tl.where(m_pair & (o_c[:, None] > o_c[None, :]), b_dm * b_decay, 0).to(k.dtype.element_ty)
    b_ds = tl.where(m_pair & (o_c[:, None] >= o_c[None, :]), b_ds * b_decay * scale, 0).to(k.dtype.element_ty)
    # the original value and beta gradients consume the local FP32 sensitivity before this dot-operand conversion.
    b_p = b_p.to(k.dtype.element_ty)
    b_dg = tl.zeros((BT,), tl.float32)
    b_dg_last = tl.full((), 0, tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        b_k = tl.load(k + o_th[:, None] * K + o_k[None, :], mask=m_k, other=0)
        b_q = tl.load(q + o_th[:, None] * K + o_k[None, :], mask=m_k, other=0)
        if STATE_V_FIRST:
            o_h = (i_ht * H + i_h) * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            o_h = (i_ht * H + i_h) * K * V + o_k[None, :] * V + o_v[:, None]
        m_h = (o_v[:, None] < V) & (o_k[None, :] < K)
        b_h = tl.load(h + o_h, mask=m_h, other=0)
        b_dq = tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False) * (b_exp_g * scale)[:, None]
        b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
        tl.store(dq + o_th[:, None] * K + o_k[None, :], b_dq, mask=m_k)
        b_dg += tl.sum(b_dq * b_q, axis=1)

        b_de = -tl.dot(b_p, b_h.to(b_p.dtype), allow_tf32=False) * b_exp_g[:, None]
        b_left = tl.dot(b_dm, b_k, allow_tf32=False)
        b_erase = b_de + b_left
        b_db_piece = tl.sum(b_erase * b_k, axis=1)
        b_db += b_db_piece
        b_dk_residual = b_erase * b_beta[:, None]
        b_dg += b_beta * b_db_piece
        b_right = tl.dot(tl.trans(b_dm), (b_k * b_beta[:, None]).to(b_k.dtype), allow_tf32=False)
        b_dk_residual += b_right
        # reuse the key contractions for the gate reduction instead of materializing a key Gram matrix.
        b_dg -= tl.sum(b_right * b_k, axis=1)

        b_dh = tl.load(dh + o_h, mask=m_h, other=0)
        b_dk_state = tl.dot(b_r, b_dh.to(b_r.dtype), allow_tf32=False) * b_exp_tail[:, None]
        b_dg_last += tl.sum(b_h.to(tl.float32) * b_dh.to(tl.float32)) * b_exp_last + tl.sum(b_dk_state * b_k)
        b_dk_direct = b_dk_state + tl.dot(tl.trans(b_ds), b_q, allow_tf32=False)
        b_dg -= tl.sum(b_dk_direct * b_k, axis=1)
        # preserve the two original low-precision key-gradient stores before their final addition.
        b_dk = b_dk_direct.to(dk.dtype.element_ty).to(tl.float32) + b_dk_residual.to(dk.dtype.element_ty).to(tl.float32)
        tl.store(dk + o_th[:, None] * K + o_k[None, :], b_dk, mask=m_k)
    tl.store(db + o_th, b_db, mask=m_t)
    b_dg = tl.where(o_t == i_last, b_dg + b_dg_last, b_dg)
    tl.store(dg + o_th, b_dg, mask=m_t)


@register(chunk_gated_delta_rule_bwd_dqkvbg)
class ReWYBackend(BaseBackend):
    """Opt-in residual WY and Gram-free GDN backward on NVIDIA GPUs."""

    backend_type = 'rewy'
    env_var = 'FLA_REWY'
    default_enable = False

    @classmethod
    def is_available(cls) -> bool:
        return IS_NVIDIA

    def chunk_gated_delta_rule_bwd_dqkvbg_verifier(
        self, q, k, v, g, beta, A, w, v_new, h, dh, dv, do, scale,
        state_v_first=False, cu_seqlens=None, chunk_indices=None, cp_context=None, chunk_size=64,
    ) -> tuple[bool, str | None]:
        if cp_context is not None:
            return False, 'ReWY context parallelism has not been validated'
        if q.dtype not in (torch.float16, torch.bfloat16) or not (q.dtype == k.dtype == v.dtype):
            return False, 'ReWY requires matching FP16 or BF16 inputs'
        if q.shape[2] != v.shape[2]:
            return False, 'ReWY grouped value heads have not been validated'
        if v.shape[-1] > 128:
            return False, 'ReWY keeps the full value tile live and currently supports V <= 128'
        if chunk_size not in (16, 32, 64):
            return False, 'ReWY supports chunk sizes 16, 32 and 64'
        return True, None

    def chunk_gated_delta_rule_bwd_dqkvbg(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        w: torch.Tensor,
        v_new: torch.Tensor,
        h: torch.Tensor,
        dh: torch.Tensor,
        dv: torch.Tensor,
        do: torch.Tensor,
        scale: float,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        chunk_size: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = v.shape[-1]
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        NT = triton.cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv_out = torch.empty_like(v)
        db = torch.empty_like(beta)
        dg = torch.empty_like(g, dtype=torch.float32)
        chunk_gated_delta_rule_bwd_kernel_rewy[(NT, B * H)](
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            dr=dv,
            r=v_new,
            h=h,
            dh=dh,
            do=do,
            dq=dq,
            dk=dk,
            dv=dv_out,
            db=db,
            dg=dg,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=chunk_size,
            BK=32,
            BV=max(16, triton.next_power_of_2(V)),
            scale=scale,
            IS_VARLEN=cu_seqlens is not None,
            STATE_V_FIRST=state_v_first,
            num_warps=4,
            num_stages=1,
        )
        return dq, dk, dv_out, db, dg
