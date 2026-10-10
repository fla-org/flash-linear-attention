# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Precond-KDA backward kernels adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import check_shared_mem

_NUM_WARPS = 4


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_precond_kda_bwd_kernel_dAv_npu(
    q,
    k,
    v,
    A,
    do,
    dv,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    dA += (bos * H + i_h) * BT

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_r = tl.arange(0, BT)
    p_A = A + (bos * H + i_h) * BT + o_r[:, None] + o_t[None, :] * (H*BT)
    # NPU delta: fold the triangular + token mask into the load - the
    # tl.where form miscompiles. Both axes shift by o_t, so compare with o_r.
    m_A = (o_r[:, None] <= o_r[None, :]) & (m_t[:, None] & m_t[None, :])
    b_A = tl.load(p_A, mask=m_A, other=0.0).to(do.dtype.element_ty)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        # [BV, BT] transposed
        p_v = v + o_v[:, None] + o_t[None, :] * (H*V)
        p_do = do + o_t[:, None] * (H*V) + o_v[None, :]
        p_dv = dv + o_t[:, None] * (H*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v[:, None] & m_t[None, :], other=0.0)
        # [BT, BV]
        b_do = tl.load(p_do, mask=m_tv, other=0.0)
        # [BT, BT]
        b_dA += tl.dot(b_do, b_v)
        # [BT, BV]
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(dv.dtype.element_ty), mask=m_tv)

    o_A = tl.arange(0, BT)
    p_dA = dA + o_t[:, None] * (H*BT) + o_A[None, :]
    b_dA = tl.where(o_t[:, None] >= o_t, b_dA * scale, 0.)
    tl.store(p_dA, b_dA.to(dA.dtype.element_ty), mask=m_t[:, None] & (o_A[None, :] < BT))


def chunk_precond_kda_bwd_dAv_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    A: torch.Tensor | None = None,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, do.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    # H100 can have larger block size
    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem():
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    # float(scale): a unit int constant miscompiles under the CANN-bundled hivmc (#1298).
    if scale is not None:
        scale = float(scale)

    dA = v.new_empty(B, T, H, BT, dtype=torch.float)
    dv = torch.empty_like(do)
    grid = (NT, B * H)
    chunk_precond_kda_bwd_kernel_dAv_npu[grid](
        q=q,
        k=k,
        v=v,
        A=A,
        do=do,
        dv=dv,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        num_warps=_NUM_WARPS,
        num_stages=2,
    )
    return dA, dv


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_precond_kda_bwd_kernel_wy_dqkg_npu(
    q,
    k,           # original k (for WY backward)
    k_precond,   # preconditioned k (for inter backward)
    v,
    v_new,
    g,
    beta,
    A,
    h,
    do,
    dh,
    dq,
    dk,          # dk from WY backward (original k)
    dkg,         # dkg from inter backward (k_precond)
    dv,
    dv2,
    dg,
    db,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = (o_t == min(T, i_t * BT + BT) - 1)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    k_precond += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    v_new += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    beta += bos * H + i_h
    A += (bos * H + i_h) * BT
    h += (i_tg * H + i_h) * K*V
    do += (bos * H + i_h) * V
    dh += (i_tg * H + i_h) * K*V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dkg += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    dg += (bos * H + i_h) * K
    db += bos * H + i_h
    dA += (bos * H + i_h) * BT

    b_beta = tl.load(beta + o_t*H, mask=m_t, other=0.0)

    o_r = tl.arange(0, BT)
    p_A = A + o_r[:, None] + o_t[None, :] * (H * BT)
    b_A = tl.load(p_A, mask=(o_r[:, None] < BT) & m_t[None, :], other=0.0)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_db = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        m_tk = m_t[:, None] & m_k[None, :]
        p_k = k + o_t[:, None] * (H*K) + o_k[None, :]
        p_kp = k_precond + o_t[:, None] * (H*K) + o_k[None, :]
        p_g = g + o_t[:, None] * (H*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_kp = tl.load(p_kp, mask=m_tk, other=0.0)
        b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)

        p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * H*K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dkg_raw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            m_v = o_v < V
            m_tv = m_t[:, None] & m_v[None, :]
            m_vk = m_v[:, None] & m_k[None, :]
            p_v_new = v_new + o_t[:, None] * (H*V) + o_v[None, :]
            p_do = do + o_t[:, None] * (H*V) + o_v[None, :]
            if TRANSPOSE_STATE:
                p_h = h + o_v[:, None] * K + o_k[None, :]
                p_dh = dh + o_v[:, None] * K + o_k[None, :]
            else:
                p_h = h + o_v[:, None] + o_k[None, :] * V
                p_dh = dh + o_v[:, None] + o_k[None, :] * V
            p_dv = dv + o_t[:, None] * (H*V) + o_v[None, :]
            # [BT, BV]
            b_v_new = tl.load(p_v_new, mask=m_tv, other=0.0)
            b_do = tl.load(p_do, mask=m_tv, other=0.0)
            # [BV, BK]
            b_h = tl.load(p_h, mask=m_vk, other=0.0)
            b_dh = tl.load(p_dh, mask=m_vk, other=0.0)
            # [BT, BV]
            b_dv = tl.load(p_dv, mask=m_tv, other=0.0)

            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            b_dkg_raw += tl.dot(b_v_new, b_dh.to(b_v_new.dtype))
            b_dw += tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype))
            tl.debug_barrier()  # DO NOT REMOVE THIS LINE!
            if i_k == 0:
                p_v = v + o_t[:, None] * (H*V) + o_v[None, :]
                p_dv2 = dv2 + o_t[:, None] * (H*V) + o_v[None, :]

                b_v = tl.load(p_v, mask=m_tv, other=0.0)

                b_dA += tl.dot(b_dv, tl.trans(b_v))

                b_dvb = tl.dot(b_A, b_dv)
                b_dv2 = b_dvb * b_beta[:, None]
                b_db += tl.sum(b_dvb * b_v, 1)

                tl.store(p_dv2, b_dv2.to(dv2.dtype.element_ty), mask=m_tv)

        b_gk_exp = exp2(b_g)
        b_gb = b_gk_exp * b_beta[:, None]
        b_dgk *= exp2(b_gn)
        b_dq = b_dq * b_gk_exp * scale

        # WY backward: uses original k
        b_kg_orig = b_k * b_gk_exp

        b_dw = -b_dw.to(b_A.dtype)
        b_dA += tl.dot(b_dw, tl.trans(b_kg_orig.to(b_A.dtype)))

        b_dkgb = tl.dot(b_A, b_dw)
        b_db += tl.sum(b_dkgb * b_kg_orig, 1)

        # dk from WY backward (original k)
        b_dk = b_dkgb * b_gb

        # Inter backward: uses k_precond
        f_t = m_t[:, None].to(tl.float32)
        # Mask folded into the exponent: inf * 0 would be NaN.
        b_gn_g = exp2((b_gn[None, :] - b_g) * f_t - 127.0 * (1.0 - f_t)) * f_t
        b_dkg = b_dkg_raw * b_gn_g

        b_kg_precond = b_kp * b_gn_g
        b_kp_dkg = b_kg_precond * b_dkg_raw
        b_dgk += tl.sum(b_kp_dkg, axis=0)

        p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_tk, other=0.0)

        b_dg = (b_q * b_dq                           # inter query
                - b_kp_dkg                              # inter write key
                + m_last[:, None] * b_dgk               # accumulated last position
                + b_kg_orig * b_dkgb * b_beta[:, None])  # WY backward

        p_dq = dq + o_t[:, None] * (H*K) + o_k[None, :]
        p_dk = dk + o_t[:, None] * (H*K) + o_k[None, :]
        p_dkg = dkg + o_t[:, None] * (H*K) + o_k[None, :]
        p_dg = dg + o_t[:, None] * (H*K) + o_k[None, :]
        tl.store(p_dq, b_dq.to(dq.dtype.element_ty), mask=m_tk)
        tl.store(p_dk, b_dk.to(dk.dtype.element_ty), mask=m_tk)
        tl.store(p_dkg, b_dkg.to(dkg.dtype.element_ty), mask=m_tk)
        tl.store(p_dg, b_dg.to(dg.dtype.element_ty), mask=m_tk)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    f_A = m_A.to(tl.float32)
    b_dA = (b_dA * b_beta[None, :]) * f_A
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = -b_dA * f_A

    o_A = tl.arange(0, BT)
    p_dA = dA + o_t[:, None] * (H * BT) + o_A[None, :]
    tl.store(p_dA, b_dA.to(dA.dtype.element_ty), mask=m_t[:, None] & (o_A[None, :] < BT))
    tl.store(db + o_t*H, b_db.to(db.dtype.element_ty), mask=m_t)


def _chunk_precond_kda_bwd_wy_dqkg_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    k_precond: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None,
    BT: int,
    transpose_state_layout: bool,
):
    """Torch mirror of chunk_precond_kda_bwd_kernel_wy_dqkg_npu, full K/V width.

    b_A[r, c] reads A[token(o_t[c]), r] and b_h is [V, K], matching the
    kernel's transposed views. One (sequence, chunk) per iteration, batched
    over heads.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    dev = k.device
    varlen = cu_seqlens is not None

    dq = torch.zeros(B, T, H, K, dtype=torch.float, device=dev)
    dk = torch.zeros(B, T, H, K, dtype=torch.float, device=dev)
    dkg = torch.zeros(B, T, H, K, dtype=torch.float, device=dev)
    dv2 = torch.zeros(B, T, H, V, dtype=v.dtype, device=dev)
    dg = torch.zeros(B, T, H, K, dtype=torch.float, device=dev)
    db = torch.zeros(B, T, H, dtype=torch.float, device=dev)
    dA = torch.zeros(B, T, H, BT, dtype=torch.float, device=dev)

    o = torch.arange(BT, device=dev)

    def state_view(s):
        # s: [H, K, V] or [H, V, K] -> [H, V, K] view matching b_h/b_dh
        return s if transpose_state_layout else s.transpose(-1, -2)

    if varlen:
        bos_all = cu_seqlens[:-1].to(torch.long)
        eos_all = cu_seqlens[1:].to(torch.long)
        seqs = chunk_indices.tolist()
    else:
        bos_all = torch.zeros(B, dtype=torch.long, device=dev)
        eos_all = torch.full((B,), T, dtype=torch.long, device=dev)
        seqs = [(b, i_t) for b in range(B) for i_t in range(triton.cdiv(T, BT))]

    for row, (i_n, i_t) in enumerate(seqs):
        bos, eos = int(bos_all[i_n]), int(eos_all[i_n])
        seq_t = eos - bos
        i_tg = row if varlen else i_n * triton.cdiv(T, BT) + i_t

        o_t = i_t * BT + o
        valid = o_t < seq_t                                  # [BT]
        idx = torch.clamp(bos + o_t, max=eos - 1)             # global tokens, clamped
        m_last = (o_t == min(seq_t, i_t * BT + BT) - 1).to(torch.float)

        def gather(x):
            # rows of this chunk -> [BT, H, *] -> [H, BT, *]
            g_ = x[0, idx] if varlen else x[i_n, idx]
            return g_.transpose(0, 1).float()

        q_r, k_r, kp_r, g_r = gather(q), gather(k), gather(k_precond), gather(g)
        v_r, vn_r, do_r, dv_r = gather(v), gather(v_new), gather(do), gather(dv)
        vld = valid.view(1, BT, 1)
        v_r, vn_r, do_r, dv_r = v_r * vld, vn_r * vld, do_r * vld, dv_r * vld

        beta_r = (beta[0, idx] if varlen else beta[i_n, idx]).transpose(0, 1).float() * valid  # [H, BT]

        # b_A[r, c] = A[token(o_t[c]), r] - transposed like the kernel load
        b_A = (A[0, idx] if varlen else A[i_n, idx]).permute(1, 2, 0).float()  # [H, BT, BT]

        # h/dh are [B, NT, H, K, V]; the kernel indexes them flat, so flatten.
        h_flat = h.reshape(-1, H, h.shape[-2], h.shape[-1])
        dh_flat = dh.reshape(-1, H, dh.shape[-2], dh.shape[-1])
        b_h = state_view(h_flat[i_tg].float())              # [H, V, K]
        b_dh = state_view(dh_flat[i_tg].float())

        b_dgk = (b_h * b_dh).sum(1)                           # [H, K]
        b_dq = torch.matmul(do_r, b_h)                        # [H, BT, K]
        b_dkg_raw = torch.matmul(vn_r, b_dh)
        b_dw = torch.matmul(dv_r, b_h)

        b_dA = torch.matmul(dv_r, v_r.transpose(1, 2))        # [H, BT, BT]
        b_dvb = torch.matmul(b_A, dv_r)                       # [H, BT, V]
        b_dv2 = b_dvb * beta_r.unsqueeze(-1)
        b_db = (b_dvb * v_r).sum(-1)                          # [H, BT]
        # only valid rows: clamped invalid rows would clobber real tokens
        sel = (0, idx[valid]) if varlen else (i_n, idx[valid])
        dv2[sel] = b_dv2.transpose(0, 1)[valid].to(dv2.dtype)

        gk_exp = torch.exp2(torch.clamp(g_r, max=88.0))
        b_dq = b_dq * gk_exp * scale
        kg_orig = k_r * gk_exp
        b_dw = -b_dw
        b_dA = b_dA + torch.matmul(b_dw, kg_orig.transpose(1, 2))
        b_dkgb = torch.matmul(b_A, b_dw)                      # [H, BT, K]
        b_db = b_db + (b_dkgb * kg_orig).sum(-1)
        b_dk = b_dkgb * (gk_exp * beta_r.unsqueeze(-1))

        gn_tok = min(bos + min(seq_t, i_t * BT + BT) - 1, eos - 1)
        gn = (g[0, gn_tok] if varlen else g[i_n, gn_tok]).float()   # [H, K]
        gn_g = torch.exp2(torch.clamp(gn.unsqueeze(1) - g_r, max=88.0)) * vld
        b_dkg = b_dkg_raw * gn_g
        kp_dkg = kp_r * gn_g * b_dkg_raw
        b_dgk = b_dgk * torch.exp2(torch.clamp(gn, max=88.0)) + kp_dkg.sum(1)

        b_dg = (q_r * b_dq - kp_dkg
                + m_last.view(1, BT, 1) * b_dgk.unsqueeze(1)
                + kg_orig * b_dkgb * beta_r.unsqueeze(-1))

        for out, src in ((dq, b_dq), (dk, b_dk), (dkg, b_dkg), (dg, b_dg)):
            out[sel] = src.transpose(0, 1)[valid]

        m_A = ((o.view(-1, 1) > o.view(1, -1)) & valid.view(-1, 1) & valid.view(1, -1)).float()
        fin = (b_dA * beta_r.unsqueeze(1)) * m_A
        fin = torch.matmul(torch.matmul(b_A, fin), b_A)
        fin = -fin * m_A
        dA[sel] = fin.transpose(0, 1)[valid]
        db[sel] = b_db.transpose(0, 1)[valid]

    return dq, dk, dkg, dv2, db, dg, dA


def chunk_precond_kda_bwd_wy_dqkg_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    k_precond: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
):
    """
    Inter-chunk + WY backward for preconditioned KDA.

    Matches KDA's chunk_kda_bwd_wy_dqkg_fused but with k/k_precond asymmetry:
    - WY backward uses original k (for w = A @ (k * beta * exp(gk)))
    - Inter backward uses k_precond (for kg = k_precond * exp(gn - gk))

    Args:
        q: [B, T, H, K] - queries
        k: [B, T, H, K] - original k (for WY backward)
        k_precond: [B, T, H, K] - preconditioned k (for inter backward)
        v: [B, T, H, V] - original v (for WY backward)
        v_new: [B, T, H, V] - corrected v (for inter backward)
        g: [B, T, H, K] - gate cumsum (log2 space)
        beta: [B, T, H] - beta scaling
        A: [B, T, H, BT] - WY inverse matrix
        h: [NT, H, K, V] - per-chunk hidden states
        do: [B, T, H, V] - output gradient
        dh: [NT, H, K, V] - hidden state gradient
        dv: [B, T, H, V] - dv from h backward (= du for WY)

    Returns:
        dq, dk, dkg, dv, db, dg, dA
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    # float(scale): a unit int constant miscompiles under the CANN-bundled hivmc (#1298).
    if scale is not None:
        scale = float(scale)

    # The CANN-bundled hivmc cannot build the triton kernel for half
    # dtypes with K > 64; fall back to torch (as the intra backward does).
    if k.dtype in (torch.float16, torch.bfloat16) and K > 64:
        return _chunk_precond_kda_bwd_wy_dqkg_torch(
            q, k, k_precond, v, v_new, g, beta, A, h, do, dh, dv, scale,
            cu_seqlens, chunk_indices, BT, transpose_state_layout,
        )

    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    dkg = torch.empty_like(k_precond, dtype=torch.float)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float)
    db = torch.empty_like(beta, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)

    grid = (NT, B * H)
    chunk_precond_kda_bwd_kernel_wy_dqkg_npu[grid](
        q=q,
        k=k,
        k_precond=k_precond,
        v=v,
        v_new=v_new,
        g=g,
        beta=beta,
        A=A,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dkg=dkg,
        dv=dv,
        dv2=dv2,
        dg=dg,
        db=db,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        # BV=128 crashes the CANN-bundled hivmc; BK=32/BV=64 is verified working.
        BK=32,
        BV=64,
        TRANSPOSE_STATE=transpose_state_layout,
        num_warps=_NUM_WARPS,
        num_stages=2,
        # multibuffer=False: multi-buffering deadlocks the aicore on fp16
        # inputs under the CANN-bundled hivmc.
        multibuffer=False,
    )
    dv = dv2
    return dq, dk, dkg, dv, db, dg, dA
