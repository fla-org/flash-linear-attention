# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Dispatched ``common`` primitives for the momentum delta rule chunk path.

Mirrors the delta-rule decomposition (``fla/ops/delta_rule/chunk.py``): the op-level
``chunk_momentum_delta_rule`` composes these primitives, and on Ascend NPU the
triton-ascend backend (``fla/ops/common/backends/triton_ascend``) overrides them
with kernels. The base implementations below are plain torch and numerically follow
``fla/ops/momentum_delta_rule/naive.py``'s chunk reference (fp32 throughout).

Layout conventions:
  - gate vectors are `[B, H, NT, BT]` fp32 contiguous (one `[BT]` vector per chunk).
  - the kkt ``A`` tensor is `[B, T, H, BT]`: chunk row ``t`` stores its ``BT`` columns,
    i.e. the strictly-lower entries of ``attn``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from fla.backends import dispatch


def _pad_t(x: torch.Tensor, pT: int) -> torch.Tensor:
    pad = pT - x.shape[1]
    if not pad:
        return x
    if x.dim() == 3:
        return F.pad(x, (0, 0, 0, pad))
    return F.pad(x, (0, 0, 0, 0, 0, pad))


def _gamma_mask(b_t_row: torch.Tensor, cfac: torch.Tensor, lct1: torch.Tensor, lct_row: torch.Tensor) -> torch.Tensor:
    """Rank-1 mask: mask[i, j] = b_t_row[i] * cfac[j] * (1 - exp(lct1[j] - lct_row[i])).

    ``b_t_row``/``lct_row`` are the per-chunk row factors (`b_t`/`b_tm1` for
    gamma_mask_q/gamma_mask and `log_ct`/`lct1` for the row shift).
    """
    diff = lct1[..., None, :] - lct_row[..., :, None]
    return b_t_row[..., :, None] * cfac[..., None, :] * (1.0 - torch.exp(diff))


@dispatch('common')
def chunk_momentum_delta_kkt_fwd(
    p_eff: torch.Tensor,
    k_eta: torch.Tensor,
    cfac: torch.Tensor,
    lct1: torch.Tensor,
    b_tm1: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """``attn = (p_eff @ k_eta^T) * gamma_mask`` packed into the ``A`` layout.

    ``gamma_mask[i, j] = b_tm1[i] * cfac[j] * (1 - exp(lct1[j] - lct1[i]))`` is strictly
    lower-triangular, so the inverse is ``(I + attn)^{-1}``.
    """
    B, T, H, K = p_eff.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    pT = NT * BT
    p_eff = _pad_t(p_eff, pT).to(torch.float32)
    k_eta = _pad_t(k_eta, pT).to(torch.float32)

    p_c = p_eff.reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)   # [B, H, NT, BT, K]
    k_c = k_eta.reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)
    attn = p_c @ k_c.transpose(-1, -2)                           # [B, H, NT, BT, BT]
    mask = _gamma_mask(b_tm1, cfac, lct1, lct1)
    attn = attn * torch.tril(mask, diagonal=-1)
    return attn.permute(0, 2, 3, 1, 4).reshape(B, pT, H, BT).contiguous()


@dispatch('common')
def chunk_momentum_delta_wy_fwd(
    A: torch.Tensor,
    v: torch.Tensor,
    p_eff: torch.Tensor,
    bar_a_tm1: torch.Tensor,
    b_tm1: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The inverse's three RHS: ``u_c = Ai @ v``, ``y_c = Ai @ (bar_a_tm1 * p_eff)``,
    ``z_c = Ai @ (b_tm1 * p_eff)`` with ``Ai = (I + attn)^{-1}``.

    The inverse is built with the exact substitution loop from the naive reference
    (``naive.py:156-159``): the chunk matrices are ill-conditioned (entries up to 1e4),
    and a different summation order diverges ~4e-4 relative, which the state recurrence
    amplifies past the parity tolerance. The NPU kernel reproduces this loop.
    """
    B, T, H, K = p_eff.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    pT = NT * BT
    A_c = A.reshape(B, NT, BT, H, BT).permute(0, 3, 1, 2, 4).to(torch.float32)     # [B, H, NT, BT, BT]
    v_c = _pad_t(v, pT).to(torch.float32).reshape(B, NT, BT, H, V).permute(0, 3, 1, 2, 4)
    p_c = _pad_t(p_eff, pT).to(torch.float32).reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)

    attn_inv = -A_c
    for i in range(1, BT):
        attn_inv[..., i, :i] += (attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()).sum(-2)
    attn_inv = attn_inv + torch.eye(BT, dtype=torch.float32, device=A.device)

    u_c = attn_inv @ v_c                                           # [B, H, NT, BT, V]
    y_c = attn_inv @ (bar_a_tm1[..., :, None] * p_c)               # [B, H, NT, BT, K]
    z_c = attn_inv @ (b_tm1[..., :, None] * p_c)

    def back(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1, 4).reshape(B, pT, H, x.shape[-1]).contiguous()

    return back(u_c), back(y_c), back(z_c)


@dispatch('common')
def chunk_momentum_delta_fwd_h(
    k_eta: torch.Tensor,
    u_c: torch.Tensor,
    y_c: torch.Tensor,
    z_c: torch.Tensor,
    cfac: torch.Tensor,
    lct1: torch.Tensor,
    lm_cum: torch.Tensor,
    a_last: torch.Tensor,
    b_last: torch.Tensor,
    ct_last: torch.Tensor,
    lm_last: torch.Tensor,
    initial_S: torch.Tensor | None,
    initial_M: torch.Tensor | None,
    output_final_state: bool,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Two-state inter-chunk recurrence.

    Per chunk ``i``: ``v_i = u_c - y_c @ S_pre + z_c @ M_pre``, then the S/M updates
    with the ``decay_s``/``decay_m`` vectors (built from the precomputed scalars).
    ``h_s``/``h_m`` store the state *entering* each chunk (same convention as
    ``chunk_gated_delta_rule_fwd_h``). Returns ``(h_s, h_m, v_i, final_S, final_M)``.
    """
    B, T, H, K = k_eta.shape
    V = u_c.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    pT = NT * BT
    k_eta = _pad_t(k_eta, pT).to(torch.float32)
    u_c = _pad_t(u_c, pT).to(torch.float32)
    y_c = _pad_t(y_c, pT).to(torch.float32)
    z_c = _pad_t(z_c, pT).to(torch.float32)

    k_c = k_eta.reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)   # [B, H, NT, BT, K]
    u_c_c = u_c.reshape(B, NT, BT, H, V).permute(0, 3, 1, 2, 4)
    y_c_c = y_c.reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)
    z_c_c = z_c.reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)

    S = torch.zeros(B, H, K, V, dtype=torch.float32, device=k_eta.device)
    M = torch.zeros(B, H, K, V, dtype=torch.float32, device=k_eta.device)
    if initial_S is not None:
        S = S + initial_S.to(torch.float32)
    if initial_M is not None:
        M = M + initial_M.to(torch.float32)

    h_s = torch.empty(B, NT, H, K, V, dtype=torch.float32, device=k_eta.device)
    h_m = torch.empty(B, NT, H, K, V, dtype=torch.float32, device=k_eta.device)
    v_i = torch.empty(B, pT, H, V, dtype=torch.float32, device=k_eta.device)

    for i in range(NT):
        h_s[:, i] = S
        h_m[:, i] = M
        v_cur = u_c_c[:, :, i] - y_c_c[:, :, i] @ S + z_c_c[:, :, i] @ M            # [B, H, BT, V]
        v_i[:, i * BT:(i + 1) * BT] = v_cur.transpose(1, 2)
        # decay_s[r] = b_last * cfac[r] * (1 - exp(lct1[r] - ct_last)); decay_m[r] = exp(lm_last - lm_cum[r])
        decay_s = b_last[:, :, i:i + 1] * cfac[:, :, i] * (1.0 - torch.exp(lct1[:, :, i] - ct_last[:, :, i:i + 1]))
        decay_m = torch.exp(lm_last[:, :, i:i + 1] - lm_cum[:, :, i])
        k_cur = k_c[:, :, i]                                                        # [B, H, BT, K]
        S = a_last[:, :, i:i + 1, None] * S - b_last[:, :, i:i + 1, None] * \
            M + (k_cur * decay_s[..., None]).transpose(-1, -2) @ v_cur
        M = torch.exp(lm_last[:, :, i:i + 1, None]) * M - (k_cur * decay_m[..., None]).transpose(-1, -2) @ v_cur

    if output_final_state:
        return h_s, h_m, v_i, S, M
    return h_s, h_m, v_i, None, None


@dispatch('common')
def chunk_momentum_delta_fwd_o(
    q: torch.Tensor,
    k_eta: torch.Tensor,
    v_i: torch.Tensor,
    h_s: torch.Tensor,
    h_m: torch.Tensor,
    a_cum: torch.Tensor,
    b_t: torch.Tensor,
    cfac: torch.Tensor,
    lct: torch.Tensor,
    lct1: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> torch.Tensor:
    """``o = (q*a_cum) @ h_s - (q*b_t) @ h_m + ((q @ k_eta^T) * gamma_mask_q) @ v_i``."""
    B, T, H, K = q.shape
    V = v_i.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    pT = NT * BT
    q = _pad_t(q, pT).to(torch.float32) * scale
    k_eta = _pad_t(k_eta, pT).to(torch.float32)
    v_i = _pad_t(v_i, pT).to(torch.float32)

    q_c = q.reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)         # [B, H, NT, BT, K]
    k_c = k_eta.reshape(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)
    v_c = v_i.reshape(B, NT, BT, H, V).permute(0, 3, 1, 2, 4)
    h_s_c = h_s.transpose(1, 2)                                     # [B, H, NT, K, V]
    h_m_c = h_m.transpose(1, 2)

    qs = q_c * a_cum[..., :, None]                                  # bar_alpha_t * q
    qb = q_c * b_t[..., :, None]
    o = qs @ h_s_c - qb @ h_m_c                                     # [B, H, NT, BT, V]
    mask_q = _gamma_mask(b_t, cfac, lct1, lct)
    attn_inner = (q_c @ k_c.transpose(-1, -2)) * torch.tril(mask_q, diagonal=0)
    o = o + attn_inner @ v_c
    return o.permute(0, 2, 3, 1, 4).reshape(B, pT, H, V)[:, :T].contiguous()
