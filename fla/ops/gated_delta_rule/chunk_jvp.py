# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

r"""
Forward-mode derivative (JVP) of the chunked gated delta rule.

Notation (one sequence, one head; chunk n of C positions r, s; key features kappa; value features mu;
a dot marks a tangent). All gates here are in natural-log units; the Triton kernels store the within-chunk
cumulative gate in log2 units (``chunk_local_cumsum(..., scale=RCP_LN2)``) so that ``exp2(g_cum) = exp(G)``.

Forward (as the library computes it):
    G_{n,r}          = sum_{s <= r, s in chunk n} g_s                       (within-chunk cumulative gate)
    M_{n,rs}         = beta_r (k_r . k_s) exp(G_r - G_s)       for r > s, else 0
    A_n              = (I + M_n)^{-1}                                       (unit lower triangular)
    u_{n,r mu}       = sum_s A_{rs} beta_s v_{s mu}
    w_{n,r kappa}    = sum_s A_{rs} beta_s exp(G_s) k_{s kappa}
    nu_{n,r mu}      = u_{n,r mu} - sum_kappa w_{n,r kappa} h_{n,kappa mu}  (the library's ``v_new``)
    h_{n+1,kappa mu} = exp(G_L) h_{n,kappa mu} + sum_r k_{r kappa} exp(G_L - G_r) nu_{r mu}   (L = last row of chunk n)
    o_{r mu}         = scale [exp(G_r) sum_kappa q_{r kappa} h_{n,kappa mu}
                              + sum_{s <= r} (q_r . k_s) exp(G_r - G_s) nu_{s mu}]       =: O(q, k, nu, h)_{r mu}

Tangent:
    M_dot_{rs} = [beta_dot_r (k_r.k_s) + beta_r (k_dot_r.k_s + k_r.k_dot_s)
                  + beta_r (k_r.k_s)(G_dot_r - G_dot_s)] exp(G_r - G_s)                 (r > s)
    A_dot      = -A M_dot A, hence
    u_dot      = A [beta_dot v + beta v_dot - M_dot u]
    w_dot      = A [exp(G) (beta_dot k + beta G_dot k + beta k_dot) - M_dot w]
    nu_dot     = u_dot - w_dot h_n - w h_dot_n
    h_dot_{n+1} = exp(G_L) (h_dot_n + G_dot_L h_n)
                  + sum_r k_r exp(G_L - G_r) [nu_dot_r + (G_dot_L - G_dot_r) nu_r]
                  + sum_r k_dot_r exp(G_L - G_r) nu_r
    o_dot      = O(q_dot, k, nu, h) + O(q, k_dot, nu, 0) + O(q, k, nu_dot - G_dot nu, h_dot) + G_dot o
(the last line: O is linear in q, k, (nu, h) separately, and d/dG of the decays gives G_dot_r o_r - [terms with
G_dot_s nu_s], which is the -G_dot nu inside the third call.)
"""

import torch
import torch.nn.functional as F

__all__ = ['chunk_gated_delta_rule_jvp_ref']


def _chunk(x: torch.Tensor, C: int, lead: int) -> torch.Tensor:
    r"""[*lead, B, T, H, *f] -> [*lead, B, H, N, C, *f], zero-padding T up to a multiple of C."""
    T = x.shape[lead + 1]
    pad = (-T) % C
    if pad:
        # F.pad pads from the last dim backwards: we need to pad dim (lead + 1)
        nf = x.dim() - (lead + 2)  # number of dims after T (H and features)
        x = F.pad(x, (0, 0) * nf + (0, pad))
    shape = x.shape
    x = x.view(*shape[:lead + 1], (T + pad) // C, C, *shape[lead + 2:])
    # [*lead, B, N, C, H, *f] -> [*lead, B, H, N, C, *f]
    return x.movedim(lead + 3, lead + 1)


def _unchunk(x: torch.Tensor, T: int, lead: int) -> torch.Tensor:
    r"""[*lead, B, H, N, C, *f] -> [*lead, B, T, H, *f]."""
    x = x.movedim(lead + 1, lead + 3)
    shape = x.shape
    x = x.reshape(*shape[:lead + 1], shape[lead + 1] * shape[lead + 2], *shape[lead + 3:])
    return x[(slice(None),) * (lead + 1) + (slice(0, T),)]


def _intra_chunk_tangents(
    k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, G: torch.Tensor,
    A: torch.Tensor, u: torch.Tensor, w: torch.Tensor,
    k_dot: torch.Tensor, v_dot: torch.Tensor, beta_dot: torch.Tensor, G_dot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Tangents of the WY representation, batched over chunks (no loop).

    Primal (chunked layout, ``...`` = [B, H, N]): k [..., C, K], v [..., C, V], beta [..., C], G [..., C] (natural log),
    A [..., C, C] = (I + M)^{-1}, u [..., C, V], w [..., C, K].
    Tangents carry one extra leading direction dim D: k_dot [D, ..., C, K], etc.
    Returns u_dot [D, ..., C, V], w_dot [D, ..., C, K].
    """
    C = k.shape[-2]
    strict = torch.ones(C, C, dtype=torch.bool, device=k.device).tril(-1)
    # decay_{rs} = exp(G_r - G_s) on r > s, 0 elsewhere (masked before exp: the upper triangle can overflow)
    dG = G[..., :, None] - G[..., None, :]
    decay = torch.where(strict, dG, float('-inf')).exp()
    kk = k @ k.transpose(-1, -2)                                   # (k_r . k_s)
    kdk = k_dot @ k.transpose(-1, -2)                              # (k_dot_r . k_s); its transpose is (k_r . k_dot_s)
    # M_dot_{rs} = [beta_dot_r kk + beta_r (kdk + kdk^T) + beta_r kk (G_dot_r - G_dot_s)] decay_{rs}
    M_dot = (beta_dot[..., :, None] * kk
             + beta[..., :, None] * (kdk + kdk.transpose(-1, -2)
                                     + kk * (G_dot[..., :, None] - G_dot[..., None, :]))) * decay
    eG = G.exp()
    u_dot = A @ (beta_dot[..., None] * v + beta[..., None] * v_dot - M_dot @ u)
    w_dot = A @ ((eG[..., None] * (beta_dot[..., None] * k + (beta * G_dot)[..., None] * k + beta[..., None] * k_dot))
                 - M_dot @ w)
    return u_dot, w_dot


def _chunk_o(q, k, nu, h, G, scale):
    r"""
    O(q, k, nu, h)_{r mu} = scale [exp(G_r) q_r . h_{:, mu} + sum_{s <= r} (q_r . k_s) exp(G_r - G_s) nu_{s mu}].
    Chunked layout; h [..., N, K, V] or None (zero state); broadcasting over a leading direction dim is allowed.
    """
    C = q.shape[-2]
    causal = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    decay = torch.where(causal, G[..., :, None] - G[..., None, :], float('-inf')).exp()
    o = ((q @ k.transpose(-1, -2)) * decay) @ nu
    if h is not None:
        o = o + G.exp()[..., None] * (q @ h)
    return o * scale


def chunk_gated_delta_rule_jvp_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    q_dot: torch.Tensor,
    k_dot: torch.Tensor,
    v_dot: torch.Tensor,
    g_dot: torch.Tensor,
    beta_dot: torch.Tensor,
    scale: float | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Pure-PyTorch chunked reference of the gated delta rule and its forward-mode derivative.

    Computes in the input dtype (use float64 for a reference). Vectorized over chunks except for the
    sequential state recurrence (a loop over chunks, never over tokens).

    Args:
        q, k: [B, T, H, K]; v: [B, T, H, V]; g: [B, T, H] per-step log decay (not cumulative); beta: [B, T, H].
        q_dot, k_dot, v_dot, g_dot, beta_dot: the same shapes with a leading direction dim D.
    Returns:
        o [B, T, H, V], o_dot [D, B, T, H, V]
    """
    B, T, H, K = q.shape
    C = chunk_size
    if scale is None:
        scale = K ** -0.5
    qc, kc, vc = (_chunk(x, C, 0) for x in (q, k, v))                      # [B, H, N, C, F]
    bc, gc = (_chunk(x, C, 0) for x in (beta, g))                          # [B, H, N, C]
    qdc, kdc, vdc = (_chunk(x, C, 1) for x in (q_dot, k_dot, v_dot))      # [D, B, H, N, C, F]
    bdc, gdc = (_chunk(x, C, 1) for x in (beta_dot, g_dot))                # [D, B, H, N, C]
    N = qc.shape[2]

    # within-chunk cumulative gate (padded positions have g = 0, so G stays at its last valid value)
    G, G_dot = gc.cumsum(-1), gdc.cumsum(-1)

    # primal WY representation
    strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    decay = torch.where(strict, G[..., :, None] - G[..., None, :], float('-inf')).exp()
    M = bc[..., :, None] * (kc @ kc.transpose(-1, -2)) * decay
    eye = torch.eye(C, dtype=q.dtype, device=q.device)
    A = torch.linalg.solve_triangular(eye + M, eye.expand_as(M), upper=False, unitriangular=True)
    u = A @ (bc[..., None] * vc)
    w = A @ ((bc * G.exp())[..., None] * kc)
    u_dot, w_dot = _intra_chunk_tangents(kc, vc, bc, G, A, u, w, kdc, vdc, bdc, G_dot)

    # sequential state recurrence over chunks (h_n is the state entering chunk n)
    D = q_dot.shape[0]
    V = v.shape[-1]
    h = q.new_zeros(B, H, K, V)
    h_dot = q.new_zeros(D, B, H, K, V)
    hs, hds, nus, nuds = [], [], [], []
    for n in range(N):
        hs.append(h)
        hds.append(h_dot)
        nu = u[:, :, n] - w[:, :, n] @ h
        nu_dot = u_dot[:, :, :, n] - w_dot[:, :, :, n] @ h - w[:, :, n] @ h_dot
        nus.append(nu)
        nuds.append(nu_dot)
        GL, GdL = G[:, :, n, -1], G_dot[:, :, :, n, -1]                     # [B, H], [D, B, H]
        dec = (GL[..., None] - G[:, :, n]).exp()                             # exp(G_L - G_r): [B, H, C]
        kdec = kc[:, :, n] * dec[..., None]                                  # [B, H, C, K]
        h_new = GL.exp()[..., None, None] * h + kdec.transpose(-1, -2) @ nu
        h_dot = (GL.exp()[..., None, None] * (h_dot + GdL[..., None, None] * h)
                 + kdec.transpose(-1, -2) @ (nu_dot + (GdL[..., None] - G_dot[:, :, :, n])[..., None] * nu)
                 + (kdc[:, :, :, n] * dec[..., None]).transpose(-1, -2) @ nu)
        h = h_new
    hs, hds = torch.stack(hs, 2), torch.stack(hds, 3)                        # [B, H, N, K, V], [D, B, H, N, K, V]
    nus, nuds = torch.stack(nus, 2), torch.stack(nuds, 3)                    # [B, H, N, C, V], [D, B, H, N, C, V]

    o = _chunk_o(qc, kc, nus, hs, G, scale)
    o_dot = (_chunk_o(qdc, kc, nus, hs, G, scale)
             + _chunk_o(qc, kdc, nus, None, G, scale)
             + _chunk_o(qc, kc, nuds - G_dot[..., None] * nus, hds, G, scale)
             + G_dot[..., None] * o)
    return _unchunk(o, T, 0), _unchunk(o_dot, T, 1)
