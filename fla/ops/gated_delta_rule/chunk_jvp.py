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

import math
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp2
from fla.utils import IS_NVIDIA_BLACKWELL, IS_TF32_SUPPORTED, input_guard

__all__ = ['chunk_gated_delta_rule_jvp', 'chunk_gated_delta_rule_jvp_fwd', 'chunk_gated_delta_rule_jvp_ref']


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


##################################################################################################################
# Fast path: the library's forward kernels for the primal, then three Triton kernels (intra-chunk tangents u_dot,
# w_dot; the sequential tangent recurrence; the output tangent), with the D directions as a grid dimension so
# that the primal tensors (q, k, v, w, u, nu, h, A, g) are never copied per direction.
##################################################################################################################

LN2 = math.log(2.0)

# The forward state kernel of the library stays at num_warps=2 on Blackwell (a tl.dot recurrence race in Triton
# before 3.8); the tangent recurrence below has the same structure and follows the same rule.
JVP_H_NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]


def _resolve_precision(input_precision: str | None) -> str:
    r"""fp32 tl.dot precision of the JVP kernels: explicit argument, else TRITON_F32_DEFAULT, else tf32 (Triton default)."""
    if input_precision is None:
        input_precision = os.environ.get('TRITON_F32_DEFAULT') or ('tf32' if IS_TF32_SUPPORTED else 'ieee')
    if input_precision not in ('tf32', 'tf32x3', 'ieee'):
        raise ValueError(f"input_precision must be 'tf32', 'tf32x3' or 'ieee', got {input_precision}")
    if input_precision != 'ieee' and not IS_TF32_SUPPORTED:
        input_precision = 'ieee'
    return input_precision


def _prune_ieee(configs, named_args, **kwargs):
    r"""IEEE fp32 tl.dot compiles to FMA loops that take minutes per config to build: tune only the smallest tiles."""
    if kwargs.get('DOT_PRECISION') != 'ieee':
        return configs
    small = [c for c in configs if c.kwargs.get('BV', 0) == min(x.kwargs.get('BV', 0) for x in configs)
             and c.kwargs.get('BK', 0) == min(x.kwargs.get('BK', 0) for x in configs)
             and c.num_stages == min(x.num_stages for x in configs)]
    return small or configs[:1]


@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [16, 32, 64]
        for num_warps in JVP_H_NUM_WARPS
        for num_stages in [1, 2, 3]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'FUSE_O', 'DOT_PRECISION'],
    prune_configs_by={'early_config_prune': _prune_ieee},
)
@triton.jit(do_not_specialize=['T', 'B'])
def chunk_gated_delta_rule_jvp_fwd_kernel_h(
    q, k, w, nu, h, g, P,
    kd, wd, ud, gd,
    hd, nud, od,
    s_kd_d, s_kd_b, s_kd_t, s_kd_h,
    s_wd_d, s_wd_b, s_wd_t, s_wd_h,
    s_ud_d, s_ud_b, s_ud_t, s_ud_h,
    s_gd_d, s_gd_b, s_gd_t, s_gd_h,
    scale,
    T,
    B,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    FUSE_O: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    r"""
    Sequential tangent recurrence over chunks, one program per (value block, batch*head, direction).

    Primal inputs (stored by the forward): k, w [B, T, H, K]; nu = v_new [B, T, H, V] (before gating);
    h [B, NT, H, K, V] (state entering each chunk); g [B, T, H] within-chunk cumulative gate in log2 units.
    Tangent inputs (strided, feature stride 1): kd, wd [D, B, T, H, K]; ud [D, B, T, H, V];
    gd [D, B, T, H] within-chunk cumulative gate tangent in NATURAL units (d/de exp2(g) = exp2(g) * gd).
    Per chunk (L = last valid row):
        nu_dot = u_dot - w_dot h - w h_dot
        h_dot <- exp2(g_L) (h_dot + gd_L h)
                 + k^T [exp2(g_L - g) (nu_dot + (gd_L - gd) nu)] + k_dot^T [exp2(g_L - g) nu]
    If FUSE_O: writes the part of the output tangent that needs h_dot and nu_dot,
        od = scale [exp2(g) (q h_dot) + P nu_dot]     (P [B, T, H, BT]: P_rs = (q_r . k_s) exp2(g_r - g_s), s <= r)
    so that h_dot and nu_dot are never stored; chunk_gated_delta_rule_jvp_fwd_kernel_o then adds the rest.
    Else: stores hd [D, B, NT, H, K, V] (h_dot entering each chunk) and nud [D, B, T, H, V] (nu_dot).
    """
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v, i_bhd = pid % NV, pid // NV
    i_bh, i_d = i_bhd % (B * H), i_bhd // (B * H)
    i_b, i_h = i_bh // H, i_bh % H
    NT = tl.cdiv(T, BT)
    bos = i_b * T

    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    nu += (bos * H + i_h) * V
    g += bos * H + i_h
    h += (i_b * NT * H + i_h) * K * V
    kd += i_d * s_kd_d + i_b * s_kd_b + i_h * s_kd_h
    wd += i_d * s_wd_d + i_b * s_wd_b + i_h * s_wd_h
    ud += i_d * s_ud_d + i_b * s_ud_b + i_h * s_ud_h
    gd += i_d * s_gd_d + i_b * s_gd_b + i_h * s_gd_h
    if FUSE_O:
        q += (bos * H + i_h) * K
        P += (bos * H + i_h) * BT
        od += ((i_d * B + i_b) * T * H + i_h) * V
    else:
        hd += ((i_d * B + i_b) * NT * H + i_h) * K * V
        nud += ((i_d * B + i_b) * T * H + i_h) * V

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]

    # h_dot entering chunk 0 is zero (no initial state)
    b_hd = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t0 in range(NT):
        i_t = i_t0.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tv = m_t[:, None] & m_v[None, :]
        m_tk = m_t[:, None] & m_k[None, :]

        if not FUSE_O:
            tl.store(hd + i_t * H * K * V + o_k[:, None] * V + o_v[None, :], b_hd.to(hd.dtype.element_ty), mask=m_kv)
        # [BK, BV] primal state entering this chunk
        b_h = tl.load(h + i_t * H * K * V + o_k[:, None] * V + o_v[None, :], mask=m_kv, other=0.).to(tl.float32)
        # [BT, BK]
        b_w = tl.load(w + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.)
        b_wd = tl.load(wd + o_t[:, None] * s_wd_t + o_k[None, :], mask=m_tk, other=0.).to(b_w.dtype)
        # nu_dot = u_dot - w_dot h - w h_dot: [BT, BV]
        b_nud = tl.load(ud + o_t[:, None] * s_ud_t + o_v[None, :], mask=m_tv, other=0.).to(tl.float32)
        b_nud -= tl.dot(b_wd, b_h.to(b_w.dtype), input_precision=DOT_PRECISION)
        b_nud -= tl.dot(b_w, b_hd.to(b_w.dtype), input_precision=DOT_PRECISION)

        last_idx = tl.minimum((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + last_idx * H).to(tl.float32)
        b_g = tl.load(g + o_t * H, mask=m_t, other=0.).to(tl.float32)
        if FUSE_O:
            # output part with h_dot (entering this chunk) and nu_dot: scale [exp2(g) (q h_dot) + P nu_dot]
            b_q = tl.load(q + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.)
            b_P = tl.load(P + o_t[:, None] * (H * BT) + tl.arange(0, BT)[None, :], mask=m_t[:, None], other=0.)
            b_od = tl.dot(b_q, b_hd.to(b_q.dtype), input_precision=DOT_PRECISION) * exp2(b_g)[:, None]
            b_od = tl.dot(b_P.to(b_q.dtype), b_nud.to(b_q.dtype), b_od, input_precision=DOT_PRECISION)
            tl.store(od + o_t[:, None] * (H * V) + o_v[None, :], (b_od * scale).to(od.dtype.element_ty), mask=m_tv)
        else:
            tl.store(nud + o_t[:, None] * (H * V) + o_v[None, :], b_nud.to(nud.dtype.element_ty), mask=m_tv)
        b_gd_last = tl.load(gd + last_idx * s_gd_t).to(tl.float32)
        b_gd = tl.load(gd + o_t * s_gd_t, mask=m_t, other=0.).to(tl.float32)
        # exp(G_L - G_r), zero on padded rows (also avoids 0 * inf there)
        b_dec = tl.where(m_t, exp2(b_g_last - b_g), 0.)
        b_nu = tl.load(nu + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.).to(tl.float32)
        b_x = (b_nud + (b_gd_last - b_gd)[:, None] * b_nu) * b_dec[:, None]
        b_y = b_nu * b_dec[:, None]

        # [BK, BT]
        b_kt = tl.load(k + o_k[:, None] + o_t[None, :] * (H * K), mask=m_k[:, None] & m_t[None, :], other=0.)
        b_kdt = tl.load(kd + o_k[:, None] + o_t[None, :] * s_kd_t, mask=m_k[:, None] & m_t[None, :], other=0.)
        b_hd = exp2(b_g_last) * (b_hd + b_gd_last * b_h)
        b_hd = tl.dot(b_kt, b_x.to(b_kt.dtype), b_hd, input_precision=DOT_PRECISION)
        b_hd = tl.dot(b_kdt.to(b_kt.dtype), b_y.to(b_kt.dtype), b_hd, input_precision=DOT_PRECISION)


@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [4, 8]
        for num_stages in [2, 3]
    ],
    key=['H', 'K', 'V', 'BT', 'FUSE_O', 'DOT_PRECISION'],
    prune_configs_by={'early_config_prune': _prune_ieee},
    restore_value=['od'],
)
@triton.jit(do_not_specialize=['T', 'B'])
def chunk_gated_delta_rule_jvp_fwd_kernel_o(
    q, k, nu, h, g,
    qd, kd, nud, hd, gd,
    od,
    s_qd_d, s_qd_b, s_qd_t, s_qd_h,
    s_kd_d, s_kd_b, s_kd_t, s_kd_h,
    s_gd_d, s_gd_b, s_gd_t, s_gd_h,
    scale,
    T,
    B,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    FUSE_O: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    r"""
    Output tangent, one program per (value block * chunk, batch*head*direction). With P_{rs} = (q_r . k_s)
    exp(G_r - G_s) [s <= r] and P_dot from (q_dot_r . k_s + q_r . k_dot_s):
        o_dot_r = scale [exp(G_r) (q_dot_r h + q_r h_dot + G_dot_r q_r h)
                         + sum_s (P_dot_{rs} + P_{rs} (G_dot_r - G_dot_s)) nu_s + sum_s P_{rs} nu_dot_s]
    which equals O(q_dot, k, nu, h) + O(q, k_dot, nu, 0) + O(q, k, nu_dot - G_dot nu, h_dot) + G_dot o
    in one pass, without materializing the primal o or per-direction copies of q, k, nu, h.
    If FUSE_O, the terms with h_dot and nu_dot were written into od by the recurrence kernel: add the rest to them.
    """
    i_v, i_t = tl.program_id(0).to(tl.int64) % tl.cdiv(V, BV), tl.program_id(0).to(tl.int64) // tl.cdiv(V, BV)
    i_bhd = tl.program_id(1).to(tl.int64)
    i_bh, i_d = i_bhd % (B * H), i_bhd // (B * H)
    i_b, i_h = i_bh // H, i_bh % H
    NT = tl.cdiv(T, BT)
    bos = i_b * T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    nu += (bos * H + i_h) * V
    g += bos * H + i_h
    h += ((i_b * NT + i_t) * H + i_h) * K * V
    qd += i_d * s_qd_d + i_b * s_qd_b + i_h * s_qd_h
    kd += i_d * s_kd_d + i_b * s_kd_b + i_h * s_kd_h
    gd += i_d * s_gd_d + i_b * s_gd_b + i_h * s_gd_h
    if not FUSE_O:
        nud += ((i_d * B + i_b) * T * H + i_h) * V
        hd += (((i_d * B + i_b) * NT + i_t) * H + i_h) * K * V
    od += ((i_d * B + i_b) * T * H + i_h) * V

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    b_o0 = tl.zeros([BT, BV], dtype=tl.float32)   # q h
    b_o1 = tl.zeros([BT, BV], dtype=tl.float32)   # q_dot h (+ q h_dot)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)    # q k^T
    b_Ad = tl.zeros([BT, BT], dtype=tl.float32)   # q_dot k^T + q k_dot^T
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        m_kt = m_k[:, None] & m_t[None, :]
        m_kv = m_k[:, None] & m_v[None, :]
        b_q = tl.load(q + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.)
        b_qd = tl.load(qd + o_t[:, None] * s_qd_t + o_k[None, :], mask=m_tk, other=0.).to(b_q.dtype)
        b_kt = tl.load(k + o_k[:, None] + o_t[None, :] * (H * K), mask=m_kt, other=0.)
        b_kdt = tl.load(kd + o_k[:, None] + o_t[None, :] * s_kd_t, mask=m_kt, other=0.).to(b_q.dtype)
        b_h = tl.load(h + o_k[:, None] * V + o_v[None, :], mask=m_kv, other=0.).to(b_q.dtype)
        b_o0 = tl.dot(b_q, b_h, b_o0, input_precision=DOT_PRECISION)
        b_o1 = tl.dot(b_qd, b_h, b_o1, input_precision=DOT_PRECISION)
        if not FUSE_O:
            b_hd = tl.load(hd + o_k[:, None] * V + o_v[None, :], mask=m_kv, other=0.).to(b_q.dtype)
            b_o1 = tl.dot(b_q, b_hd, b_o1, input_precision=DOT_PRECISION)
        b_A = tl.dot(b_q, b_kt, b_A, input_precision=DOT_PRECISION)
        b_Ad = tl.dot(b_qd, b_kt, b_Ad, input_precision=DOT_PRECISION)
        b_Ad = tl.dot(b_q, b_kdt, b_Ad, input_precision=DOT_PRECISION)

    b_g = tl.load(g + o_t * H, mask=m_t, other=0.).to(tl.float32)
    b_gd = tl.load(gd + o_t * s_gd_t, mask=m_t, other=0.).to(tl.float32)
    b_o = exp2(b_g)[:, None] * (b_o1 + b_gd[:, None] * b_o0)
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_dec = tl.where(m_A, exp2(b_g[:, None] - b_g[None, :]), 0.)
    b_Ad = (b_Ad + b_A * (b_gd[:, None] - b_gd[None, :])) * b_dec
    b_A = b_A * b_dec

    m_tv = m_t[:, None] & m_v[None, :]
    b_nu = tl.load(nu + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.)
    b_o = tl.dot(b_Ad.to(b_nu.dtype), b_nu, b_o, input_precision=DOT_PRECISION)
    if FUSE_O:
        b_o = b_o * scale + tl.load(od + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.).to(tl.float32)
    else:
        b_nud = tl.load(nud + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.).to(b_nu.dtype)
        b_o = tl.dot(b_A.to(b_nu.dtype), b_nud, b_o, input_precision=DOT_PRECISION) * scale
    tl.store(od + o_t[:, None] * (H * V) + o_v[None, :], b_o.to(od.dtype.element_ty), mask=m_tv)


@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [4, 8]
        for num_stages in [2, 3]
    ],
    key=['H', 'K', 'V', 'BT', 'DOT_PRECISION'],
    prune_configs_by={'early_config_prune': _prune_ieee},
)
@triton.jit(do_not_specialize=['T', 'B'])
def chunk_gated_delta_rule_jvp_fwd_kernel_intra(
    k, v, beta, g, A, u, w,
    kd, vd, bd, gd,
    ud, wd,
    s_kd_d, s_kd_b, s_kd_t, s_kd_h,
    s_vd_d, s_vd_b, s_vd_t, s_vd_h,
    s_bd_d, s_bd_b, s_bd_t, s_bd_h,
    s_gd_d, s_gd_b, s_gd_t, s_gd_h,
    T,
    B,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    r"""
    Intra-chunk tangents of the WY representation, one program per (chunk, batch*head*direction):
        M_dot = [beta_dot_r (k_r.k_s) + beta_r (k_dot_r.k_s + k_r.k_dot_s) + beta_r (k_r.k_s)(G_dot_r - G_dot_s)]
                * exp2(g_r - g_s)  (r > s)
        u_dot = A [beta_dot v + beta v_dot - M_dot u]
        w_dot = A [exp2(g) ((beta_dot + beta G_dot) k + beta k_dot) - M_dot w]
    Primal: k [B, T, H, K], v, u [B, T, H, V], w [B, T, H, K], beta [B, T, H], g [B, T, H] (log2 cumulative),
    A [B, T, H, BT] = (I + M)^{-1}. Tangents strided (kd, vd [D, B, T, H, *], bd, gd [D, B, T, H]; gd natural units).
    Outputs ud [D, B, T, H, V], wd [D, B, T, H, K] (contiguous).
    """
    i_t, i_bhd = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_bh, i_d = i_bhd % (B * H), i_bhd // (B * H)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    u += (bos * H + i_h) * V
    beta += bos * H + i_h
    g += bos * H + i_h
    A += (bos * H + i_h) * BT
    kd += i_d * s_kd_d + i_b * s_kd_b + i_h * s_kd_h
    vd += i_d * s_vd_d + i_b * s_vd_b + i_h * s_vd_h
    bd += i_d * s_bd_d + i_b * s_bd_b + i_h * s_bd_h
    gd += i_d * s_gd_d + i_b * s_gd_b + i_h * s_gd_h
    ud += ((i_d * B + i_b) * T * H + i_h) * V
    wd += ((i_d * B + i_b) * T * H + i_h) * K

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_c = tl.arange(0, BT)

    b_b = tl.load(beta + o_t * H, mask=m_t, other=0.).to(tl.float32)
    b_bd = tl.load(bd + o_t * s_bd_t, mask=m_t, other=0.).to(tl.float32)
    b_g = tl.load(g + o_t * H, mask=m_t, other=0.).to(tl.float32)
    b_gd = tl.load(gd + o_t * s_gd_t, mask=m_t, other=0.).to(tl.float32)
    b_A = tl.load(A + o_t[:, None] * (H * BT) + o_c[None, :], mask=m_t[:, None], other=0.).to(tl.float32)

    # (k_r . k_s) and (k_dot_r . k_s), accumulated over key blocks
    b_kk = tl.zeros([BT, BT], dtype=tl.float32)
    b_kdk = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k < K)[None, :]
        b_k = tl.load(k + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.).to(tl.float32)
        b_kd = tl.load(kd + o_t[:, None] * s_kd_t + o_k[None, :], mask=m_tk, other=0.).to(tl.float32)
        b_kt = tl.trans(b_k)
        b_kk = tl.dot(b_k, b_kt, b_kk, input_precision=DOT_PRECISION)
        b_kdk = tl.dot(b_kd, b_kt, b_kdk, input_precision=DOT_PRECISION)
    m_s = (o_c[:, None] > o_c[None, :]) & (m_t[:, None] & m_t[None, :])
    b_dec = tl.where(m_s, exp2(b_g[:, None] - b_g[None, :]), 0.)
    b_Md = (b_bd[:, None] * b_kk
            + b_b[:, None] * (b_kdk + tl.trans(b_kdk) + b_kk * (b_gd[:, None] - b_gd[None, :]))) * b_dec

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v < V)[None, :]
        b_v = tl.load(v + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.).to(tl.float32)
        b_vd = tl.load(vd + o_t[:, None] * s_vd_t + o_v[None, :], mask=m_tv, other=0.).to(tl.float32)
        b_u = tl.load(u + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.).to(tl.float32)
        b_rhs = b_bd[:, None] * b_v + b_b[:, None] * b_vd
        b_rhs -= tl.dot(b_Md, b_u, input_precision=DOT_PRECISION)
        b_ud = tl.dot(b_A, b_rhs, input_precision=DOT_PRECISION)
        tl.store(ud + o_t[:, None] * (H * V) + o_v[None, :], b_ud.to(ud.dtype.element_ty), mask=m_tv)

    b_eg = exp2(b_g)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k < K)[None, :]
        b_k = tl.load(k + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.).to(tl.float32)
        b_kd = tl.load(kd + o_t[:, None] * s_kd_t + o_k[None, :], mask=m_tk, other=0.).to(tl.float32)
        b_w = tl.load(w + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.).to(tl.float32)
        b_rhs = b_eg[:, None] * ((b_bd + b_b * b_gd)[:, None] * b_k + b_b[:, None] * b_kd)
        b_rhs -= tl.dot(b_Md, b_w, input_precision=DOT_PRECISION)
        b_wd = tl.dot(b_A, b_rhs, input_precision=DOT_PRECISION)
        tl.store(wd + o_t[:, None] * (H * K) + o_k[None, :], b_wd.to(wd.dtype.element_ty), mask=m_tk)


@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [2, 4]
        for num_stages in [2, 3]
    ],
    key=['H', 'K', 'BT', 'DOT_PRECISION'],
    prune_configs_by={'early_config_prune': _prune_ieee},
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_jvp_fwd_kernel_qk(
    q, k, g, P,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    r"""P [B, T, H, BT]: P_rs = (q_r . k_s) exp2(g_r - g_s) for s <= r within a chunk, else 0 (direction-independent)."""
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += bos * H + i_h
    P += (bos * H + i_h) * BT
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_q = tl.load(q + o_t[:, None] * (H * K) + o_k[None, :], mask=m_t[:, None] & m_k[None, :], other=0.)
        b_kt = tl.load(k + o_k[:, None] + o_t[None, :] * (H * K), mask=m_k[:, None] & m_t[None, :], other=0.)
        b_A = tl.dot(b_q, b_kt, b_A, input_precision=DOT_PRECISION)
    b_g = tl.load(g + o_t * H, mask=m_t, other=0.).to(tl.float32)
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_A = tl.where(m_A, b_A * exp2(b_g[:, None] - b_g[None, :]), 0.)
    tl.store(P + o_t[:, None] * (H * BT) + tl.arange(0, BT)[None, :], b_A.to(P.dtype.element_ty), mask=m_t[:, None])


def _strides4(x: torch.Tensor, chunked: bool) -> tuple[int, int, int, int]:
    r"""(direction, batch, time, head) strides of a tangent: [D, B, T, H, *] or, if chunked, [D, B, H, N, C, *]."""
    if chunked:
        # [D, B, H, N, C, *]: time t = n * C + c has stride C-stride (N and C are contiguous with each other)
        assert x.stride(3) == x.shape[4] * x.stride(4)
        return x.stride(0), x.stride(1), x.stride(4), x.stride(2)
    return x.stride(0), x.stride(1), x.stride(2), x.stride(3)


def chunk_gated_delta_rule_jvp_fwd(
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
    scale: float,
    chunk_size: int = 64,
    input_precision: str | None = None,
    intra_impl: str = 'triton',
    fuse_output: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Implementation of :func:`chunk_gated_delta_rule_jvp` (no argument checking). Primal inputs contiguous;
    tangents with feature stride 1. Returns (o [B, T, H, V], o_dot [D, B, T, H, V]).
    ``intra_impl``: 'triton' (fused kernel) or 'torch' (batched PyTorch ops; the slow-but-simple path).
    ``fuse_output``: compute the output terms that need h_dot and nu_dot inside the recurrence kernel, so neither
    is stored (saves D * B * T * H * (K * V / BT + V) elements); False stores them for a separate output pass.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    D = q_dot.shape[0]
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    precision = _resolve_precision(input_precision)

    # ---- primal forward: the library's own kernels (identical to chunk_gated_delta_rule_fwd without initial state)
    g_cum = chunk_local_cumsum(g, chunk_size=BT, scale=RCP_LN2)                    # [B, T, H], log2 units, fp32
    w, u, A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g_cum, beta=beta, chunk_size=BT)
    h, nu, _ = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g_cum, chunk_size=BT)  # h [B, NT, H, K, V], nu = v_new
    o = chunk_fwd_o(q=q, k=k, v=nu, h=h, g=g_cum, scale=scale, chunk_size=BT)

    # ---- tangent of the within-chunk cumulative gate, natural units: [D, B, T, H] fp32
    G_dot = chunk_local_cumsum(g_dot.reshape(D * B, T, H).contiguous(), chunk_size=BT).view(D, B, T, H)

    # ---- intra-chunk tangents u_dot, w_dot, batched over chunks and directions
    if intra_impl == 'triton':
        u_dot = torch.empty(D, B, T, H, V, dtype=k.dtype, device=k.device)
        w_dot = torch.empty(D, B, T, H, K, dtype=k.dtype, device=k.device)
        chunk_gated_delta_rule_jvp_fwd_kernel_intra[(NT, D * B * H)](
            k, v, beta, g_cum, A, u, w,
            k_dot, v_dot, beta_dot, G_dot,
            u_dot, w_dot,
            *_strides4(k_dot, False),
            *_strides4(v_dot, False),
            *_strides4(beta_dot, False),
            *_strides4(G_dot, False),
            T, B,
            H=H, K=K, V=V, BT=BT,
            DOT_PRECISION=precision,
        )
        ud_chunked = wd_chunked = False
    else:
        # slow-but-simple PyTorch path (fp32; IEEE unless the caller enabled TF32 matmuls in PyTorch, or
        # input_precision='ieee' forces it here)
        f32 = torch.float32
        kc, vc = _chunk(k, BT, 0).to(f32), _chunk(v, BT, 0).to(f32)                # [B, H, N, C, *]
        bc = _chunk(beta, BT, 0).to(f32)                                            # [B, H, N, C]
        Gc = _chunk(g_cum, BT, 0) * LN2                                             # natural units, exp(Gc) = exp2(g_cum)
        Ac = _chunk(A, BT, 0).to(f32)                                               # [B, H, N, C, C]
        uc, wc = _chunk(u, BT, 0).to(f32), _chunk(w, BT, 0).to(f32)
        kdc, vdc = _chunk(k_dot, BT, 1).to(f32), _chunk(v_dot, BT, 1).to(f32)      # [D, B, H, N, C, *]
        bdc, Gdc = _chunk(beta_dot, BT, 1).to(f32), _chunk(G_dot, BT, 1)            # [D, B, H, N, C]
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        if precision == 'ieee':
            torch.backends.cuda.matmul.allow_tf32 = False
        try:
            u_dot, w_dot = _intra_chunk_tangents(kc, vc, bc, Gc, Ac, uc, wc, kdc, vdc, bdc, Gdc)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        del kc, vc, bc, Gc, Ac, uc, wc, kdc, vdc, bdc, Gdc
        if k.dtype != f32:
            u_dot, w_dot = u_dot.to(k.dtype), w_dot.to(k.dtype)
        ud_chunked = wd_chunked = True

    # ---- sequential tangent recurrence (Triton), and with fuse_output the output terms that need h_dot, nu_dot
    BK = max(triton.next_power_of_2(K), 16)
    assert K <= 256, "the JVP state kernel supports head dimension K <= 256"
    o_dot = torch.empty(D, B, T, H, V, dtype=v.dtype, device=v.device)
    if fuse_output:
        P = torch.empty(B, T, H, BT, dtype=k.dtype, device=k.device)
        chunk_gated_delta_rule_jvp_fwd_kernel_qk[(NT, B * H)](
            q, k, g_cum, P, T, H=H, K=K, BT=BT, DOT_PRECISION=precision,
        )
        h_dot = nu_dot = None
    else:
        P = None
        h_dot = k.new_empty(D, B, NT, H, K, V)
        nu_dot = torch.empty(D, B, T, H, V, dtype=nu.dtype, device=nu.device)

    def grid_h(meta): return (triton.cdiv(V, meta['BV']) * B * H * D,)
    chunk_gated_delta_rule_jvp_fwd_kernel_h[grid_h](
        q, k, w, nu, h, g_cum, P,
        k_dot, w_dot, u_dot, G_dot,
        h_dot, nu_dot, o_dot,
        *_strides4(k_dot, False),
        *_strides4(w_dot, wd_chunked),
        *_strides4(u_dot, ud_chunked),
        *_strides4(G_dot, False),
        scale,
        T, B,
        H=H, K=K, V=V, BT=BT, BK=BK,
        FUSE_O=fuse_output,
        DOT_PRECISION=precision,
    )
    del w_dot, u_dot, P

    # ---- output tangent (Triton): all of it, or (fuse_output) the terms without h_dot and nu_dot, added to o_dot
    def grid_o(meta): return (triton.cdiv(V, meta['BV']) * NT, D * B * H)
    chunk_gated_delta_rule_jvp_fwd_kernel_o[grid_o](
        q, k, nu, h, g_cum,
        q_dot, k_dot, nu_dot, h_dot, G_dot,
        o_dot,
        *_strides4(q_dot, False),
        *_strides4(k_dot, False),
        *_strides4(G_dot, False),
        scale,
        T, B,
        H=H, K=K, V=V, BT=BT,
        FUSE_O=fuse_output,
        DOT_PRECISION=precision,
    )
    return o.to(q.dtype), o_dot


@input_guard(no_guard_contiguous=['q_dot', 'k_dot', 'v_dot', 'g_dot', 'beta_dot'])
def chunk_gated_delta_rule_jvp(
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
    input_precision: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Forward-mode derivative (JVP) of :func:`chunk_gated_delta_rule` along D tangent directions at once.

    Args:
        q, k (torch.Tensor): `[B, T, H, K]` (the caller applies any l2 normalization).
        v (torch.Tensor): `[B, T, H, V]`.
        g (torch.Tensor): `[B, T, H]` per-step log decay (NOT cumulative).
        beta (torch.Tensor): `[B, T, H]`.
        q_dot, k_dot, v_dot, g_dot, beta_dot (torch.Tensor): tangents of the same shapes with a leading
            direction dimension `D`, e.g. `q_dot` is `[D, B, T, H, K]`. Any strides with unit feature stride.
        scale (float): defaults to `K ** -0.5`.
        chunk_size (int): 16, 32 or 64.
        input_precision (str): fp32 tl.dot precision of the three JVP kernels (intra-chunk tangents, tangent
            recurrence, output tangent): 'ieee', 'tf32' or 'tf32x3'. Default (None): TRITON_F32_DEFAULT if set,
            else 'tf32' (Triton's default). The library's forward kernels, run unchanged for the primal, use
            Triton's default (TRITON_F32_DEFAULT) and keep TF32 in the solve_tril block merge. 'ieee' compiles
            slowly (minutes on first use per shape) and runs slower; 'tf32x3' is close to IEEE accuracy at
            tensor-core speed.

    No initial state, no final state, no variable lengths, no GVA (v has the heads of q and k).

    Returns:
        o (torch.Tensor): `[B, T, H, V]`, equal to `chunk_gated_delta_rule(q, k, v, g, beta, scale)[0]`.
        o_dot (torch.Tensor): `[D, B, T, H, V]`, the derivative of o along each direction.
    """
    B, T, H, K = q.shape
    if k.shape != q.shape or v.shape[:3] != (B, T, H) or g.shape != (B, T, H) or beta.shape != (B, T, H):
        raise ValueError(f"bad shapes q {tuple(q.shape)} k {tuple(k.shape)} v {tuple(v.shape)} "
                         f"g {tuple(g.shape)} beta {tuple(beta.shape)}")
    D = q_dot.shape[0]
    for name, x, p in (('q_dot', q_dot, q), ('k_dot', k_dot, k), ('v_dot', v_dot, v),
                       ('g_dot', g_dot, g), ('beta_dot', beta_dot, beta)):
        if x.shape != (D, *p.shape):
            raise ValueError(f"{name} must have shape {(D, *p.shape)}, got {tuple(x.shape)}")
    if chunk_size not in (16, 32, 64):
        raise ValueError(f"chunk_size must be 16, 32 or 64, got {chunk_size}")
    if scale is None:
        scale = K ** -0.5

    def unit_feature_stride(x):
        return x if x.stride(-1) == 1 else x.contiguous()
    q_dot, k_dot, v_dot = (unit_feature_stride(x) for x in (q_dot, k_dot, v_dot))
    # g_dot, beta_dot are reshaped/chunked in PyTorch; any strides work there
    return chunk_gated_delta_rule_jvp_fwd(
        q, k, v, g, beta, q_dot, k_dot, v_dot, g_dot, beta_dot,
        scale=scale, chunk_size=chunk_size, input_precision=input_precision,
    )
