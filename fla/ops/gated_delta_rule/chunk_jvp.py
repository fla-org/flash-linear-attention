# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

r"""
Forward-mode derivative (JVP) of the chunked gated delta rule.

The primal runs through the library's forward kernels; three Triton kernels then compute the intra-chunk tangents,
the sequential tangent recurrence and the output tangent, with the tangent directions as a grid dimension.
See README.md in this directory for the derivation and the notation used here.
"""

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import BLACKWELL_8_WARP_RACE, chunk_fwd_o
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp2
from fla.utils import IS_NVIDIA_BLACKWELL, IS_TF32_SUPPORTED, TRITON_ABOVE_3_8_0, autotune_cache_kwargs, input_guard

__all__ = ['chunk_gated_delta_rule_jvp', 'chunk_gated_delta_rule_jvp_fwd', 'chunk_gated_delta_rule_jvp_ref']


def _chunk(x: torch.Tensor, C: int, lead: int) -> torch.Tensor:
    r"""[*lead, B, T, H, *f] -> [*lead, B, H, N, C, *f], zero-padding T up to a multiple of C."""
    T = x.shape[lead + 1]
    pad = (-T) % C
    if pad:
        # F.pad pads from the last dim backwards; skip H and the feature dims to reach T
        x = F.pad(x, (0, 0) * (x.dim() - lead - 2) + (0, pad))
    shape = x.shape
    # [*lead, B, N, C, H, *f]
    x = x.view(*shape[:lead + 1], (T + pad) // C, C, *shape[lead + 2:])
    return x.movedim(lead + 3, lead + 1)


def _unchunk(x: torch.Tensor, T: int, lead: int) -> torch.Tensor:
    r"""[*lead, B, H, N, C, *f] -> [*lead, B, T, H, *f]."""
    x = x.movedim(lead + 1, lead + 3)
    shape = x.shape
    x = x.reshape(*shape[:lead + 1], shape[lead + 1] * shape[lead + 2], *shape[lead + 3:])
    return x[(slice(None),) * (lead + 1) + (slice(0, T),)]


def _intra_chunk_tangents(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    G: torch.Tensor,
    A: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    k_dot: torch.Tensor,
    v_dot: torch.Tensor,
    beta_dot: torch.Tensor,
    G_dot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Tangents (u_dot, w_dot) of the WY representation, batched over chunks.
    Primal inputs are chunked as [..., C, *] with `...` = [B, H, N] and G in natural log units;
    tangents carry an extra leading direction dim.
    """
    C = k.shape[-2]
    strict = torch.ones(C, C, dtype=torch.bool, device=k.device).tril(-1)
    # exp(G_r - G_s) on r > s and 0 elsewhere, masked before exp since the upper triangle can overflow
    decay = torch.where(strict, G[..., :, None] - G[..., None, :], float('-inf')).exp()
    kk = k @ k.transpose(-1, -2)
    # (k_dot_r . k_s); its transpose is (k_r . k_dot_s)
    kdk = k_dot @ k.transpose(-1, -2)
    M_dot = (beta_dot[..., :, None] * kk
             + beta[..., :, None] * (kdk + kdk.transpose(-1, -2)
                                     + kk * (G_dot[..., :, None] - G_dot[..., None, :]))) * decay
    eG = G.exp()
    u_dot = A @ (beta_dot[..., None] * v + beta[..., None] * v_dot - M_dot @ u)
    w_dot = A @ ((eG[..., None] * (beta_dot[..., None] * k + (beta * G_dot)[..., None] * k + beta[..., None] * k_dot))
                 - M_dot @ w)
    return u_dot, w_dot


def _chunk_o(
    q: torch.Tensor,
    k: torch.Tensor,
    nu: torch.Tensor,
    h: torch.Tensor | None,
    G: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    r"""O(q, k, nu, h) of README.md in the chunked layout; `h=None` is a zero state."""
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
    Pure-PyTorch chunked reference of :func:`chunk_gated_delta_rule_jvp`, computed in the input dtype.
    It loops over chunks for the state recurrence and is batched otherwise.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            Per-step log decay of shape `[B, T, H]`.
        beta (torch.Tensor):
            Betas of shape `[B, T, H]`.
        q_dot (torch.Tensor):
            Tangent of `q` of shape `[D, B, T, H, K]`.
        k_dot (torch.Tensor):
            Tangent of `k` of shape `[D, B, T, H, K]`.
        v_dot (torch.Tensor):
            Tangent of `v` of shape `[D, B, T, H, V]`.
        g_dot (torch.Tensor):
            Tangent of `g` of shape `[D, B, T, H]`.
        beta_dot (torch.Tensor):
            Tangent of `beta` of shape `[D, B, T, H]`.
        scale (float, Optional):
            Scale of the attention scores. Default: `K ** -0.5`.
        chunk_size (int, Optional):
            Chunk size. Default: 64.

    Returns:
        o (torch.Tensor):
            Output of shape `[B, T, H, V]`.
        o_dot (torch.Tensor):
            Derivative of `o` along each direction, of shape `[D, B, T, H, V]`.
    """
    B, T, H, K = q.shape
    C = chunk_size
    if scale is None:
        scale = K ** -0.5
    # [B, H, N, C, *]
    qc, kc, vc, bc, gc = (_chunk(x, C, lead=0) for x in (q, k, v, beta, g))
    # [D, B, H, N, C, *]
    qdc, kdc, vdc, bdc, gdc = (_chunk(x, C, lead=1) for x in (q_dot, k_dot, v_dot, beta_dot, g_dot))
    N = qc.shape[2]

    # padded positions have g = 0, so G stays at its last valid value there
    G, G_dot = gc.cumsum(-1), gdc.cumsum(-1)

    strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    decay = torch.where(strict, G[..., :, None] - G[..., None, :], float('-inf')).exp()
    M = bc[..., :, None] * (kc @ kc.transpose(-1, -2)) * decay
    eye = torch.eye(C, dtype=q.dtype, device=q.device)
    A = torch.linalg.solve_triangular(eye + M, eye.expand_as(M), upper=False, unitriangular=True)
    u = A @ (bc[..., None] * vc)
    w = A @ ((bc * G.exp())[..., None] * kc)
    u_dot, w_dot = _intra_chunk_tangents(
        k=kc,
        v=vc,
        beta=bc,
        G=G,
        A=A,
        u=u,
        w=w,
        k_dot=kdc,
        v_dot=vdc,
        beta_dot=bdc,
        G_dot=G_dot,
    )

    # h is the state entering chunk n
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
        # [B, H], [D, B, H]
        GL, GdL = G[:, :, n, -1], G_dot[:, :, :, n, -1]
        # [B, H, C]
        dec = (GL[..., None] - G[:, :, n]).exp()
        # [B, H, C, K]
        kdec = kc[:, :, n] * dec[..., None]
        h_new = GL.exp()[..., None, None] * h + kdec.transpose(-1, -2) @ nu
        h_dot = (GL.exp()[..., None, None] * (h_dot + GdL[..., None, None] * h)
                 + kdec.transpose(-1, -2) @ (nu_dot + (GdL[..., None] - G_dot[:, :, :, n])[..., None] * nu)
                 + (kdc[:, :, :, n] * dec[..., None]).transpose(-1, -2) @ nu)
        h = h_new
    # [B, H, N, K, V], [D, B, H, N, K, V]
    hs, hds = torch.stack(hs, 2), torch.stack(hds, 3)
    # [B, H, N, C, V], [D, B, H, N, C, V]
    nus, nuds = torch.stack(nus, 2), torch.stack(nuds, 3)

    o = _chunk_o(q=qc, k=kc, nu=nus, h=hs, G=G, scale=scale)
    o_dot = (_chunk_o(q=qdc, k=kc, nu=nus, h=hs, G=G, scale=scale)
             + _chunk_o(q=qc, k=kdc, nu=nus, h=None, G=G, scale=scale)
             + _chunk_o(q=qc, k=kc, nu=nuds - G_dot[..., None] * nus, h=hds, G=G, scale=scale)
             + G_dot[..., None] * o)
    return _unchunk(o, T, lead=0), _unchunk(o_dot, T, lead=1)


# 2 warps on Blackwell before Triton 3.8: the recurrence has the tl.dot race that chunk_delta_h avoids the same way
JVP_H_NUM_WARPS = [2] if (IS_NVIDIA_BLACKWELL and not TRITON_ABOVE_3_8_0) else [2, 4]
# the intra-chunk and output kernels have chunk_fwd_o's structure, so they drop 8 warps where chunk_o.py does
JVP_NUM_WARPS = [4] if BLACKWELL_8_WARP_RACE else [4, 8]


def _resolve_precision(input_precision: str | None) -> str:
    r"""fp32 tl.dot precision of the JVP kernels: the argument, else TRITON_F32_DEFAULT, else tf32 where supported."""
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
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'B'])
def chunk_gated_delta_rule_jvp_fwd_kernel_h(
    q,
    k,
    w,
    nu,
    h,
    g,
    P,
    kd,
    wd,
    ud,
    gd,
    hd,
    nud,
    od,
    s_kd_d,
    s_kd_b,
    s_kd_t,
    s_kd_h,
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
    Sequential tangent recurrence over chunks, one program per (value block, head, direction).

    Primal: k, w [B, T, H, K]; nu (v_new) [B, T, H, V]; h [B, NT, H, K, V] (state entering each chunk);
    g [B, T, H] within-chunk cumulative gate in log2 units. Tangents: kd [D, B, T, H, K] with strides s_kd_*;
    wd [D, B, T, H, K], ud [D, B, T, H, V] and gd [D, B, T, H] (natural units), contiguous.
    If FUSE_O, writes into od the output terms with h_dot and nu_dot, scale [exp2(g) (q h_dot) + P nu_dot],
    from P [B, T, H, BT]; else stores hd [D, B, NT, H, K, V] (h_dot entering each chunk) and nud [D, B, T, H, V].
    """
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v, i_bhd = pid % NV, pid // NV
    i_bh, i_d = i_bhd % (B * H), i_bhd // (B * H)
    i_b, i_h = i_bh // H, i_bh % H
    i_db = i_d * B + i_b
    NT = tl.cdiv(T, BT)
    bos = i_b * T

    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    nu += (bos * H + i_h) * V
    g += bos * H + i_h
    h += (i_b * NT * H + i_h) * K * V
    kd += i_d * s_kd_d + i_b * s_kd_b + i_h * s_kd_h
    wd += (i_db * T * H + i_h) * K
    ud += (i_db * T * H + i_h) * V
    gd += i_db * T * H + i_h
    if FUSE_O:
        q += (bos * H + i_h) * K
        P += (bos * H + i_h) * BT
        od += (i_db * T * H + i_h) * V
    else:
        hd += (i_db * NT * H + i_h) * K * V
        nud += (i_db * T * H + i_h) * V

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]

    # [BK, BV] h_dot entering chunk 0 is zero without an initial state
    b_hd = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t0 in range(NT):
        i_t = i_t0.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tv = m_t[:, None] & m_v[None, :]
        m_tk = m_t[:, None] & m_k[None, :]

        if not FUSE_O:
            tl.store(hd + i_t * H * K * V + o_k[:, None] * V + o_v[None, :], b_hd.to(hd.dtype.element_ty), mask=m_kv)
        # [BK, BV]
        b_h = tl.load(h + i_t * H * K * V + o_k[:, None] * V + o_v[None, :], mask=m_kv, other=0.).to(tl.float32)
        # [BT, BK]
        b_w = tl.load(w + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.)
        b_wd = tl.load(wd + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.).to(b_w.dtype)
        # [BT, BV] nu_dot = u_dot - w_dot h - w h_dot
        b_nud = tl.load(ud + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.).to(tl.float32)
        b_nud -= tl.dot(b_wd, b_h.to(b_w.dtype), input_precision=DOT_PRECISION)
        b_nud -= tl.dot(b_w, b_hd.to(b_w.dtype), input_precision=DOT_PRECISION)

        last_idx = tl.minimum((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + last_idx * H).to(tl.float32)
        b_g = tl.load(g + o_t * H, mask=m_t, other=0.).to(tl.float32)
        if FUSE_O:
            b_q = tl.load(q + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.)
            b_P = tl.load(P + o_t[:, None] * (H * BT) + tl.arange(0, BT)[None, :], mask=m_t[:, None], other=0.)
            b_od = tl.dot(b_q, b_hd.to(b_q.dtype), input_precision=DOT_PRECISION) * exp2(b_g)[:, None]
            b_od = tl.dot(b_P.to(b_q.dtype), b_nud.to(b_q.dtype), b_od, input_precision=DOT_PRECISION)
            tl.store(od + o_t[:, None] * (H * V) + o_v[None, :], (b_od * scale).to(od.dtype.element_ty), mask=m_tv)
        else:
            tl.store(nud + o_t[:, None] * (H * V) + o_v[None, :], b_nud.to(nud.dtype.element_ty), mask=m_tv)
        b_gd_last = tl.load(gd + last_idx * H).to(tl.float32)
        b_gd = tl.load(gd + o_t * H, mask=m_t, other=0.).to(tl.float32)
        # exp(G_L - G_r), zeroed on padded rows so that 0 * inf cannot occur there
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
        for num_warps in JVP_NUM_WARPS
        for num_stages in [2, 3]
    ],
    key=['H', 'K', 'V', 'BT', 'FUSE_O', 'DOT_PRECISION'],
    prune_configs_by={'early_config_prune': _prune_ieee},
    restore_value=['od'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'B'])
def chunk_gated_delta_rule_jvp_fwd_kernel_o(
    q,
    k,
    nu,
    h,
    g,
    qd,
    kd,
    nud,
    hd,
    gd,
    od,
    s_qd_d,
    s_qd_b,
    s_qd_t,
    s_qd_h,
    s_kd_d,
    s_kd_b,
    s_kd_t,
    s_kd_h,
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
    Output tangent o_dot [D, B, T, H, V], one program per (value block, chunk, head, direction), without
    materializing the primal output or per-direction copies of q, k, nu, h.
    Tangents: qd, kd with strides s_qd_*, s_kd_*; gd [D, B, T, H] contiguous; nud and hd as written by the
    recurrence kernel. If FUSE_O, od already holds the terms with h_dot and nu_dot and the rest is added to it.
    """
    pid = tl.program_id(0).to(tl.int64)
    NV, NT = tl.cdiv(V, BV), tl.cdiv(T, BT)
    i_v, i_t, i_bhd = pid % NV, (pid // NV) % NT, pid // (NV * NT)
    i_bh, i_d = i_bhd % (B * H), i_bhd // (B * H)
    i_b, i_h = i_bh // H, i_bh % H
    i_db = i_d * B + i_b
    bos = i_b * T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    nu += (bos * H + i_h) * V
    g += bos * H + i_h
    h += ((i_b * NT + i_t) * H + i_h) * K * V
    qd += i_d * s_qd_d + i_b * s_qd_b + i_h * s_qd_h
    kd += i_d * s_kd_d + i_b * s_kd_b + i_h * s_kd_h
    gd += i_db * T * H + i_h
    if not FUSE_O:
        nud += (i_db * T * H + i_h) * V
        hd += ((i_db * NT + i_t) * H + i_h) * K * V
    od += (i_db * T * H + i_h) * V

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    # accumulators of q h, q_dot h (+ q h_dot), q k^T and q_dot k^T + q k_dot^T
    b_o0 = tl.zeros([BT, BV], dtype=tl.float32)
    b_o1 = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    b_Ad = tl.zeros([BT, BT], dtype=tl.float32)
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
    b_gd = tl.load(gd + o_t * H, mask=m_t, other=0.).to(tl.float32)
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
        for num_warps in JVP_NUM_WARPS
        for num_stages in [2, 3]
    ],
    key=['H', 'K', 'V', 'BT', 'DOT_PRECISION'],
    prune_configs_by={'early_config_prune': _prune_ieee},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'B'])
def chunk_gated_delta_rule_jvp_fwd_kernel_intra(
    k,
    v,
    beta,
    g,
    A,
    u,
    w,
    kd,
    vd,
    bd,
    gd,
    ud,
    wd,
    s_kd_d,
    s_kd_b,
    s_kd_t,
    s_kd_h,
    s_vd_d,
    s_vd_b,
    s_vd_t,
    s_vd_h,
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
    Intra-chunk tangents ud [D, B, T, H, V] and wd [D, B, T, H, K] of the WY representation, one program per
    (chunk, head, direction). Primal: k, w [B, T, H, K]; v, u [B, T, H, V]; beta and g (log2 cumulative)
    [B, T, H]; A [B, T, H, BT]. Tangents: kd, vd with strides s_kd_*, s_vd_*; bd and gd [D, B, T, H] contiguous.
    """
    pid = tl.program_id(0).to(tl.int64)
    NT = tl.cdiv(T, BT)
    i_t, i_bhd = pid % NT, pid // NT
    i_bh, i_d = i_bhd % (B * H), i_bhd // (B * H)
    i_b, i_h = i_bh // H, i_bh % H
    i_db = i_d * B + i_b
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
    bd += i_db * T * H + i_h
    gd += i_db * T * H + i_h
    ud += (i_db * T * H + i_h) * V
    wd += (i_db * T * H + i_h) * K

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_c = tl.arange(0, BT)

    b_b = tl.load(beta + o_t * H, mask=m_t, other=0.).to(tl.float32)
    b_bd = tl.load(bd + o_t * H, mask=m_t, other=0.).to(tl.float32)
    b_g = tl.load(g + o_t * H, mask=m_t, other=0.).to(tl.float32)
    b_gd = tl.load(gd + o_t * H, mask=m_t, other=0.).to(tl.float32)
    b_A = tl.load(A + o_t[:, None] * (H * BT) + o_c[None, :], mask=m_t[:, None], other=0.).to(tl.float32)

    # [BT, BT] (k_r . k_s) and (k_dot_r . k_s), accumulated over key blocks
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
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_jvp_fwd_kernel_qk(
    q,
    k,
    g,
    P,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    r"""P [B, T, H, BT]: P_rs = (q_r . k_s) exp2(g_r - g_s) for s <= r within a chunk, else 0."""
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
    fuse_output: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    :func:`chunk_gated_delta_rule_jvp` without argument checks.
    The primal inputs, `g_dot` and `beta_dot` are contiguous; `q_dot`, `k_dot` and `v_dot` have unit feature stride.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    D = q_dot.shape[0]
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    BK = max(triton.next_power_of_2(K), 16)
    assert K <= 256, "the JVP state kernel supports head dimension K <= 256"
    precision = _resolve_precision(input_precision)

    # the library's forward kernels, as in chunk_gated_delta_rule_fwd without an initial state
    g_cum = chunk_local_cumsum(g=g, chunk_size=BT, scale=RCP_LN2)
    w, u, A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g_cum, beta=beta, chunk_size=BT)
    # h [B, NT, H, K, V] is the state entering each chunk and nu is v_new
    h, nu, _ = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g_cum, chunk_size=BT)
    o = chunk_fwd_o(q=q, k=k, v=nu, h=h, g=g_cum, scale=scale, chunk_size=BT)

    # [D, B, T, H] within-chunk cumulative gate tangent in natural units, fp32
    G_dot = chunk_local_cumsum(g=g_dot.view(D * B, T, H), chunk_size=BT).view(D, B, T, H)

    u_dot = torch.empty(D, B, T, H, V, dtype=k.dtype, device=k.device)
    w_dot = torch.empty(D, B, T, H, K, dtype=k.dtype, device=k.device)
    # 1D grid: D * B * H can exceed the 65535 limit of the second grid dimension
    chunk_gated_delta_rule_jvp_fwd_kernel_intra[(NT * D * B * H,)](
        k=k,
        v=v,
        beta=beta,
        g=g_cum,
        A=A,
        u=u,
        w=w,
        kd=k_dot,
        vd=v_dot,
        bd=beta_dot,
        gd=G_dot,
        ud=u_dot,
        wd=w_dot,
        s_kd_d=k_dot.stride(0),
        s_kd_b=k_dot.stride(1),
        s_kd_t=k_dot.stride(2),
        s_kd_h=k_dot.stride(3),
        s_vd_d=v_dot.stride(0),
        s_vd_b=v_dot.stride(1),
        s_vd_t=v_dot.stride(2),
        s_vd_h=v_dot.stride(3),
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BT=BT,
        DOT_PRECISION=precision,
    )

    o_dot = torch.empty(D, B, T, H, V, dtype=v.dtype, device=v.device)
    if fuse_output:
        P = torch.empty(B, T, H, BT, dtype=k.dtype, device=k.device)
        chunk_gated_delta_rule_jvp_fwd_kernel_qk[(NT, B * H)](
            q=q,
            k=k,
            g=g_cum,
            P=P,
            T=T,
            H=H,
            K=K,
            BT=BT,
            DOT_PRECISION=precision,
        )
        h_dot = nu_dot = None
    else:
        P = None
        h_dot = k.new_empty(D, B, NT, H, K, V)
        nu_dot = torch.empty(D, B, T, H, V, dtype=nu.dtype, device=nu.device)

    def grid_h(meta): return (triton.cdiv(V, meta['BV']) * B * H * D,)
    chunk_gated_delta_rule_jvp_fwd_kernel_h[grid_h](
        q=q,
        k=k,
        w=w,
        nu=nu,
        h=h,
        g=g_cum,
        P=P,
        kd=k_dot,
        wd=w_dot,
        ud=u_dot,
        gd=G_dot,
        hd=h_dot,
        nud=nu_dot,
        od=o_dot,
        s_kd_d=k_dot.stride(0),
        s_kd_b=k_dot.stride(1),
        s_kd_t=k_dot.stride(2),
        s_kd_h=k_dot.stride(3),
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        FUSE_O=fuse_output,
        DOT_PRECISION=precision,
    )
    del w_dot, u_dot, P

    def grid_o(meta): return (triton.cdiv(V, meta['BV']) * NT * D * B * H,)
    chunk_gated_delta_rule_jvp_fwd_kernel_o[grid_o](
        q=q,
        k=k,
        nu=nu,
        h=h,
        g=g_cum,
        qd=q_dot,
        kd=k_dot,
        nud=nu_dot,
        hd=h_dot,
        gd=G_dot,
        od=o_dot,
        s_qd_d=q_dot.stride(0),
        s_qd_b=q_dot.stride(1),
        s_qd_t=q_dot.stride(2),
        s_qd_h=q_dot.stride(3),
        s_kd_d=k_dot.stride(0),
        s_kd_b=k_dot.stride(1),
        s_kd_t=k_dot.stride(2),
        s_kd_h=k_dot.stride(3),
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BT=BT,
        FUSE_O=fuse_output,
        DOT_PRECISION=precision,
    )
    return o.to(q.dtype), o_dot


@input_guard(no_guard_contiguous=['q_dot', 'k_dot', 'v_dot'])
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
    fuse_output: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Forward-mode derivative (JVP) of :func:`chunk_gated_delta_rule` along `D` tangent directions at once.
    Initial and final states, variable lengths and grouped value heads are not supported.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`, with any l2 normalization already applied.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`, with any l2 normalization already applied.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            Per-step log decay of shape `[B, T, H]`, not cumulative.
        beta (torch.Tensor):
            Betas of shape `[B, T, H]`.
        q_dot (torch.Tensor):
            Tangent of `q` of shape `[D, B, T, H, K]`. Any strides with a unit feature stride are read in place.
        k_dot (torch.Tensor):
            Tangent of `k` of shape `[D, B, T, H, K]`. Any strides with a unit feature stride are read in place.
        v_dot (torch.Tensor):
            Tangent of `v` of shape `[D, B, T, H, V]`. Any strides with a unit feature stride are read in place.
        g_dot (torch.Tensor):
            Tangent of `g` of shape `[D, B, T, H]`.
        beta_dot (torch.Tensor):
            Tangent of `beta` of shape `[D, B, T, H]`.
        scale (float, Optional):
            Scale of the attention scores. Default: `K ** -0.5`.
        chunk_size (int, Optional):
            Chunk size, one of 16, 32 and 64. Default: 64.
        input_precision (str, Optional):
            Precision of the fp32 `tl.dot` in the JVP kernels: `'ieee'`, `'tf32'` or `'tf32x3'`.
            `None` uses `TRITON_F32_DEFAULT` if set, else `'tf32'` where supported and `'ieee'` elsewhere.
            The primal runs through the library's forward kernels and is not affected.
            `'ieee'` takes minutes to compile per shape on NVIDIA GPUs; `'tf32x3'` is close to it in accuracy.
            Default: `None`.
        fuse_output (bool, Optional):
            Whether the recurrence kernel computes the output terms that need the state tangent,
            so that the state tangent `[D, B, NT, H, K, V]` is never stored.
            With 16-bit inputs, `True` rounds those terms to the output dtype before the rest is added. Default: `True`.

    Returns:
        o (torch.Tensor):
            Output of shape `[B, T, H, V]`, equal to `chunk_gated_delta_rule(q, k, v, g, beta, scale)[0]`.
        o_dot (torch.Tensor):
            Derivative of `o` along each direction, of shape `[D, B, T, H, V]`.
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
    return chunk_gated_delta_rule_jvp_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        q_dot=q_dot,
        k_dot=k_dot,
        v_dot=v_dot,
        g_dot=g_dot,
        beta_dot=beta_dot,
        scale=scale,
        chunk_size=chunk_size,
        input_precision=input_precision,
        fuse_output=fuse_output,
    )
