# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Reference PyTorch implementation of Gated DeltaNet 2 (GDN-2).
#
# Update rule (per token, on the matrix state S in R^{K x V}):
#
#   S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T
#
# where `*` is the elementwise (Hadamard) product, `b_t in R^K` is the
# channel-wise erase gate on the key axis, `w_t in R^V` is the channel-wise
# write gate on the value axis, and `g_t in R^K` is the channel-wise log-decay.
# Collapsing `b_t = w_t = beta` to a scalar recovers KDA.

import torch
import torch.nn.functional as F


def naive_recurrent_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    """Token-by-token reference forward pass for GDN-2.

    Args:
        q: queries of shape [B, T, H, K].
        k: keys of shape [B, T, H, K].
        v: values of shape [B, T, H, V].
        g: log-decay of shape [B, T, H, K] (already in the natural-log base).
        b: channel-wise erase gate of shape [B, T, H, K] (range typically [0, 1]).
        w: channel-wise write gate of shape [B, T, H, V] (range typically [0, 1]).
        scale: attention scale; defaults to 1 / sqrt(K).
        initial_state: optional [B, H, K, V] initial state in float32.
        output_final_state: whether to return the final state.

    Returns:
        o: outputs of shape [B, T, H, V].
        final_state: [B, H, K, V] if output_final_state else None.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q, k, v, g, b, w = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, g, b, w))
    B, H, T, K = k.shape
    V = v.shape[-1]
    o = torch.zeros(B, H, T, V, device=v.device, dtype=torch.float32)
    h = torch.zeros(B, H, K, V, device=v.device, dtype=torch.float32)
    if initial_state is not None:
        h = initial_state.to(torch.float32).clone()
    q = q * scale

    for t in range(T):
        b_q = q[:, :, t]            # [B, H, K]
        b_k = k[:, :, t]            # [B, H, K]
        b_v = v[:, :, t]            # [B, H, V]
        b_g = g[:, :, t]            # [B, H, K]
        b_b = b[:, :, t]            # [B, H, K]
        b_w = w[:, :, t]            # [B, H, V]

        # Per-channel decay applied along the K axis of the state.
        h = h * b_g.exp().unsqueeze(-1)
        # Gated read at key (b * k): pulls the existing contribution at this key.
        erase = ((b_b * b_k).unsqueeze(-1) * h).sum(-2)     # [B, H, V]
        # Gated write: commit (w * v) with the erase contribution subtracted.
        b_v_new = b_w * b_v - erase                          # [B, H, V]
        # Rank-1 outer-product update.
        h = h + b_k.unsqueeze(-1) * b_v_new.unsqueeze(-2)
        o[:, :, t] = (b_q.unsqueeze(-1) * h).sum(-2)

    o = o.transpose(1, 2).contiguous().to(v.dtype)
    if not output_final_state:
        h = None
    return o, h


def naive_chunk_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    """Chunkwise PyTorch reference for GDN-2.

    The within-chunk recurrence is expanded via the WY representation:
    `A = (I + tril(K' D K^T, -1))^{-1}` where `K' = b * k * exp(g)`, then
    `u = A (w * v)` and `w_wy = A (b * k * exp(g))`. The inter-chunk state
    recurrence carries the rank-`chunk_size` update across chunks.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    BT = chunk_size

    q, k, v, g, b, w = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, g, b, w))
    B, H, T, K = k.shape
    V = v.shape[-1]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        g = F.pad(g, (0, 0, 0, pad_len))
        b = F.pad(b, (0, 0, 0, pad_len))
        w = F.pad(w, (0, 0, 0, pad_len))
    T_pad = q.shape[2]
    NT = T_pad // BT

    q = q * scale
    # Reshape to chunks: [B, H, NT, BT, D]

    def chunk(x):
        return x.view(B, H, NT, BT, -1)
    q, k, v, g, b, w = (chunk(x) for x in (q, k, v, g, b, w))

    # Cumulative log-decay within each chunk (per channel).
    g_cum = g.cumsum(-2)                              # [B, H, NT, BT, K]
    g_last = g_cum[..., -1:, :]                       # [B, H, NT, 1, K] — final state decay
    # Pre-decayed key for the WY block.
    k_g = k * g_cum.exp()                             # k * exp(g)
    k_g_b = k_g * b                                   # (b * k) * exp(g)

    # Strictly lower-triangular gated kk score matrix used by the WY inverse.
    # Each (i, j) entry has decay factor exp(g_cum[i] - g_cum[j]).
    decay_ij = (g_cum.unsqueeze(-2) - g_cum.unsqueeze(-3))            # [B, H, NT, BT, BT, K]
    decay_ij_exp = decay_ij.exp()
    tril_mask = torch.tril(
        torch.ones(BT, BT, device=q.device, dtype=torch.bool),
        diagonal=-1,
    )
    # The strictly lower-triangular K^T K product, decay-weighted, b-modulated.
    # The contraction is along the K axis of k and (b * k).
    bk = b * k                                                       # [B, H, NT, BT, K]
    # T_ij = sum_d (b*k)_i_d * k_j_d * exp(g_i_d - g_j_d)
    decay_for_T = decay_ij_exp                                       # [..., i, j, K]
    T_lower = torch.einsum('bhnik,bhnjk,bhnijk->bhnij',
                           bk, k, decay_for_T)
    T_lower = T_lower.masked_fill(~tril_mask, 0.0)
    # Blocked forward substitution to build A_inv = (I + T_lower)^{-1} in place,
    # mirroring the WY-block solve in fla.ops.gated_delta_rule.naive.
    A_inv = -T_lower
    for i in range(1, BT):
        A_inv[..., i, :i] = A_inv[..., i, :i].clone() + (
            A_inv[..., i, :i, None].clone() * A_inv[..., :i, :i].clone()
        ).sum(-2)
    A_inv = A_inv + torch.eye(BT, device=q.device, dtype=torch.float32)

    # Within-chunk auxiliaries.
    u_wy = A_inv @ (w * v)                                            # [B, H, NT, BT, V]
    w_wy = A_inv @ k_g_b                                              # [B, H, NT, BT, K]
    # Decayed key for the inter-chunk recurrence: k * exp(g_last - g_cum).
    k_tail = k * (g_last - g_cum).exp()                               # [B, H, NT, BT, K]

    # Output path: causal intra-chunk QK^T (with decay) times v_new, plus
    # the contribution from the carried state.
    decay_qk = (g_cum.unsqueeze(-2) - g_cum.unsqueeze(-3)).exp()      # [B, H, NT, BT, BT, K]
    causal_mask = torch.tril(
        torch.ones(BT, BT, device=q.device, dtype=torch.bool),
        diagonal=0,
    )

    S = torch.zeros(B, H, K, V, device=v.device, dtype=torch.float32)
    if initial_state is not None:
        S = initial_state.to(torch.float32).clone()
    o = torch.zeros_like(v)
    for n in range(NT):
        q_n = q[:, :, n]                                              # [B, H, BT, K]
        k_n = k[:, :, n]
        g_n = g_cum[:, :, n]
        g_last_n = g_last[:, :, n].squeeze(-2)                        # [B, H, K]
        w_n = w_wy[:, :, n]                                           # [B, H, BT, K]
        u_n = u_wy[:, :, n]                                           # [B, H, BT, V]
        k_tail_n = k_tail[:, :, n]                                    # [B, H, BT, K]

        # v_new = u - w_wy @ S (delta-rule write minus carried contribution).
        v_new = u_n - w_n @ S                                          # [B, H, BT, V]
        # Intra-chunk attention with channel-wise decay on the K axis.
        # A_qk[i, j] = sum_d q_i_d * k_j_d * exp(g_i_d - g_j_d)
        A_qk = torch.einsum('bhik,bhjk,bhijk->bhij',
                            q_n, k_n, decay_qk[:, :, n]).masked_fill(~causal_mask, 0.0)
        o_intra = A_qk @ v_new
        # Inter-chunk contribution: q decayed up to chunk start times the carried state.
        o_inter = (q_n * g_n.exp()) @ S
        o[:, :, n] = o_intra + o_inter
        # State update: S <- Diag(exp(g_last)) S + k_tail^T v_new.
        S = S * g_last_n.unsqueeze(-1).exp() + k_tail_n.transpose(-1, -2) @ v_new

    o = o.reshape(B, H, T_pad, V)[:, :, :T].transpose(1, 2).contiguous().to(v.dtype)
    if not output_final_state:
        S = None
    return o, S
