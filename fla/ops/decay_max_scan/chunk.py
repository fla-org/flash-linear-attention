# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F

from fla.utils import input_guard

try:
    from .triton_backend import triton_decay_max_scan
    TRITON_BACKEND = True
except Exception:        # ImportError or triton compile issues on unsupported stacks
    TRITON_BACKEND = False

CHUNK = 128


def _scan_chunk(xc, gc, s):
    """Hillis-Steele scan over one chunk under the (max, *) semiring.

    Per-position pairs (m, d) init to (x_t, g_t); combine((m_l, d_l), (m_r, d_r))
    = (max(m_r, d_r * m_l), d_r * d_l) is associative, so after log2(C) doubling
    steps ``m_t = max_{s<=t} x_s * prod_{r=s+1..t} g_r`` and ``d_t = prod_{r<=t} g_r``.
    The decay product is clamped at 1e-30: once the chunk-cumulative decay is
    below that, the history term is negligible for any bounded input magnitude.
    """
    m = xc.clone()
    d = gc.clone()
    off = 1
    while off < m.shape[1]:
        mp = m[:, :-off]
        dp = d[:, :-off]
        m = torch.cat([m[:, :off], torch.maximum(m[:, off:], d[:, off:] * mp)], dim=1)
        d = torch.cat([d[:, :off], d[:, off:] * dp], dim=1)
        d = d.clamp_min(1e-30)
        off *= 2
    e = torch.maximum(m, d * s.unsqueeze(1))
    return e, e[:, -1]


@input_guard
def chunk_decay_max_scan(
    x: torch.Tensor,
    decay_v: torch.Tensor | None = None,
    decay_b: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked decay-max scan: ``e_t = max(x_t, g_t * e_{t-1})`` with
    ``g_t = exp(-softplus(x_t * v_d + b_d))``, evaluated with an associative
    (max, *) semiring doubling scan per chunk — vectorized, no cumprod division,
    no underflow-sensitive path.

    Supports varlen packing via ``cu_seqlens`` (state resets at each boundary).

    Args:
        x (torch.Tensor):
            Scan inputs `[B, T, D, H]`.
        decay_v (torch.Tensor, Optional):
            Per-channel decay slope `[D]`. Default: `None`.
        decay_b (torch.Tensor, Optional):
            Per-channel decay bias `[D]`. Default: `None`.
        initial_state (torch.Tensor, Optional):
            Running state `[B, D, H]`. Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence offsets. Default: `None`.

    Returns:
        e (torch.Tensor):
            Scan outputs `[B, T, D, H]` in float32.
        final_state (torch.Tensor):
            Final running state `[B, D, H]`.
    """
    B, T, D, H = x.shape
    # dense CUDA batches take the Triton backend; varlen packing and non-CUDA
    # tensors run the vectorized torch scan below
    if cu_seqlens is None and TRITON_BACKEND and x.is_cuda:
        return triton_decay_max_scan(x, decay_v, decay_b, initial_state)
    xf = x.float()
    if decay_v is not None:
        pre = xf * decay_v.float().view(1, 1, D, 1) + decay_b.float().view(1, 1, D, 1)
        g = torch.exp(-F.softplus(pre))
    else:
        g = torch.ones_like(xf)
    bounds = cu_seqlens.tolist() if cu_seqlens is not None else [0, T]
    e = torch.empty(B, T, D, H, dtype=torch.float32, device=x.device)
    # -1e30 (not -inf): keeps d * state gradients finite on the cold-start branch
    s = torch.full((B, D, H), -1e30, dtype=torch.float32, device=x.device) \
        if initial_state is None else initial_state.float().clone()
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if i > 0:
            s = torch.full_like(s, -1e30)
        for ck in range(lo, hi, CHUNK):
            hi2 = min(ck + CHUNK, hi)
            e[:, ck:hi2], s = _scan_chunk(xf[:, ck:hi2], g[:, ck:hi2], s)
    return e, s
