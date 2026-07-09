# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Per-kernel tests for the Gated Delta Net (GDN) Triton kernels.

Each test compares a single Triton kernel against a pure-PyTorch (torch) baseline
that implements the same math, so that every kernel can be validated in isolation.
"""

import pytest
import torch
import torch.nn.functional as F

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.gated_delta_rule.gate import gdn_gate_bwd, gdn_gate_chunk_cumsum, gdn_gate_fwd
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils.constant import RCP_LN2
from fla.utils import assert_close, device


def _make_wy_inverse(B: int, T: int, HV: int, BT: int, dtype: torch.dtype) -> torch.Tensor:
    """Build a realistic per-chunk (I + strictly-lower-triangular)^{-1} matrix A.

    Shape: [B, T, HV, BT]. Within each chunk of length BT, A is lower-triangular
    with 1s on the diagonal and small random values below the diagonal.
    """
    A = torch.randn(B, T, HV, BT, dtype=dtype, device=device) * 0.1
    mask = torch.tril(torch.ones(BT, BT, device=device, dtype=dtype)).view(1, BT, 1, BT)
    eye = torch.eye(BT, device=device, dtype=dtype).view(1, BT, 1, BT)
    for it in range(0, T, BT):
        A[:, it:it + BT] = A[:, it:it + BT] * mask + eye
    return A


def _make_gate(B: int, T: int, HV: int, chunk_size: int = 64) -> torch.Tensor:
    """Build a realistic GDN log-space chunk-cumsum gate (monotonically
    decreasing within each chunk, reset at chunk boundaries), matching what the
    real GDN op feeds into the chunk kernels. Generated via the already-tested
    `gdn_gate_chunk_cumsum` so the values are faithful to the kernel's contract.
    """
    raw = torch.randn(B, T, HV, device=device)
    A_log = torch.randn(HV, device=device)
    g = gdn_gate_chunk_cumsum(raw, A_log, chunk_size=chunk_size, scale=1.0 / 0.6931471805599453)
    return g.float()


def recompute_w_u_fwd_ref(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch baseline for `recompute_w_u_fwd_kernel`.

    For each chunk [i_t*BT, (i_t+1)*BT):
        u[t] = sum_s A[t, s] * (v[s] * beta[s])
        w[t] = sum_s A[t, s] * (k[s] * beta[s] [* exp2(g[s])])

    where `s` indexes the position within the chunk. For GVA (HV > H), each key
    head is shared by `HV // H` value heads.
    """
    B, T, H, K = k.shape
    HV = v.shape[2]
    V = v.shape[3]
    BT = A.shape[-1]
    assert T % BT == 0, "the reference only supports T divisible by BT"

    if HV != H:
        k_hv = k.repeat_interleave(HV // H, dim=2)
    else:
        k_hv = k

    w = torch.empty(B, T, HV, K, dtype=k.dtype, device=k.device)
    u = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    beta_f = beta.float()
    g_f = torch.exp2(g.float()) if g is not None else None

    for it in range(0, T, BT):
        s, e = it, it + BT
        # [B, BT, HV, BT] -> [B, HV, BT, BT]
        A_c = A[:, s:e].permute(0, 2, 1, 3).float()

        v_c = v[:, s:e].float() * beta_f[:, s:e, :, None]
        v_c = v_c.permute(0, 2, 1, 3)  # [B, HV, BT, V]
        u_c = torch.matmul(A_c, v_c)  # [B, HV, BT, V]
        u[:, s:e] = u_c.permute(0, 2, 1, 3).to(v.dtype)

        k_c = k_hv[:, s:e].float() * beta_f[:, s:e, :, None]
        if g_f is not None:
            k_c = k_c * g_f[:, s:e, :, None]
        k_c = k_c.permute(0, 2, 1, 3)  # [B, HV, BT, K]
        w_c = torch.matmul(A_c, k_c)  # [B, HV, BT, K]
        w[:, s:e] = w_c.permute(0, 2, 1, 3).to(k.dtype)

    return w, u


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'use_g', 'dtype'),
    [
        pytest.param(B, T, H, HV, D, use_g, dtype,
                     id=f"B{B}-T{T}-H{H}-HV{HV}-D{D}-use_g{use_g}-{dtype}")
        for (B, T, H, HV, D, use_g, dtype) in [
            (2, 128, 2, 2, 64, False, torch.bfloat16),
            (2, 128, 2, 4, 64, False, torch.bfloat16),
            (2, 128, 2, 2, 64, True, torch.bfloat16),
            (1, 256, 4, 4, 32, True, torch.float16),
        ]
    ],
)
def test_recompute_w_u_fwd(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    use_g: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    BT = 64
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    beta = torch.rand(B, T, HV, dtype=dtype, device=device).sigmoid()
    A = _make_wy_inverse(B, T, HV, BT, dtype)
    g = torch.randn(B, T, HV, dtype=torch.float32, device=device) * 0.1 if use_g else None

    w_ref, u_ref = recompute_w_u_fwd_ref(k, v, beta, A, g)
    w_tri, u_tri = recompute_w_u_fwd(k, v, beta, A, g)

    assert_close('u', u_ref, u_tri, 0.005)
    assert_close('w', w_ref, w_tri, 0.005)


def chunk_kkt_solve_ref(
    k: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    chunk_size: int = 64,
    use_exp2: bool = True,
) -> torch.Tensor:
    """Torch baseline for `chunk_gated_delta_rule_fwd_kkt_solve_kernel`.

    For each chunk:
        L[i, j] = beta[i] * <k[i], k[j]> * exp2(g[i] - g[j])   (i > j)
        L[i, j] = 0                                              (i <= j)
        A = (I + L)^{-1}
    """
    B, T, H, K = k.shape
    HV = beta.shape[2]
    BT = chunk_size
    assert T % BT == 0

    if HV != H:
        k_hv = k.repeat_interleave(HV // H, dim=2)
    else:
        k_hv = k

    A_out = torch.zeros(B, T, HV, BT, dtype=k.dtype, device=k.device)
    m = torch.tril(torch.ones(BT, BT, device=k.device), diagonal=-1)
    I = torch.eye(BT, device=k.device, dtype=torch.float32)

    for it in range(0, T, BT):
        s, e = it, it + BT
        k_c = k_hv[:, s:e].float().permute(0, 2, 1, 3)  # [B, HV, BT, K]
        kkt = torch.matmul(k_c, k_c.transpose(-1, -2))  # [B, HV, BT, BT]

        if g is not None:
            g_c = g[:, s:e].float().permute(0, 2, 1)  # [B, HV, BT]
            gdiff = g_c[:, :, :, None] - g_c[:, :, None, :]  # [B, HV, BT, BT]
            # strictly mask the upper-triangular part to 0 *before* exp2, so that
            # exp2 never sees a large positive value (which would overflow to inf
            # and turn into NaN after the zero mask is applied).
            gate = torch.exp2(gdiff.masked_fill(~m.bool(), 0.0)) if use_exp2 \
                else torch.exp(gdiff.masked_fill(~m.bool(), 0.0))
        else:
            gate = 1.0

        beta_c = beta[:, s:e].float().permute(0, 2, 1)  # [B, HV, BT]
        L = kkt * gate * beta_c[:, :, :, None] * m
        A_c = torch.linalg.inv(I + L)  # [B, HV, BT, BT]
        A_out[:, s:e] = A_c.permute(0, 2, 1, 3).to(k.dtype)

    return A_out


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'use_g', 'dtype'),
    [
        pytest.param(B, T, H, HV, D, use_g, dtype,
                     id=f"B{B}-T{T}-H{H}-HV{HV}-D{D}-use_g{use_g}-{dtype}")
        for (B, T, H, HV, D, use_g, dtype) in [
            (2, 128, 2, 2, 64, True, torch.bfloat16),
            (2, 128, 2, 4, 64, True, torch.bfloat16),
            (1, 256, 4, 4, 32, True, torch.float16),
            (2, 128, 2, 2, 64, False, torch.bfloat16),
        ]
    ],
)
def test_chunk_kkt_solve(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    use_g: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    BT = 64
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    beta = torch.rand(B, T, HV, dtype=dtype, device=device).sigmoid()
    g = torch.randn(B, T, HV, dtype=torch.float32, device=device) * 0.1 if use_g else None

    A_ref = chunk_kkt_solve_ref(k, g, beta, BT)
    # `chunk_gated_delta_rule_fwd_kkt_solve_kernel` is launched inside the
    # supported wrapper (for chunk_size == 64); only its A output is checked here.
    # The kernel always uses exp2 for the gate on main.
    v = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    _, _, A_tri = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        chunk_size=BT,
    )

    assert_close('A', A_ref, A_tri, 0.005)


def gdn_gate_fwd_ref(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch baseline for `gdn_gate_fwd_kernel`: yg = -exp(A_log) * softplus(g + dt_bias)."""
    x = g.float()
    if dt_bias is not None:
        x = x + dt_bias.float()
    return -torch.exp(A_log.float()) * F.softplus(x)


def gdn_gate_chunk_cumsum_ref(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int = 64,
    scale: float | None = None,
    dt_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch baseline for `gdn_gate_chunk_cumsum_scalar_kernel` (forward, REVERSE=False).

    Computes the per-chunk cumulative sum of the gate activation.
    """
    gate = gdn_gate_fwd_ref(g, A_log, dt_bias)  # [B, T, H]
    B, T, H = g.shape
    BT = chunk_size
    assert T % BT == 0
    o = torch.cumsum(gate.reshape(B, T // BT, BT, H), dim=2).reshape(B, T, H)
    if scale is not None:
        o = o * scale
    return o


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'use_bias', 'dtype'),
    [
        pytest.param(B, T, H, use_bias, dtype,
                     id=f"B{B}-T{T}-H{H}-use_bias{use_bias}-{dtype}")
        for (B, T, H, use_bias, dtype) in [
            (2, 128, 4, False, torch.bfloat16),
            (2, 128, 4, True, torch.bfloat16),
            (1, 256, 8, True, torch.float32),
        ]
    ],
)
def test_gdn_gate_fwd(B: int, T: int, H: int, use_bias: bool, dtype: torch.dtype):
    torch.manual_seed(42)
    g = torch.randn(B, T, H, dtype=dtype, device=device)
    A_log = torch.randn(H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H, dtype=torch.float32, device=device) if use_bias else None

    ref = gdn_gate_fwd_ref(g, A_log, dt_bias).to(dtype)
    tri = gdn_gate_fwd(g, A_log, dt_bias, output_dtype=dtype)

    assert_close('yg', ref, tri, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'chunk_size', 'use_scale', 'use_bias', 'dtype'),
    [
        pytest.param(B, T, H, chunk_size, use_scale, use_bias, dtype,
                     id=f"B{B}-T{T}-H{H}-BT{chunk_size}-scale{use_scale}-bias{use_bias}-{dtype}")
        for (B, T, H, chunk_size, use_scale, use_bias, dtype) in [
            (2, 128, 4, 64, False, False, torch.float32),
            (2, 128, 4, 64, True, True, torch.float32),
            (1, 256, 8, 64, True, False, torch.float32),
        ]
    ],
)
def test_gdn_gate_chunk_cumsum(
    B: int,
    T: int,
    H: int,
    chunk_size: int,
    use_scale: bool,
    use_bias: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    g = torch.randn(B, T, H, dtype=dtype, device=device)
    A_log = torch.randn(H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H, dtype=torch.float32, device=device) if use_bias else None
    scale = 1.442695041 if use_scale else None  # RCP_LN2

    ref = gdn_gate_chunk_cumsum_ref(g, A_log, chunk_size, scale, dt_bias).to(dtype)
    tri = gdn_gate_chunk_cumsum(
        g=g,
        A_log=A_log,
        chunk_size=chunk_size,
        scale=scale,
        dt_bias=dt_bias,
        output_dtype=dtype,
    )

    assert_close('g_cumsum', ref, tri, 0.005)


def chunk_fwd_o_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None,
    scale: float,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Torch baseline for `chunk_fwd_kernel_o`.

    For each chunk [i_t*BT, (i_t+1)*BT) with hidden state h[i_t] (shape [K, V]):
        o_inter[t] = (q[t] @ h[i_t]) * exp2(g[t])
        A[t, s]    = (q[t] @ k[s]) * exp2(g[t] - g[s])   (s <= t)
        o_intra[t] = sum_{s <= t} A[t, s] * v[s]
        o[t]       = (o_inter[t] + o_intra[t]) * scale
    """
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    BT = chunk_size
    assert T % BT == 0
    NT = T // BT

    if HV != H:
        q_hv = q.repeat_interleave(HV // H, dim=2)
        k_hv = k.repeat_interleave(HV // H, dim=2)
    else:
        q_hv, k_hv = q, k

    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)
    m = torch.tril(torch.ones(BT, BT, device=v.device))

    for it in range(NT):
        s, e = it * BT, (it + 1) * BT
        q_c = q_hv[:, s:e].float()  # [B, BT, HV, K]
        k_c = k_hv[:, s:e].float()
        v_c = v[:, s:e].float()
        h_c = h[:, it].float()      # [B, HV, K, V]

        # inter-chunk: (q @ h) * exp2(g)
        o_inter = torch.einsum('bthk,bhkv->bthv', q_c, h_c)
        if g is not None:
            g_c = g[:, s:e].float()
            o_inter = o_inter * torch.exp2(g_c)[:, :, :, None]

        # intra-chunk attention scores
        scores = torch.matmul(q_c.permute(0, 2, 1, 3), k_c.permute(0, 2, 1, 3).transpose(-1, -2))
        if g is not None:
            g_ch = g[:, s:e].float().permute(0, 2, 1)  # [B, HV, BT]
            gdiff = g_ch[:, :, :, None] - g_ch[:, :, None, :]
            scores = scores * torch.exp2(gdiff.masked_fill(~m.bool(), 0.0))
        scores = scores * m

        o_intra = torch.matmul(scores, v_c.permute(0, 2, 1, 3))  # [B, HV, BT, V]
        o_c = (o_inter.permute(0, 2, 1, 3) + o_intra) * scale
        o[:, s:e] = o_c.permute(0, 2, 1, 3).to(v.dtype)

    return o


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'use_g', 'dtype'),
    [
        pytest.param(B, T, H, HV, D, use_g, dtype,
                     id=f"B{B}-T{T}-H{H}-HV{HV}-D{D}-use_g{use_g}-{dtype}")
        for (B, T, H, HV, D, use_g, dtype) in [
            (2, 128, 2, 2, 64, True, torch.bfloat16),
            (2, 128, 2, 4, 64, True, torch.bfloat16),
            (1, 256, 4, 4, 32, True, torch.float16),
            (2, 128, 2, 2, 64, False, torch.bfloat16),
        ]
    ],
)
def test_chunk_fwd_o(B: int, T: int, H: int, HV: int, D: int, use_g: bool, dtype: torch.dtype):
    torch.manual_seed(42)
    BT = 64
    NT = T // BT
    scale = D ** -0.5
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    # chunk_fwd_kernel_o expects h to share q/k's dtype (tl.dot has no cast).
    h = torch.randn(B, NT, HV, D, D, dtype=dtype, device=device)
    g = torch.randn(B, T, HV, dtype=torch.float32, device=device) * 0.1 if use_g else None

    ref = chunk_fwd_o_ref(q, k, v, h, g, scale, BT)
    tri = chunk_fwd_o(q=q, k=k, v=v, h=h, g=g, scale=scale, chunk_size=BT)

    assert_close('o', ref, tri, 0.005)


def chunk_gated_delta_rule_fwd_h_ref(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch baseline for `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`.

    Recurrence over chunks (state_v_first=False, USE_G, USE_GK=False). For each
    chunk [s, e) with last gate g_last = g[e-1]:
        h[it]          = h_state                         (state at chunk start)
        v_new[t]       = (u[t] - w[t] @ h_state) * exp2(g_last - g[t])
        h_state        = h_state * exp2(g_last) + k^T @ v_new
    The final `h_state` is the returned final state (fp32).
    """
    B, T, H, K = k.shape
    HV = u.shape[2]
    V = u.shape[3]
    BT = chunk_size
    assert T % BT == 0
    NT = T // BT

    if HV != H:
        k_hv = k.repeat_interleave(HV // H, dim=2)
    else:
        k_hv = k

    h_all = torch.zeros(B, NT, HV, K, V, dtype=k.dtype, device=k.device)
    v_new = torch.zeros(B, T, HV, V, dtype=u.dtype, device=u.device)
    if initial_state is not None:
        h_state = initial_state.float().clone()
    else:
        h_state = torch.zeros(B, HV, K, V, device=k.device)

    for it in range(NT):
        s, e = it * BT, (it + 1) * BT
        h_all[:, it] = h_state.to(k.dtype)
        w_c = w[:, s:e].float()       # [B, BT, HV, K]
        u_c = u[:, s:e].float()       # [B, BT, HV, V]
        k_c = k_hv[:, s:e].float()    # [B, BT, HV, K]

        b_v = u_c - torch.einsum('bthk,bhkv->bthv', w_c, h_state)
        # v_new is the *ungated* residual (stored before the gate is applied to b_v).
        v_new[:, s:e] = b_v.to(u.dtype)
        if g is not None:
            g_c = g[:, s:e].float()   # [B, BT, HV]
            g_last = g_c[:, -1]       # [B, HV]
            b_v = b_v * torch.exp2(g_last[:, None, :, None] - g_c[:, :, :, None])
            h_state = h_state * torch.exp2(g_last)[:, :, None, None]
        h_state = h_state + torch.einsum('bthk,bthv->bhkv', k_c, b_v)

    return h_all, v_new, h_state


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'use_h0', 'dtype'),
    [
        pytest.param(B, T, H, HV, D, use_h0, dtype,
                     id=f"B{B}-T{T}-H{H}-HV{HV}-D{D}-use_h0{use_h0}-{dtype}")
        for (B, T, H, HV, D, use_h0, dtype) in [
            (2, 128, 2, 2, 64, True, torch.bfloat16),
            (2, 128, 2, 4, 64, False, torch.bfloat16),
            (1, 256, 4, 4, 32, True, torch.float16),
        ]
    ],
)
def test_chunk_gated_delta_rule_fwd_h(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    use_h0: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    BT = 64
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    w = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    u = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    g = _make_gate(B, T, HV)
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32, device=device) if use_h0 else None

    h_ref, vn_ref, fs_ref = chunk_gated_delta_rule_fwd_h_ref(k, w, u, g, h0, BT)
    h_tri, vn_tri, fs_tri = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=h0,
        output_final_state=True,
        chunk_size=BT,
    )

    assert_close('h', h_ref, h_tri, 0.005)
    assert_close('v_new', vn_ref, vn_tri, 0.005)
    assert_close('final_state', fs_ref, fs_tri, 0.005)


def chunk_bwd_dv_local_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor | None,
    scale: float,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Torch baseline for `chunk_bwd_kernel_dv_local`.

    For each chunk, for t <= s:
        A[t, s] = scale * <k[t], q[s]> * exp2(g[s] - g[t])
        dv[t]   = sum_{s >= t} A[t, s] * do[s]
    """
    B, T, H, K = q.shape
    HV = do.shape[2]
    BT = chunk_size
    assert T % BT == 0

    if HV != H:
        q_hv = q.repeat_interleave(HV // H, dim=2)
        k_hv = k.repeat_interleave(HV // H, dim=2)
    else:
        q_hv, k_hv = q, k

    dv = torch.empty(B, T, HV, do.shape[-1], dtype=do.dtype, device=do.device)
    m = torch.triu(torch.ones(BT, BT, device=do.device))

    for it in range(0, T, BT):
        s, e = it, it + BT
        q_c = q_hv[:, s:e].float()
        k_c = k_hv[:, s:e].float()
        do_c = do[:, s:e].float()

        scores = torch.matmul(k_c.permute(0, 2, 1, 3), q_c.permute(0, 2, 1, 3).transpose(-1, -2)) * scale
        if g is not None:
            g_c = g[:, s:e].float().permute(0, 2, 1)  # [B, HV, BT]
            gdiff = g_c[:, :, None, :] - g_c[:, :, :, None]  # g[s] - g[t]
            scores = scores * torch.exp2(gdiff.masked_fill(~m.bool(), 0.0))
        scores = scores * m

        dv_c = torch.matmul(scores, do_c.permute(0, 2, 1, 3))
        dv[:, s:e] = dv_c.permute(0, 2, 1, 3).to(do.dtype)

    return dv


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'use_g', 'dtype'),
    [
        pytest.param(B, T, H, HV, D, use_g, dtype,
                     id=f"B{B}-T{T}-H{H}-HV{HV}-D{D}-use_g{use_g}-{dtype}")
        for (B, T, H, HV, D, use_g, dtype) in [
            (2, 128, 2, 2, 64, True, torch.bfloat16),
            (2, 128, 2, 4, 64, True, torch.bfloat16),
            (1, 256, 4, 4, 32, True, torch.float16),
            (2, 128, 2, 2, 64, False, torch.bfloat16),
        ]
    ],
)
def test_chunk_bwd_dv_local(B: int, T: int, H: int, HV: int, D: int, use_g: bool, dtype: torch.dtype):
    torch.manual_seed(42)
    BT = 64
    scale = D ** -0.5
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    do = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    g = torch.randn(B, T, HV, dtype=torch.float32, device=device) * 0.1 if use_g else None

    ref = chunk_bwd_dv_local_ref(q, k, do, g, scale, BT)
    tri = chunk_bwd_dv_local(q=q, k=k, do=do, g=g, scale=scale, chunk_size=BT)

    assert_close('dv', ref, tri, 0.005)


def chunk_bwd_dqkwg_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor | None,
    dv: torch.Tensor | None,
    g: torch.Tensor | None,
    scale: float,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Torch baseline for the `chunk_bwd_dqkwg` wrapper (gated, state_v_first=False).

    Obtained by reverse-mode autodiff of the forward sub-loss that the kernel
    differentiates:

        o      = chunk_fwd_o(q, k, v_new, h_start, g, scale)   # local output
        h_end  = state_update(h_start, k, v_new, g)            # state at chunk end
        L      = <do, o> + <dh, h_end>

    with `v_new`, `h_start` (= the chunk-start states `h`), `do` and `dh` treated
    as constants. `dh` is the gradient of the total loss w.r.t. the state at the
    END of each chunk (as produced by `chunk_gated_delta_rule_bwd_dhu`), so it
    already folds in all downstream propagation; the kernel only has to backprop
    through the local chunk_fwd_o and the local state update.

    Returns (dq, dk, dw, dg) matching the wrapper's output contract:
        dq, dk : [B, T, H, K]    (summed over the GVA head group)
        dw     : [B, T, HV, K]   (= -dv @ h_start, if w is not None)
        dg     : [B, T, HV]      (gradient w.r.t. the chunk-cumsum gate)
    """
    B, T, H, K = q.shape
    HV = do.shape[2]
    V = v_new.shape[3]
    BT = chunk_size
    assert T % BT == 0
    NT = T // BT

    q_r = q.detach().float().requires_grad_()
    k_r = k.detach().float().requires_grad_()
    g_r = g.detach().float().requires_grad_() if g is not None else None

    if HV != H:
        q_hv = q_r.repeat_interleave(HV // H, dim=2)
        k_hv = k_r.repeat_interleave(HV // H, dim=2)
    else:
        q_hv, k_hv = q_r, k_r

    v_new_c = v_new.detach().float()
    h_start = h.detach().float()          # [B, NT, HV, K, V]
    dh_c = dh.detach().float()            # [B, NT, HV, K, V], dL/d(state at chunk end)
    do_c = do.detach().float()

    # local output (q_hv/k_hv are HV-shaped, so the ref sees H == HV)
    o = chunk_fwd_o_ref(q_hv, k_hv, v_new_c, h_start, g_r, scale, BT)

    # per-chunk state update: h_end[it] = exp2(g_last)*h_start[it] + k^T @ (v_new * exp2(g_last - g))
    h_end = torch.empty(B, NT, HV, K, V, dtype=torch.float32, device=q.device)
    for it in range(NT):
        s, e = it * BT, (it + 1) * BT
        hs = h_start[:, it]               # [B, HV, K, V]
        kc = k_hv[:, s:e]                 # [B, BT, HV, K]
        vc = v_new_c[:, s:e]              # [B, BT, HV, V]
        if g_r is not None:
            gc = g_r[:, s:e]              # [B, BT, HV]
            gl = gc[:, -1]                # [B, HV]
            bv = vc * torch.exp2(gl[:, None, :, None] - gc[:, :, :, None])
            he = hs * torch.exp2(gl)[:, :, None, None]
        else:
            bv = vc
            he = hs
        he = he + torch.einsum('bthk,bthv->bhkv', kc, bv)
        h_end[:, it] = he

    loss = (do_c * o.float()).sum() + (dh_c * h_end).sum()
    vars_ = [q_r, k_r] + ([g_r] if g_r is not None else [])
    grads = torch.autograd.grad(loss, vars_, allow_unused=True)
    dq = grads[0]
    dk = grads[1]
    dg = grads[2] if g_r is not None else None
    # The kernel computes dg via the identity sum(dq*q) - sum(dk*k), which drops the
    # ln2 factor that autodiff through exp2(g) produces. The missing RCP_LN2 is
    # compensated by the orchestration's reverse cumsum (see chunk_gated_delta_rule_bwd),
    # so the per-kernel dg contract is dL/d(g_cumsum) * RCP_LN2.
    if dg is not None:
        dg = dg * RCP_LN2

    # dw = -dv @ h_start (the kernel computes this from the `dv` input; `w` only sets the shape)
    if w is not None and dv is not None:
        du_c = dv.detach().float()        # [B, T, HV, V]
        dw = torch.empty(B, T, HV, K, dtype=torch.float32, device=q.device)
        for it in range(NT):
            s, e = it * BT, (it + 1) * BT
            dw[:, s:e] = -torch.einsum('bthv,bhkv->bthk', du_c[:, s:e], h_start[:, it])
        dw = dw.to(w.dtype)
    else:
        dw = None

    return dq.to(q.dtype), dk.to(k.dtype), dw, dg


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'dtype'),
    [
        pytest.param(B, T, H, HV, D, dtype, id=f"B{B}-T{T}-H{H}-HV{HV}-D{D}-{dtype}")
        for (B, T, H, HV, D, dtype) in [
            (2, 128, 2, 2, 64, torch.bfloat16),
            (2, 128, 2, 4, 64, torch.bfloat16),
            (1, 256, 4, 4, 32, torch.float16),
        ]
    ],
)
def test_chunk_bwd_dqkwg(B: int, T: int, H: int, HV: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    BT = 64
    NT = T // BT
    scale = D ** -0.5
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v_new = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    do = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    h = torch.randn(B, NT, HV, D, D, dtype=dtype, device=device)
    dh = torch.randn(B, NT, HV, D, D, dtype=dtype, device=device)
    w = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    dv = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    g = torch.randn(B, T, HV, dtype=torch.float32, device=device) * 0.1

    dq_ref, dk_ref, dw_ref, dg_ref = chunk_bwd_dqkwg_ref(q, k, v_new, do, h, dh, w, dv, g, scale, BT)
    dq_tri, dk_tri, dw_tri, dg_tri = chunk_bwd_dqkwg(
        q=q, k=k, v=v_new, do=do, h=h, dh=dh, w=w, dv=dv, g=g, scale=scale, chunk_size=BT,
    )

    assert_close('dq', dq_ref, dq_tri, 0.006)
    assert_close('dk', dk_ref, dk_tri, 0.006)
    assert_close('dw', dw_ref, dw_tri, 0.006)
    assert_close('dg', dg_ref, dg_tri, 0.006)


def gdn_fwd_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None,
    h0: torch.Tensor | None,
    scale: float,
    chunk_size: int = 64,
):
    """Full differentiable GDN forward in torch (chains the per-kernel torch
    baselines). Returns (o, final_state, intermediates) so that torch.autograd can
    produce reference gradients for every backward kernel and intermediate."""
    A = chunk_kkt_solve_ref(k, g, beta, chunk_size)
    w, u = recompute_w_u_fwd_ref(k, v, beta, A, g)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h_ref(k, w, u, g, h0, chunk_size)
    o = chunk_fwd_o_ref(q, k, v_new, h, g, scale, chunk_size)
    return o, final_state, dict(A=A, w=w, u=u, h=h, v_new=v_new)


def gdn_bwd_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
) -> dict[str, torch.Tensor]:
    """Reference input gradients for the GDN backward, obtained by
    torch.autograd on the gold-standard `naive_recurrent_gated_delta_rule`
    forward (the same reference used by `tests/ops/test_gdn.py`).

    Returns gradients of ``sum(do*o) + sum(dht*final_state)`` w.r.t. the inputs.
    These serve as the ground truth both for the end-to-end backward test and for
    cross-checking the per-kernel backward baselines.
    """
    q_r = q.detach().requires_grad_()
    k_r = k.detach().requires_grad_()
    v_r = v.detach().requires_grad_()
    beta_r = beta.detach().requires_grad_()
    g_r = g.detach().requires_grad_()
    h0_r = h0.detach().requires_grad_() if h0 is not None else None

    H, HV = q.shape[2], v.shape[2]
    # The naive reference does not handle GVA internally; expand q/k to HV heads
    # (as tests/ops/test_gdn.py does). torch.autograd folds the expanded-head
    # gradients back into q_r.grad / k_r.grad automatically.
    gqa = HV // H
    if gqa > 1:
        q_in = q_r.repeat_interleave(gqa, dim=2)
        k_in = k_r.repeat_interleave(gqa, dim=2)
    else:
        q_in, k_in = q_r, k_r

    o, final_state = naive_recurrent_gated_delta_rule(
        q_in, k_in, v_r, beta_r, g_r, scale, h0_r, output_final_state=True,
    )
    loss = (do * o).sum()
    if dht is not None:
        loss = loss + (dht * final_state).sum()
    loss.backward()

    return dict(
        dq=q_r.grad, dk=k_r.grad, dv=v_r.grad, dbeta=beta_r.grad,
        dg=g_r.grad, dh0=h0_r.grad if h0_r is not None else None,
    )


def gdn_gate_bwd_ref(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dyg: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Torch baseline for `gdn_gate_bwd_kernel`.

    gate = -exp(A_log) * softplus(g + dt_bias) = yg
    dg       = -exp(A_log) * dyg * sigmoid(g + dt_bias)
    dA_log[h] = sum_t dyg[t, h] * yg[t, h]
    dbias[h]  = sum_t dg[t, h]   (if dt_bias is not None)
    """
    x = g.float()
    if dt_bias is not None:
        x = x + dt_bias.float()
    neg_expA = -torch.exp(A_log.float())
    yg = neg_expA * F.softplus(x)
    dg = neg_expA * dyg.float() * torch.sigmoid(x)
    reduce_dims = tuple(range(dyg.ndim - 1))
    dA_log = (dyg.float() * yg).sum(dim=reduce_dims)
    dbias = dg.sum(dim=reduce_dims) if dt_bias is not None else None
    return dg, dA_log, dbias


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'use_bias', 'dtype'),
    [
        pytest.param(B, T, H, use_bias, dtype, id=f"B{B}-T{T}-H{H}-use_bias{use_bias}-{dtype}")
        for (B, T, H, use_bias, dtype) in [
            (2, 128, 4, False, torch.bfloat16),
            (2, 128, 4, True, torch.bfloat16),
            (1, 256, 8, True, torch.float32),
        ]
    ],
)
def test_gdn_gate_bwd(B: int, T: int, H: int, use_bias: bool, dtype: torch.dtype):
    torch.manual_seed(42)
    g = torch.randn(B, T, H, dtype=dtype, device=device)
    A_log = torch.randn(H, dtype=torch.float32, device=device)
    dyg = torch.randn(B, T, H, dtype=dtype, device=device)
    dt_bias = torch.randn(H, dtype=torch.float32, device=device) if use_bias else None

    dg_ref, dA_ref, db_ref = gdn_gate_bwd_ref(g, A_log, dyg, dt_bias)
    dg_tri, dA_tri, db_tri = gdn_gate_bwd(g=g, A_log=A_log, dt_bias=dt_bias, dyg=dyg)

    assert_close('dg', dg_ref.to(dtype), dg_tri, 0.005)
    assert_close('dA_log', dA_ref, dA_tri, 0.005)
    if use_bias:
        assert_close('dt_bias', db_ref.to(dtype), db_tri, 0.005)


def prepare_wy_repr_bwd_ref(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch baseline for the `prepare_wy_repr_bwd` wrapper (state_v_first=False).

    The kernel is the exact backward of `recompute_w_u_fwd`, including the gradient
    through the WY inverse ``A = (I + L)^{-1}`` (which depends on k, beta, g). We
    therefore recompute the *true* A from (k, g, beta) and let reverse-mode
    autodiff flow through it:

        A      = (I + L(k, g, beta))^{-1}
        w, u   = recompute_w_u_fwd(k, v, beta, A, g)
        L_loss = <dw, w> + <du, u>
        dk, dv, dbeta = autograd(L_loss, [k, v, beta])

    The passed `A` is intentionally not used for the gradient (the kernel likewise
    assumes A is the true inverse of (I + L)); the test feeds the same true A to
    the kernel so the forward values agree. Returns (dk, dv, dbeta) matching the
    wrapper's output contract: dk is summed over the GVA head group.
    """
    k_r = k.detach().float().requires_grad_()
    v_r = v.detach().float().requires_grad_()
    beta_r = beta.detach().float().requires_grad_()
    g_r = g.detach().float().requires_grad_() if g is not None else None

    # true WY inverse, differentiable w.r.t. k, beta, g
    A_r = chunk_kkt_solve_ref(k_r, g_r, beta_r, chunk_size)
    w, u = recompute_w_u_fwd_ref(k_r, v_r, beta_r, A_r, g_r)

    loss = (dw.detach().float() * w.float()).sum() + (du.detach().float() * u.float()).sum()
    dk, dv, db = torch.autograd.grad(loss, [k_r, v_r, beta_r])
    return dk.to(k.dtype), dv.to(v.dtype), db


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'use_g', 'dtype'),
    [
        pytest.param(B, T, H, HV, D, use_g, dtype,
                     id=f"B{B}-T{T}-H{H}-HV{HV}-D{D}-use_g{use_g}-{dtype}")
        for (B, T, H, HV, D, use_g, dtype) in [
            (2, 128, 2, 2, 64, True, torch.bfloat16),
            (2, 128, 2, 4, 64, True, torch.bfloat16),
            (1, 256, 4, 4, 32, True, torch.float16),
            (2, 128, 2, 2, 64, False, torch.bfloat16),
        ]
    ],
)
def test_prepare_wy_repr_bwd(B: int, T: int, H: int, HV: int, D: int, use_g: bool, dtype: torch.dtype):
    torch.manual_seed(42)
    BT = 64
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    # normalize k so the (I + L) inverse is well-conditioned (production always
    # L2-normalizes q/k before the op, as in test_gdn_full_bwd); random k makes
    # the inverse ill-conditioned and blows up autodiff through it.
    k = F.normalize(k, p=2, dim=-1)
    v = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    beta = torch.rand(B, T, HV, dtype=dtype, device=device).sigmoid()
    g = torch.randn(B, T, HV, dtype=torch.float32, device=device) * 0.1 if use_g else None
    # the kernel's backward assumes A is the true (I + L)^{-1}; feed the true A so
    # the kernel and the autograd baseline see identical forward values.
    A = chunk_kkt_solve_ref(k, g, beta, BT)
    dw = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    du = torch.randn(B, T, HV, D, dtype=dtype, device=device)

    dk_ref, dv_ref, db_ref = prepare_wy_repr_bwd_ref(k, v, beta, A, dw, du, g, BT)
    dk_tri, dv_tri, db_tri, _ = prepare_wy_repr_bwd(k=k, v=v, beta=beta, A=A, dw=dw, du=du, g=g)

    assert_close('dk', dk_ref, dk_tri, 0.006)
    assert_close('dv', dv_ref, dv_tri, 0.006)
    assert_close('db', db_ref, db_tri, 0.006)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'use_h0', 'dtype'),
    [
        pytest.param(B, T, H, HV, D, use_h0, dtype,
                     id=f"B{B}-T{T}-H{H}-HV{HV}-D{D}-use_h0{use_h0}-{dtype}")
        for (B, T, H, HV, D, use_h0, dtype) in [
            (2, 128, 2, 2, 64, True, torch.bfloat16),
            (2, 128, 2, 4, 64, False, torch.bfloat16),
            (1, 128, 4, 4, 32, True, torch.float16),
        ]
    ],
)
def test_gdn_full_bwd(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    use_h0: bool,
    dtype: torch.dtype,
):
    """Validate the full GDN backward pipeline against torch-fwd + autograd.

    This is the same reference style as `tests/ops/test_gdn.py` (torch forward
    + autograd), and is used here as the ground truth to (a) validate the whole
    backward pipeline end-to-end and (b) cross-check the per-kernel backward
    baselines. A real Triton kernel bug (e.g. the prepare_wy_repr_bwd issue from
    #984) shows up here as a mismatched input gradient.
    """
    torch.manual_seed(42)
    scale = D ** -0.5
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    beta = torch.rand(B, T, HV, dtype=dtype, device=device).sigmoid()
    g = _make_gate(B, T, HV)
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32, device=device) if use_h0 else None
    do = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    dht = torch.randn(B, HV, D, D, dtype=torch.float32, device=device)

    # match tests/ops/test_gdn.py: L2-normalize q and k before the op so the
    # gradients are comparable and the chunk-kernel inputs stay bounded.
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    ref = gdn_bwd_autograd(q, k, v, beta, g, h0, do, dht, scale)

    q_t = q.detach().requires_grad_()
    k_t = k.detach().requires_grad_()
    v_t = v.detach().requires_grad_()
    beta_t = beta.detach().requires_grad_()
    g_t = g.detach().requires_grad_()
    h0_t = h0.detach().requires_grad_() if h0 is not None else None
    o, final_state = chunk_gated_delta_rule(
        q_t, k_t, v_t, g_t, beta_t, scale, h0_t, output_final_state=True,
    )
    loss = (do * o).sum() + (dht * final_state).sum()
    loss.backward()

    assert_close('dq', ref['dq'].to(dtype), q_t.grad, 0.006)
    assert_close('dk', ref['dk'].to(dtype), k_t.grad, 0.008)
    assert_close('dv', ref['dv'].to(dtype), v_t.grad, 0.006)
    assert_close('dbeta', ref['dbeta'].to(dtype), beta_t.grad, 0.008)
    assert_close('dg', ref['dg'].to(dtype), g_t.grad, 0.008)
    if use_h0:
        assert_close('dh0', ref['dh0'], h0_t.grad, 0.006)


def chunk_gated_delta_rule_bwd_dhu_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    h0: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    dv_local: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Torch baseline for `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`.

    The kernel's `dh` is the gradient w.r.t. the chunk-*end* states, produced by a
    reverse recurrence that autodiff on the forward states cannot reproduce
    directly (the reverse-pass gradient flows through a running accumulator, not
    through the stored forward states). We therefore evaluate that same reverse
    recurrence in plain torch (state_v_first=False, USE_GK=False):

        b_dh = dht
        for it = NT-1 ... 0:
            dh[it]  = b_dh
            b_dv    = (k[it] @ b_dh) * exp2(g_last[it] - g[it]) + dv_local[it]
            dv2[it] = b_dv
            b_dh    = b_dh * exp2(g_last[it])
                      + (q_gated[it]^T @ do[it]) * scale
                      - w[it]^T @ b_dv
        dh0   = b_dh

    Returns (dh, dh0, dv2) matching the kernel. `u` is unused (the kernel consumes
    `dv_local` instead) and kept only for a uniform call signature.
    """
    del u  # unused: the kernel differentiates w.r.t. dv_local, not u
    B, T, H, K = k.shape
    HV = do.shape[2]
    V = do.shape[3]
    BT = chunk_size
    assert T % BT == 0
    NT = T // BT

    if HV != H:
        q_hv = q.float().repeat_interleave(HV // H, dim=2)
        k_hv = k.float().repeat_interleave(HV // H, dim=2)
    else:
        q_hv = q.float()
        k_hv = k.float()
    w_f = w.float()
    do_f = do.float()
    dv_f = dv_local.float()
    g_f = g.float() if g is not None else None

    dh = torch.zeros(B, NT, HV, K, V, dtype=torch.float32, device=k.device)
    dv2 = torch.zeros(B, T, HV, V, dtype=torch.float32, device=k.device)
    if dht is not None:
        b_dh = dht.float().clone()
    else:
        b_dh = torch.zeros(B, HV, K, V, dtype=torch.float32, device=k.device)

    for it in range(NT - 1, -1, -1):
        s, e = it * BT, (it + 1) * BT
        dh[:, it] = b_dh
        q_c = q_hv[:, s:e]            # [B, BT, HV, K]
        k_c = k_hv[:, s:e]            # [B, BT, HV, K]
        w_c = w_f[:, s:e]             # [B, BT, HV, K]
        do_c = do_f[:, s:e]           # [B, BT, HV, V]
        dv_c = dv_f[:, s:e]           # [B, BT, HV, V]
        kdh = torch.einsum('bthk,bhkv->bthv', k_c, b_dh)   # [B, BT, HV, V]
        if g_f is not None:
            g_c = g_f[:, s:e]         # [B, BT, HV]
            g_last = g_c[:, -1]       # [B, HV]
            b_dv = kdh * torch.exp2(g_last[:, None, :, None] - g_c[:, :, :, None]) + dv_c
            q_gated = q_c * torch.exp2(g_c)[:, :, :, None]
            b_dh = b_dh * torch.exp2(g_last)[:, :, None, None]
        else:
            b_dv = kdh + dv_c
            q_gated = q_c
        dv2[:, s:e] = b_dv
        b_dh = b_dh + torch.einsum('bthk,bthv->bhkv', q_gated, do_c) * scale
        b_dh = b_dh - torch.einsum('bthk,bthv->bhkv', w_c, b_dv)

    dh0 = b_dh if h0 is not None else None
    return dh, dh0, dv2


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'use_h0', 'dtype'),
    [
        pytest.param(B, T, H, HV, D, use_h0, dtype,
                     id=f"B{B}-T{T}-H{H}-HV{HV}-D{D}-use_h0{use_h0}-{dtype}")
        for (B, T, H, HV, D, use_h0, dtype) in [
            (2, 128, 2, 2, 64, True, torch.bfloat16),
            (2, 128, 2, 4, 64, False, torch.bfloat16),
            (1, 256, 4, 4, 32, True, torch.float16),
        ]
    ],
)
def test_chunk_gated_delta_rule_bwd_dhu(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    use_h0: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    BT = 64
    scale = D ** -0.5
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    w = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    u = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    do = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    g = _make_gate(B, T, HV)
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32, device=device) if use_h0 else None
    dht = torch.randn(B, HV, D, D, dtype=torch.float32, device=device)

    # the kernel takes dv_local as input; compute it via its own baseline
    dv_local = chunk_bwd_dv_local_ref(q, k, do, g, scale, BT)
    dh_ref, dh0_ref, dv2_ref = chunk_gated_delta_rule_bwd_dhu_ref(
        q, k, w, u, g, h0, do, dht, dv_local, scale, BT,
    )
    dh_tri, dh0_tri, dv2_tri = chunk_gated_delta_rule_bwd_dhu(
        q=q, k=k, w=w, g=g, h0=h0, dht=dht, do=do, dv=dv_local, scale=scale, chunk_size=BT,
    )

    assert_close('dh', dh_ref.to(dtype), dh_tri, 0.006)
    assert_close('dv2', dv2_ref.to(dtype), dv2_tri, 0.006)
    if use_h0:
        assert_close('dh0', dh0_ref, dh0_tri, 0.006)
