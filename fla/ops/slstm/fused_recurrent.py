# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# The sLSTM of xLSTM (https://arxiv.org/abs/2405.04517): exponential input gates and sigmoid forget gates with a
# stabilizer state m, and a block-diagonal recurrent matrix R that mixes the hidden state within each head,
#     [i~, f~, z~, o~]_t = x_t + h_{t-1} R,
#     m_t = max(i~_t, log sigmoid(f~_t) + m_{t-1}),
#     c_t = exp(log sigmoid(f~_t) + m_{t-1} - m_t) c_{t-1} + exp(i~_t - m_t) tanh(z~_t),
#     n_t = exp(log sigmoid(f~_t) + m_{t-1} - m_t) n_{t-1} + exp(i~_t - m_t),
#     h_t = sigmoid(o~_t) c_t / n_t.
# The recurrence is nonlinear in h, so every program scans one head token by token. The stabilizer rescales c and n
# alike and cancels in h, so the backward pass treats m as a constant and is still exact for h.

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp, log
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard


@triton.jit
def logsigmoid(x):
    return tl.minimum(x, 0.) - log(1. + exp(-tl.abs(x)))


@triton.jit
def tanh(x):
    return 2. * tl.sigmoid(2. * x) - 1.


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_slstm_fwd_kernel(
    x,
    R,
    h0,
    c0,
    n0,
    m0,
    y,
    gs,
    cs,
    ns,
    ms,
    ht,
    ct,
    nt,
    mt,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_bh = tl.program_id(0).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_R = m_d[:, None] & m_d[None, :]
    p_R = R + i_h * 4 * D * D + o_d[:, None] * D + o_d[None, :]

    b_h = tl.zeros([BD], dtype=tl.float32)
    b_c = tl.zeros([BD], dtype=tl.float32)
    b_n = tl.zeros([BD], dtype=tl.float32)
    b_m = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        o_s = i_bh * D + o_d
        b_h += tl.load(h0 + o_s, mask=m_d, other=0.).to(tl.float32)
        b_c += tl.load(c0 + o_s, mask=m_d, other=0.).to(tl.float32)
        b_n += tl.load(n0 + o_s, mask=m_d, other=0.).to(tl.float32)
        b_m += tl.load(m0 + o_s, mask=m_d, other=0.).to(tl.float32)

    for i_t in range(0, T):
        # [4, H, D] gates of token t, and [H, D] states
        o_g = (i_b * T + i_t) * 4 * H * D + i_h * D + o_d
        o_s = ((i_b * T + i_t) * H + i_h) * D + o_d
        b_hk = b_h[:, None]
        b_ri = tl.load(x + o_g, mask=m_d, other=0.).to(tl.float32) + \
            tl.sum(b_hk * tl.load(p_R, mask=m_R, other=0.).to(tl.float32), 0)
        b_rf = tl.load(x + o_g + H * D, mask=m_d, other=0.).to(tl.float32) + \
            tl.sum(b_hk * tl.load(p_R + D * D, mask=m_R, other=0.).to(tl.float32), 0)
        b_rz = tl.load(x + o_g + 2 * H * D, mask=m_d, other=0.).to(tl.float32) + \
            tl.sum(b_hk * tl.load(p_R + 2 * D * D, mask=m_R, other=0.).to(tl.float32), 0)
        b_ro = tl.load(x + o_g + 3 * H * D, mask=m_d, other=0.).to(tl.float32) + \
            tl.sum(b_hk * tl.load(p_R + 3 * D * D, mask=m_R, other=0.).to(tl.float32), 0)

        b_logf = logsigmoid(b_rf)
        b_m_new = tl.maximum(b_ri, b_logf + b_m)
        b_i = exp(b_ri - b_m_new)
        b_f = exp(b_logf + b_m - b_m_new)
        b_c = b_f * b_c + b_i * tanh(b_rz)
        b_n = b_f * b_n + b_i
        b_m = b_m_new
        b_h = tl.sigmoid(b_ro) * b_c / b_n

        tl.store(y + o_s, b_h.to(y.dtype.element_ty), mask=m_d)
        tl.store(gs + o_g, b_ri, mask=m_d)
        tl.store(gs + o_g + H * D, b_rf, mask=m_d)
        tl.store(gs + o_g + 2 * H * D, b_rz, mask=m_d)
        tl.store(gs + o_g + 3 * H * D, b_ro, mask=m_d)
        tl.store(cs + o_s, b_c, mask=m_d)
        tl.store(ns + o_s, b_n, mask=m_d)
        tl.store(ms + o_s, b_m, mask=m_d)

    if STORE_FINAL_STATE:
        o_s = i_bh * D + o_d
        tl.store(ht + o_s, b_h, mask=m_d)
        tl.store(ct + o_s, b_c, mask=m_d)
        tl.store(nt + o_s, b_n, mask=m_d)
        tl.store(mt + o_s, b_m, mask=m_d)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_slstm_bwd_kernel(
    R,
    c0,
    n0,
    m0,
    gs,
    cs,
    ns,
    ms,
    dy,
    dgs,
    dh0,
    dc0,
    dn0,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
):
    i_bh = tl.program_id(0).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_R = m_d[:, None] & m_d[None, :]
    p_R = R + i_h * 4 * D * D + o_d[:, None] * D + o_d[None, :]

    # gradients of h_{t}, c_t and n_t from later tokens
    b_dh = tl.zeros([BD], dtype=tl.float32)
    b_dc = tl.zeros([BD], dtype=tl.float32)
    b_dn = tl.zeros([BD], dtype=tl.float32)
    for i_t in range(T - 1, -1, -1):
        o_g = (i_b * T + i_t) * 4 * H * D + i_h * D + o_d
        o_s = ((i_b * T + i_t) * H + i_h) * D + o_d
        b_ri = tl.load(gs + o_g, mask=m_d, other=0.)
        b_rf = tl.load(gs + o_g + H * D, mask=m_d, other=0.)
        b_rz = tl.load(gs + o_g + 2 * H * D, mask=m_d, other=0.)
        b_ro = tl.load(gs + o_g + 3 * H * D, mask=m_d, other=0.)
        b_c = tl.load(cs + o_s, mask=m_d, other=0.)
        b_n = tl.load(ns + o_s, mask=m_d, other=1.)
        b_m = tl.load(ms + o_s, mask=m_d, other=0.)
        if i_t > 0:
            b_c_prev = tl.load(cs + o_s - H * D, mask=m_d, other=0.)
            b_n_prev = tl.load(ns + o_s - H * D, mask=m_d, other=0.)
            b_m_prev = tl.load(ms + o_s - H * D, mask=m_d, other=0.)
        elif USE_INITIAL_STATE:
            b_c_prev = tl.load(c0 + i_bh * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_n_prev = tl.load(n0 + i_bh * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_m_prev = tl.load(m0 + i_bh * D + o_d, mask=m_d, other=0.).to(tl.float32)
        else:
            b_c_prev = tl.zeros([BD], dtype=tl.float32)
            b_n_prev = tl.zeros([BD], dtype=tl.float32)
            b_m_prev = tl.zeros([BD], dtype=tl.float32)

        b_dh += tl.load(dy + o_s, mask=m_d, other=0.).to(tl.float32)
        b_o = tl.sigmoid(b_ro)
        b_z = tanh(b_rz)
        b_i = exp(b_ri - b_m)
        b_f = exp(logsigmoid(b_rf) + b_m_prev - b_m)
        b_r = b_c / b_n
        # h_t = o c_t / n_t
        b_dro = b_dh * b_r * b_o * (1 - b_o)
        b_dc += b_dh * b_o / b_n
        b_dn -= b_dh * b_o * b_r / b_n
        # c_t = f c_{t-1} + i z, n_t = f n_{t-1} + i
        b_dri = (b_dc * b_z + b_dn) * b_i
        b_drz = b_dc * b_i * (1 - b_z * b_z)
        b_drf = (b_dc * b_c_prev + b_dn * b_n_prev) * b_f * (1 - tl.sigmoid(b_rf))
        tl.store(dgs + o_g, b_dri, mask=m_d)
        tl.store(dgs + o_g + H * D, b_drf, mask=m_d)
        tl.store(dgs + o_g + 2 * H * D, b_drz, mask=m_d)
        tl.store(dgs + o_g + 3 * H * D, b_dro, mask=m_d)

        # the gates read h_{t-1} R
        b_dh = tl.sum(tl.load(p_R, mask=m_R, other=0.).to(tl.float32) * b_dri[None, :], 1)
        b_dh += tl.sum(tl.load(p_R + D * D, mask=m_R, other=0.).to(tl.float32) * b_drf[None, :], 1)
        b_dh += tl.sum(tl.load(p_R + 2 * D * D, mask=m_R, other=0.).to(tl.float32) * b_drz[None, :], 1)
        b_dh += tl.sum(tl.load(p_R + 3 * D * D, mask=m_R, other=0.).to(tl.float32) * b_dro[None, :], 1)
        b_dc = b_dc * b_f
        b_dn = b_dn * b_f

    if USE_INITIAL_STATE:
        o_s = i_bh * D + o_d
        tl.store(dh0 + o_s, b_dh, mask=m_d)
        tl.store(dc0 + o_s, b_dc, mask=m_d)
        tl.store(dn0 + o_s, b_dn, mask=m_d)


def fused_recurrent_slstm_fwd(x, R, initial_state, output_final_state):
    B, T, _, H, D = x.shape
    y = x.new_empty(B, T, H, D)
    gs = x.new_empty(B, T, 4, H, D, dtype=torch.float)
    cs, ns, ms = (x.new_empty(B, T, H, D, dtype=torch.float) for _ in range(3))
    h0, c0, n0, m0 = initial_state if initial_state is not None else (None,) * 4
    final_state = tuple(x.new_empty(B, H, D, dtype=torch.float) for _ in range(4)) if output_final_state else (None,) * 4
    fused_recurrent_slstm_fwd_kernel[(B * H,)](
        x, R, h0, c0, n0, m0, y, gs, cs, ns, ms, *final_state, T=T, H=H, D=D, BD=triton.next_power_of_2(D),
    )
    return y, gs, cs, ns, ms, (final_state if output_final_state else None)


def fused_recurrent_slstm_bwd(R, initial_state, y, gs, cs, ns, ms, dy):
    B, T, _, H, D = gs.shape
    dgs = torch.empty_like(gs)
    _, c0, n0, m0 = initial_state if initial_state is not None else (None,) * 4
    dh0, dc0, dn0 = (gs.new_empty(B, H, D) for _ in range(3)) if initial_state is not None else (None,) * 3
    fused_recurrent_slstm_bwd_kernel[(B * H,)](
        R, c0, n0, m0, gs, cs, ns, ms, dy, dgs, dh0, dc0, dn0, T=T, H=H, D=D, BD=triton.next_power_of_2(D),
    )
    # [B, T, H, D] hidden states read by every token
    h_prev = torch.cat([
        initial_state[0][:, None].float() if initial_state is not None else y.new_zeros(B, 1, H, D, dtype=torch.float),
        y[:, :-1].float(),
    ], 1)
    dR = torch.einsum('bthk,btghj->hgkj', h_prev, dgs)
    return dgs, dR, dh0, dc0, dn0


class FusedRecurrentSLSTMFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, R, h0, c0, n0, m0, output_final_state):
        initial_state = (h0, c0, n0, m0) if h0 is not None else None
        y, gs, cs, ns, ms, final_state = fused_recurrent_slstm_fwd(x, R, initial_state, output_final_state)
        ctx.save_for_backward(R, h0, c0, n0, m0, y, gs, cs, ns, ms)
        if final_state is not None:
            ctx.mark_non_differentiable(*final_state)
        return y, final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dy, *_):
        R, h0, c0, n0, m0, y, gs, cs, ns, ms = ctx.saved_tensors
        initial_state = (h0, c0, n0, m0) if h0 is not None else None
        dgs, dR, dh0, dc0, dn0 = fused_recurrent_slstm_bwd(R, initial_state, y, gs, cs, ns, ms, dy)
        if initial_state is not None:
            dh0, dc0, dn0 = dh0.to(h0), dc0.to(c0), dn0.to(n0)
        return dgs.to(dy.dtype), dR.to(R), dh0, dc0, dn0, None, None


@torch.compiler.disable
def fused_recurrent_slstm(
    x: torch.Tensor,
    R: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    r"""
    The sLSTM of xLSTM (https://arxiv.org/abs/2405.04517) with a head-wise recurrent matrix:

    .. math::
        [\tilde{i}, \tilde{f}, \tilde{z}, \tilde{o}]_t = x_t + h_{t-1} R, \quad
        h_t = \sigma(\tilde{o}_t) \frac{c_t}{n_t},

    where `c_t` and `n_t` accumulate `tanh(z~_t)` and 1 with exponential input gates `exp(i~_t)` and sigmoid forget
    gates `sigmoid(f~_t)`, stabilized by `m_t`.

    Args:
        x (torch.Tensor):
            gate pre-activations of shape `[B, T, 4, H, D]`, ordered as input, forget, cell and output gates,
            usually a projection of the input plus the gate biases.
        R (torch.Tensor):
            recurrent weights of shape `[H, 4, D, D]`, where `R[h, g]` maps the hidden state of head `h` to gate `g`.
        initial_state (Optional[tuple[torch.Tensor, ...]]):
            initial states `(h, c, n, m)`, each of shape `[B, H, D]`. Gradients flow to `h`, `c` and `n`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final states `(h, c, n, m)` in fp32, which carry no gradient. Default: `False`.

    Returns:
        y (torch.Tensor):
            hidden states of shape `[B, T, H, D]`.
        final_state (Optional[tuple[torch.Tensor, ...]]):
            final states `(h, c, n, m)` if `output_final_state=True` else `None`.
    """
    if x.dim() != 5 or x.shape[2] != 4:
        raise ValueError(f"`x` is expected to have shape [B, T, 4, H, D], got {tuple(x.shape)}.")
    h0, c0, n0, m0 = initial_state if initial_state is not None else (None,) * 4
    return FusedRecurrentSLSTMFunction.apply(x, R, h0, c0, n0, m0, output_final_state)
