# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp, logaddexp, tanh
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard

StateTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@triton.jit
def logsigmoid(x):
    return -logaddexp(0., -x)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_slstm_fwd_kernel(
    x,
    recurrent_weight,
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
    cu_seqlens,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(0).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_rw = m_d[:, None] & m_d[None, :]
    p_rw = recurrent_weight + i_h * 4 * D * D + o_d[:, None] * D + o_d[None, :]

    b_h = tl.zeros([BD], dtype=tl.float32)
    b_c = tl.zeros([BD], dtype=tl.float32)
    b_n = tl.zeros([BD], dtype=tl.float32)
    b_m = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        o_s = i_nh * D + o_d
        b_h += tl.load(h0 + o_s, mask=m_d, other=0.).to(tl.float32)
        b_c += tl.load(c0 + o_s, mask=m_d, other=0.).to(tl.float32)
        b_n += tl.load(n0 + o_s, mask=m_d, other=0.).to(tl.float32)
        b_m += tl.load(m0 + o_s, mask=m_d, other=0.).to(tl.float32)

    for i_t in range(0, T):
        # [4, H, D] gates of token t, and [H, D] states
        o_g = (bos + i_t) * 4 * H * D + i_h * D + o_d
        o_s = ((bos + i_t) * H + i_h) * D + o_d
        b_hk = b_h[:, None]
        b_ri = tl.load(x + o_g, mask=m_d, other=0.).to(tl.float32) + \
            tl.sum(b_hk * tl.load(p_rw, mask=m_rw, other=0.).to(tl.float32), 0)
        b_rf = tl.load(x + o_g + H * D, mask=m_d, other=0.).to(tl.float32) + \
            tl.sum(b_hk * tl.load(p_rw + D * D, mask=m_rw, other=0.).to(tl.float32), 0)
        b_rz = tl.load(x + o_g + 2 * H * D, mask=m_d, other=0.).to(tl.float32) + \
            tl.sum(b_hk * tl.load(p_rw + 2 * D * D, mask=m_rw, other=0.).to(tl.float32), 0)
        b_ro = tl.load(x + o_g + 3 * H * D, mask=m_d, other=0.).to(tl.float32) + \
            tl.sum(b_hk * tl.load(p_rw + 3 * D * D, mask=m_rw, other=0.).to(tl.float32), 0)

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
        o_s = i_nh * D + o_d
        tl.store(ht + o_s, b_h, mask=m_d)
        tl.store(ct + o_s, b_c, mask=m_d)
        tl.store(nt + o_s, b_n, mask=m_d)
        tl.store(mt + o_s, b_m, mask=m_d)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_slstm_bwd_kernel(
    recurrent_weight,
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
    cu_seqlens,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(0).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_rw = m_d[:, None] & m_d[None, :]
    p_rw = recurrent_weight + i_h * 4 * D * D + o_d[:, None] * D + o_d[None, :]

    # gradients of h_t, c_t and n_t from later tokens
    b_dh = tl.zeros([BD], dtype=tl.float32)
    b_dc = tl.zeros([BD], dtype=tl.float32)
    b_dn = tl.zeros([BD], dtype=tl.float32)
    for i_t in range(T - 1, -1, -1):
        o_g = (bos + i_t) * 4 * H * D + i_h * D + o_d
        o_s = ((bos + i_t) * H + i_h) * D + o_d
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
            b_c_prev = tl.load(c0 + i_nh * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_n_prev = tl.load(n0 + i_nh * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_m_prev = tl.load(m0 + i_nh * D + o_d, mask=m_d, other=0.).to(tl.float32)
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

        # the gates read h_{t-1} @ recurrent_weight
        b_dh = tl.sum(tl.load(p_rw, mask=m_rw, other=0.).to(tl.float32) * b_dri[None, :], 1)
        b_dh += tl.sum(tl.load(p_rw + D * D, mask=m_rw, other=0.).to(tl.float32) * b_drf[None, :], 1)
        b_dh += tl.sum(tl.load(p_rw + 2 * D * D, mask=m_rw, other=0.).to(tl.float32) * b_drz[None, :], 1)
        b_dh += tl.sum(tl.load(p_rw + 3 * D * D, mask=m_rw, other=0.).to(tl.float32) * b_dro[None, :], 1)
        b_dc = b_dc * b_f
        b_dn = b_dn * b_f

    if USE_INITIAL_STATE:
        o_s = i_nh * D + o_d
        tl.store(dh0 + o_s, b_dh, mask=m_d)
        tl.store(dc0 + o_s, b_dc, mask=m_d)
        tl.store(dn0 + o_s, b_dn, mask=m_d)


def fused_recurrent_slstm_fwd(
    x: torch.Tensor,
    recurrent_weight: torch.Tensor,
    initial_state: StateTuple | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StateTuple | None]:
    B, T, _, H, D = x.shape
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    y = x.new_empty(B, T, H, D)
    gs = x.new_empty(B, T, 4, H, D, dtype=torch.float)
    cs, ns, ms = (x.new_empty(B, T, H, D, dtype=torch.float) for _ in range(3))
    h0, c0, n0, m0 = initial_state if initial_state is not None else (None, None, None, None)
    if output_final_state:
        ht, ct, nt, mt = (x.new_empty(N, H, D, dtype=torch.float) for _ in range(4))
    else:
        ht = ct = nt = mt = None
    fused_recurrent_slstm_fwd_kernel[(N * H,)](
        x=x,
        recurrent_weight=recurrent_weight,
        h0=h0,
        c0=c0,
        n0=n0,
        m0=m0,
        y=y,
        gs=gs,
        cs=cs,
        ns=ns,
        ms=ms,
        ht=ht,
        ct=ct,
        nt=nt,
        mt=mt,
        cu_seqlens=cu_seqlens,
        T=T,
        H=H,
        D=D,
        BD=triton.next_power_of_2(D),
    )
    final_state = (ht, ct, nt, mt) if output_final_state else None
    return y, gs, cs, ns, ms, final_state


def fused_recurrent_slstm_bwd(
    recurrent_weight: torch.Tensor,
    initial_state: StateTuple | None,
    gs: torch.Tensor,
    cs: torch.Tensor,
    ns: torch.Tensor,
    ms: torch.Tensor,
    dy: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    B, T, _, H, D = gs.shape
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    dgs = torch.empty_like(gs)
    if initial_state is not None:
        h0, c0, n0, m0 = initial_state
        dh0, dc0, dn0 = (gs.new_empty(N, H, D) for _ in range(3))
    else:
        h0 = c0 = n0 = m0 = None
        dh0 = dc0 = dn0 = None
    fused_recurrent_slstm_bwd_kernel[(N * H,)](
        recurrent_weight=recurrent_weight,
        c0=c0,
        n0=n0,
        m0=m0,
        gs=gs,
        cs=cs,
        ns=ns,
        ms=ms,
        dy=dy,
        dgs=dgs,
        dh0=dh0,
        dc0=dc0,
        dn0=dn0,
        cu_seqlens=cu_seqlens,
        T=T,
        H=H,
        D=D,
        BD=triton.next_power_of_2(D),
    )
    # Reconstruct fp32 hidden states from saved fp32 gates/states to avoid bf16/fp16 rounding in d_recurrent_weight
    h_fp32 = torch.sigmoid(gs[:, :, 3]) * cs / ns
    if cu_seqlens is None:
        h_first = h0[:, None].float() if h0 is not None else h_fp32.new_zeros(B, 1, H, D)
        h_prev = torch.cat([h_first, h_fp32[:, :-1]], dim=1)
    else:
        h_prev = torch.cat([h_fp32.new_zeros(1, 1, H, D), h_fp32[:, :-1]], dim=1)
        bos = cu_seqlens[:-1]
        valid = bos < T
        if h0 is not None:
            h_prev[0, bos[valid]] = h0[valid].float()
        else:
            h_prev[0, bos[valid]] = 0.0
    d_recurrent_weight = torch.einsum('bthk,btghj->hgkj', h_prev, dgs)
    return dgs, d_recurrent_weight, dh0, dc0, dn0


class FusedRecurrentSLSTMFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        recurrent_weight: torch.Tensor,
        h0: torch.Tensor | None,
        c0: torch.Tensor | None,
        n0: torch.Tensor | None,
        m0: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None,
    ):
        initial_state = (h0, c0, n0, m0) if h0 is not None else None
        y, gs, cs, ns, ms, final_state = fused_recurrent_slstm_fwd(
            x=x,
            recurrent_weight=recurrent_weight,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(recurrent_weight, h0, c0, n0, m0, gs, cs, ns, ms)
        ctx.cu_seqlens = cu_seqlens
        if final_state is not None:
            ctx.mark_non_differentiable(*final_state)
        return y, final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dy: torch.Tensor, *_):
        recurrent_weight, h0, c0, n0, m0, gs, cs, ns, ms = ctx.saved_tensors
        initial_state = (h0, c0, n0, m0) if h0 is not None else None
        dgs, d_recurrent_weight, dh0, dc0, dn0 = fused_recurrent_slstm_bwd(
            recurrent_weight=recurrent_weight,
            initial_state=initial_state,
            gs=gs,
            cs=cs,
            ns=ns,
            ms=ms,
            dy=dy,
            cu_seqlens=ctx.cu_seqlens,
        )
        if initial_state is not None:
            dh0, dc0, dn0 = dh0.to(h0), dc0.to(c0), dn0.to(n0)
        return dgs.to(dy.dtype), d_recurrent_weight.to(recurrent_weight), dh0, dc0, dn0, None, None, None


@torch.compiler.disable
def fused_recurrent_slstm(
    x: torch.Tensor,
    recurrent_weight: torch.Tensor,
    initial_state: StateTuple | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, StateTuple | None]:
    r"""
    The sLSTM of xLSTM (https://arxiv.org/abs/2405.04517) with a head-wise recurrent matrix:

    .. math::
        [\tilde{i}, \tilde{f}, \tilde{z}, \tilde{o}]_t = x_t + h_{t-1} R, \quad
        h_t = \sigma(\tilde{o}_t) \frac{c_t}{n_t},

    where `c_t` and `n_t` accumulate `tanh(z~_t)` and 1 with exponential input gates `exp(i~_t)` and sigmoid forget
    gates `sigmoid(f~_t)`, stabilized by `m_t`. Because `m_t` rescales `c_t` and `n_t` alike and cancels in `h_t`,
    treating `m_t` as constant during backward is exact for `h_t`.

    Supported dtypes for `x` and `recurrent_weight`: `torch.float32`, `torch.float16`, `torch.bfloat16`.
    State gradients flow through `h0`, `c0`, and `n0`; `m0` and `final_state` are non-differentiable.

    Args:
        x (torch.Tensor):
            Gate pre-activations of shape `[B, T, 4, H, D]`, ordered as input, forget, cell, and output gates,
            usually a projection of the input plus the gate biases.
        recurrent_weight (torch.Tensor):
            Recurrent weights of shape `[H, 4, D, D]`, where `recurrent_weight[h, g]` maps the hidden state of
            head `h` to gate `g`.
        initial_state (tuple[torch.Tensor, ...], Optional):
            Initial states `(h0, c0, n0, m0)`, each of shape `[N, H, D]` for `N` input sequences (`N = B` for
            dense inputs, `N = len(cu_seqlens) - 1` for packed varlen inputs). Gradients flow to `h0`, `c0`, and `n0`.
            Default: `None`.
        output_final_state (bool, Optional):
            Whether to output the final states `(ht, ct, nt, mt)` of shape `[N, H, D]` in fp32 (non-differentiable).
            Default: `False`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N + 1]` for packed variable-length inputs (requires `B == 1`),
            consistent with the FlashAttention API. Default: `None`.

    Returns:
        y (torch.Tensor):
            Hidden states of shape `[B, T, H, D]`.
        final_state (tuple[torch.Tensor, ...], Optional):
            Final states `(ht, ct, nt, mt)` if `output_final_state=True` else `None`.
    """
    if x.dim() != 5 or x.shape[2] != 4:
        raise ValueError(f"`x` is expected to have shape [B, T, 4, H, D], got {tuple(x.shape)}.")
    B, _, _, H, D = x.shape
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    if x.dtype not in supported_dtypes:
        raise ValueError(f"`x` dtype must be one of {supported_dtypes}, got {x.dtype}.")
    if recurrent_weight.shape != (H, 4, D, D):
        raise ValueError(
            f"`recurrent_weight` is expected to have shape {(H, 4, D, D)}, got {tuple(recurrent_weight.shape)}.",
        )
    if recurrent_weight.dtype not in supported_dtypes:
        raise ValueError(
            f"`recurrent_weight` dtype must be one of {supported_dtypes}, got {recurrent_weight.dtype}.",
        )
    if recurrent_weight.device != x.device:
        raise ValueError(
            f"`recurrent_weight` and `x` must be on the same device, got {recurrent_weight.device} and {x.device}.",
        )
    if cu_seqlens is not None:
        if x.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {x.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if cu_seqlens.dim() != 1 or len(cu_seqlens) < 2:
            raise ValueError(
                f"`cu_seqlens` is expected to be a 1D tensor with at least 2 elements, got shape {tuple(cu_seqlens.shape)}.",
            )
        if cu_seqlens.device != x.device:
            raise ValueError(
                f"`cu_seqlens` and `x` must be on the same device, got {cu_seqlens.device} and {x.device}.",
            )
        N = len(cu_seqlens) - 1
    else:
        N = B

    if initial_state is not None:
        if not isinstance(initial_state, (tuple, list)) or len(initial_state) != 4:
            raise ValueError("`initial_state` must be a 4-tuple of tensors `(h0, c0, n0, m0)`.")
        for name, state in zip(('h0', 'c0', 'n0', 'm0'), initial_state):
            if not isinstance(state, torch.Tensor):
                raise ValueError(f"`initial_state` element `{name}` must be a torch.Tensor, got {type(state).__name__}.")
            if state.shape != (N, H, D):
                raise ValueError(
                    f"`initial_state` element `{name}` is expected to have shape {(N, H, D)}, got {tuple(state.shape)}.",
                )
            if state.device != x.device:
                raise ValueError(
                    f"`initial_state` element `{name}` and `x` must be on the same device, got {state.device} and {x.device}.",
                )
        h0, c0, n0, m0 = initial_state
    else:
        h0 = c0 = n0 = m0 = None

    return FusedRecurrentSLSTMFunction.apply(
        x,
        recurrent_weight,
        h0,
        c0,
        n0,
        m0,
        output_final_state,
        cu_seqlens,
    )
