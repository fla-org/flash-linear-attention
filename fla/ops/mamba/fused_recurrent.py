# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# The selective scan of Mamba (https://arxiv.org/abs/2312.00752) with a diagonal state matrix A of shape [D, N]:
#     dt_t = softplus(dt_t + dt_bias),    h_t = exp(dt_t A) * h_{t-1} + (dt_t x_t) B_t^T,    y_t = h_t C_t + D * x_t.
# The decay differs for every (channel, state) pair, so there is no chunked form; every program scans its channels
# token by token. The forward pass checkpoints the state every BT tokens, and the backward pass rebuilds the states of
# one chunk at a time from its checkpoint into a scratch buffer before scanning it in reverse.

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_offsets
from fla.ops.utils.op import exp, log
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard

BD = 32
BT = 64


@triton.jit
def softplus(x):
    # the threshold of torch.nn.functional.softplus
    return tl.where(x <= 20., log(1. + exp(x)), x)


@triton.heuristics({
    'USE_DT_BIAS': lambda args: args['dt_bias'] is not None,
    'USE_D': lambda args: args['D_skip'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=['D', 'N'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_mamba_fwd_kernel(
    x,
    dt,
    A,
    B,
    C,
    D_skip,
    dt_bias,
    y,
    h0,
    ht,
    hc,
    cu_seqlens,
    chunk_offsets,
    T,
    D: tl.constexpr,
    N: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
    BT: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    USE_DT_BIAS: tl.constexpr,
    USE_D: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        boc = i_n * tl.cdiv(T, BT)

    o_d = i_d * BD + tl.arange(0, BD)
    o_n = tl.arange(0, BN)
    m_d, m_n = o_d < D, o_n < N
    m_h = m_d[:, None] & m_n[None, :]
    o_h = o_d[:, None] * N + o_n[None, :]

    b_A = tl.load(A + o_h, mask=m_h, other=0.).to(tl.float32)
    if USE_DT_BIAS:
        b_dt_bias = tl.load(dt_bias + o_d, mask=m_d, other=0.).to(tl.float32)
    if USE_D:
        b_D = tl.load(D_skip + o_d, mask=m_d, other=0.).to(tl.float32)
    b_h = tl.zeros([BD, BN], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h += tl.load(h0 + i_n * D*N + o_h, mask=m_h, other=0.).to(tl.float32)

    for i_t in range(0, T):
        if i_t % BT == 0:
            tl.store(hc + (boc + i_t // BT) * D*N + o_h, b_h, mask=m_h)
        b_x = tl.load(x + (bos + i_t) * D + o_d, mask=m_d, other=0.).to(tl.float32)
        b_dt = tl.load(dt + (bos + i_t) * D + o_d, mask=m_d, other=0.).to(tl.float32)
        b_B = tl.load(B + (bos + i_t) * N + o_n, mask=m_n, other=0.).to(tl.float32)
        b_C = tl.load(C + (bos + i_t) * N + o_n, mask=m_n, other=0.).to(tl.float32)
        if USE_DT_BIAS:
            b_dt += b_dt_bias
        if DT_SOFTPLUS:
            b_dt = softplus(b_dt)

        b_h = exp(b_dt[:, None] * b_A) * b_h + (b_dt * b_x)[:, None] * b_B[None, :]
        b_y = tl.sum(b_h * b_C[None, :], 1)
        if USE_D:
            b_y += b_D * b_x
        tl.store(y + (bos + i_t) * D + o_d, b_y.to(y.dtype.element_ty), mask=m_d)

    if STORE_FINAL_STATE:
        tl.store(ht + i_n * D*N + o_h, b_h, mask=m_h)


@triton.heuristics({
    'USE_DT_BIAS': lambda args: args['dt_bias'] is not None,
    'USE_D': lambda args: args['D_skip'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=['D', 'N'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_mamba_bwd_kernel(
    x,
    dt,
    A,
    B,
    C,
    D_skip,
    dt_bias,
    hc,
    hs,
    dy,
    dht,
    dx,
    ddt,
    dA,
    dB,
    dC,
    dD,
    dh0,
    cu_seqlens,
    chunk_offsets,
    T,
    D: tl.constexpr,
    N: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
    BT: tl.constexpr,
    NDB: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    USE_DT_BIAS: tl.constexpr,
    USE_D: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        boc = i_n * tl.cdiv(T, BT)

    o_d = i_d * BD + tl.arange(0, BD)
    o_n = tl.arange(0, BN)
    m_d, m_n = o_d < D, o_n < N
    m_h = m_d[:, None] & m_n[None, :]
    o_h = o_d[:, None] * N + o_n[None, :]
    # this program's scratch buffer of BT states
    p_hs = hs + (i_n * NDB + i_d).to(tl.int64) * BT * BD*BN + tl.arange(0, BD)[:, None] * BN + o_n[None, :]

    b_A = tl.load(A + o_h, mask=m_h, other=0.).to(tl.float32)
    if USE_DT_BIAS:
        b_dt_bias = tl.load(dt_bias + o_d, mask=m_d, other=0.).to(tl.float32)
    if USE_D:
        b_D = tl.load(D_skip + o_d, mask=m_d, other=0.).to(tl.float32)
    b_dA = tl.zeros([BD, BN], dtype=tl.float32)
    b_dD = tl.zeros([BD], dtype=tl.float32)
    # gradient of the state after the current token
    b_dh = tl.zeros([BD, BN], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        b_dh += tl.load(dht + i_n * D*N + o_h, mask=m_h, other=0.).to(tl.float32)

    for i_c in range(tl.cdiv(T, BT) - 1, -1, -1):
        # rebuild the states before every token of the chunk
        b_h = tl.load(hc + (boc + i_c) * D*N + o_h, mask=m_h, other=0.)
        for i_t in range(i_c * BT, min(i_c * BT + BT, T)):
            tl.store(p_hs + (i_t - i_c * BT) * BD*BN, b_h)
            b_x = tl.load(x + (bos + i_t) * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_dt = tl.load(dt + (bos + i_t) * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_B = tl.load(B + (bos + i_t) * N + o_n, mask=m_n, other=0.).to(tl.float32)
            if USE_DT_BIAS:
                b_dt += b_dt_bias
            if DT_SOFTPLUS:
                b_dt = softplus(b_dt)
            b_h = exp(b_dt[:, None] * b_A) * b_h + (b_dt * b_x)[:, None] * b_B[None, :]
        tl.debug_barrier()

        for i_t in range(min(i_c * BT + BT, T) - 1, i_c * BT - 1, -1):
            b_h_prev = tl.load(p_hs + (i_t - i_c * BT) * BD*BN)
            b_x = tl.load(x + (bos + i_t) * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_dt_raw = tl.load(dt + (bos + i_t) * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_B = tl.load(B + (bos + i_t) * N + o_n, mask=m_n, other=0.).to(tl.float32)
            b_C = tl.load(C + (bos + i_t) * N + o_n, mask=m_n, other=0.).to(tl.float32)
            b_dy = tl.load(dy + (bos + i_t) * D + o_d, mask=m_d, other=0.).to(tl.float32)
            if USE_DT_BIAS:
                b_dt_raw += b_dt_bias
            b_dt = softplus(b_dt_raw) if DT_SOFTPLUS else b_dt_raw
            b_decay = exp(b_dt[:, None] * b_A)
            b_dBx = b_x[:, None] * b_B[None, :]
            b_h = b_decay * b_h_prev + b_dt[:, None] * b_dBx

            # y_t = h_t C_t + D * x_t
            b_dx = tl.zeros([BD], dtype=tl.float32)
            if USE_D:
                b_dx += b_D * b_dy
                b_dD += b_x * b_dy
            tl.store(dC + ((bos + i_t) * NDB + i_d) * N + o_n, tl.sum(b_h * b_dy[:, None], 0), mask=m_n)
            b_dh += b_dy[:, None] * b_C[None, :]

            # h_t = exp(dt_t A) * h_{t-1} + dt_t x_t B_t
            b_dh_decay = b_dh * b_decay * b_h_prev
            b_ddt = tl.sum(b_dh_decay * b_A + b_dh * b_dBx, 1)
            b_dA += b_dh_decay * b_dt[:, None]
            b_dx += b_dt * tl.sum(b_dh * b_B[None, :], 1)
            tl.store(dB + ((bos + i_t) * NDB + i_d) * N + o_n, tl.sum(b_dh * (b_dt * b_x)[:, None], 0), mask=m_n)
            if DT_SOFTPLUS:
                b_ddt = b_ddt * tl.where(b_dt_raw <= 20., tl.sigmoid(b_dt_raw), 1.)
            tl.store(dx + (bos + i_t) * D + o_d, b_dx.to(dx.dtype.element_ty), mask=m_d)
            tl.store(ddt + (bos + i_t) * D + o_d, b_ddt, mask=m_d)
            b_dh = b_dh * b_decay
        tl.debug_barrier()

    tl.store(dA + i_n * D*N + o_h, b_dA, mask=m_h)
    tl.store(dD + i_n * D + o_d, b_dD, mask=m_d)
    if USE_INITIAL_STATE:
        tl.store(dh0 + i_n * D*N + o_h, b_dh, mask=m_h)


def prepare_scan(x, cu_seqlens):
    B, T, D = x.shape
    if cu_seqlens is None:
        return B, triton.cdiv(T, BT) * B, None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    return len(cu_seqlens) - 1, int(chunk_offsets[-1]), chunk_offsets


def fused_recurrent_mamba_fwd(x, dt, A, B, C, D_skip, dt_bias, dt_softplus, initial_state, cu_seqlens):
    T, D, N = x.shape[1], x.shape[-1], A.shape[-1]
    NS, NC, chunk_offsets = prepare_scan(x, cu_seqlens)
    y = torch.empty_like(x)
    ht = x.new_empty(NS, D, N, dtype=torch.float)
    hc = x.new_empty(NC, D, N, dtype=torch.float)
    fused_recurrent_mamba_fwd_kernel[(triton.cdiv(D, BD), NS)](
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D_skip=D_skip,
        dt_bias=dt_bias,
        y=y,
        h0=initial_state,
        ht=ht,
        hc=hc,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        D=D,
        N=N,
        BD=BD,
        BN=triton.next_power_of_2(N),
        BT=BT,
        DT_SOFTPLUS=dt_softplus,
    )
    return y, ht, hc


def fused_recurrent_mamba_bwd(x, dt, A, B, C, D_skip, dt_bias, dt_softplus, initial_state, hc, dy, dht, cu_seqlens):
    T, D, N = x.shape[1], x.shape[-1], A.shape[-1]
    NS, _, chunk_offsets = prepare_scan(x, cu_seqlens)
    NDB, BN = triton.cdiv(D, BD), triton.next_power_of_2(N)
    hs = x.new_empty(NS, NDB, BT, BD, BN, dtype=torch.float)
    dx = torch.empty_like(x)
    ddt = torch.empty_like(dt, dtype=torch.float)
    dA = x.new_empty(NS, D, N, dtype=torch.float)
    # partial gradients of B and C from every block of channels
    dB = x.new_empty(*x.shape[:2], NDB, N, dtype=torch.float)
    dC = x.new_empty(*x.shape[:2], NDB, N, dtype=torch.float)
    dD = x.new_empty(NS, D, dtype=torch.float)
    dh0 = torch.empty_like(initial_state, dtype=torch.float) if initial_state is not None else None
    fused_recurrent_mamba_bwd_kernel[(NDB, NS)](
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D_skip=D_skip,
        dt_bias=dt_bias,
        hc=hc,
        hs=hs,
        dy=dy,
        dht=dht,
        dx=dx,
        ddt=ddt,
        dA=dA,
        dB=dB,
        dC=dC,
        dD=dD,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        D=D,
        N=N,
        BD=BD,
        BN=BN,
        BT=BT,
        NDB=NDB,
        DT_SOFTPLUS=dt_softplus,
    )
    ddt_bias = ddt.sum((0, 1)) if dt_bias is not None else None
    dD = dD.sum(0) if D_skip is not None else None
    return dx, ddt, dA.sum(0), dB.sum(2), dC.sum(2), dD, ddt_bias, dh0


class FusedRecurrentMambaFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, dt, A, B, C, D_skip, dt_bias, dt_softplus, initial_state, output_final_state, cu_seqlens):
        y, ht, hc = fused_recurrent_mamba_fwd(x, dt, A, B, C, D_skip, dt_bias, dt_softplus, initial_state, cu_seqlens)
        ctx.save_for_backward(x, dt, A, B, C, D_skip, dt_bias, initial_state, hc)
        ctx.dt_softplus, ctx.cu_seqlens = dt_softplus, cu_seqlens
        return y, (ht if output_final_state else None)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dy, dht):
        x, dt, A, B, C, D_skip, dt_bias, initial_state, hc = ctx.saved_tensors
        dx, ddt, dA, dB, dC, dD, ddt_bias, dh0 = fused_recurrent_mamba_bwd(
            x, dt, A, B, C, D_skip, dt_bias, ctx.dt_softplus, initial_state, hc, dy, dht, ctx.cu_seqlens,
        )
        return (
            dx, ddt.to(dt), dA.to(A), dB.to(B), dC.to(C),
            dD.to(D_skip) if dD is not None else None,
            ddt_bias.to(dt_bias) if ddt_bias is not None else None,
            None, dh0, None, None,
        )


@torch.compiler.disable
def fused_recurrent_mamba(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = True,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    The selective scan of Mamba (https://arxiv.org/abs/2312.00752):

    .. math::
        \Delta_t = \text{softplus}(dt_t + dt_{bias}), \quad
        h_t = \exp(\Delta_t A) \odot h_{t-1} + (\Delta_t x_t) B_t^\top, \quad
        y_t = h_t C_t + D \odot x_t.

    Args:
        x (torch.Tensor):
            inputs of shape `[B, T, D]`.
        dt (torch.Tensor):
            time steps of shape `[B, T, D]`, before `dt_bias` and softplus.
        A (torch.Tensor):
            diagonal state matrix of shape `[D, N]`, usually `-exp(A_log)`.
        B (torch.Tensor):
            input projections of shape `[B, T, N]`.
        C (torch.Tensor):
            output projections of shape `[B, T, N]`.
        D (Optional[torch.Tensor]):
            skip connection of shape `[D]`. Default: `None`.
        dt_bias (Optional[torch.Tensor]):
            bias of shape `[D]` added to `dt`. Default: `None`.
        dt_softplus (bool):
            Whether to apply softplus to `dt`. Default: `True`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N_seq, D, N]` for `N_seq` input sequences. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N_seq, D, N]` in fp32. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N_seq+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        y (torch.Tensor):
            Outputs of shape `[B, T, D]`.
        final_state (torch.Tensor):
            Final state of shape `[N_seq, D, N]` if `output_final_state=True` else `None`.
    """
    if cu_seqlens is not None:
        if x.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {x.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    return FusedRecurrentMambaFunction.apply(
        x, dt, A, B, C, D, dt_bias, dt_softplus, initial_state, output_final_state, cu_seqlens,
    )
