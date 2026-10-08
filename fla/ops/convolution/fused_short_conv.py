# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.modules.conv.causal_conv1d import causal_conv1d
from fla.modules.conv.triton.kernels import causal_conv1d_fwd_tile
from fla.modules.conv.triton.ops import causal_conv1d_bwd, causal_conv1d_update_states
from fla.modules.l2norm import l2norm, l2norm_bwd
from fla.ops.utils import prepare_chunk_indices
from fla.utils import IS_NVIDIA, input_guard


@triton.heuristics({
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_short_conv_fwd_kernel(
    x,
    y,
    rstd,
    weight,
    bias,
    residual,
    initial_state,
    cu_seqlens,
    chunk_indices,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    D: tl.constexpr,
    W: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    BW: tl.constexpr,
    EPS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_h, i_t, i_b = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int64)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        T = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
        p_x = x + bos * stride_x_t
    else:
        i_n = i_b
        bos = i_b * T
        p_x = x + i_b * stride_x_n
    o_t = i_t * BT + tl.arange(0, BT).to(tl.int64)
    o_d = i_h * HEAD_DIM + tl.arange(0, BD).to(tl.int64)
    m_d = tl.arange(0, BD) < HEAD_DIM
    b_y = causal_conv1d_fwd_tile(
        p_x=p_x,
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=initial_state,
        bos=bos,
        i_n=i_n,
        i_t=i_t,
        o_t=o_t,
        o_d=o_d,
        m_d=m_d,
        T=T,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        BD=BD,
        ACTIVATION=ACTIVATION,
        HAS_WEIGHT=True,
        HAS_BIAS=HAS_BIAS,
        HAS_RESIDUAL=HAS_RESIDUAL,
        USE_INITIAL_STATE=USE_INITIAL_STATE,
    )
    # preserve the rounding between the unfused convolution and L2 normalization
    b_y = b_y.to(y.dtype.element_ty).to(tl.float32)
    b_y = tl.where(m_d[None, :], b_y, 0.0)
    b_rstd = 1 / tl.sqrt(tl.sum(b_y * b_y, 1) + EPS)
    b_y *= b_rstd[:, None]
    tl.store(y + (bos + o_t[:, None]) * D + o_d[None, :], b_y, mask=(o_t < T)[:, None] & m_d[None, :])
    tl.store(rstd + (bos + o_t) * (D // HEAD_DIM) + i_h, b_rstd, mask=o_t < T)


class FusedShortConvFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        residual,
        initial_state,
        output_final_state,
        activation,
        cu_seqlens,
        chunk_indices,
        norm_eps,
        head_dim,
        cu_seqlens_cpu,
    ):
        B, T, D = x.shape
        W = weight.shape[1]
        BT = 64
        if cu_seqlens is not None and chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
        NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
        y = torch.empty_like(x, memory_format=torch.contiguous_format)
        rstd = torch.empty((B, T, D // head_dim), device=x.device, dtype=torch.float32)
        if B * NT:
            fused_short_conv_fwd_kernel[(D // head_dim, NT, B)](
                x=x,
                y=y,
                rstd=rstd,
                weight=weight,
                bias=bias,
                residual=residual,
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                T=T,
                stride_x_n=x.stride(0),
                stride_x_t=x.stride(1),
                stride_x_d=x.stride(2),
                D=D,
                W=W,
                HEAD_DIM=head_dim,
                BT=BT,
                BD=triton.next_power_of_2(head_dim),
                BW=triton.next_power_of_2(W),
                EPS=norm_eps,
                ACTIVATION=activation,
                num_warps=4,
            )
        final_state = None
        if output_final_state:
            final_state = causal_conv1d_update_states(
                x=x,
                state_len=W,
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
            )
        ctx.save_for_backward(x, weight, bias, residual, initial_state, y, rstd, cu_seqlens, chunk_indices)
        ctx.activation = activation
        ctx.norm_eps = norm_eps
        ctx.head_dim = head_dim
        ctx.cu_seqlens_cpu = cu_seqlens_cpu
        return y, final_state

    @staticmethod
    @input_guard
    def backward(ctx, dy, dht):
        x, weight, bias, residual, initial_state, y, rstd, cu_seqlens, chunk_indices = ctx.saved_tensors
        dy = l2norm_bwd(
            y=y.view(*y.shape[:2], y.shape[-1] // ctx.head_dim, ctx.head_dim),
            rstd=rstd,
            dy=dy.view(*dy.shape[:2], dy.shape[-1] // ctx.head_dim, ctx.head_dim),
            eps=ctx.norm_eps,
        ).view_as(y)
        dx, dw, db, dr, dh0 = causal_conv1d_bwd(
            x=x,
            dy=dy,
            dht=dht,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=ctx.activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=ctx.cu_seqlens_cpu,
            chunk_indices=chunk_indices,
        )
        return dx, dw, db, dr, dh0, None, None, None, None, None, None, None


@input_guard(no_guard_contiguous=['x'])
def fused_short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_norm: bool = False,
    norm_eps: float = 1e-6,
    head_dim: int | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
):
    """Apply causal convolution, optionally followed by head-wise L2 normalization.

    Normalization includes the residual and preserves the intermediate rounding of
    separate convolution and L2 operators. Final states contain convolution input history.
    NVIDIA uses a fused forward kernel for head dimensions up to 256; other cases use the existing operators.

    Args:
        x (torch.Tensor):
            Input of shape `[B, T, D]`; packed variable-length inputs require `B=1`.
        weight (torch.Tensor):
            Depthwise convolution weights of shape `[D, W]`.
        bias (torch.Tensor, Optional):
            Bias of shape `[D]`. Default: `None`.
        residual (torch.Tensor, Optional):
            Residual of shape `[B, T, D]`, added before normalization. Default: `None`.
        initial_state (torch.Tensor, Optional):
            Convolution state of shape `[N, D, W]`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the final convolution state. Default: `False`.
        activation (str, Optional):
            `silu`, `swish`, or no activation. Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Packed sequence boundaries of shape `[N+1]`. Default: `None`.
        chunk_indices (torch.LongTensor, Optional):
            Packed chunk indices for a chunk size of 64. Default: `None`.
        use_norm (bool, Optional):
            Whether to normalize each head. Default: `False`.
        norm_eps (float, Optional):
            Positive epsilon added to the sum of squared features. Default: `1e-6`.
        head_dim (int, Optional):
            Positive divisor of `D`, required when `use_norm=True`. Default: `None`.
        cu_seqlens_cpu (torch.LongTensor, Optional):
            CPU copy of packed sequence boundaries. Default: `None`.
    """
    if x.ndim != 3 or weight.ndim != 2 or x.shape[-1] != weight.shape[0] or weight.shape[1] < 1:
        raise ValueError('Expected x of shape [B, T, D] and weight of shape [D, W] with W > 0')
    if activation not in (None, 'silu', 'swish'):
        raise ValueError('activation must be None, silu, or swish')
    if cu_seqlens is not None and x.shape[0] != 1:
        raise ValueError('Packed variable-length inputs require batch size 1')
    if use_norm:
        if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0 or x.shape[-1] % head_dim:
            raise ValueError('head_dim must be a positive divisor of the channel dimension')
        if not norm_eps > 0:
            raise ValueError('norm_eps must be positive')
    if not use_norm or not IS_NVIDIA or head_dim > 256:
        y, final_state = causal_conv1d(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            output_final_state=output_final_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
        )
        if use_norm:
            y = l2norm(y.view(*y.shape[:2], y.shape[-1] // head_dim, head_dim), eps=norm_eps).view_as(y)
        return y, final_state
    return FusedShortConvFunction.apply(
        x, weight, bias, residual, initial_state, output_final_state, activation,
        cu_seqlens, chunk_indices, norm_eps, head_dim, cu_seqlens_cpu,
    )
