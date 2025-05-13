# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import torch.nn as nn
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.utils import prepare_chunk_indices
from fla.utils import get_multiprocessor_count, input_guard


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps, num_stages=num_stages)
        for BD in [128, 256]
        for num_warps in [4, 8, 16, 32]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['B', 'D', 'W', 'NB'],
)
@triton.jit
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n), tl.load(cu_seqlens + i_n + 1)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_b * T, i_b * T + T

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    for i_w in range(-W + 1, 1):
        p_x = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))

        # [BT, BD]
        b_x = tl.load(p_x, boundary_check=(0, 1))
        if HAS_WEIGHT:
            b_x *= tl.load(weight + o_d * W + i_w + W - 1, mask=m_d, other=0)
        b_y += b_x
    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d)

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        p_residual = tl.make_block_ptr(residual + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_residual = tl.load(p_residual, boundary_check=(0, 1))
        b_y += b_residual

    p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps, num_stages=num_stages)
        for BD in [128, 256]
        for num_warps in [4, 8, 16, 32]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['B', 'D', 'W', 'NB'],
)
@triton.jit
def causal_conv1d_bwd_kernel(
    x,
    weight,
    bias,
    residual,
    dy,
    dx,
    dw,
    db,
    dr,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n), tl.load(cu_seqlens + i_n + 1)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_b * T, i_b * T + T
    i_bt = i_b * tl.num_programs(1) + i_t

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    if HAS_WEIGHT:
        p_x = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_x = tl.load(p_x, boundary_check=(0, 1))

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)
    for i_w in range(0, W):
        p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))

        # [BT, BD]
        b_dy = tl.load(p_dy, boundary_check=(0, 1))

        if HAS_WEIGHT:
            o_w = W - i_w - 1
            # [BT, BD]
            b_wdy = b_dy * tl.load(weight + o_d * W + o_w, mask=m_d, other=0)
            # [BD]
            b_dw = tl.sum(b_dy * b_x, 0)
            b_dw = tl.cast(b_dw, dtype=dw.dtype.element_ty, fp_downcast_rounding='rtne')
            tl.store(dw + i_bt * D*W + o_d * W + o_w, b_dw, mask=m_d)
        if HAS_BIAS and i_w == 0:
            b_db += tl.sum(b_dy, 0)
        b_dx += b_wdy
    if HAS_BIAS:
        b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding='rtne')
        tl.store(db + i_bt * D + o_d, b_db, mask=m_d)

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_dx = b_dx * tl.sigmoid(b_dx)

    if HAS_RESIDUAL:
        p_residual = tl.make_block_ptr(residual + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_residual = tl.load(p_residual, boundary_check=(0, 1))
        b_dx += b_residual

    p_dx = tl.make_block_ptr(dx + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    tl.store(p_dx, tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))


def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
    B, T, D, W = *x.shape, weight.shape[1]
    BT = min(64, triton.next_power_of_2(triton.cdiv(T, get_multiprocessor_count(x.device.index))))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    NB = triton.cdiv(T, 1024)

    y = torch.empty_like(x)
    def grid(meta): return (triton.cdiv(D, meta['BD']), NT, B)
    causal_conv1d_fwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        residual=residual,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        NB=NB,
        ACTIVATION=activation,
    )
    return y.view(shape)


def causal_conv1d_bwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    dy: torch.Tensor,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None
):
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
    B, T, D, W = *x.shape, weight.shape[1]
    BT = min(64, triton.next_power_of_2(triton.cdiv(T, get_multiprocessor_count(x.device.index))))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    NB = triton.cdiv(T, 1024)

    dx = torch.empty_like(x)
    dw = weight.new_empty(B*NT, *weight.shape) if weight is not None else None
    db = bias.new_empty(B*NT, *bias.shape) if bias is not None else None
    dr = torch.empty_like(residual) if residual is not None else None
    def grid(meta): return (triton.cdiv(D, meta['BD']), NT, B)
    causal_conv1d_bwd_kernel[grid](
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        dy=dy,
        dx=dx,
        dw=dw,
        db=db,
        dr=dr,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        NB=NB,
        ACTIVATION=activation,
    )
    if weight is not None:
        dw = dw.sum(0).to(weight)
    if bias is not None:
        db = db.sum(0).to(bias)

    return dx.view(shape), dw, db, dr


class CausalConv1dFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        activation: Optional[str] = None,
        cu_seqlens: Optional[torch.Tensor] = None
    ):
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.save_for_backward(x, weight, bias, residual)
        return causal_conv1d_fwd(x, weight, bias, residual, activation, cu_seqlens)

    @staticmethod
    @input_guard
    def backward(ctx, dy: torch.Tensor):
        x, weight, bias, residual = ctx.saved_tensors
        dx, dw, db, dr = causal_conv1d_bwd(
            x=x,
            weight=weight,
            bias=bias,
            dy=dy,
            residual=residual,
            activation=ctx.activation,
            cu_seqlens=ctx.cu_seqlens
        )
        return dx, dw, db, dr, None, None


def causal_conv1d(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None
):
    """
    Args:
        x:
            Input tensor of shape [B, T, D].
        weight:
            Weight tensor of shape [D, W]. Default: `None`.
        bias:
            Bias tensor of shape [D]. Default: `None`.
        residual:
            Residual tensor of shape [B, T, D]. Default: `None`.
        activation:
            Activations applied to output, only `swish`/`silu` or `None` (i.e., no activation) are supported.
            Default: `None`.
        cu_seqlens:
            Cumulative sequence lengths (optional)

    Returns:
        Tensor of same shape as input with CausalConv1dFunction applied
    """
    return CausalConv1dFunction.apply(x, weight, bias, residual, activation, cu_seqlens)


class CausalConv1d(nn.Conv1d):
    """
    Implementation of CausalConv1d layer as mentioned in https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: Optional[str] = 'silu',
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )

        self.hidden_size = hidden_size
        self.activation = None
        if activation is not None:
            assert activation in ['silu', 'swish'], f"Activation `{activation}` not supported yet."
            self.activation = activation

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.activation is not None:
            s += ', activation={activation}'
        if not self.use_fast_conv1d:
            s += ', use_fast_conv1d={use_fast_conv1d}'
        return s.format(**self.__dict__)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[B, T, D]`.
                If `seq_idx` is provided, `B` must be 1.
            residual (`Optional[torch.Tensor]`):
                Residual tensor of shape `[B, T, D]`. Default: `None`.
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, D, W]`. Default: `False`.
            cu_seqlens (Optional[torch.LongTensor]):
                Cumulative sequence lengths for each batch. Used for varlen. Default: `None`.
                Shape: [B+1]

        Returns:
            Tensor of shape `[B, T, D]`.
        """

        B, _, D, W = *x.shape, self.kernel_size[0]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")
            x = x.mul_(mask.unsqueeze(-1))
        if output_final_state and cache is None:
            cache = x.new_zeros(N, D, W)
        y = causal_conv1d(x, rearrange(self.weight, "d 1 w -> d w"), self.bias, residual, self.activation, cu_seqlens)
        return y, cache

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size
