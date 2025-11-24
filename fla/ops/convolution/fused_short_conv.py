# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.utils import prepare_chunk_indices
from fla.utils import get_multiprocessor_count, input_guard

@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def fused_short_conv_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    EPS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_NORM: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    if HAS_WEIGHT:
        # [BD, BW]
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    if not USE_INITIAL_STATE:
        for i_w in tl.static_range(-W + 1, 1):
            p_yi = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            # [BT, BD]
            b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    elif i_t * BT >= W:
        # to make Triton compiler happy, we need to copy codes
        for i_w in tl.static_range(-W + 1, 1):
            p_yi = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            # [BT, BD]
            b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    else:
        o_t = i_t * BT + tl.arange(0, BT)
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d
            m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d

            b_yi = tl.load(x + bos * D + o_x[:, None] * D + o_d, mask=m_x, other=0).to(tl.float32)

            b_yi += tl.load(initial_state + i_n * D*W + o_d * W + (o_x + W)[:, None], mask=m_c, other=0).to(tl.float32)

            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * tl.sigmoid(b_y)

    if USE_NORM:
        # L2 norm over the head dimension (BD)
        # b_y is [BT, BD]
        b_var = tl.sum(b_y * b_y, axis=1)
        b_std = tl.sqrt(b_var + EPS)
        b_y = b_y / b_std[:, None]

    if HAS_RESIDUAL:
        p_residual = tl.make_block_ptr(residual + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_residual = tl.load(p_residual, boundary_check=(0, 1))
        b_y += b_residual

    p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['dw'] is not None,
    'HAS_BIAS': lambda args: args['db'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def fused_short_conv_bwd_kernel(
    x,
    y,
    weight,
    bias,
    initial_state,
    dh0,
    dht,
    dy,
    dx,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    EPS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_NORM: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_tg = i_b * tl.num_programs(1) + i_t
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    if HAS_WEIGHT:
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    if HAS_WEIGHT:
        p_x = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)

    for i_w in range(0, W):
        p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
        b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
        
        if USE_NORM:
            # Recompute y_conv at T_global = i_t*BT + i_w + t_local
            # We need to loop over k (kernel support) to compute convolution
            b_y_conv = tl.zeros((BT, BD), dtype=tl.float32)
            t_local = tl.arange(0, BT)
            
            for k in range(0, W):
                w_k = tl.sum(b_w * (o_w[None, :] == k), 1)
                # Forward: y[t] = sum_{j=0}^{W-1} x[t - W + 1 + j] * w[j]
                # Here t = i_t * BT + i_w + t_local, j = k
                # So x index = t - W + 1 + k = (i_t * BT + i_w + t_local) - W + 1 + k
                x_offset = i_t * BT + i_w - W + 1 + k
                m_x_valid = (x_offset + t_local >= 0) & (x_offset + t_local < T)
                
                # We need to reload x from memory as it's not in registers.
                # Constructing pointers manually to allow random access in loop
                # This is efficient enough for small W.
                val_x = tl.load(x + bos * D + (x_offset + t_local)[:, None] * D + o_d[None, :],
                               mask=m_x_valid[:, None] & m_d[None, :], other=0.0).to(tl.float32)
                b_y_conv += val_x * w_k[None, :]

            if HAS_BIAS:
                b_y_conv += tl.load(bias + o_d, mask=m_d).to(tl.float32)
            
            b_y_act = b_y_conv
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                 b_y_act = b_y_conv * tl.sigmoid(b_y_conv)

            b_var = tl.sum(b_y_act * b_y_act, 1)
            b_std = tl.sqrt(b_var + EPS)
            b_inv_std = 1.0 / b_std
            b_y_out = b_y_act * b_inv_std[:, None]
            b_dot = tl.sum(b_dy * b_y_out, 1)
            b_dy = (b_dy - b_y_out * b_dot[:, None]) * b_inv_std[:, None]
            
            # For activation backward
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                 b_sig = tl.sigmoid(b_y_conv)
                 b_dy = b_dy * b_sig * (1 + b_y_conv * (1 - b_sig))

        b_wdy = b_dy
        if HAS_WEIGHT:
            b_wdy = b_wdy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)
            b_dw = tl.sum(b_dy * b_x, 0)
            tl.store(dw + i_tg * D*W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)

        if HAS_BIAS and i_w == 0:
            b_db += tl.sum(b_dy, 0)
        
        b_dx += b_wdy

    p_dx = tl.make_block_ptr(dx + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    tl.store(p_dx, tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
    
    if HAS_BIAS:
         tl.store(db + i_tg * D + o_d, b_db.to(db.dtype.element_ty), mask=m_d)


class FusedShortConvFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        activation: str | None = None,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        use_norm: bool = False,
        norm_eps: float = 1e-5,
        head_dim: int | None = None,
    ):
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        ctx.use_norm = use_norm
        ctx.norm_eps = norm_eps
        ctx.head_dim = head_dim
        
        # Save tensors for backward
        # We use recomputation strategy: don't save y_act, recompute in backward
        ctx.save_for_backward(x, weight, bias, residual, initial_state)

        shape = x.shape
        if x.shape[-1] != weight.shape[0]:
            x = rearrange(x, 'b t ... -> b t (...)')
        B, T, D, W = *x.shape, weight.shape[1]
        BT = min(64, triton.next_power_of_2(triton.cdiv(max(16, B*T), get_multiprocessor_count(x.device.index))))
        BW = triton.next_power_of_2(W)
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
        
        # Determine BD
        if use_norm:
            assert head_dim is not None, "head_dim must be provided when use_norm is True"
            BD = head_dim
            # Check BD is power of 2? Triton prefers it.
            # If not, next power of 2 and mask handles it.
            BD = triton.next_power_of_2(head_dim)
        else:
            BD = 32 # Default fallback or simple value since we don't autotune
        
        y = torch.empty_like(x)
        def grid(meta): return (triton.cdiv(D, meta['BD']), NT, B)
        
        fused_short_conv_fwd_kernel[grid](
            x=x,
            y=y,
            weight=weight,
            bias=bias,
            residual=residual,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            chunk_indices=chunk_indices,
            B=B,
            T=T,
            D=D,
            W=W,
            BT=BT,
            BW=BW,
            BD=BD,
            EPS=norm_eps,
            ACTIVATION=activation,
            USE_NORM=use_norm,
        )
        return y, None # final_state not implemented for now

    @staticmethod
    @input_guard
    def backward(ctx, dy: torch.Tensor, dht: torch.Tensor | None = None):
        x, weight, bias, residual, initial_state = ctx.saved_tensors
        use_norm = ctx.use_norm
        norm_eps = ctx.norm_eps
        head_dim = ctx.head_dim
        activation = ctx.activation
        
        # Similar setup
        shape = x.shape
        if x.shape[-1] != weight.shape[0]:
            x = rearrange(x, 'b t ... -> b t (...)')
        B, T, D = x.shape
        W = weight.shape[1]
        BT = min(64, triton.next_power_of_2(triton.cdiv(max(16, B*T), get_multiprocessor_count(x.device.index))))
        BW = triton.next_power_of_2(W)
        if ctx.chunk_indices is None and ctx.cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(ctx.cu_seqlens, BT)
        else:
            chunk_indices = ctx.chunk_indices
        NT = len(chunk_indices) if ctx.cu_seqlens is not None else triton.cdiv(T, BT)
        if use_norm:
            BD = triton.next_power_of_2(head_dim)
        else:
            BD = 32

        dx = torch.empty_like(x)
        dh0 = None # Not implemented
        dr = dy if residual is not None else None
        
        # Always use recomputation strategy (best performance + memory efficiency)
        y = None
        
        # Standard backward kernel
        dw = weight.new_empty(B*NT, *weight.shape, dtype=torch.float) if weight is not None else None
        db = bias.new_empty(B*NT, *bias.shape, dtype=torch.float) if bias is not None else None
        
        def grid(meta): return (triton.cdiv(D, meta['BD']), NT, B)
        fused_short_conv_bwd_kernel[grid](
            x=x,
            y=y,
            weight=weight,
            bias=bias,
            initial_state=initial_state,
            dh0=dh0,
            dht=dht,
            dy=dy,
            dx=dx,
            dw=dw,
            db=db,
            cu_seqlens=ctx.cu_seqlens,
            chunk_indices=chunk_indices,
            B=B,
            T=T,
            D=D,
            W=W,
            BT=BT,
            BW=BW,
            BD=BD,
            EPS=norm_eps,
            ACTIVATION=activation,
            USE_NORM=use_norm,
        )
        
        if weight is not None:
            dw = dw.sum(0).to(weight)
        if bias is not None:
            db = db.sum(0).to(bias)
            
        return dx, dw, db, dr, dh0, None, None, None, None, None, None, None

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
    norm_eps: float = 1e-5,
    head_dim: int | None = None,
):
    """
    Fused short convolution with optional L2 normalization.
    
    Uses recomputation strategy in backward: activations are recomputed on-the-fly
    instead of being saved, providing both speed and memory benefits.
    """
    return FusedShortConvFunction.apply(
        x, weight, bias, residual, initial_state, output_final_state, activation, cu_seqlens, chunk_indices, use_norm, norm_eps, head_dim
    )
