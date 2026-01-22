
import torch
import triton
from einops import rearrange

from fla.ops.utils import prepare_chunk_indices
from fla.utils import input_guard

from .kernels import (
    STATIC_WARPS,
    causal_conv1d_bwd_kernel,
    causal_conv1d_fwd_kernel,
    causal_conv1d_states_fwd_kernel,
    causal_conv1d_update_kernel,
    compute_dh0_kernel,
)


@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    BT: int = 64,
) -> torch.Tensor:
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
    B, T, D = x.shape[0], x.shape[1], weight.shape[0]
    W = weight.shape[1]
    stride_x_n, stride_x_t, stride_x_d = x.stride()

    BW = triton.next_power_of_2(W)
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    NB = triton.cdiv(B*T, 1024)

    y = torch.empty_like(x, memory_format=torch.contiguous_format)

    def grid(meta): return (triton.cdiv(D, meta['BD']), NT, B)
    causal_conv1d_fwd_kernel[grid](
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
        NB=NB,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        ACTIVATION=activation,
    )
    final_state = None
    if output_final_state:
        final_state = causal_conv1d_update_states(
            x=x,
            state_len=W,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )
    return y.view(shape), final_state


def compute_dh0_triton(
    dy: torch.Tensor,
    y: torch.Tensor | None,
    weight: torch.Tensor,
    initial_state: torch.Tensor,
    activation: str | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """
    Compute dh0 (gradient w.r.t. initial_state) using a separate Triton kernel.
    This is a workaround for Triton compiler bugs on some architectures (e.g., GB200).
    """
    D, W = weight.shape
    N = initial_state.shape[0]
    T = dy.shape[1]

    # Initialize dh0
    dh0 = torch.zeros_like(initial_state)

    BD = 32
    grid = (triton.cdiv(D, BD), N)

    y_to_pass = y if activation in ('swish', 'silu') else None
    # dy is [B, T, D], stride_n = T*D, stride_t = D
    stride_dy_n = dy.stride(0)
    stride_dy_t = dy.stride(1)

    compute_dh0_kernel[grid](
        dy=dy,
        y=y_to_pass,
        weight=weight,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        stride_dy_n=stride_dy_n,
        stride_dy_t=stride_dy_t,
        T=T,
        D=D,
        W=W,
        BD=BD,
    )

    return dh0


def causal_conv1d_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    BT: int = 64,
):
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
    B, T, D = x.shape
    W = weight.shape[1] if weight is not None else None

    stride_x_n, stride_x_t, stride_x_d = x.stride()

    BW = triton.next_power_of_2(W)
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    NB = triton.cdiv(B*T, 1024)

    y = None
    if activation is not None:
        y, _ = causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=initial_state,
            activation=None,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            output_final_state=False,
            chunk_indices=chunk_indices,
        )
    dx = torch.empty_like(x)
    dw = weight.new_empty(B*NT, *weight.shape, dtype=torch.float) if weight is not None else None
    db = bias.new_empty(B*NT, *bias.shape, dtype=torch.float) if bias is not None else None
    dr = dy if residual is not None else None

    stride_dx_n, stride_dx_t, stride_dx_d = dx.stride()

    def grid(meta): return (triton.cdiv(D, meta['BD']), NT, B)
    causal_conv1d_bwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        initial_state=initial_state,
        dht=dht,
        dy=dy,
        dx=dx,
        dw=dw,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        NB=NB,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        stride_dx_n=stride_dx_n,
        stride_dx_t=stride_dx_t,
        stride_dx_d=stride_dx_d,
        ACTIVATION=activation,
    )
    if weight is not None:
        dw = dw.sum(0).to(weight)
    if bias is not None:
        db = db.sum(0).to(bias)

    # Compute dh0 using separate Triton kernel to avoid compiler bugs on some architectures (e.g., GB200)
    dh0 = None
    if initial_state is not None:
        dh0 = compute_dh0_triton(
            dy=dy,
            y=y,
            weight=weight,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
        )

    return dx.view(shape), dw, db, dr, dh0


@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_update_states(
    x: torch.Tensor,
    state_len: int,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        if x.dim() == 2:
            stride_x_n = 0
            stride_x_t, stride_x_d = x.stride()
            T = x.shape[0]
        else:
            stride_x_n = x.stride(0)
            stride_x_t, stride_x_d = x.stride(1), x.stride(2)
            T = x.shape[1]
        D = x.shape[-1]
    else:
        B, T, D = x.shape
        N = B
        stride_x_n, stride_x_t, stride_x_d = x.stride()

    W = state_len
    final_state = torch.empty(N, D, W, dtype=x.dtype, device=x.device)

    BD = min(triton.next_power_of_2(D), 256)
    BW = triton.next_power_of_2(W)

    grid = (triton.cdiv(D, BD), N)

    causal_conv1d_states_fwd_kernel[grid](
        x=x,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
        W=W,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        BW=BW,
        BD=BD,
    )
    return final_state


@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor,
    residual: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    shape = x.shape
    if weight is not None and x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')

    D = x.shape[-1]
    N = x.numel() // D
    W = weight.shape[1] if weight is not None else None
    BD = 8
    BW = triton.next_power_of_2(W)

    if x.dim() == 2:
        # Case: (N, D)
        stride_x_n = x.stride(0)
        stride_x_d = x.stride(1)
    elif x.dim() == 3 and x.shape[0] == 1:
        # Case: (1, N, D) -> Time=1, Batch=N, Dim=D
        # Batch 在 dim 1
        stride_x_n = x.stride(1)
        stride_x_d = x.stride(2)
    elif x.dim() == 3:
        # Case: (N, 1, D) -> Batch=N, Time=1, Dim=D
        # Batch 在 dim 0
        stride_x_n = x.stride(0)
        stride_x_d = x.stride(2)
    else:
        # Fallback / Error case
        raise ValueError(f"Unsupported input shape: {x.shape}")

    y = torch.empty_like(x, memory_format=torch.contiguous_format)

    if y.dim() == 2:
        stride_y_n, stride_y_d = y.stride(0), y.stride(1)
    elif y.dim() == 3 and y.shape[0] == 1:
        stride_y_n, stride_y_d = y.stride(1), y.stride(2)
    elif y.dim() == 3:
        stride_y_n, stride_y_d = y.stride(0), y.stride(2)

    def grid(meta): return (triton.cdiv(D, meta['BD']), N)

    causal_conv1d_update_kernel[grid](
        x=x,
        cache=cache,
        residual=residual,
        y=y,
        weight=weight,
        bias=bias,
        stride_x_n=stride_x_n,
        stride_x_d=stride_x_d,
        stride_y_n=stride_y_n,
        stride_y_d=stride_y_d,
        D=D,
        W=W,
        BD=BD,
        BW=BW,
        ACTIVATION=activation,
        num_warps=STATIC_WARPS,
    )
    return y.view(shape), cache


class CausalConv1dFunction(torch.autograd.Function):

    @staticmethod
    @input_guard(no_guard_contiguous=["x"])
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool | None = False,
        activation: str | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        chunk_size: int = 64,
    ):
        BT = chunk_size
        if cu_seqlens is not None and chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_cpu = cu_seqlens_cpu
        ctx.chunk_indices = chunk_indices
        ctx.save_for_backward(x, weight, bias, residual, initial_state)
        y, final_state = causal_conv1d_fwd(
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
            BT=BT,
        )
        return y, final_state

    @staticmethod
    @input_guard(no_guard_contiguous=["dy"])
    def backward(ctx, dy: torch.Tensor, dht: torch.Tensor | None = None):
        x, weight, bias, residual, initial_state = ctx.saved_tensors
        dx, dw, db, dr, dh0 = causal_conv1d_bwd(
            x=x,
            dy=dy,
            dht=dht,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=ctx.activation,
            cu_seqlens=ctx.cu_seqlens,
            cu_seqlens_cpu=ctx.cu_seqlens_cpu,
            chunk_indices=ctx.chunk_indices,
        )
        return dx, dw, db, dr, dh0, None, None, None, None, None, None
