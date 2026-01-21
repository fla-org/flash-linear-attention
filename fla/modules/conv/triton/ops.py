import torch
import triton
from einops import rearrange

from fla.ops.utils import prepare_chunk_indices
from fla.utils import get_multiprocessor_count, input_guard

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
) -> torch.Tensor:
    """
    Compute the forward pass of a Triton-accelerated causal 1D convolution with optional bias, residual connection, initial state, and activation.
    
    Parameters:
        x (torch.Tensor): Input tensor with shape (..., T, D) or broadcastable variants; last dimension must match weight.shape[0] after optional rearrange.
        weight (torch.Tensor): Convolution weights with shape (D, W).
        bias (torch.Tensor): Bias tensor broadcastable to output channels.
        residual (torch.Tensor): Residual tensor added to the convolution output.
        initial_state (torch.Tensor | None): Optional initial state tensor of shape (N, D, W) used for sequence continuation.
        output_final_state (bool): If True, compute and return the final state for each sequence.
        activation (str | None): Optional activation name to apply elementwise (e.g., 'swish', 'silu'); passed to the kernel.
        cu_seqlens (torch.LongTensor | None): Optional cumulative sequence lengths for packed variable-length sequences; length should be N+1.
        cu_seqlens_cpu (torch.LongTensor | None): CPU-side cu_seqlens used to prepare chunk indices if provided.
        chunk_indices (torch.LongTensor | None): Optional precomputed chunk indices used to tile long sequences; if None and cu_seqlens is provided, indices are derived internally.
    
    Returns:
        (torch.Tensor, torch.Tensor | None): A tuple where the first element is the convolution output reshaped to the original input shape, and the second element is the final state tensor of shape (N, D, W) if `output_final_state` is True, otherwise `None`.
    """
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
    B, T, D = x.shape[0], x.shape[1], weight.shape[0]
    W = weight.shape[1]
    stride_x_n, stride_x_t, stride_x_d = x.stride()

    BT = min(64, triton.next_power_of_2(triton.cdiv(max(16, B*T), get_multiprocessor_count(x.device.index))))
    BW = triton.next_power_of_2(W)
    if chunk_indices is None and (cu_seqlens is not None or cu_seqlens_cpu is not None):
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    NB = triton.cdiv(B*T, 1024)

    y = torch.empty_like(x, memory_format=torch.contiguous_format)

    def grid(meta): """
Compute the Triton kernel grid dimensions for the current convolution launch.

Parameters:
    meta (dict): Kernel meta parameters; must contain 'BD' (block size along the D dimension).

Returns:
    tuple: (num_blockD, NT, B) where `num_blockD` is ceil(D / meta['BD']), `NT` is the number of time chunks, and `B` is the batch size.
"""
return (triton.cdiv(D, meta['BD']), NT, B)
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
    Compute the gradient with respect to the initial state (dh0) by invoking a Triton kernel.
    
    This routine produces a tensor of the same shape as `initial_state` containing dL/d(h0). `y` must be provided when `activation` is 'swish' or 'silu' because the kernel requires the post-activation values for those activations. The implementation uses a Triton kernel and serves as a workaround for compiler issues on some architectures.
    
    Parameters:
        dy (torch.Tensor): Gradient of the output with shape (B, T, D).
        y (torch.Tensor | None): Optional forward-output (post-activation) with shape compatible with `dy`; required for 'swish'/'silu', otherwise may be None.
        weight (torch.Tensor): Convolution weight with shape (D, W).
        initial_state (torch.Tensor): Initial state with shape (N, D, W); determines the shape of the returned gradient.
        activation (str | None): Activation name; affects whether `y` is required.
        cu_seqlens (torch.Tensor | None): Optional cumulative sequence lengths for packed/batched sequences.
    
    Returns:
        torch.Tensor: `dh0`, the gradient w.r.t. `initial_state`, with the same shape and dtype as `initial_state`.
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
):
    """
    Compute gradients for a causal 1D convolution with optional parameter, residual, and initial-state gradients.
    
    Parameters:
        x (torch.Tensor): Input tensor of shape (B, T, D) or broadcastable equivalent used in the forward pass.
        dy (torch.Tensor): Gradient of the loss w.r.t. the forward output y (same layout as y).
        dht (torch.Tensor): Gradient w.r.t. any provided final hidden/state passed back into backward (can be None).
        weight (torch.Tensor | None): Convolution weight tensor of shape (D_in, W); when provided, per-weight gradients are computed.
        bias (torch.Tensor | None): Bias tensor for the convolution; when provided, per-bias gradients are computed.
        residual (torch.Tensor | None): Residual input from the forward pass; when provided, its gradient is returned (as dr).
        initial_state (torch.Tensor | None): Initial hidden/state tensor used in the forward pass; when provided, gradient w.r.t. this state (dh0) is computed.
        activation (str | None): Activation name used in the forward pass; when provided, internal forward activations may be recomputed to support gradient calculation.
        cu_seqlens (torch.Tensor | None): Optional cumulative sequence lengths for packed variable-length sequences; used to compute chunking and per-sequence gradients.
        cu_seqlens_cpu (torch.LongTensor | None): CPU copy of cu_seqlens to assist chunk index preparation when needed.
        chunk_indices (torch.LongTensor | None): Precomputed chunk indices for time-chunking; if not provided and cu_seqlens is present, indices will be prepared internally.
    
    Returns:
        tuple: A 5-tuple containing:
            - dx (torch.Tensor): Gradient w.r.t. the input x, shaped like the original x.
            - dw (torch.Tensor | None): Gradient w.r.t. weight (same dtype and shape as weight) or None if weight was not provided.
            - db (torch.Tensor | None): Gradient w.r.t. bias (same dtype and shape as bias) or None if bias was not provided.
            - dr (torch.Tensor | None): Gradient w.r.t. the residual (same shape as residual) or None if residual was not provided.
            - dh0 (torch.Tensor | None): Gradient w.r.t. the initial_state or None if initial_state was not provided.
    """
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
    B, T, D = x.shape
    W = weight.shape[1] if weight is not None else None

    stride_x_n, stride_x_t, stride_x_d = x.stride()

    BT = min(64, triton.next_power_of_2(triton.cdiv(max(16, B*T), get_multiprocessor_count(x.device.index))))
    BW = triton.next_power_of_2(W)
    if chunk_indices is None and (cu_seqlens is not None or cu_seqlens_cpu is not None):
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
        )
    dx = torch.empty_like(x)
    dw = weight.new_empty(B*NT, *weight.shape, dtype=torch.float) if weight is not None else None
    db = bias.new_empty(B*NT, *bias.shape, dtype=torch.float) if bias is not None else None
    dr = dy if residual is not None else None

    stride_dx_n, stride_dx_t, stride_dx_d = dx.stride()

    def grid(meta): """
Compute the Triton kernel grid dimensions for the current convolution launch.

Parameters:
    meta (dict): Kernel meta parameters; must contain 'BD' (block size along the D dimension).

Returns:
    tuple: (num_blockD, NT, B) where `num_blockD` is ceil(D / meta['BD']), `NT` is the number of time chunks, and `B` is the batch size.
"""
return (triton.cdiv(D, meta['BD']), NT, B)
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
    """
    Compute the final convolutional state for each sequence in `x` after a causal 1D convolution.
    
    Parameters:
        x (torch.Tensor): Input activations. Accepts either a batched tensor of shape (B, T, D) or a packed representation when `cu_seqlens` is provided — either (T, D) or (N_packed, T, D). D is the feature dimension and T is the time length.
        state_len (int): Length W of the per-feature state to produce for each sequence.
        initial_state (torch.Tensor | None): Optional initial state of shape (N, D, W) to seed state updates. If None, the kernels will assume an implicit zero initial state.
        cu_seqlens (torch.Tensor | None): Optional 1D cumulative sequence lengths tensor; when provided, N is inferred as len(cu_seqlens) - 1 and `x` is interpreted as packed per-sequence data.
    
    Returns:
        final_state (torch.Tensor): Tensor of shape (N, D, W) containing the final state for each sequence, with the same dtype and device as `x`.
    """
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
    """
    Apply a single-step causal convolution update to `x` using the provided `cache` and optional weight, bias, residual, and activation.
    
    Parameters:
        x (torch.Tensor): Input tensor. Supported shapes: (N, D), (1, N, D) interpreted as (Time=1, Batch=N, Dim=D), or (N, 1, D) interpreted as (Batch=N, Time=1, Dim=D).
        cache (torch.Tensor): Residual/cache tensor used by the kernel and returned alongside the output.
        residual (torch.Tensor | None): Optional residual to add to the convolution output.
        weight (torch.Tensor | None): Optional convolution weight with shape (D, W). If provided and `x`'s last dimension does not match `weight.shape[0]`, `x` will be reshaped to align.
        bias (torch.Tensor | None): Optional bias added to the convolution output.
        activation (str | None): Optional activation name passed through to the kernel (e.g., "swish", "silu"); if None no activation is applied.
    
    Returns:
        tuple: (y, cache) where `y` is the updated output tensor reshaped to match the original `x` shape, and `cache` is the (possibly unchanged) cache tensor provided as input.
    """
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

    def grid(meta): """
Constructs a Triton kernel grid tuple using the block size provided in `meta`.

Parameters:
    meta (dict): Must contain key `'BD'` (block size along the D dimension) used to compute the first grid dimension.

Returns:
    tuple: `(grid_x, grid_y)` where `grid_x = ceil(D / meta['BD'])` and `grid_y = N`.
"""
return (triton.cdiv(D, meta['BD']), N)

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
    ):
        """
        Perform the forward pass of the Triton-accelerated causal 1D convolution and save tensors required for the backward pass.
        
        Parameters:
            ctx: Autograd context used to save tensors and configuration for backward.
            x (torch.Tensor): Input tensor with time and feature dimensions (shape varies by layout).
            weight (torch.Tensor | None): Convolution weight; if None, convolution is skipped and input is forwarded.
            bias (torch.Tensor | None): Optional bias added to the convolution output.
            residual (torch.Tensor | None): Optional residual to add to the convolution output.
            initial_state (torch.Tensor | None): Optional initial state for the causal convolution (shape (N, D, W)).
            output_final_state (bool | None): If True, compute and return the final state after the forward pass.
            activation (str | None): Optional activation name applied after convolution (e.g., "swish", "silu"); pass None for no activation.
            cu_seqlens (torch.Tensor | None): Optional packed sequence lengths for variable-length (packed) inputs.
            cu_seqlens_cpu (torch.LongTensor | None): CPU copy of cu_seqlens when provided.
            chunk_indices (torch.LongTensor | None): Optional precomputed chunk indices for packed inputs to control kernel tiling.
        
        Returns:
            y (torch.Tensor): Convolution output tensor, shaped to match the input layout.
            final_state (torch.Tensor | None): Final state tensor when `output_final_state` is True (shape (N, D, W)), otherwise `None`.
        """
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
        )
        return y, final_state

    @staticmethod
    @input_guard(no_guard_contiguous=["dy"])
    def backward(ctx, dy: torch.Tensor, dht: torch.Tensor | None = None):
        """
        Compute and return gradients for tensors saved in the forward pass using the causal_conv1d backward routine.
        
        Parameters:
            dy (torch.Tensor): Gradient of the loss with respect to the output produced in forward.
            dht (torch.Tensor | None): Optional gradient with respect to the final state returned by forward.
        
        Returns:
            tuple: A tuple of gradients corresponding to the forward arguments:
                - dx (torch.Tensor): Gradient with respect to the input `x`.
                - dw (torch.Tensor | None): Gradient with respect to `weight`, or `None` if `weight` was not provided.
                - db (torch.Tensor | None): Gradient with respect to `bias`, or `None` if `bias` was not provided.
                - dr (torch.Tensor | None): Gradient with respect to `residual`, or `None` if `residual` was not provided.
                - dh0 (torch.Tensor | None): Gradient with respect to `initial_state`, or `None` if `initial_state` was not provided.
                - Followed by `None` placeholders for gradients corresponding to non-tensor or unused forward arguments.
        """
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
        return dx, dw, db, dr, dh0, None, None, None, None, None