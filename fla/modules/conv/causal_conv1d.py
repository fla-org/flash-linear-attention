# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Main interface for causal 1D convolution operations."""

import torch
from einops import rearrange

from fla.ops.cp import FLACPContext
from fla.ops.utils import prepare_sequence_ids
from fla.utils import input_guard

try:
    from causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_cuda
except ImportError:
    causal_conv1d_fn_cuda = None

try:
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
except ImportError:
    causal_conv1d_bwd_function = None


@input_guard(no_guard_contiguous=["x"])
def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool | None = False,
    activation: str | None = None,
    backend: str | None = 'triton',
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    **kwargs,
):
    """
    Compute a causal 1D convolution over batched sequences using one of several backends.
    
    Supports an optional residual (implements the Canon operation), an initial state / cache for streaming,
    optional emission of the final convolution state, and an optional activation (`'swish'`/`'silu'`).
    Selects among CP, Triton, mixed (requires external causal-conv1d), or CUDA backends via `cp_context`/`backend`.
    
    Parameters:
        x (torch.Tensor): Input tensor of shape [B, T, D].
        weight (torch.Tensor | None): Convolution kernel of shape [D, W] or `None` for identity-like behavior.
        bias (torch.Tensor | None): Per-channel bias of shape [D] or `None`.
        residual (torch.Tensor | None): Optional residual to add to the output, shape [B, T, D].
        initial_state (torch.Tensor | None): Optional cache tensor of shape [N, D, W] used to initialize streaming state.
        output_final_state (bool | None): If true, return the final state corresponding to the processed input.
        activation (str | None): Activation name to apply to outputs; supported: `'swish'`/`'silu'` or `None`.
        backend (str | None): Backend selector: `'triton'`, `'mix'`, or `'cuda'`. Default: `'triton'`.
        cu_seqlens (torch.Tensor | None): Optional cumulative sequence lengths for variable-length batches.
        cu_seqlens_cpu (torch.LongTensor | None): CPU copy of `cu_seqlens` when needed by some backends.
        chunk_indices (torch.LongTensor | None): Optional chunk indices for variable-length processing.
        cp_context (FLACPContext | None): When provided, run the CP (causal-parallel) implementation; in this mode
            `initial_state` must be `None`, `output_final_state` must be `False`, and `cu_seqlens` is required.
        **kwargs: Backwards-compatible extras (e.g., `seq_idx` for some backends).
    
    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: (output, final_state_or_cache)
        - output: Tensor of shape [B, T, D] containing the convolution result.
        - final_state_or_cache: Final state tensor (shape matching input cache) when requested or applicable; otherwise `None`.
    """
    # Import here to avoid circular dependencies
    from fla.modules.conv.cp import causal_conv1d_cp
    from fla.modules.conv.cuda import fast_causal_conv1d_fn
    from fla.modules.conv.triton import CausalConv1dFunction, causal_conv1d_update_states

    if cp_context is not None:
        assert initial_state is None, "Initial state is not supported for CP"
        assert output_final_state is False, "Output final state is not supported for CP"
        assert cu_seqlens is not None, "cu_seqlens is required for CP"
        output = causal_conv1d_cp(
            x=x,
            weight=weight,
            bias=bias,
            activation=activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            cp_context=cp_context,
        )
        return output, None

    if backend == 'triton':
        y, final_state = CausalConv1dFunction.apply(
            x,
            weight,
            bias,
            residual,
            initial_state,
            output_final_state,
            activation,
            cu_seqlens,
            cu_seqlens_cpu,
            chunk_indices,
        )
        return y, final_state
    elif backend == 'mix':
        if causal_conv1d_bwd_function is None:
            raise ImportError(
                "causal_conv1d is required for backend='mix', but it is not installed. "
                "Please install it with: pip install causal-conv1d\n"
                "For more details, see: https://github.com/Dao-AILab/causal-conv1d"
            )
        seq_idx = kwargs.get('seq_idx')
        return fast_causal_conv1d_fn(
            x,
            weight,
            bias,
            residual,
            initial_state,
            output_final_state,
            activation,
            cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            seq_idx=seq_idx,
        )

    # CUDA backend
    W = weight.shape[-1]
    if initial_state is not None:
        # Case: Has initial_state -> Must be Channel-Last (physically B, T, D)
        if x.stride(-1) != 1:
            x = x.contiguous()
        x = rearrange(x, 'b t d -> b d t')
    else:
        # Case: No initial_state -> Prefer Contiguous (physically B, D, T)
        x = rearrange(x, 'b t d -> b d t').contiguous()

    # check if cu_seqlens and cache are both provided
    # Sequence index for each token. Used for varlen.
    # Suppose a batch consists of two sequences with lengths 3 and 4,
    # seq_idx=[0, 0, 0, 1, 1, 1, 1] for this batch.
    # NOTE: No need to provide this arg if `cu_seqlens` is passed.
    # This arg is just for BC, and will be removed in the future.
    # [B, T]
    seq_idx = kwargs.get('seq_idx')
    if cu_seqlens is not None and seq_idx is None:
        seq_idx = prepare_sequence_ids(cu_seqlens).to(torch.int32).unsqueeze(0)

    # equivalent to:
    # y = _conv_forward(x, weight, bias)[..., :x.shape[-1]]
    # if activation is not None:
    #     y = ACT2FN[activation](x)

    cache, initial_state = initial_state, None
    if cache is not None:
        # To make causal-conv1d happy
        initial_state = (
            cache[:, :, -(W-1):]   # [N, D, W-1]
            .transpose(1, 2).contiguous()  # [N, W-1, D] and stride(2)==1
            .transpose(1, 2)               # [N, D, W-1] and stride(1)==1
        )

    y = causal_conv1d_fn_cuda(
        x=x,
        weight=weight,
        bias=bias,
        activation=activation,
        seq_idx=seq_idx,
        initial_states=initial_state,
        return_final_states=False,
    )

    y = rearrange(y, 'b d t -> b t d')
    if output_final_state:
        final_state = causal_conv1d_update_states(
            x=x,
            state_len=W,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )
    if residual is not None:
        y.add_(residual)

    return y, cache