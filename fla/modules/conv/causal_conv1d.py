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
    A causal 1D convolution implementation that powers Mamba/Mamba2 and DeltaNet architectures.

    When a residual connection is provided, this implements the Canon operation
    described in the paper at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330.

    Args:
        x (torch.Tensor):
            Input tensor of shape [B, T, D].
        weight (Optional[torch.Tensor]):
            Weight tensor of shape [D, W]. Default: `None`.
        bias (Optional[torch.Tensor]):
            Bias tensor of shape [D]. Default: `None`.
        residual (Optional[torch.Tensor]):
            Residual tensor of shape [B, T, D]. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state tensor of shape [N, D, W],
            where `N` is the number of sequences in the batch and `W` is the kernel size.
            If provided, the initial state is used to initialize the cache. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape [N, D, W]. Default: `False`.
        activation (Optional[str]):
            Activations applied to output, only `swish`/`silu` or `None` (i.e., no activation) are supported.
            Default: `None`.
        backend (Optional[str]):
            Specifies the backend to use for the convolution operation. Supported values are `'cuda'` 、 `'triton'` and `'mix'`.
            Default: `'triton'`.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths (optional)
        chunk_indices (Optional[torch.LongTensor]):
            Chunk indices for variable-length sequences (optional)

    Returns:
        Tuple of (output, final_state).
        If `output_final_state` is `False`, the final state is `None`.
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
