# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""CUDA-based mixed-mode implementation for causal convolution."""

import torch
from einops import rearrange

from fla.modules.conv.triton import causal_conv1d_update_states
from fla.ops.utils import prepare_sequence_ids
from fla.utils import input_guard

try:
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
except ImportError:
    causal_conv1d_bwd_function = None

try:
    from causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_cuda
except ImportError:
    causal_conv1d_fn_cuda = None


class FastCausalConv1dFn(torch.autograd.Function):
    """
    Mixed-mode (Mix) Causal Convolution Implementation - Combining Triton Forward and CUDA Backward Propagation

    This class implements forward propagation using FLA's Triton kernel, while using the optimized
    implementation from TriDao's causal_conv1d CUDA package for backward propagation.
    This hybrid strategy combines the advantages of both technologies:

    - Forward: Uses FLA's Triton implementation, optimized for the FLA framework
    - Backward: Uses TriDao's causal_conv1d_bwd_function CUDA implementation for faster speed

    Performance Benefits:
    - CUDA backward implementation is typically faster than the Triton version, reducing training time
    - Maintains the flexibility and compatibility of forward propagation

    Note:
    - Input/Output format is (batch, seqlen, dim)
    - Backward propagation requires causal_conv1d package: pip install causal-conv1d
    - Supports SILU/Swish activation functions
    - Current limitations (not yet supported):
        * output_final_state must be False
        * initial_states must be None
        * residual must be None
    """
    @staticmethod
    @input_guard(no_guard_contiguous=["x"])
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        residual: torch.Tensor | None = None,
        initial_states=None,
        output_final_state=False,
        activation=None,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        seq_idx: torch.LongTensor | None = None,
    ):
        if activation not in [None, "silu", "swish"]:
            raise NotImplementedError("activation must be None, silu, or swish")
        assert output_final_state is False, "output_final_state must be False for FastCausalConv1dFn"
        assert initial_states is None, "initial_states must be None for FastCausalConv1dFn"
        assert residual is None, "residual must be None for FastCausalConv1dFn"

        bias = bias.contiguous() if bias is not None else None
        if cu_seqlens is not None and seq_idx is None:
            seq_idx = prepare_sequence_ids(cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu).to(
                torch.int32).unsqueeze(0)
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None

        # Import here to avoid circular dependency
        from fla.modules.conv.triton.ops import causal_conv1d_fwd

        ctx.activation = activation in ["silu", "swish"]
        out, _ = causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=None,
            output_final_state=output_final_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
        )

        ctx.save_for_backward(x, weight, bias, seq_idx, initial_states)
        ctx.return_final_states = output_final_state
        ctx.return_dinitial_states = (
            initial_states is not None and initial_states.requires_grad
        )
        return out, None

    @staticmethod
    @input_guard
    def backward(ctx, dout, *args):
        x, weight, bias, seq_idx, initial_states = ctx.saved_tensors
        dx = torch.empty_like(x, memory_format=torch.contiguous_format)
        x = rearrange(x, 'b t d -> b d t')
        dx = rearrange(dx, 'b t d -> b d t')
        dout = rearrange(dout, 'b t d -> b d t')
        dfinal_states = args[0] if ctx.return_final_states else None

        if dout.stride(2) != 1 and dout.stride(1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        # Here we just pass in None and dx will be allocated in the C++ code.
        dx, dweight, dbias, dinitial_states = causal_conv1d_bwd_function(
            x,
            weight,
            bias,
            dout,
            seq_idx,
            initial_states,
            dfinal_states,
            dx,
            ctx.return_dinitial_states,
            ctx.activation,
        )
        dx = rearrange(dx, 'b d t -> b t d')
        return (
            dx,
            dweight,
            dbias if bias is not None else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fast_causal_conv1d_fn(
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
    seq_idx: torch.LongTensor | None = None,
):
    """
    x: (batch, seqlen, dim)
    weight: (dim, width)
    bias: (dim,)
    seq_idx: (batch, seqlen)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1), to be written to
    activation: either None or "silu" or "swish"

    out: (batch, seqlen, dim)
    """
    assert causal_conv1d_bwd_function is not None, "causal_conv1d_bwd_function is not available"
    return FastCausalConv1dFn.apply(
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
        seq_idx,
    )


def causal_conv1d_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool | None = False,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    **kwargs,
):
    assert causal_conv1d_fn_cuda is not None, "causal_conv1d_fn_cuda is not available"
    seq_idx = kwargs.get('seq_idx')
    if cu_seqlens is not None or seq_idx is not None:
        assert initial_state is None, "For CUDA backend, initial_state must be None if cu_seqlens or seq_idx is provided"
    W = weight.shape[-1]
    if x.stride(-1) != 1:
        x = x.contiguous()
    x_conv1d = rearrange(x, 'b t d -> b d t')
    if cu_seqlens is not None and seq_idx is None:
        seq_idx = prepare_sequence_ids(cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu).to(torch.int32).unsqueeze(0)

    y = causal_conv1d_fn_cuda(
        x=x_conv1d,
        weight=weight,
        bias=bias,
        activation=activation,
        seq_idx=seq_idx,
        initial_states=None,
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
    else:
        final_state = None
    if residual is not None:
        y.add_(residual)

    return y, final_state
