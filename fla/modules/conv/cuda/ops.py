# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""CUDA-based mixed-mode implementation for causal convolution."""

import torch
from einops import rearrange

from fla.ops.utils import prepare_sequence_ids
from fla.utils import input_guard

try:
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
except ImportError:
    causal_conv1d_bwd_function = None


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
        """
        Compute the forward pass of a mixed-mode causal 1D convolution and prepare context for the backward pass.
        
        Parameters:
            ctx: Autograd context for saving tensors and metadata for backward.
            x (Tensor): Input tensor of shape (batch, seqlen, dim).
            weight (Tensor): Convolution weight tensor of shape (dim, width).
            bias (Tensor, optional): Bias tensor of shape (dim,).
            residual (Tensor or None): Must be None for this implementation.
            initial_states: Must be None for this implementation.
            output_final_state (bool): Must be False for this implementation.
            activation (str or None): Activation applied in forward; allowed values: None, "silu", "swish".
            cu_seqlens (LongTensor or None): Optional cumulative sequence lengths for packed sequences.
            cu_seqlens_cpu (LongTensor or None): Optional CPU-side cu_seqlens used when preparing sequence ids.
            chunk_indices (LongTensor or None): Optional chunk mapping for variable-length sequence processing.
            seq_idx (LongTensor or None): Optional precomputed sequence id tensor; if not provided and cu_seqlens is given, it will be computed.
        
        Returns:
            tuple:
                - out (Tensor): Output tensor of the forward convolution.
                - None: Placeholder second return value.
        
        Side effects:
            - Saves (x, weight, bias, seq_idx, initial_states) and flags in ctx for use by backward.
            - Normalizes `bias` and `seq_idx` contiguity and may compute `seq_idx` from `cu_seqlens`.
        
        Raises:
            NotImplementedError: If `activation` is not one of None, "silu", or "swish".
            AssertionError: If `output_final_state` is True or if `initial_states` or `residual` is provided.
        """
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
        """
        Compute gradients for the FastCausalConv1dFn autograd function.
        
        This backward implementation produces gradients for the inputs of the forward call and invokes the configured CUDA backward kernel. If ctx.return_final_states is True, the first extra arg is interpreted as the gradient w.r.t. the final states.
        
        Parameters:
            ctx: Autograd context with saved tensors and flags from forward.
            dout (Tensor): Upstream gradient with shape (batch, time, dim).
            *args: Optional additional gradients (first element used for final-state gradients when enabled).
        
        Returns:
            tuple: Gradients aligned with the forward inputs:
                (dx, dweight, dbias_or_None, None, None, None, None, None, None, None, None)
        
            - dx (Tensor): Gradient w.r.t. input `x` or None if not computed.
            - dweight (Tensor): Gradient w.r.t. `weight`.
            - dbias_or_None (Tensor or None): Gradient w.r.t. `bias` if bias was provided, otherwise `None`.
            - Remaining entries are `None` placeholders corresponding to unused forward arguments and optional outputs.
        """
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
    Apply a mixed-mode causal 1D convolution (Triton forward, CUDA backward when available) to the input sequence.
    
    Parameters:
        x (torch.Tensor): Input tensor of shape (batch, seqlen, dim).
        weight (torch.Tensor | None): Convolution kernel of shape (dim, width).
        bias (torch.Tensor | None): Optional bias of shape (dim,).
        residual (torch.Tensor | None): Unused in this implementation; kept for API compatibility.
        initial_state (torch.Tensor | None): Optional initial states with shape (batch, dim, width - 1); not used by the forward path.
        output_final_state (bool | None): If True would request final states from forward; must be False for this implementation.
        activation (str | None): Activation applied inside the forward path; allowed values are None, "silu", or "swish".
        cu_seqlens (torch.Tensor | None): Optional cumulative sequence lengths for packed variable-length sequences (1D tensor).
        cu_seqlens_cpu (torch.LongTensor | None): Same as `cu_seqlens` but guaranteed to be on CPU when provided.
        chunk_indices (torch.LongTensor | None): Optional chunking indices for sequence packing.
        seq_idx (torch.LongTensor | None): Optional per-element sequence id tensor of shape (batch, seqlen); if omitted and `cu_seqlens` is provided, it will be derived automatically.
    
    Returns:
        torch.Tensor: Output tensor of shape (batch, seqlen, dim) containing the causal convolution result.
    """
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