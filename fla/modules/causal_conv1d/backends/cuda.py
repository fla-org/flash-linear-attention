# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CUDA-based mixed-mode implementation for causal convolution."""

import torch
from einops import rearrange

from fla.modules.causal_conv1d import causal_conv1d_update_states
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
    """Causal convolution with Triton forward and CUDA backward on `[B, T, D]` inputs.

    Backward requires the `causal-conv1d` package. Activation may be `None`, `silu`, or `swish`.
    Residuals and initial/final states are unsupported.
    """
    @staticmethod
    @input_guard(no_guard_contiguous=["x"])
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        initial_states: torch.Tensor | None = None,
        output_final_state: bool = False,
        activation: str | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        seq_idx: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if activation not in [None, "silu", "swish"]:
            raise NotImplementedError("activation must be None, silu, or swish")
        assert output_final_state is False, "output_final_state must be False for FastCausalConv1dFn"
        assert initial_states is None, "initial_states must be None for FastCausalConv1dFn"
        assert residual is None, "residual must be None for FastCausalConv1dFn"

        bias = bias.contiguous() if bias is not None else None
        if cu_seqlens is not None and seq_idx is None:
            seq_idx = prepare_sequence_ids(cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu).to(torch.int32).unsqueeze(0)
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None

        # import here to avoid circular dependency.
        from fla.modules.causal_conv1d import causal_conv1d_fwd

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
        ctx.return_dinitial_states = (initial_states is not None and initial_states.requires_grad)
        return out, None

    @staticmethod
    @input_guard
    def backward(ctx, dout: torch.Tensor, *args) -> tuple[torch.Tensor | None, ...]:
        x, weight, bias, seq_idx, initial_states = ctx.saved_tensors
        dx = torch.empty_like(x, memory_format=torch.contiguous_format)
        x = rearrange(x, 'b t d -> b d t')
        dx = rearrange(dx, 'b t d -> b d t')
        dout = rearrange(dout, 'b t d -> b d t')
        dfinal_states = args[0] if ctx.return_final_states else None

        if dout.stride(2) != 1 and dout.stride(1) != 1:
            dout = dout.contiguous()
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
        return (dx, dweight, dbias if bias is not None else None, None, None, None, None, None, None, None, None)


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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply mixed Triton/CUDA causal convolution to `[B, T, D]` inputs.

    Args:
        x (torch.Tensor):
            Input of shape `[B, T, D]`.
        weight (torch.Tensor, Optional):
            Convolution weights of shape `[D, W]`. Default: `None`.
        bias (torch.Tensor, Optional):
            Bias of shape `[D]`. Default: `None`.
        residual (torch.Tensor, Optional):
            Unsupported by this backend; must be `None`. Default: `None`.
        initial_state (torch.Tensor, Optional):
            Unsupported by this backend; must be `None`. Default: `None`.
        output_final_state (bool, Optional):
            Unsupported by this backend; must be `False`. Default: `False`.
        activation (str, Optional):
            Activation applied to the output: `None`, `silu`, or `swish`. Default: `None`.
        cu_seqlens (torch.Tensor, Optional):
            Cumulative sequence lengths for packed inputs. Default: `None`.
        cu_seqlens_cpu (torch.LongTensor, Optional):
            CPU copy of `cu_seqlens`. Default: `None`.
        chunk_indices (torch.LongTensor, Optional):
            Precomputed sequence chunk indices. Default: `None`.
        seq_idx (torch.LongTensor, Optional):
            Sequence IDs of shape `[B, T]`. Derived from `cu_seqlens` when omitted. Default: `None`.

    Returns:
        tuple[torch.Tensor, None]:
            Output of shape `[B, T, D]` and no final state.
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        final_state = causal_conv1d_update_states(x=x, state_len=W, initial_state=initial_state, cu_seqlens=cu_seqlens)
    else:
        final_state = None
    if residual is not None:
        y.add_(residual)

    return y, final_state


__all__ = ['FastCausalConv1dFn', 'causal_conv1d_cuda', 'fast_causal_conv1d_fn']
