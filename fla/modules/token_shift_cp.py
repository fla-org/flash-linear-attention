# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Context Parallel support for Token Shift.

Token shift has a 1-token dependency on previous tokens:
    y[t] = x[t-1] - x[t]  (for t > 0)
    y[0] = cache - x[0]   (cache is the last token from previous rank)

In CP mode, non-first ranks need the last token from the previous rank as cache.
Backward: non-last ranks need to send the last token's gradient to previous rank.
"""

import torch
import torch.distributed as dist

from fla.modules.token_shift import token_shift_bwd, token_shift_fwd
from fla.ops.cp import FLACPContext, conv_cp_send_recv_bwd, conv_cp_send_recv_fwd


class TokenShiftCPFunction(torch.autograd.Function):
    """
    Context Parallel version of TokenShift.

    Forward:
        1. Get last token from previous rank to construct cache
        2. Call token_shift_fwd with cache

    Backward:
        1. Call token_shift_bwd to get dx
        2. Sync communication: add next rank's first token gradient to current rank's last token
    """

    @staticmethod
    def _prepare_cache_for_cp(
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        context: FLACPContext,
        group: dist.ProcessGroup | None,
    ) -> tuple[torch.Tensor | None, int]:
        """Prepare cache for CP forward pass by communicating with previous rank.

        Args:
            x: Input tensor of shape [1, T, D]
            cu_seqlens: Cumulative sequence lengths
            context: CP context
            group: Process group for communication

        Returns:
            cache: Cache tensor of shape [N, D] or None
            pre_num_tokens: Number of tokens from previous rank for the first sequence
        """
        if group is None:
            return None, 0

        D = x.shape[-1]
        cache = None
        pre_num_tokens = 0

        if not context.is_first_rank:
            # Non-first rank: need cache from previous rank
            assert x.dim() == 3 and x.shape[0] == 1, f"CP requires [1, T, D], got {x.shape}"
            x_2d = x.squeeze(0)  # [T, D]
            last_token = x_2d[-1:].contiguous()  # [1, D]
            prev_last_token = conv_cp_send_recv_fwd(last_token, group)  # [1, D]

            # For varlen: only the first sequence needs cache from prev rank
            N = len(cu_seqlens) - 1 if cu_seqlens is not None else 1
            cache = torch.zeros(N, D, device=x.device, dtype=x.dtype)

            # pre_num_conv_tokens tells us how many tokens from prev rank
            # belong to the first sequence on this rank
            pre_num_tokens = getattr(context, 'pre_num_conv_tokens', 0)
            if pre_num_tokens > 0:
                # The prev rank's last token is used as cache for first sequence
                cache[0] = prev_last_token[0]
        else:
            # First rank: participate in send but don't use received data
            x_2d = x.squeeze(0)
            last_token = x_2d[-1:].contiguous()
            _ = conv_cp_send_recv_fwd(last_token, group)

        return cache, pre_num_tokens

    @staticmethod
    def _correct_dx_for_cp(
        dx: torch.Tensor,
        grad_cache: torch.Tensor | None,
        group: dist.ProcessGroup | None,
        is_first_rank: bool,
        pre_num_tokens: int = 0,
    ) -> None:
        """Correct dx gradients for CP backward pass.

        Args:
            dx: Gradient tensor to be corrected, shape [1, T, D]
            grad_cache: Gradient w.r.t. cache, shape [N, D] or None
            group: Process group
            is_first_rank: Whether this is the first rank
            pre_num_tokens: Number of tokens from previous rank for first sequence
        """
        if group is None:
            return

        D = dx.shape[-1]

        # Prepare gradient to send to previous rank
        if grad_cache is not None and pre_num_tokens > 0:
            # Only first sequence's cache gradient is relevant
            d_cache = grad_cache[0:1]  # [1, D]
        else:
            d_cache = torch.zeros(1, D, device=dx.device, dtype=dx.dtype)

        # Send to previous rank, receive from next rank
        recv_grad = conv_cp_send_recv_bwd(d_cache, group)  # [1, D]

        # Add received gradient to current rank's last token
        dx[0, -1, :].add_(recv_grad[0])

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        chunk_indices: torch.Tensor | None,
        cp_context: FLACPContext | None,
    ):
        if cp_context is None:
            raise ValueError("cp_context must be provided for TokenShiftCPFunction")

        cu_seqlens = cp_context.cu_seqlens
        group = cp_context.group

        # Prepare cache for CP
        cache, pre_num_tokens = TokenShiftCPFunction._prepare_cache_for_cp(
            x=x,
            cu_seqlens=cu_seqlens,
            context=cp_context,
            group=group,
        )

        # Save for backward
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        ctx.group = group
        ctx.has_cache = cache is not None
        ctx.is_first_rank = cp_context.is_first_rank
        ctx.pre_num_tokens = pre_num_tokens

        # Call original forward
        y, N, T, use_short_kernel, cache_out = token_shift_fwd(
            x=x,
            cu_seqlens=cu_seqlens,
            cache=cache,
            output_cache=True,
            chunk_indices=chunk_indices,
        )

        ctx.N = N
        ctx.T = T
        ctx.use_short_kernel = use_short_kernel

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        group = ctx.group

        # Prepare dcache for backward
        # For CP: non-last rank needs to receive gradient from next rank
        # This is handled in _correct_dx_for_cp after computing dx
        dcache = None  # Will be computed by token_shift_bwd

        # Call original backward
        dx, grad_cache = token_shift_bwd(
            dy=dy,
            N=ctx.N,
            T=ctx.T,
            dcache=dcache,
            cu_seqlens=ctx.cu_seqlens,
            use_short_kernel=ctx.use_short_kernel,
            has_init_cache=ctx.has_cache,
            chunk_indices=ctx.chunk_indices,
        )

        # Correct dx gradients for CP
        TokenShiftCPFunction._correct_dx_for_cp(
            dx=dx,
            grad_cache=grad_cache,
            group=group,
            is_first_rank=ctx.is_first_rank,
            pre_num_tokens=ctx.pre_num_tokens,
        )

        return dx, None, None, None


def token_shift_cp(
    x: torch.Tensor,
    cp_context: FLACPContext,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
):
    """
    Context Parallel version of token_shift.

    Args:
        x: Input tensor of shape [1, T, D]
        cp_context: CP context (required for CP mode)
        cu_seqlens: Cumulative sequence lengths
        chunk_indices: Chunk indices for variable-length sequences

    Returns:
        output: Tensor of shape [1, T, D] after applying token-shift
    """
    if cp_context is None:
        raise ValueError("cp_context must be provided for token_shift_cp")

    assert cp_context.cu_seqlens is not None, "cu_seqlens must be provided for token_shift_cp"

    return TokenShiftCPFunction.apply(
        x, cu_seqlens, chunk_indices, cp_context
    )
