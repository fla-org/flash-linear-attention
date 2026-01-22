import torch
import torch.distributed as dist

from fla.ops.cp import FLACPContext, conv_cp_send_recv_bwd, conv_cp_send_recv_fwd
from fla.ops.utils import prepare_chunk_indices


class CausalConv1dFunctionCP(torch.autograd.Function):
    """
    Context Parallel version of CausalConv1dFunction.

    Forward:
        1. Get tails from previous rank to construct initial_state
        2. Call causal_conv1d_fwd

    Backward:
        1. Call causal_conv1d_bwd to get dx
        2. Sync communication: add next rank's first W-1 token gradients to current rank's last W-1 tokens
    """

    @staticmethod
    def _prepare_initial_state_for_cp(
        x: torch.Tensor,
        weight: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        context: FLACPContext,
        group: dist.ProcessGroup | None,
    ) -> torch.Tensor | None:
        """Prepare initial_state for CP forward pass by communicating with previous rank.

        Args:
            x: Input tensor of shape [1, T, D]
            weight: Weight tensor of shape [D, W]
            cu_seqlens: Cumulative sequence lengths
            context: CP context
            group: Process group for communication

        Returns:
            initial_state: Initial state tensor of shape [N, D, W] or None
        """
        if group is None:
            return None

        W = weight.shape[-1]  # weight: [D, W]
        D = weight.shape[0]
        initial_state = None
        if not context.is_first_rank:
            # Non-first rank needs initial_state
            assert x.dim() == 3 and x.shape[0] == 1, f"CP requires [1, T, D], got {x.shape}"
            x_2d = x.squeeze(0)  # [T, D]
            tails = x_2d[-(W-1):].contiguous()  # [W-1, D]
            heads = conv_cp_send_recv_fwd(tails, group)  # [W-1, D]
            # Construct initial_state: [N, D, W]
            N = len(cu_seqlens) - 1
            initial_state = torch.zeros(N, D, W, device=x.device, dtype=x.dtype)
            valid_len = min(W - 1, context.pre_num_conv_tokens)
            if valid_len > 0:
                # heads[-valid_len:]: [valid_len, D] -> [D, valid_len]
                initial_state[0, :, -valid_len:] = heads[-valid_len:].T
        else:
            # First rank also needs to participate in communication (send tails)
            x_2d = x.squeeze(0)
            tails = x_2d[-(W-1):].contiguous()
            _ = conv_cp_send_recv_fwd(tails, group)  # Send but don't use

        return initial_state

    @staticmethod
    def _correct_dx_for_cp(
        dx: torch.Tensor,
        dh0: torch.Tensor | None,
        W: int,
        group: dist.ProcessGroup | None,
        is_first_rank: bool,
    ) -> None:
        """Correct dx gradients for CP backward pass by communicating with next rank.

        Args:
            dx: Gradient tensor to be corrected, shape [1, T, D]
            dh0: Gradient w.r.t. initial_state, shape [N, D, W] or None
            W: Kernel size
            group: Process group for communication
            is_first_rank: Whether this is the first rank in the sequence's processing chain
        """
        if group is None:
            return

        D = dx.shape[-1]
        # dh0: [N, D, W] or None
        # We only care about the first sequence's initial_state gradient
        if dh0 is not None:
            # Get first sequence's d_initial_state: [D, W] -> last W-1 cols -> [D, W-1] -> [W-1, D]
            d_initial_state = dh0[0, :, -(W-1):].T.contiguous()  # [W-1, D]
        else:
            # dh0 is None only when this is the first rank (no initial_state needed)
            assert is_first_rank, "dh0 should not be None when is_first_rank=False"
            d_initial_state = torch.zeros(W-1, D, device=dx.device, dtype=dx.dtype)
        # Sync communication: send d_initial_state to previous rank, receive from next rank
        recv_d_init = conv_cp_send_recv_bwd(d_initial_state, group)  # [W-1, D]
        # Add to current rank's last W-1 tokens (these tokens are used as initial_state by next rank)
        dx[0, -(W-1):, :].add_(recv_d_init)

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        chunk_indices: torch.Tensor | None,
        cp_context: FLACPContext | None,
        chunk_size: int | None,
        backend: str = 'triton',
    ):
        # Import here to avoid circular dependency
        from fla.modules.conv.triton.ops import causal_conv1d_fwd

        if cp_context is None:
            raise ValueError("cp_context must be provided for CausalConv1dFunctionCP")
        cu_seqlens = cp_context.cu_seqlens
        cu_seqlens_cpu = cp_context.cu_seqlens_cpu
        group = cp_context.group

        # Get kernel_size
        W = weight.shape[-1]  # weight: [D, W]
        # Prepare initial_state for CP
        initial_state = CausalConv1dFunctionCP._prepare_initial_state_for_cp(
            x=x,
            weight=weight,
            cu_seqlens=cu_seqlens,
            context=cp_context,
            group=group,
        )

        ctx.save_for_backward(x, weight, bias, initial_state)
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_cpu = cu_seqlens_cpu
        ctx.chunk_indices = chunk_indices
        ctx.chunk_size = chunk_size
        ctx.group = group
        ctx.W = W
        ctx.is_first_rank = cp_context.is_first_rank

        # Call original forward
        y, _ = causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=initial_state,
            output_final_state=False,
            activation=activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            BT=chunk_size,
        )

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        # Import here to avoid circular dependency
        from fla.modules.conv.triton.ops import causal_conv1d_bwd

        x, weight, bias, initial_state = ctx.saved_tensors
        group = ctx.group
        W = ctx.W

        # Call original backward
        dx, dw, db, _, dh0 = causal_conv1d_bwd(
            x=x,
            dy=dy,
            dht=None,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=initial_state,
            activation=ctx.activation,
            cu_seqlens=ctx.cu_seqlens,
            cu_seqlens_cpu=ctx.cu_seqlens_cpu,
            chunk_indices=ctx.chunk_indices,
            BT=ctx.chunk_size,
        )

        # Correct dx gradients for CP
        CausalConv1dFunctionCP._correct_dx_for_cp(
            dx=dx,
            dh0=dh0,
            W=W,
            group=group,
            is_first_rank=ctx.is_first_rank,
        )

        return dx, dw, db, None, None, None, None, None


def causal_conv1d_cp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    chunk_indices: torch.Tensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_size: int | None = None,
    backend: str = 'triton',
):
    """
    Context Parallel version of causal_conv1d.

    Automatically handles communication in CP environment:
    - Forward: get initial_state from previous rank
    - Backward: correct dx gradients

    Args:
        x: Input tensor of shape [1, T, D]
        weight: Weight tensor of shape [D, W]
        bias: Bias tensor of shape [D] or None
        activation: Activation function name or None
        cu_seqlens: Cumulative sequence lengths
        cu_seqlens_cpu: Cumulative sequence lengths on CPU
        chunk_indices: Chunk indices for variable-length sequences
        cp_context: CP context (required for CP mode)
    """
    if cp_context is None:
        raise ValueError("cp_context must be provided for causal_conv1d_cp")

    assert cp_context.conv1d_kernel_size is not None, "conv1d_kernel_size must be provided for causal_conv1d_cp"
    assert cp_context.cu_seqlens is not None, "cu_seqlens must be provided for causal_conv1d_cp"
    assert backend in ['triton'], "backend must be 'triton'"
    chunk_size = chunk_size or 64
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cp_context.cu_seqlens, chunk_size, cu_seqlens_cpu=cp_context.cu_seqlens_cpu)

    return CausalConv1dFunctionCP.apply(
        x, weight, bias, activation,
        chunk_indices, cp_context, chunk_size, backend
    )
