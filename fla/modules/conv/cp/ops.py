import torch
import torch.distributed as dist

from fla.ops.cp import FLACPContext, conv_cp_send_recv_bwd, conv_cp_send_recv_fwd


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
        """
        Propagate and apply initial-state gradients across context-parallel ranks to correct dx.
        
        If a process group is provided, this exchanges the gradient of the forward initial_state (shape [W-1, D]) with neighboring ranks and adds the received gradient to dx's last W-1 timesteps. If `dh0` is None, it must be because this is the first CP rank; a zero tensor is used as the outgoing initial-state gradient in that case. No action is taken when `group` is None.
        
        Parameters:
            dx: Gradient tensor to be corrected; expected layout with last dim D.
            dh0: Gradient w.r.t. the saved initial_state for the first sequence (shape [N, D, W]) or None when first rank has no initial_state.
            W: Convolution kernel width.
            group: Process group used for CP communication; when None the function is a no-op.
            is_first_rank: True if this rank is the first in the CP chain (required when `dh0` is None).
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
        cu_seqlens: torch.Tensor | None,
        cu_seqlens_cpu: torch.Tensor | None,
        chunk_indices: torch.Tensor | None,
        cp_context: FLACPContext | None,
    ):
        # Import here to avoid circular dependency
        """
        Perform the forward pass for a context-parallel (CP) 1D causal convolution and save tensors/metadata required for backward.
        
        Parameters:
            x (torch.Tensor): Input tensor.
            weight (torch.Tensor): Convolution weight tensor with shape [..., W].
            bias (torch.Tensor | None): Optional bias tensor.
            activation (str | None): Activation name or None.
            cu_seqlens (torch.Tensor | None): Optional GPU cumulative sequence lengths used for packed sequences.
            cu_seqlens_cpu (torch.Tensor | None): Optional CPU cumulative sequence lengths mirror.
            chunk_indices (torch.Tensor | None): Optional tensor of chunk indices used by the underlying op.
            cp_context (FLACPContext | None): Context-parallel context providing the process group and rank metadata; required for CP mode.
        
        Returns:
            y (torch.Tensor): Output tensor produced by the causal convolution.
        
        Raises:
            ValueError: If `cp_context` is None.
        """
        from fla.modules.conv.triton.ops import causal_conv1d_fwd

        if cp_context is None:
            raise ValueError("cp_context must be provided for CausalConv1dFunctionCP")
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
        )

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        # Import here to avoid circular dependency
        """
        Compute gradients for the CP-enabled causal 1D convolution and propagate corrections across CP ranks.
        
        Calls the underlying causal_conv1d backward implementation to compute gradients for inputs and parameters, then applies context-parallel (CP) gradient synchronization to adjust the input gradient that corresponds to the exchanged initial state between ranks.
        
        Parameters:
            dy (torch.Tensor): Gradient of the convolution output with respect to the downstream loss.
        
        Returns:
            dx (torch.Tensor): Gradient with respect to `x`.
            dw (torch.Tensor): Gradient with respect to `weight`.
            db (torch.Tensor or None): Gradient with respect to `bias`, or `None` if `bias` was not provided.
            None: Placeholder for `activation` gradient (not computed).
            None: Placeholder for `cu_seqlens` gradient (not computed).
            None: Placeholder for `cu_seqlens_cpu` gradient (not computed).
            None: Placeholder for `chunk_indices` gradient (not computed).
            None: Placeholder for `cp_context` gradient (not computed).
        """
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
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    cp_context: FLACPContext | None = None,
):
    """
    Context-parallel causal 1D convolution that wires CP context into autograd.
    
    Performs a causal_conv1d forward using the provided CP context to initialize cross-rank state during forward and to correct gradients across ranks during backward. The caller must pass a valid `cp_context` when running in CP mode.
    
    Parameters:
        x: Input tensor; expected layout [1, T, D] when participating in CP communication.
        weight: Convolution weights; kernel width is inferred from its second dimension.
        cp_context: FLACPContext that carries CP metadata and the process group; required for CP mode.
        cu_seqlens: Cumulative sequence lengths used for packed/variable-length inputs (optional, forwarded to the op).
        cu_seqlens_cpu: CPU-side cumulative sequence lengths (optional, forwarded to the op).
        chunk_indices: Chunk indices for variable-length or chunked inputs (optional).
        bias: Optional bias vector forwarded to the op.
        activation: Optional activation name forwarded to the op.
    
    Returns:
        The convolved output tensor with the same leading batch/sequence layout as the underlying causal_conv1d implementation.
    """
    return CausalConv1dFunctionCP.apply(
        x, weight, bias, activation,
        cu_seqlens, cu_seqlens_cpu, chunk_indices, cp_context
    )