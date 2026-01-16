import torch
import torch.distributed as dist

from fla.modules.convolution import causal_conv1d_bwd, causal_conv1d_fwd
from fla.ops.cp import get_cp_context
from fla.ops.cp.comm import conv_cp_send_recv_bwd, conv_cp_send_recv_fwd


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
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        cu_seqlens: torch.Tensor | None,
        cu_seqlens_cpu: torch.Tensor | None,
        chunk_indices: torch.Tensor | None,
    ):
        context = get_cp_context()
        group = context.group

        # Get kernel_size
        W = weight.shape[-1]  # weight: [D, W]
        D = weight.shape[0]

        rank = dist.get_rank(group) if group is not None else 0

        # === CP Forward: construct initial_state ===
        initial_state = None
        if group is not None and context.pre_num_ranks > 0:
            # Non-first rank needs initial_state
            assert x.dim() == 3 and x.shape[0] == 1, f"CP requires [1, T, D], got {x.shape}"

            x_2d = x.squeeze(0)  # [T, D]
            tails = x_2d[-(W-1):]  # [W-1, D]
            heads = conv_cp_send_recv_fwd(tails, group)  # [W-1, D]

            # Construct initial_state: [N, D, W]
            N = len(cu_seqlens) - 1
            initial_state = torch.zeros(N, D, W, device=x.device, dtype=x.dtype)

            valid_len = min(W - 1, context.pre_num_conv_tokens)

            if valid_len > 0:
                # heads[-valid_len:]: [valid_len, D] -> [D, valid_len]
                initial_state[0, :, -valid_len:] = heads[-valid_len:].T
        elif group is not None:
            # First rank also needs to participate in communication (send tails)
            x_2d = x.squeeze(0)
            tails = x_2d[-(W-1):].contiguous()
            _ = conv_cp_send_recv_fwd(tails, group)  # Send but don't use

        ctx.save_for_backward(x, weight, bias, initial_state)
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_cpu = cu_seqlens_cpu
        ctx.chunk_indices = chunk_indices
        ctx.group = group
        ctx.W = W

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

        # === CP Backward: correct dx ===
        # Key: send d_initial_state (dh0), not front_dx
        # dh0 is gradient w.r.t. initial_state, which comes from previous rank's tails
        if group is not None:
            D = x.shape[-1]

            # dh0: [N, D, W] or None
            # We only care about the first sequence's initial_state gradient
            if dh0 is not None:
                # Get first sequence's d_initial_state: [D, W] -> last W-1 cols -> [D, W-1] -> [W-1, D]
                d_initial_state = dh0[0, :, -(W-1):].T.contiguous()  # [W-1, D]
            else:
                d_initial_state = torch.zeros(W-1, D, device=x.device, dtype=x.dtype)

            # Sync communication: send d_initial_state to previous rank, receive from next rank
            recv_d_init = conv_cp_send_recv_bwd(d_initial_state, group)  # [W-1, D]

            # Add to current rank's last W-1 tokens (these tokens are used as initial_state by next rank)
            dx[0, -(W-1):, :].add_(recv_d_init)

        return dx, dw, db, None, None, None, None


def causal_conv1d_cp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
):
    """
    Context Parallel version of causal_conv1d.

    Automatically handles communication in CP environment:
    - Forward: get initial_state from previous rank
    - Backward: correct dx gradients
    """
    return CausalConv1dFunctionCP.apply(
        x, weight, bias, activation,
        cu_seqlens, cu_seqlens_cpu, chunk_indices
    )
