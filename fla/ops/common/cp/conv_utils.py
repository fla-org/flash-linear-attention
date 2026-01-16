import torch
import torch.distributed as dist

from fla.ops.common.cp.cp_chunk_delta_h import get_gdn_cp_context


def all_gather(inp, out=None, group=None, async_op=False):
    world_size = dist.get_world_size(group=group)
    if out is None:
        out = torch.empty(world_size, *inp.shape, device=inp.device, dtype=inp.dtype)
    handle = dist.all_gather_into_tensor(out, inp, group=group, async_op=async_op)
    return out, handle


def cp_send_recv_fwd(tails, group):
    """
    Forward 时：每个 rank 发送 tails，接收前一个 rank 的 tails 作为 heads

    Args:
        tails: [W-1, D] 当前 rank 的尾部 tokens
        group: 通信组
    Returns:
        heads: [W-1, D] 前一个 rank 的尾部 tokens
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    ag_tensor, _ = all_gather(tails, group=group, async_op=False)

    if rank == 0:
        # 首 rank 没有前置 tokens，返回 zeros
        return torch.zeros_like(tails)
    else:
        return ag_tensor[rank - 1].clone()


def cp_send_recv_bwd(front_dx, group):
    """
    Backward 时：每个 rank 发送前 W-1 个 token 的 dx，接收后一个 rank 的

    Args:
        front_dx: [W-1, D] 当前 rank 前 W-1 个 token 的梯度
        group: 通信组
    Returns:
        recv_dx: [W-1, D] 后一个 rank 发来的梯度
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    ag_tensor, _ = all_gather(front_dx, group=group, async_op=False)

    if rank == world_size - 1:
        # 最后一个 rank 没有后续 rank，返回 zeros
        return torch.zeros_like(front_dx)
    else:
        return ag_tensor[rank + 1].clone()


class CausalConv1dFunctionCP(torch.autograd.Function):
    """
    Context Parallel 版本的 CausalConv1dFunction

    Forward:
        1. 从前一个 rank 获取 tails 构造 initial_state
        2. 调用 causal_conv1d_fwd

    Backward:
        1. 调用 causal_conv1d_bwd 获取 dx
        2. 同步通信：后一个 rank 的前 W-1 个 token 的 dx 加到当前 rank 的最后 W-1 个 token
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
        from fla.modules.convolution import causal_conv1d_fwd

        context = get_gdn_cp_context()
        group = context.group

        # 获取 kernel_size
        W = weight.shape[-1]  # weight: [D, W]
        D = weight.shape[0]

        rank = dist.get_rank(group) if group is not None else 0

        # === CP Forward: 构造 initial_state ===
        initial_state = None
        if group is not None and context.pre_num_ranks > 0:
            # 非首 rank 需要 initial_state
            assert x.dim() == 3 and x.shape[0] == 1, f"CP requires [1, T, D], got {x.shape}"

            x_2d = x.squeeze(0)  # [T, D]
            tails = x_2d[-(W-1):].contiguous()  # [W-1, D]
            heads = cp_send_recv_fwd(tails, group)  # [W-1, D]

            # 构造 initial_state: [N, D, W]
            N = len(cu_seqlens) - 1
            initial_state = torch.zeros(N, D, W, device=x.device, dtype=x.dtype)

            valid_len = min(W - 1, context.pre_num_conv_tokens)

            # === DEBUG ===
            print(f"[Rank {rank}] Forward: pre_num_ranks={context.pre_num_ranks}, pre_num_conv_tokens={context.pre_num_conv_tokens}")
            print(f"[Rank {rank}] Forward: valid_len={valid_len}, N={N}, D={D}, W={W}")
            print(f"[Rank {rank}] Forward: heads shape={heads.shape}, heads[:3,:3]={heads[:3, :3]}")

            if valid_len > 0:
                # heads[-valid_len:]: [valid_len, D] -> [D, valid_len]
                initial_state[0, :, -(valid_len):] = heads[-valid_len:].T
                print(f"[Rank {rank}] Forward: initial_state[0,:3,-valid_len:]={initial_state[0, :3, -(valid_len):]}")
        elif group is not None:
            # 首 rank 也需要参与通信（发送 tails）
            x_2d = x.squeeze(0)
            tails = x_2d[-(W-1):].contiguous()
            _ = cp_send_recv_fwd(tails, group)  # 发送但不使用
            print(f"[Rank {rank}] Forward: first rank, no initial_state needed")

        # 保存用于 backward（需要 clone，因为 causal_conv1d_fwd 可能修改 initial_state）
        saved_initial_state = initial_state.clone() if initial_state is not None else None
        ctx.save_for_backward(x, weight, bias, saved_initial_state)
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_cpu = cu_seqlens_cpu
        ctx.chunk_indices = chunk_indices
        ctx.group = group
        ctx.W = W

        # 调用原始 forward
        y, final_state = causal_conv1d_fwd(
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
        from fla.modules.convolution import causal_conv1d_bwd

        x, weight, bias, initial_state = ctx.saved_tensors
        group = ctx.group
        W = ctx.W

        rank = dist.get_rank(group) if group is not None else 0
        world_size = dist.get_world_size(group) if group is not None else 1

        # === DEBUG: 检查 initial_state ===
        if initial_state is not None:
            print(f"[Rank {rank}] Backward: initial_state shape={initial_state.shape}")
            print(f"[Rank {rank}] Backward: initial_state[0,:3,1:]={initial_state[0, :3, 1:]}")
        else:
            print(f"[Rank {rank}] Backward: initial_state is None")

        # 调用原始 backward
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

        # === CP Backward: 修正 dx ===
        # 关键：发送 d_initial_state (dh0)，而不是 front_dx
        # dh0 是对 initial_state 的梯度，而 initial_state 来自前一个 rank 的 tails
        if group is not None:
            D = x.shape[-1]

            # dh0: [N, D, W] 或 None
            # 我们只关心第一个序列的 initial_state 的梯度
            if dh0 is not None:
                # 取第一个序列的 d_initial_state: [D, W] -> 取最后 W-1 列 -> [D, W-1] -> [W-1, D]
                d_initial_state = dh0[0, :, -(W-1):].T.contiguous()  # [W-1, D]
            else:
                d_initial_state = torch.zeros(W-1, D, device=x.device, dtype=x.dtype)

            # === DEBUG ===
            print(f"[Rank {rank}] Backward: dh0={dh0 is not None}, d_initial_state[:,:3]={d_initial_state[:, :3] if d_initial_state is not None else None}")
            print(f"[Rank {rank}] Backward: dx[0,-(W-1):,:3] before={dx[0, -(W-1):, :3]}")

            # 同步通信：发送 d_initial_state 给前一个 rank，接收后一个 rank 的 d_initial_state
            recv_d_init = cp_send_recv_bwd(d_initial_state, group)  # [W-1, D]

            print(f"[Rank {rank}] Backward: recv_d_init[:,:3]={recv_d_init[:, :3]}")

            # 加到当前 rank 的最后 W-1 个 token（这些 token 是被后一个 rank 用作 initial_state 的）
            dx[0, -(W-1):, :] = dx[0, -(W-1):, :] + recv_d_init

            print(f"[Rank {rank}] Backward: dx[0,-(W-1):,:3] after={dx[0, -(W-1):, :3]}")

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
    Context Parallel 版本的 causal_conv1d

    在 CP 环境下自动处理通信：
    - Forward: 从前一个 rank 获取 initial_state
    - Backward: 修正 dx 梯度
    """
    return CausalConv1dFunctionCP.apply(
        x, weight, bias, activation,
        cu_seqlens, cu_seqlens_cpu, chunk_indices
    )
