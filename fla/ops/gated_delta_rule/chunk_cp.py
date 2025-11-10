import warnings
from typing import Optional

import torch
import torch.distributed as dist

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_fwd,
    chunk_gated_delta_rule_bwd,
)

class ChunkGatedDeltaRuleFunctionCP(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
        cp_rank: int = 0,
        cp_size: int = 1,
        cp_group = None,
    ):
        device = q.device
        B, T, H, K = q.shape
        B, T, H, V = v.shape

        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        # Handle initial state for CP
        if cp_size > 1:
            if cp_rank == 0:
                h0_local = initial_state.to(torch.float32) if initial_state is not None else torch.zeros(
                    (B, H, K, V), dtype=torch.float32, device=device
                )
            else:
                h0_local = torch.empty((B, H, K, V), dtype=torch.float32, device=device)
                recv_req = dist.irecv(h0_local, src=cp_rank - 1, group=cp_group)
                recv_req.wait()
        else:
            h0_local = initial_state.to(torch.float32) if initial_state is not None else torch.zeros(
                (B, H, K, V), dtype=torch.float32, device=device
            )

        # Force output_final_state=True for CP to get the final state
        force_output_final_state = True if cp_size > 1 else output_final_state

        # Use the received/initialized state as the initial state for this chunk
        g, o, A, final_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=h0_local,
            output_final_state=force_output_final_state,  # Force True for CP
            cu_seqlens=cu_seqlens,
        )

        # Send final state to next rank (only if we have next rank)
        if cp_size > 1 and cp_rank < cp_size - 1:
            if final_state is not None:
                send_req = dist.isend(final_state, dst=cp_rank + 1, group=cp_group)
                send_req.wait()  # Wait for send to complete

        # Save for backward 
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, h0_local)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cu_seqlens = cu_seqlens
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.cp_group = cp_group
        ctx.output_final_state = output_final_state  # Store original value
        ctx.initial_state_requires_grad = initial_state is not None and initial_state.requires_grad  # Track if gradient needed

        # Return final_state as None if original output_final_state was False
        return_final_state = final_state if output_final_state else None
        return o.to(q.dtype), return_final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, q_rstd, k, k_rstd, v, g, beta, A, h0_local = ctx.saved_tensors
        cp_rank, cp_size, cp_group = ctx.cp_rank, ctx.cp_size, ctx.cp_group
        cu_seqlens = ctx.cu_seqlens

        B, T, H, K = q.shape
        B, T, H, V = v.shape

        # Handle gradient of final state for CP
        if cp_size > 1:
            if cp_rank == cp_size - 1:
                # Last rank: use provided dht or zero
                if ctx.output_final_state and dht is not None:
                    dht_local = dht.to(torch.float32)
                else:
                    dht_local = torch.zeros((B, H, K, V), dtype=torch.float32, device=q.device)
            else:
                # Intermediate ranks: receive from next rank
                dht_local = torch.empty((B, H, K, V), dtype=torch.float32, device=q.device)
                # torch.cuda.synchronize()
                # dist.recv(dht_local, src=cp_rank + 1, group=cp_group)
                # torch.cuda.synchronize()

                recv_req = dist.irecv(dht_local, src=cp_rank + 1, group=cp_group)
                recv_req.wait()
        else:
            # Single rank case
            dht_local = dht.to(torch.float32) if dht is not None else torch.zeros(
                (B, H, K, V), dtype=torch.float32, device=q.device
            )

        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=h0_local,
            do=do,
            dht=dht_local,
            cu_seqlens=cu_seqlens,
        )

        # Send gradient of initial state to previous rank
        if cp_size > 1 and cp_rank > 0:
            if dh0 is not None:
                send_req = dist.isend(dh0, dst=cp_rank - 1, group=cp_group)
                send_req.wait()  # Wait for send to complete


        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
            
        # Only return gradient for initial_state if it required grad in the forward pass
        dh0_return = dh0 if ctx.initial_state_requires_grad else None
        
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0_return, None, None, None, None, None, None

@torch.compiler.disable
def chunk_gated_delta_rule_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    cp_rank: int = 0,
    cp_size: int = 1,
    cp_group = None,
):
    """
    Context Parallel version of chunk_gated_delta_rule
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."

    if head_first:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
        
    o, final_state = ChunkGatedDeltaRuleFunctionCP.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        cp_rank,
        cp_size,
        cp_group
    )
    return o, final_state
