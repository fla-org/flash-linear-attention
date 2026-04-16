# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified for QuasarAttention

import torch
import triton

from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.quasar.chunk_intra import chunk_quasar_fwd_intra
from fla.ops.quasar.gate import fused_quasar_gate, fast_quasar_alpha
from fla.utils import IS_AMD, autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, check_shared_mem, input_guard
from fla.ops.common.chunk_o import chunk_fwd_o, chunk_bwd_dv_local, chunk_bwd_dqkwg

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]
BT_LIST_AUTOTUNE = [32, 64, 128]
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [4, 8, 16, 32]


@input_guard
def chunk_quasar_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Kernelized chunk-wise QuasarAttention forward pass."""
    B, T, H, S = q.shape
    BT = chunk_size
    if BT != 64:
        raise ValueError("Only chunk_size=64 is currently supported in the kernelized Quasar chunk path")

    # Prepare chunk indices for varlen
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    # Quasar-specific per-token alpha
    # alpha[t] = (1 - exp(-beta * ||k_t||^2)) / (||k_t||^2 + eps)
    # beta is head-wise [H]
    
    # Ensure high precision for stability components
    k_f32 = k.float()
    k_norm_sq = (k_f32 * k_f32).sum(dim=-1)  # [B, T, H]
    
    # Aggressive clamping to prevent exp() instability
    k_norm_sq = torch.clamp(k_norm_sq, min=0.1, max=10.0)
    
    # Flexible beta shape: support head-wise [H] or token-wise [B, T, H]
    if beta.dim() == 1:
        beta_h = beta.view(1, 1, H).float()
    else:
        beta_h = beta.float()
    
    # Quasar-style decay computation with per-dim dt_bias
    # dt_bias is [H*K], we keep it full dimensional like Quasar does
    
    if A_log is not None:
        A = A_log.float().exp().view(1, 1, H, 1)  # [1, 1, H, 1] for broadcasting
    else:
        A = 1.0
    
    # Expand beta to [B, T, H, 1] to match key dim
    beta_expanded = beta_h.unsqueeze(-1)  # [B, T, H, 1]
    
    # Reshape dt_bias to [H, K] and add batch/time dims
    if dt_bias is not None:
        K = q.shape[-1]  # key dimension
        dt_bias_full = dt_bias.float().view(1, 1, H, K)  # [1, 1, H, K]
    else:
        dt_bias_full = 0.0
        K = q.shape[-1]
    
    # Expand k_norm_sq to [B, T, H, 1] for broadcasting
    k_norm_sq_expanded = k_norm_sq.unsqueeze(-1)  # [B, T, H, 1]
    
    # Compute Quasar-style gate per-dimension: -exp(A_log) * softplus(beta + dt_bias)
    g_quasar = -A * torch.nn.functional.softplus(beta_expanded + dt_bias_full)  # [B, T, H, K]
    
    # Convert to decay factor
    decay = torch.exp(g_quasar)  # [B, T, H, K]
    
    # Quasar alpha formula adapted per-dimension
    alpha = (1.0 - decay) / (k_norm_sq_expanded + 1e-6)  # [B, T, H, K]
    
    # For Quasar's kernel which expects beta_tok as [B, T, H], we take mean across K
    # This is a compromise - ideally the kernel would handle per-dim
    beta_tok = alpha.mean(dim=-1).to(dtype=q.dtype)  # [B, T, H]

    # Use a zero decay tensor to reuse kernels without additional gating.
    # Shape-compatible with log-space decay, but equals 0 -> exp(0)=1.
    g_zero = torch.zeros_like(q)

    scale = S ** -0.5

    # Intra-chunk: compute Aqk + Akk^{-1} representation and WY factors (w/u).
    w, u, qg, kg, Aqk, Akk = chunk_quasar_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g_zero,
        beta=beta_tok,  # FIXED: pass per-token alpha, not head-wise beta
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        chunk_indices=chunk_indices,
        safe_gate=False,
        disable_recompute=True,
        beta_out=beta_tok,  # Output is same as input for Quasar
    )

    # Recurrence (kernelized, no Python loop): produces per-chunk states h and updated values v_new.
    if initial_state is not None and initial_state.dtype != torch.float32:
        initial_state_f32 = initial_state.float()
    else:
        initial_state_f32 = initial_state

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        g=None,
        gk=None,
        initial_state=initial_state_f32,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )

    # Output (kernelized): o = q @ h + Aqk @ v_new (implemented via efficient SRAM standard kernel)
    o = chunk_fwd_o(
        q=q,
        k=kg,  # standard k was normalized, use scaled kg here
        v=v_new,
        h=h,
        g=None,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )

    return o, final_state


class ChunkQuasarFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ):
        chunk_size = 64
        chunk_indices = prepare_chunk_indices(
            cu_seqlens, chunk_size) if cu_seqlens is not None else None
        
        o, final_state = chunk_quasar_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
        )
        
        ctx.save_for_backward(q, k, v, beta, A_log, dt_bias, initial_state, cu_seqlens, chunk_indices)
        ctx.chunk_size = chunk_size
        ctx.output_final_state = output_final_state
        
        return o, final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, d_final_state: torch.Tensor | None):
        q, k, v, beta, A_log, dt_bias, initial_state, cu_seqlens, chunk_indices = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        
        # Recompute forward intermediates (simpler than saving all)
        B, T, H, S = q.shape
        
        # Recompute alpha
        eps = 1e-6
        k_norm_sq = (k.float() * k.float()).sum(dim=-1)  # [B, T, H]
        k_norm_sq = torch.clamp(k_norm_sq, min=0.1, max=10.0)
        
        if beta.dim() == 1:
            beta_h = beta.view(1, 1, H).to(k_norm_sq.dtype)
        else:
            beta_h = beta.to(k_norm_sq.dtype)
            
        beta_h = torch.clamp(beta_h, min=0.01, max=10.0)
        # Compute alpha with numerical stability
        exp_term = torch.exp(-beta_h * k_norm_sq)
        alpha = (1.0 - exp_term) / (k_norm_sq + eps)
        beta_tok = alpha.to(dtype=q.dtype)
        
        g_zero = torch.zeros_like(q)
        scale = S ** -0.5
        
        # Allocate beta_out for Quasar alpha computation
        beta_out = torch.empty_like(beta_tok)
        
        # Recompute forward intermediates
        w, u, qg, kg, Aqk, Akk = chunk_quasar_fwd_intra(
            q=q,
            k=k,
            v=v,
            gk=g_zero,
            beta=beta_tok,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            safe_gate=False,
            disable_recompute=True,
            beta_out=beta_out,
        )
        
        if initial_state is not None and initial_state.dtype != torch.float32:
            initial_state_f32 = initial_state.float()
        else:
            initial_state_f32 = initial_state
        
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u,
            g=None,
            gk=None,
            initial_state=initial_state_f32,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
        )
        # Backward: output kernel (dA, dv)
        from fla.ops.quasar.chunk_bwd import chunk_quasar_bwd_dAv
        dA, dv = chunk_quasar_bwd_dAv(
            q=q,
            k=k,
            v=v_new,
            do=do,
            A=Aqk,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        
        # Backward: recurrence (dh)
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
        dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu(
            q=q,
            k=kg,
            w=w,
            do=do,
            dv=dv,
            g=None,
            gk=None,
            h0=initial_state_f32,
            dht=None,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            use_exp2=True,
        )
        dv = dv2
        
        # Backward: WY recompute + intra (dq, dk, dbeta)
        from fla.ops.quasar.chunk_bwd import chunk_quasar_bwd_wy_dqkb_fused
        dq, dk, dv3, db, dA2 = chunk_quasar_bwd_wy_dqkb_fused(
            q=q,
            k=k,
            v=v,
            v_new=v_new,
            beta=beta_tok,
            A=Akk,
            h=h,
            do=do,
            dh=dh,
            dv=dv,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        
        # Combine gradients
        dv = dv + dv3
        dA = dA + dA2
        
        # Backward: alpha formula (dbeta from dk)
        # db is the gradient of Loss w.r.t alpha, shape [B, T, H]
        db_f32 = db.float()
        
        # Aggressive clamping for gradient stability
        k_norm_sq = torch.clamp(k_norm_sq, min=0.1, max=10.0)
        
        if beta.dim() == 1:
            beta_h = beta.view(1, 1, H).float()
            beta_h = torch.clamp(beta_h, min=0.01, max=10.0)
            # Chain rule: dL/dbeta_head = sum( dL/dalpha * dalpha/dbeta )
            dalpha_dbeta = k_norm_sq * exp_term / (k_norm_sq + eps)
            dbeta = (db_f32 * dalpha_dbeta).sum(dim=(0, 1)) / T
            dbeta = torch.clamp(dbeta, min=-1.0, max=1.0)
        else:
            beta_h = beta.float()
            beta_h = torch.clamp(beta_h, min=0.01, max=10.0)
            # Chain rule: dL/dbeta_token = dL/dalpha * dalpha/dbeta
            dalpha_dbeta = k_norm_sq * exp_term / (k_norm_sq + eps)
            dbeta = db_f32 * dalpha_dbeta
            # Token-wise gradient doesn't need / T normalization if it's fed to linear layer
            dbeta = torch.clamp(dbeta, min=-1.0, max=1.0)
        
        return dq, dk, dv, dbeta, None, None, None, None, None


@torch.compiler.disable
def chunk_quasar(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Chunk-wise QuasarAttention forward pass with autograd support.
    
    Args:
        q (torch.Tensor): Query tensor of shape [B, T, H, S]
        k (torch.Tensor): Key tensor of shape [B, T, H, S]
        v (torch.Tensor): Value tensor of shape [B, T, H, S]
        beta (torch.Tensor): Beta parameter tensor of shape [H]
        A_log (torch.Tensor | None): Learnable state decay, shape [H]
        dt_bias (torch.Tensor | None): Learnable time bias, shape [H*K]
        initial_state (torch.Tensor | None): Initial state tensor of shape [B, H, S, S]
        output_final_state (bool): Whether to output the final state
        cu_seqlens (torch.Tensor | None): Cumulative sequence lengths for variable-length sequences
    
    Returns:
        o (torch.Tensor): Output tensor of shape [B, T, H, S]
        final_state (torch.Tensor | None): Final state tensor of shape [B, H, S, S] if output_final_state
    """
    return ChunkQuasarFunction.apply(q, k, v, beta, A_log, dt_bias, initial_state, output_final_state, cu_seqlens)