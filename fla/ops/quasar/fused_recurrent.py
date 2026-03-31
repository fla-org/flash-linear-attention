# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified for QuasarAttention

import torch
import triton
import triton.language as tl

from fla.utils import IS_AMD, autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, check_shared_mem, input_guard

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]
BT_LIST_AUTOTUNE = [32, 64, 128]
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [4, 8, 16, 32]


@triton.heuristics({
    'HAS_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'STORE_FINAL_STATE': lambda args: args['final_state'] is not None,
    'HAS_DT_BIAS': lambda args: args['dt_bias'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_quasar_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    o,
    initial_state,
    final_state,
    scale,
    T,
    H: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    
    # [BK, BV] fragment of the state
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    
    if HAS_INITIAL_STATE:
        p_h0 = initial_state + (i_b * H + i_h) * BK * BK + o_k[:, None] * BK + o_v[None, :]
        b_h += tl.load(p_h0).to(tl.float32)
        
    # Load Invariants Outside Loop
    b_beta_head = tl.load(beta + i_h).to(tl.float32)
    b_A = tl.load(A_log + i_h).to(tl.float32)
    b_exp_A = tl.exp(b_A)
    eps = 1e-8
    
    # Block Pointers for sequential loading
    p_q = tl.make_block_ptr(q + (i_b * T * H + i_h) * BK, (T, BK), (H * BK, 1), (0, 0), (1, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (i_b * T * H + i_h) * BK, (T, BK), (H * BK, 1), (0, 0), (1, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (i_b * T * H + i_h) * BK + i_v * BV, (T, BV), (H * BK, 1), (0, 0), (1, BV), (1, 0))
    p_g = tl.make_block_ptr(g + (i_b * T * H + i_h) * BK, (T, BK), (H * BK, 1), (0, 0), (1, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (i_b * T * H + i_h) * BK + i_v * BV, (T, BV), (H * BK, 1), (0, 0), (1, BV), (1, 0))

    for _ in range(0, T):
        # Load tokens for this step
        # [1, BK]
        b_q = tl.load(p_q).to(tl.float32)
        b_k = tl.load(p_k).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        # [1, BV]
        b_v = tl.load(p_v).to(tl.float32)
        
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            
        b_q *= scale
        
        # 1. CT Alpha Logic - Scalar reduction over BK
        b_k2 = tl.sum(b_k * b_k)
        # Use a more stable clamp to avoid NaN
        b_k2_stab = tl.maximum(b_k2, 0.05)
        b_alpha = (1.0 - tl.exp(-b_beta_head * b_k2_stab)) / (b_k2_stab + eps)
        
        # 2. Hybrid Forget Gate
        if HAS_DT_BIAS:
            b_bias = tl.load(dt_bias + i_h * BK + o_k).to(tl.float32)
            b_g += b_bias[None, :]
        
        # Softplus Gate Approximation
        b_gk = -b_exp_A * (tl.where(b_g > 20.0, b_g, tl.log1p(tl.exp(b_g))))
        
        # Apply Forget Gate to State
        # b_gk is [1, BK], b_h is [BK, BV]
        b_h *= tl.exp(b_gk[0, :, None])
        
        # 3. State Update (Rank-1 Delta Rule)
        # S_t = S_t_forgot + alpha * k @ (v - k^T @ S_t_forgot)^T
        # v_pred = k @ h -> [1, BV]
        b_v_pred = tl.dot(b_k, b_h)
        b_v_err = b_v - b_v_pred
        # Outer product: [BK, 1] @ [1, BV]
        b_h += (b_alpha * tl.trans(b_k)) @ b_v_err
        
        # 4. Output Projection
        # o = q @ h -> [1, BV]
        b_o = tl.dot(b_q, b_h)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty))
        
        # Advance pointers
        p_q = tl.advance(p_q, (1, 0))
        p_k = tl.advance(p_k, (1, 0))
        p_v = tl.advance(p_v, (1, 0))
        p_g = tl.advance(p_g, (1, 0))
        p_o = tl.advance(p_o, (1, 0))
        
    if STORE_FINAL_STATE:
        p_ht = final_state + (i_b * H + i_h) * BK * BK + o_k[:, None] * BK + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty))
        
    if STORE_FINAL_STATE:
        p_ht = final_state + (i_b * H + i_h) * BK * BK + o_k[:, None] * BK + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty))


@input_guard
def fused_recurrent_quasar_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, S = q.shape
    if scale is None:
        scale = S ** -0.5
        
    o = torch.empty_like(v)
    final_state = torch.empty(B, H, S, S, dtype=torch.float32, device=q.device) if output_final_state else None
    
    # Grid: (V_heads, B*H)
    # BV=64 often works better on A100/H100 if BK is small
    BV = 64 if S <= 64 else 32
    grid = (triton.cdiv(S, BV), B * H)
    fused_recurrent_quasar_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        initial_state=initial_state,
        final_state=final_state,
        scale=scale,
        T=T,
        H=H,
        BK=S,
        BV=BV,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=8,
        num_stages=4,
    )
    
    return o, final_state


class FusedRecurrentQuasarFunction(torch.autograd.Function):
    @staticmethod
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        scale: float | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs,
    ):
        o, final_state = fused_recurrent_quasar_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            output_final_state=output_final_state,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        return o, final_state

    @staticmethod
    def backward(ctx, do, dht):
        raise NotImplementedError("Backward pass for fused_recurrent_quasar is not implemented yet.")


@torch.compiler.disable
def fused_recurrent_quasar(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return FusedRecurrentQuasarFunction.apply(
        q, k, v, g, beta, A_log, dt_bias, initial_state, output_final_state, scale, use_qk_l2norm_in_kernel
    )