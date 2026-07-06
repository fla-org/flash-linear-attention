# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Chunked ATK forward kernel for preconditioned delta rules.
K-tiled variant: processes head dimension in BK blocks.

ATK recurrence:
    A_t = exp(g) * A_{t-1} + beta * k^2

Preconditioner (symmetric fast squash):
    ell = log(A_t + eps)
    r = ell - log_atk_scale                # deviation from learned center
    s = r / (1 + |r|)                      # fast squash, bounded in (-1, 1)
    M = exp(-log(x) * s)                   # bounded multiplier in [1/x, x]
    k_precond = k * M
"""
import math
import os

import torch
import triton
import triton.language as tl


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def _forward_chunk_summary(
    k,                # *f32 [B, T, H, D]
    beta,             # *f32 [B, T, H]
    log_g,            # *f32 [B, T, H]
    a,                # *f32 [B, C, H, D]
    sa,               # *f32 [B, C, H]
    cu_seqlens,       # *i32 [N+1] - cumulative sequence lengths (None if not varlen)
    chunk_indices,    # *i32 [NT, 2] - (sequence_idx, chunk_idx) pairs (None if not varlen)
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    k_stride_b, k_stride_t, k_stride_h, k_stride_d,
    beta_stride_b, beta_stride_t, beta_stride_h,
    log_g_stride_b, log_g_stride_t, log_g_stride_h,
    a_stride_b, a_stride_c, a_stride_h, a_stride_d,
    sa_stride_b, sa_stride_c, sa_stride_h,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t = tl.program_id(0)
    h = tl.program_id(1)
    chunk_id = tl.program_id(2)

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        chunk_id = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        b = i_n
    else:
        b = tl.program_id(0)
        bos = 0
        eos = T

    if h >= H:
        return

    if chunk_id * CHUNK_LEN >= T:
        return

    T_range = chunk_id * CHUNK_LEN + tl.arange(0, CHUNK_LEN)
    mask_T = T_range < T

    if IS_VARLEN:
        beta_ptr = beta + (bos + T_range) * beta_stride_t + h * beta_stride_h
        log_g_ptr = log_g + (bos + T_range) * log_g_stride_t + h * log_g_stride_h
    else:
        beta_ptr = beta + b * beta_stride_b + T_range * beta_stride_t + h * beta_stride_h
        log_g_ptr = log_g + b * log_g_stride_b + T_range * log_g_stride_t + h * log_g_stride_h

    beta_val = tl.load(beta_ptr, mask=mask_T, other=0.0).to(tl.float32)
    log_g_val = tl.load(log_g_ptr, mask=mask_T, other=0.0).to(tl.float32)

    sa_val = tl.sum(log_g_val)
    decays = tl.exp(sa_val - tl.cumsum(log_g_val))

    for i_k in range(tl.cdiv(D, BK)):
        d_offset = i_k * BK
        D_range = d_offset + tl.arange(0, BK)
        mask_D = D_range < D

        if IS_VARLEN:
            k_ptr = k + (bos + T_range)[:, None] * k_stride_t + h * k_stride_h + D_range[None, :] * k_stride_d
        else:
            k_ptr = k + b * k_stride_b + T_range[:, None] * k_stride_t + h * k_stride_h + D_range[None, :] * k_stride_d

        k_val = tl.load(k_ptr, mask=mask_T[:, None] * mask_D[None, :], other=0.0).to(tl.float32)

        k_sq = k_val * k_val
        U = beta_val[:, None] * k_sq
        a_val = tl.sum(decays[:, None] * U, 0)

        a_ptr = a + b * a_stride_b + h * a_stride_h + chunk_id * a_stride_c + D_range * a_stride_d
        tl.store(a_ptr, a_val, mask=mask_D)

    sa_ptr = sa + b * sa_stride_b + h * sa_stride_h + chunk_id * sa_stride_c
    tl.store(sa_ptr, sa_val)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def _forward_pass_chunks(
    a,            # *f32 [B, C, H, D]
    sa,           # *f32 [B, C, H]
    ac,           # *f32 [B, C, H, D]
    h0,           # *f32 [N, H, D] or None
    cu_seqlens,   # *i32 [N+1]
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    a_stride_b, a_stride_c, a_stride_h, a_stride_d,
    sa_stride_b, sa_stride_c, sa_stride_h,
    ac_stride_b, ac_stride_c, ac_stride_h, ac_stride_d,
    BK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(0)

    if IS_VARLEN:
        i_n = i_nh // H
        h = i_nh % H
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        b = i_n
    else:
        b = i_nh // H
        h = i_nh % H
        if b >= B:
            return

    if h >= H:
        return

    N_chunks = tl.cdiv(T, CHUNK_LEN)

    for i_k in range(tl.cdiv(D, BK)):
        d_offset = i_k * BK
        D_range = d_offset + tl.arange(0, BK)
        D_mask = D_range < D

        sa_ptr = sa + b * sa_stride_b + h * sa_stride_h
        a_ptr = a + b * a_stride_b + h * a_stride_h + D_range * a_stride_d
        ac_ptr = ac + b * ac_stride_b + h * ac_stride_h + D_range * ac_stride_d

        if USE_INITIAL_STATE:
            ac_val = tl.load(h0 + i_nh * D + D_range, mask=D_mask, other=0).to(tl.float32)
        else:
            ac_val = tl.zeros([BK], dtype=tl.float32)

        for chunk_id in tl.range(N_chunks):
            a_val = tl.load(a_ptr, D_mask).to(tl.float32)
            sa_val = tl.load(sa_ptr).to(tl.float32)

            ac_val = tl.exp(sa_val) * ac_val + a_val

            tl.store(ac_ptr, ac_val, mask=D_mask)

            sa_ptr += sa_stride_c
            a_ptr += a_stride_c
            ac_ptr += ac_stride_c


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def _forward_chunk_out(
    k,                # *f32 [B, T, H, D]
    beta,             # *f32 [B, T, H]
    log_g,            # *f32 [B, T, H]
    ac,               # *f32 [B, C, H, D]
    h0,               # *f32 [N, H, D] or None
    k_precond,        # *f32 [B, T, H, D]
    log_atk_scale,    # *f32 [H] - per-head log-space center (learnable or fixed)
    logx,             # scalar float32 - log(x) for squash range
    eps,              # scalar float32 - epsilon for log safety
    cu_seqlens,
    chunk_indices,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    k_stride_b, k_stride_t, k_stride_h, k_stride_d,
    beta_stride_b, beta_stride_t, beta_stride_h,
    log_g_stride_b, log_g_stride_t, log_g_stride_h,
    ac_stride_b, ac_stride_c, ac_stride_h, ac_stride_d,
    kp_stride_b, kp_stride_t, kp_stride_h, kp_stride_d,
    BK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t = tl.program_id(0)
    h = tl.program_id(1)
    chunk_id = tl.program_id(2)

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        chunk_id = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        b = i_n
    else:
        b = tl.program_id(0)
        bos = 0
        eos = T

    if h >= H:
        return

    if chunk_id * CHUNK_LEN >= T:
        return

    T_range = chunk_id * CHUNK_LEN + tl.arange(0, CHUNK_LEN)
    mask_T = T_range < T
    C_range = tl.arange(0, CHUNK_LEN)

    if IS_VARLEN:
        beta_ptr = beta + (bos + T_range) * beta_stride_t + h * beta_stride_h
        log_g_ptr = log_g + (bos + T_range) * log_g_stride_t + h * log_g_stride_h
    else:
        beta_ptr = beta + b * beta_stride_b + T_range * beta_stride_t + h * beta_stride_h
        log_g_ptr = log_g + b * log_g_stride_b + h * log_g_stride_h + (log_g_stride_t * T_range)

    beta_val = tl.load(beta_ptr, mask=mask_T, other=0.0).to(tl.float32)
    log_g_val = tl.load(log_g_ptr, mask=mask_T, other=0.0).to(tl.float32)
    g_val = tl.exp(log_g_val)

    la_cumsum = tl.cumsum(log_g_val)

    roll_mat = (C_range[:, None] == (C_range[None, :] + 1)).to(tl.float32)
    la_cumsum_roll = tl.sum(roll_mat[:, :] * la_cumsum[None, :], 1)
    M = tl.exp(la_cumsum_roll[:, None] - la_cumsum[None, :])
    M = tl.where(C_range[:, None] > C_range[None, :], M, 0.0)

    base_decays = tl.exp(la_cumsum_roll * (C_range > 0).to(tl.float32))

    center = tl.load(log_atk_scale + h).to(tl.float32)

    for i_k in range(tl.cdiv(D, BK)):
        d_offset = i_k * BK
        D_range = d_offset + tl.arange(0, BK)
        mask_D = D_range < D

        if IS_VARLEN:
            k_ptr = k + (bos + T_range)[:, None] * k_stride_t + h * k_stride_h + D_range[None, :] * k_stride_d
            kp_ptr = k_precond + (bos + T_range)[:, None] * kp_stride_t + h * kp_stride_h + D_range[None, :] * kp_stride_d
        else:
            k_ptr = k + b * k_stride_b + h * k_stride_h + (k_stride_t * T_range)[:, None] + (k_stride_d) * D_range[None, :]
            kp_ptr = k_precond + b * kp_stride_b + h * kp_stride_h + \
                (kp_stride_t * T_range)[:, None] + (kp_stride_d) * D_range[None, :]

        k_val = tl.load(k_ptr, mask=mask_T[:, None] * mask_D[None, :], other=0.0).to(tl.float32)

        k_sq = k_val * k_val
        U = beta_val[:, None] * k_sq

        ac_ptr = ac + b * ac_stride_b + h * ac_stride_h + (chunk_id - 1) * ac_stride_c + D_range * ac_stride_d

        if chunk_id == 0:
            if USE_INITIAL_STATE:
                ac_val = tl.load(h0 + (b * H + h) * D + D_range, mask=mask_D, other=0).to(tl.float32)
            else:
                ac_val = tl.zeros([BK], dtype=tl.float32)
        else:
            ac_val = tl.load(ac_ptr, mask=mask_D)

        raw_state = base_decays[:, None] * ac_val[None, :] + tl.dot(M, U)

        A_t = g_val[:, None] * raw_state + U

        ell = tl.log(A_t + eps)
        r = ell - center
        s = r / (1.0 + tl.abs(r))
        M_precond = tl.exp(-logx * s)
        k_precond_val = k_val * M_precond

        tl.store(kp_ptr, k_precond_val, mask=mask_T[:, None] * mask_D[None, :])


def _atk_fwd_stages(
    k: torch.Tensor,
    beta: torch.Tensor,
    log_g: torch.Tensor,
    chunk_size: int,
    initial_A_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    x: float,
    eps: float,
    log_atk_scale: torch.Tensor | None,
):
    """Run the 3-stage ATK forward algorithm.

    Returns:
        k_precond, ac, a, sa, at
    """
    B, T, H, D = k.shape
    CHUNK_LEN = chunk_size

    BK = D if os.environ.get('ATK_NO_KTILE') else 32  # K-tile size (set ATK_NO_KTILE=1 to disable)

    is_varlen = cu_seqlens is not None

    if is_varlen:
        from fla.ops.utils import prepare_chunk_indices
        N = len(cu_seqlens) - 1
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_LEN)
        NT = len(chunk_indices)

        seq_lens = [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(N)]
        max_chunks = max(triton.cdiv(s.item(), CHUNK_LEN) for s in seq_lens)

        a = torch.empty(N, max_chunks, H, D, dtype=torch.float32, device=k.device)
        sa = torch.empty(N, max_chunks, H, dtype=torch.float32, device=k.device)
        ac = torch.empty(N, max_chunks, H, D, dtype=torch.float32, device=k.device)
    else:
        N = B
        num_chunks = math.ceil(T / CHUNK_LEN)
        chunk_indices = None
        NT = B * num_chunks

        a = torch.empty(B, num_chunks, H, D, dtype=torch.float32, device=k.device)
        sa = torch.empty(B, num_chunks, H, dtype=torch.float32, device=k.device)
        ac = torch.empty(B, num_chunks, H, D, dtype=torch.float32, device=k.device)

    k_precond = torch.empty_like(k)

    k = k.contiguous()
    beta = beta.contiguous()
    log_g = log_g.contiguous()

    if log_atk_scale is None:
        log_atk_scale = torch.full((H,), -0.2, dtype=torch.float32, device=k.device)
    else:
        log_atk_scale = log_atk_scale.contiguous()

    logx = math.log(x) if x > 0 else 0.0

    if is_varlen:
        grid = (NT, H, 1)
        grid2 = (N * H,)
    else:
        grid = (B, H, num_chunks)
        grid2 = (B * H,)

    _forward_chunk_summary[grid](
        k, beta, log_g, a, sa,
        cu_seqlens, chunk_indices,
        B, T, H, D, CHUNK_LEN,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        log_g.stride(0), log_g.stride(1), log_g.stride(2),
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        sa.stride(0), sa.stride(1), sa.stride(2),
        BK, num_warps=4
    )

    _forward_pass_chunks[grid2](
        a, sa, ac,
        initial_A_state,
        cu_seqlens,
        B if not is_varlen else N, T, H, D, CHUNK_LEN,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        sa.stride(0), sa.stride(1), sa.stride(2),
        ac.stride(0), ac.stride(1), ac.stride(2), ac.stride(3),
        BK, num_warps=4
    )

    _forward_chunk_out[grid](
        k, beta, log_g, ac, initial_A_state, k_precond,
        log_atk_scale, logx, eps,
        cu_seqlens, chunk_indices,
        B, T, H, D, CHUNK_LEN,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        log_g.stride(0), log_g.stride(1), log_g.stride(2),
        ac.stride(0), ac.stride(1), ac.stride(2), ac.stride(3),
        k_precond.stride(0), k_precond.stride(1), k_precond.stride(2), k_precond.stride(3),
        BK,
        num_warps=4
    )

    at = None
    if output_final_state:
        if is_varlen:
            seq_lens = [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(N)]
            last_chunk_indices = [triton.cdiv(s.item(), CHUNK_LEN) - 1 for s in seq_lens]
            at = torch.stack([ac[i, last_chunk_indices[i], :, :] for i in range(N)], dim=0).to(k.dtype)
        else:
            at = ac[:, -1, :, :].contiguous().to(k.dtype)

    return k_precond.to(k.dtype), ac, a, sa, at


@torch._dynamo.disable
def chunk_atk_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    log_g: torch.Tensor = None,
    chunk_size: int = 64,
    initial_A_state: torch.Tensor = None,
    output_final_state: bool = True,
    cu_seqlens: torch.Tensor = None,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
):
    r"""
    Chunked ATK forward: computes preconditioned keys via 3-stage chunk algorithm.
    """
    k_precond, _, _, _, at = _atk_fwd_stages(
        k, beta, log_g, chunk_size,
        initial_A_state, output_final_state, cu_seqlens,
        x, eps, log_atk_scale,
    )
    return k_precond, at


@torch._dynamo.disable
def recompute_atk_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    log_g: torch.Tensor,
    chunk_size: int = 64,
    initial_A_state: torch.Tensor = None,
    cu_seqlens: torch.Tensor = None,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
):
    r"""
    Recompute ATK forward intermediates for use in backward pass.
    """
    k_precond, ac, a, sa, _ = _atk_fwd_stages(
        k, beta, log_g, chunk_size,
        initial_A_state, False, cu_seqlens,
        x, eps, log_atk_scale,
    )
    return k_precond, ac, a, sa
