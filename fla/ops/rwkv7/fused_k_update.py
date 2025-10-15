

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.utils import autotune_cache_kwargs, get_multiprocessor_count, input_guard, is_amd

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [2, 4, 8, 16, 32]


@torch.jit.script
def k_update_ref(k: torch.Tensor, a: torch.Tensor, ka: torch.Tensor) -> torch.Tensor:
    return k.addcmul(k * (a - 1), ka)


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in NUM_WARPS_AUTOTUNE
        for s in [1, 2, 3]
    ],
    key=['BD'],
    **autotune_cache_kwargs,
)
@triton.jit
def k_update_fwd_kernel_short(
    k, a, ka, out,
    cu_seqlens,
    T, D,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_b).to(tl.int32)
        eos = tl.load(cu_seqlens + i_b + 1).to(tl.int32)
        g_t = bos + i_t
        if g_t >= eos:
            return
        offset = g_t * D
    else:
        g_t = i_t
        offset = i_b * T * D + g_t * D

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    off = offset + o_d

    b_k = tl.load(k + off, mask=m_d, other=0.).to(tl.float32)
    b_a = tl.load(a + off, mask=m_d, other=0.).to(tl.float32)
    b_ka = tl.load(ka + o_d, mask=m_d, eviction_policy='evict_last').to(tl.float32)

    out_val = b_k * (1 + (b_a - 1) * b_ka)
    tl.store(out + off, out_val.to(out.dtype.element_ty), mask=m_d)


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in NUM_WARPS_AUTOTUNE
        for s in [1, 2, 3]
    ],
    key=['BD', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit
def k_update_fwd_kernel_long(
    k, a, ka, out,
    cu_seqlens, chunk_indices,
    T, D,
    BD: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t_blk, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t_blk = tl.load(chunk_indices + i_t_blk * 2).to(tl.int32), \
            tl.load(chunk_indices + i_t_blk * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        t_start = i_t_blk * BT
        t_end = tl.minimum(t_start + BT, eos - bos)
    else:
        bos = i_b * T
        eos = (i_b + 1) * T
        t_start = i_t_blk * BT
        t_end = tl.minimum(t_start + BT, T)

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    for t in range(t_start, t_end):
        global_t = bos + t
        off = global_t * D + o_d
        b_k = tl.load(k + off, mask=m_d, other=0.).to(tl.float32)
        b_a = tl.load(a + off, mask=m_d, other=0.).to(tl.float32)
        b_ka = tl.load(ka + o_d, mask=m_d, eviction_policy='evict_last').to(tl.float32)
        out_val = b_k * (1 + (b_a - 1) * b_ka)
        tl.store(out + off, out_val.to(out.dtype.element_ty), mask=m_d)


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=w, num_stages=s)
        for w in NUM_WARPS_AUTOTUNE
        for s in [1, 2, 3]
        for BT in [2, 4, 8]
    ],
    key=['BD'],
    **autotune_cache_kwargs,
)
@triton.jit
def k_update_bwd_kernel_short(
    grad_out, k, a, ka,
    dk, da, dka,
    cu_seqlens,
    T, D,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_b, i_t_base = tl.program_id(0), tl.program_id(1) * BT

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_b).to(tl.int32)
        eos = tl.load(cu_seqlens + i_b + 1).to(tl.int32)
        seq_len = eos - bos
    else:
        bos = i_b * T
        eos = (i_b + 1) * T
        seq_len = T

    t_vec = i_t_base + tl.arange(0, BT)
    mask_t = t_vec < seq_len
    global_t_vec = bos + t_vec

    o_d = tl.arange(0, BD)[None, :]
    m_d = o_d < D
    off = global_t_vec[:, None] * D + o_d

    b_go = tl.load(grad_out + off, mask=mask_t[:, None] & m_d, other=0.).to(tl.float32)
    b_k = tl.load(k + off, mask=mask_t[:, None] & m_d, other=0.).to(tl.float32)
    b_a = tl.load(a + off, mask=mask_t[:, None] & m_d, other=0.).to(tl.float32)
    b_ka = tl.load(ka + o_d, mask=m_d, eviction_policy='evict_last').to(tl.float32)  # [1, BD]

    dk_vec = b_go * (1 + (b_a - 1) * b_ka)
    da_vec = b_go * b_k * b_ka
    dka_vec = b_go * b_k * (b_a - 1)
    tl.store(dk + off, dk_vec.to(dk.dtype.element_ty), mask=mask_t[:, None] & m_d)
    tl.store(da + off, da_vec.to(da.dtype.element_ty), mask=mask_t[:, None] & m_d)
    tl.store(dka + off, dka_vec.to(dka.dtype.element_ty), mask=mask_t[:, None] & m_d)


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in NUM_WARPS_AUTOTUNE
        for s in [1, 2, 3]
    ],
    key=['BD', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit
def k_update_bwd_kernel_long(
    grad_out, k, a, ka,
    dk, da, dka,
    cu_seqlens, chunk_indices,
    T, D,
    BD: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t_blk, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t_blk = tl.load(chunk_indices + i_t_blk * 2).to(tl.int32), \
            tl.load(chunk_indices + i_t_blk * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        t_start = i_t_blk * BT
        t_end = tl.minimum(t_start + BT, eos - bos)
    else:
        bos = i_b * T
        eos = (i_b + 1) * T
        t_start = i_t_blk * BT
        t_end = tl.minimum(t_start + BT, T)

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    for t in range(t_start, t_end):
        global_t = bos + t
        off = global_t * D + o_d

        b_go = tl.load(grad_out + off, mask=m_d, other=0.).to(tl.float32)
        b_k = tl.load(k + off, mask=m_d, other=0.).to(tl.float32)
        b_a = tl.load(a + off, mask=m_d, other=0.).to(tl.float32)
        b_ka = tl.load(ka + o_d, mask=m_d, eviction_policy='evict_last').to(tl.float32)

        tl.store(dk + off, (b_go * (1 + (b_a - 1) * b_ka)).to(dk.dtype.element_ty), mask=m_d)
        tl.store(da + off, (b_go * b_k * b_ka).to(da.dtype.element_ty), mask=m_d)
        tl.store(dka + off, (b_go * b_k * (b_a - 1)).to(dka.dtype.element_ty), mask=m_d)


def k_update_fwd(
    k: torch.Tensor,
    a: torch.Tensor,
    ka: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    B, T, D = k.shape
    out = torch.empty_like(k)
    use_short = T <= 512

    if use_short:
        if cu_seqlens is not None:
            N = len(cu_seqlens) - 1
        else:
            N = B
        BD = triton.next_power_of_2(D)
        grid = (N, T)
        k_update_fwd_kernel_short[grid](
            k, a, ka, out,
            cu_seqlens,
            T, D,
            BD=BD,
        )
    else:
        BT = min(64, triton.next_power_of_2(
            triton.cdiv(max(16, B * T), get_multiprocessor_count(k.device.index)),
        ))
        if cu_seqlens is not None:
            chunk_idx = prepare_chunk_indices(cu_seqlens, BT)
            NT = len(chunk_idx)
            N = len(cu_seqlens) - 1
        else:
            chunk_idx = None
            NT = triton.cdiv(T, BT)
            N = B

        BD = triton.next_power_of_2(D)

        def grid(meta):
            return (triton.cdiv(D, meta['BD']), NT, N)

        k_update_fwd_kernel_long[grid](
            k, a, ka, out,
            cu_seqlens, chunk_idx,
            T, D,
            BD=BD, BT=BT,
        )

    return out, use_short, N, T


def k_update_bwd(
    grad_out: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    ka: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    use_short: bool,
    N: int,
    T: int,
):
    B, _, D = grad_out.shape
    dk = torch.empty_like(k)
    da = torch.empty_like(a)
    dka_tmp = torch.empty_like(k, dtype=torch.float32)

    if use_short:
        BD = triton.next_power_of_2(D)
        def grid(meta): return (N, triton.cdiv(T, meta['BT']))
        k_update_bwd_kernel_short[grid](
            grad_out, k, a, ka,
            dk, da, dka_tmp,
            cu_seqlens,
            T, D,
            BD=BD,
        )
    else:
        BT = min(64, triton.next_power_of_2(
            triton.cdiv(max(16, B * T), get_multiprocessor_count(grad_out.device.index)),
        ))
        if cu_seqlens is not None:
            chunk_idx = prepare_chunk_indices(cu_seqlens, BT)
            NT = len(chunk_idx)
        else:
            chunk_idx = None
            NT = triton.cdiv(T, BT)

        BD = triton.next_power_of_2(D)

        def grid(meta):
            return (triton.cdiv(D, meta['BD']), NT, N)

        k_update_bwd_kernel_long[grid](
            grad_out, k, a, ka,
            dk, da, dka_tmp,
            cu_seqlens, chunk_idx,
            T, D,
            BD=BD, BT=BT,
        )

    if dka_tmp.dim() == 3:
        dka = dka_tmp.sum(dim=(0, 1), keepdim=True).type_as(ka)
    else:
        dka = dka_tmp.sum(dim=(0, 1)).type_as(ka)

    return dk, da, dka


class KUpdateFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(ctx, k, a, ka, cu_seqlens=None):
        out, use_short, N, T = k_update_fwd(k, a, ka, cu_seqlens)
        ctx.save_for_backward(k, a, ka)
        ctx.use_short = use_short
        ctx.N = N
        ctx.T = T
        ctx.cu_seqlens = cu_seqlens
        return out

    @staticmethod
    @input_guard
    def backward(ctx, grad_output):
        k, a, ka = ctx.saved_tensors
        dk, da, dka = k_update_bwd(
            grad_output, k, a, ka,
            ctx.cu_seqlens,
            ctx.use_short,
            ctx.N,
            ctx.T,
        )
        return dk, da, dka, None


def fused_k_rwkv7(k, a, ka, cu_seqlens=None):
    if k.shape[1] == 1:
        return k_update_ref(k, a, ka)
    return KUpdateFunction.apply(k, a, ka, cu_seqlens)
