import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def chunk_cumprod_householder_fwd_kernel(
    k,
    k_new,
    w1,
    w2,
    hc_suffix,
    hc_whole,
    cu_seqlens,
    split_indices,
    chunk_offsets,
    split_offsets,
    BT: tl.constexpr,  # small chunk size
    K: tl.constexpr,
    H: tl.constexpr,
    BK: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,  # split size, aka large chunk size
    IS_VARLEN: tl.constexpr,
):
    i_ss, i_h = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n, i_s = tl.load(split_indices + i_ss * 2).to(tl.int32), tl.load(split_indices + i_ss * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NS = tl.cdiv(T, S)

        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
        boh_large = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        NS = tl.cdiv(T, S)
        i_n, i_s = i_ss // NS, i_ss % NS
        bos, eos = i_n * T, i_n * T + T

        boh = i_n * tl.cdiv(T, BT)
        boh_large = i_n * tl.cdiv(T, S)

    NT_small = tl.cdiv(min(S, T-i_s*S), BT)
    stride_h = H*K*K

    # offset calculations
    hc_whole += ((boh_large + i_s) * H + i_h) * K * K
    hc_suffix += ((boh + tl.cdiv(i_s * S, BT)) * H + i_h) * K * K

    k += (bos * H + i_h) * K
    k_new += (bos * H + i_h) * K
    w1 += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K

    b_h = tl.zeros([BK, BK], dtype=tl.float32)
    for i_t_small in range(NT_small-1, -1, -1):
        p_hc_suffix = tl.make_block_ptr(hc_suffix + i_t_small * stride_h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        tl.store(p_hc_suffix, b_h.to(hc_suffix.dtype.element_ty), boundary_check=(0, 1))
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_s * S + i_t_small * BT, 0), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k = (b_k - tl.dot(b_k, tl.trans(b_h.to(b_k.dtype))))
        p_w1 = tl.make_block_ptr(w1, (K, T), (1, H*K), (0, i_s * S + i_t_small * BT), (BK, BT), (0, 1))
        p_w2 = tl.make_block_ptr(w2, (T, K), (H*K, 1), (i_s * S + i_t_small * BT, 0), (BT, BK), (1, 0))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        b_v_new = (b_w1 - tl.dot(b_h.to(b_w1.dtype), b_w1)).to(b_w2.dtype)
        b_h += tl.dot(b_v_new, b_w2)
        p_k_new = tl.make_block_ptr(k_new, (T, K), (H*K, 1), (i_s * S + i_t_small * BT, 0), (BT, BK), (1, 0))
        tl.store(p_k_new, b_k.to(k_new.dtype.element_ty), boundary_check=(0, 1))

    p_hc_whole = tl.make_block_ptr(hc_whole, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    tl.store(p_hc_whole, b_h.to(hc_whole.dtype.element_ty), boundary_check=(0, 1))


def chunk_cumprod_householder_fwd_fn(
    k: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    S: int,  # split size, aka large chunk size
    BT: int,  # small chunk size
    cu_seqlens: torch.Tensor = None
):
    B, T, H, K = k.shape

    split_indices = prepare_chunk_indices(cu_seqlens, S) if cu_seqlens is not None else None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT) if cu_seqlens is not None else None
    split_offsets = prepare_chunk_offsets(cu_seqlens, S) if cu_seqlens is not None else None

    if cu_seqlens is None:
        N = B
        NS = N * triton.cdiv(T, S)
        NT = N * triton.cdiv(T, BT)
    else:
        N = len(cu_seqlens) - 1
        NS = split_offsets[-1]
        NT = chunk_offsets[-1]

    grid = (NS, H)
    hc_whole = torch.empty((NS, H, K, K), device=k.device, dtype=w1.dtype)
    k_new = torch.empty_like(k, dtype=k.dtype)
    hc_suffix = torch.empty((NT, H, K, K), device=k.device, dtype=w2.dtype)
    chunk_cumprod_householder_fwd_kernel[grid](
        k=k, k_new=k_new, w1=w1, w2=w2, hc_whole=hc_whole, hc_suffix=hc_suffix,
        cu_seqlens=cu_seqlens,
        split_indices=split_indices, chunk_offsets=chunk_offsets, split_offsets=split_offsets,
        BT=BT, K=K, H=H, BK=K,
        T=T, S=S,
        # SY (2025/07/08): I don't know why when K == 128 if I set num_warps=4 the result would be completely wrong
        num_warps=8 if K == 128 else 4,
    )
    return k_new, hc_suffix, hc_whole
