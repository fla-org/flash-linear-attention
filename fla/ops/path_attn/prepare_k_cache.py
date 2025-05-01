import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets


@triton.heuristics({
    'IS_VARLEN': lambda args: args['offsets'] is not None
})
@triton.jit(do_not_specialize=['T'])
def parallel_path_fwd_kernel_prepare_k_cache(
    k, k_new, h,
    offsets, indices, chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
        NT = triton.cdiv(T, BT)
        boh = i_n * NT

    # offset calculations
    k += (bos * H + i_h) * K  # GQA when H!=HQ
    k_new += (bos * H + i_h) * K  # GQA when H!=HQ
    h += (boh * H + i_h) * K * K
    # constants
    stride_h = H * K * K
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_k = tl.zeros([BT, BK], dtype=tl.float32)
    b_k += tl.load(p_k, boundary_check=(0, 1))
    for k_block_idx in range(i_t + 1, tl.cdiv(T, BT)):
        p_h = tl.make_block_ptr(h + k_block_idx * stride_h, (K, K), (1, K), (0, 0), (BK, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_k_minus = tl.dot(b_k.to(b_h.dtype), b_h)
        b_k = b_k - b_k_minus
    p_k_new = tl.make_block_ptr(k_new, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_k_new, b_k.to(p_k_new.dtype.element_ty), boundary_check=(0, 1))


def prepare_k_cache_fn(k, h, cu_seqlens, BS, use_cache=False):
    if not use_cache:
        return None
    else:
        B, T, H, K = k.shape
        k_new = torch.empty_like(k)
        indices = prepare_chunk_indices(cu_seqlens, BS) if cu_seqlens is not None else None
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BS) if cu_seqlens is not None else None
        NT = triton.cdiv(T, BS) if cu_seqlens is None else len(indices)
        grid = (NT, B * H)
        parallel_path_fwd_kernel_prepare_k_cache[grid](
            k=k,
            k_new=k_new,
            h=h,
            offsets=cu_seqlens,
            indices=indices,
            chunk_offsets=chunk_offsets,
            H=H,
            T=T,
            K=K,
            BT=BS,
            BK=triton.next_power_of_2(K)
        )
        return k_new
