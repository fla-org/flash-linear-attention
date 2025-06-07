import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def chunk_cumprod_householder_fwd_kernel(
    q,
    q_new,
    k,
    k_new,
    h,
    hc_suffix,
    hc_prefix,
    hc_whole,
    cu_seqlens,
    split_indices,
    chunk_offsets,
    split_offsets,
    BT: tl.constexpr,  # small chunk size
    K: tl.constexpr,
    G: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    BK: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,  # split size, aka large chunk size
    IS_VARLEN: tl.constexpr,
):
    i_ss, i_hq = tl.program_id(0), tl.program_id(1)
    i_h = i_hq // G

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
    h += ((boh + tl.cdiv(i_s * S, BT)) * H + i_h) * K * K
    hc_suffix += ((boh + tl.cdiv(i_s * S, BT)) * H + i_h) * K * K
    hc_prefix += ((boh + tl.cdiv(i_s * S, BT)) * H + i_h) * K * K
    hc_whole += ((boh_large + i_s) * H + i_h) * K * K

    q += (bos * HQ + i_hq) * K
    q_new += (bos * HQ + i_hq) * K
    k += (bos * H + i_h) * K
    k_new += (bos * H + i_h) * K

    # Initialize h and load first chunk
    p_h = tl.make_block_ptr(h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    b_h = tl.zeros([BK, BK], dtype=tl.float32)
    b_h += tl.load(p_h, boundary_check=(0, 1))
    # Load and store first q chunk
    p_q = tl.make_block_ptr(q, (T, K), (HQ*K, 1), (i_s * S, 0), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    p_q_new = tl.make_block_ptr(q_new, (T, K), (HQ*K, 1), (i_s * S, 0), (BT, BK), (1, 0))
    tl.store(p_q_new, b_q.to(q_new.dtype.element_ty), boundary_check=(0, 1))

    p_hc_prefix = tl.make_block_ptr(hc_prefix, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    tl.store(p_hc_prefix, tl.zeros([BK, BK], dtype=tl.float32).to(p_hc_prefix.dtype.element_ty), boundary_check=(0, 1))

    # Process remaining chunks

    for i_t_small in range(1, NT_small):
        p_q = tl.make_block_ptr(q, (T, K), (HQ*K, 1), (i_s * S + i_t_small * BT, 0), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q - tl.dot(b_q, b_h.to(b_q.dtype))).to(b_q.dtype)
        p_q_new = tl.make_block_ptr(q_new, (T, K), (HQ*K, 1),
                                    (i_s * S + i_t_small * BT, 0), (BT, BK), (1, 0))
        tl.store(p_q_new, b_q.to(q_new.dtype.element_ty), boundary_check=(0, 1))
        if HQ % G == 0:
            p_hc_prefix = tl.make_block_ptr(hc_prefix + i_t_small * stride_h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
            tl.store(p_hc_prefix, b_h.to(hc_prefix.dtype.element_ty), boundary_check=(0, 1))
        p_h_new = tl.make_block_ptr(h + i_t_small * stride_h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        b_h_new = tl.load(p_h_new, boundary_check=(0, 1))
        b_h = b_h + b_h_new - tl.dot(b_h_new, b_h.to(b_h_new.dtype))

    tl.debug_barrier()

    if HQ % G == 0:
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_s * S + (NT_small - 1) * BT, 0), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        p_k_new = tl.make_block_ptr(k_new, (T, K), (H*K, 1), (i_s * S + (NT_small - 1) * BT, 0), (BT, BK), (1, 0))
        tl.store(p_k_new, b_k.to(k_new.dtype.element_ty), boundary_check=(0, 1))
        p_hc_suffix = tl.make_block_ptr(hc_suffix + (NT_small - 1) * stride_h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        tl.store(p_hc_suffix, tl.zeros([BK, BK], dtype=tl.float32).to(p_hc_suffix.dtype.element_ty), boundary_check=(0, 1))

        p_h = tl.make_block_ptr(h + (NT_small - 1) * stride_h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        b_h = tl.zeros([BK, BK], dtype=tl.float32)
        b_h += tl.load(p_h, boundary_check=(0, 1))

        for i_t_small in range(NT_small-2, -1, -1):
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_s * S + i_t_small * BT, 0), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_k = (b_k - tl.dot(b_k, tl.trans(b_h).to(b_k.dtype))).to(b_k.dtype)
            p_k_new = tl.make_block_ptr(k_new, (T, K), (H*K, 1), (i_s * S + i_t_small * BT, 0), (BT, BK), (1, 0))
            tl.store(p_k_new, b_k.to(k_new.dtype.element_ty), boundary_check=(0, 1))
            p_hc_suffix = tl.make_block_ptr(hc_suffix + i_t_small * stride_h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
            tl.store(p_hc_suffix, b_h.to(hc_suffix.dtype.element_ty), boundary_check=(0, 1))
            p_h_new = tl.make_block_ptr(h + i_t_small * stride_h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
            b_h_new = tl.load(p_h_new, boundary_check=(0, 1))
            b_h = b_h + b_h_new - tl.dot(b_h.to(b_h_new.dtype), b_h_new)

        p_hc_whole = tl.make_block_ptr(hc_whole, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        tl.store(p_hc_whole, b_h.to(hc_whole.dtype.element_ty), boundary_check=(0, 1))


def chunk_cumprod_householder_fwd_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    h: torch.Tensor,
    S: int,  # split size, aka large chunk size
    BT: int,  # small chunk size
    cu_seqlens: torch.Tensor = None,
):
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H

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

    grid = (NS, HQ)

    hc_suffix = torch.empty((NT, H, K, K), device=q.device, dtype=q.dtype)
    hc_prefix = torch.empty((NT, H, K, K), device=q.device, dtype=q.dtype)
    hc_whole = torch.empty((NS, H, K, K), device=q.device, dtype=q.dtype)
    q_new = torch.empty_like(q)
    k_new = torch.empty_like(k)

    chunk_cumprod_householder_fwd_kernel[grid](
        q=q, q_new=q_new, k=k, k_new=k_new, h=h, hc_suffix=hc_suffix, hc_prefix=hc_prefix, hc_whole=hc_whole,
        cu_seqlens=cu_seqlens,
        split_indices=split_indices, chunk_offsets=chunk_offsets, split_offsets=split_offsets,
        BT=BT, K=K, G=G, H=H, HQ=HQ, BK=K,
        T=T, S=S
    )
    return q_new, k_new, hc_suffix, hc_prefix, hc_whole
