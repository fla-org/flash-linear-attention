import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_cumprod_householder_bwd_kernel(
    hc_suffix, dhc_whole,
    k, dk, w1, w2, dw1, dw2, dk_new,
    cu_seqlens, split_indices, chunk_offsets, split_offsets,
    BT: tl.constexpr,  # previous small chunk size
    K: tl.constexpr,
    BK: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    G: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
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

    # offset calculations
    dhc_whole += ((boh_large + i_s) * HQ + i_hq) * K * K
    hc_suffix += ((boh + tl.cdiv(i_s * S, BT)) * H + i_h) * K * K
    k += (bos * H + i_h) * K
    w1 += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K
    dw1 += (bos * HQ + i_hq) * K
    dw2 += (bos * HQ + i_hq) * K

    # dh += ((boh + tl.cdiv(i_s * S, BT)) * HQ + i_hq) * K * K
    dk += (bos * HQ + i_hq) * K
    dk_new += (bos * HQ + i_hq) * K

    stride_h = H * K * K
    NT_small = tl.cdiv(min(S, T-i_s*S), BT)
    p_dhc_whole = tl.make_block_ptr(dhc_whole, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    b_dhc = tl.zeros([BK, BK], dtype=tl.float32)
    b_dhc += tl.load(p_dhc_whole, boundary_check=(0, 1))

    # calculate dh
    for i_t_small in range(0, NT_small):
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (HQ*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))
        p_dk_new = tl.make_block_ptr(dk_new, (T, K), (HQ*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))
        p_hc = tl.make_block_ptr(hc_suffix + i_t_small * stride_h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dk = tl.load(p_dk, boundary_check=(0, 1))

        p_w1 = tl.make_block_ptr(w1, (T, K), (H*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))
        p_w2 = tl.make_block_ptr(w2, (T, K), (H*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))

        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        b_hc = tl.load(p_hc, boundary_check=(0, 1))

        b_dk_new = b_dk - tl.dot(b_dk.to(b_hc.dtype), b_hc)
        p_dk_new = tl.make_block_ptr(dk_new, (T, K), (HQ*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))
        tl.store(p_dk_new, b_dk_new.to(dk_new.dtype.element_ty), boundary_check=(0, 1))

        b_dh = b_dhc - tl.dot(tl.trans(b_hc), b_dhc.to(b_hc.dtype))
        b_dw2 = tl.dot(b_w1, b_dh.to(b_w1.dtype))
        b_dw1 = tl.dot(b_w2, tl.trans(b_dh.to(b_w2.dtype)))

        p_dw1 = tl.make_block_ptr(dw1, (T, K), (HQ*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))
        p_dw2 = tl.make_block_ptr(dw2, (T, K), (HQ*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))

        tl.store(p_dw1, b_dw1.to(dw1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dw2, b_dw2.to(dw2.dtype.element_ty), boundary_check=(0, 1))

        b_dhc = b_dhc - tl.dot(tl.dot(b_dhc.to(b_w2.dtype), tl.trans(b_w2)).to(b_w1.dtype), b_w1)
        b_dhc -= tl.dot(tl.trans(b_dk).to(b_k.dtype), b_k)


def chunk_cumprod_householder_bwd_fn(
    w1: torch.Tensor,
    w2: torch.Tensor,
    hc_suffix: torch.Tensor,
    dhc_whole: torch.Tensor,
    k: torch.Tensor,
    dk: torch.Tensor,
    S: int,  # split size, aka large chunk size
    BT: int,  # small chunk size
    cu_seqlens: torch.Tensor = None,
):
    B, T, HQ, K = dk.shape
    H = k.shape[2]
    G = HQ // H

    split_indices = prepare_chunk_indices(cu_seqlens, S) if cu_seqlens is not None else None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT) if cu_seqlens is not None else None
    split_offsets = prepare_chunk_offsets(cu_seqlens, S) if cu_seqlens is not None else None

    if cu_seqlens is None:
        N = B
        NS = N * triton.cdiv(T, S)
    else:
        N = len(cu_seqlens) - 1
        NS = split_offsets[-1].item()

    grid = (NS, HQ)
    dw1 = torch.empty_like(dk, dtype=torch.float32)
    dw2 = torch.empty_like(dk, dtype=torch.float32)
    dk_new = torch.empty_like(dk, dtype=torch.float32)

    chunk_cumprod_householder_bwd_kernel[grid](
        hc_suffix=hc_suffix, dhc_whole=dhc_whole,
        k=k, dk=dk, w1=w1, w2=w2, dw1=dw1, dw2=dw2, dk_new=dk_new,
        cu_seqlens=cu_seqlens,
        split_indices=split_indices, chunk_offsets=chunk_offsets, split_offsets=split_offsets,
        BT=BT, K=K, G=G, H=H, HQ=HQ, BK=K,
        T=T, S=S, num_stages=2,
        # SY (2025/07/08): I don't know why when K == 128 if I set num_warps=4 the result would be completely wrong
        num_warps=8 if K == 128 else 4
    )
    return dw1, dw2, dk_new
