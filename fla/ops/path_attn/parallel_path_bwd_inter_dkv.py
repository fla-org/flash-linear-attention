import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_GATE': lambda args: args['g_cumsum'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_path_bwd_dkv_kernel(
    q, k, v, g_cumsum,
    hc_whole, scale, L, D,
    dk, dv, do, dg_cumsum,
    cu_seqlens, indices, split_offsets,
    T,
    G: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr,
    BV: tl.constexpr, S: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_GATE: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        boh_large = tl.load(split_offsets + i_n).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
        boh_large = i_n * tl.cdiv(T, S)

    # offset calculations
    q += (bos * HQ + i_hq) * K
    do += (bos * HQ + i_hq) * V
    dk += (bos * HQ + i_hq) * K
    dv += (bos * HQ + i_hq) * K
    L += (bos * HQ + i_hq)
    D += (bos * HQ + i_hq)

    k += (bos * H + i_h) * K  # GQA when H!=HQ
    v += (bos * H + i_h) * V  # GQA when H!=HQ
    hc_whole += (boh_large * H + i_h) * K * K

    if USE_GATE:
        g_cumsum += (bos * HQ + i_hq)
        dg_cumsum += (bos * HQ + i_hq)

    # constants
    stride_h = H * K * K
    sm_scale = scale * 1.44269504

    # load query
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_k_origin = tl.load(p_k, boundary_check=(0, 1))
    p_v = tl.make_block_ptr(v, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    if USE_GATE:
        b_g_cumsum_k = tl.zeros([BT,], dtype=tl.float32)
        p_g_cumsum_k = tl.make_block_ptr(g_cumsum, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
        b_g_cumsum_k += tl.load(p_g_cumsum_k, boundary_check=(0, ))
        b_dg_cumsum_k = tl.zeros([BT,], dtype=tl.float32)
    else:
        b_g_cumsum_k = None
        b_dg_cumsum_k = None

    b_dk = tl.zeros([BT, K], dtype=tl.float32)
    b_dv = tl.zeros([BT, K], dtype=tl.float32)
    idx_i = (i_t * BT // S).to(tl.int32)

    last_chunk_start = tl.floor(T/S).to(tl.int32) * S

    if i_t * BT < last_chunk_start:
        # handle right most hand
        if T % S != 0:
            idx_j = (last_chunk_start // S)
            b_k_accum = tl.zeros([BT, BK], dtype=tl.float32)
            b_k_accum += b_k_origin
            for i in range(idx_i+1, idx_j):
                p_h = tl.make_block_ptr(hc_whole + i * stride_h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
                b_h = tl.load(p_h, boundary_check=(0, 1))
                b_k_accum = (b_k_accum - tl.dot(b_k_accum.to(b_h.dtype), tl.trans(b_h)))
            b_k = b_k_accum.to(b_k_origin.dtype)

            for offset in range(tl.ceil(T/BS).to(tl.int32) * BS - BS, last_chunk_start-BS, -BS):
                p_delta = tl.make_block_ptr(D, (T, ), (HQ, ), (offset, ), (BS, ), (0, ))
                p_l = tl.make_block_ptr(L, (T, ), (HQ, ), (offset, ), (BS, ), (0, ))
                b_delta = tl.load(p_delta, boundary_check=(0, ))
                b_l = tl.load(p_l, boundary_check=(0, ))
                p_q = tl.make_block_ptr(q, (T, K), (HQ*K, 1), (offset, 0), (BS, BK), (1, 0))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_A = tl.dot(b_q, tl.trans(b_k))
                if USE_GATE:
                    p_g_cumsum_q = tl.make_block_ptr(g_cumsum, (T, ), (HQ, ), (offset, ), (BS, ), (0, ))
                    b_g_cumsum_q = tl.load(p_g_cumsum_q, boundary_check=(0, ))
                    b_A = b_A + b_g_cumsum_q[:, None] - b_g_cumsum_k[None, :]
                    b_A = tl.where((offset + tl.arange(0, BS) < T)[:, None], b_A, float("-inf"))  # avoid nan
                b_A_softmax = tl.math.exp2(b_A * sm_scale - b_l[:, None])
                p_do = tl.make_block_ptr(do, (T, V), (HQ*V, 1), (offset, 0), (BS, BV), (1, 0))
                b_do = tl.load(p_do, boundary_check=(0, 1))
                b_dv += tl.dot(tl.trans(b_A_softmax.to(b_do.dtype)), b_do)
                b_dp = tl.dot(b_do, tl.trans(b_v))
                b_dA = ((b_dp - b_delta[:, None]) * b_A_softmax * scale)
                if USE_GATE:
                    b_dg_cumsum_k -= tl.sum(b_dA, axis=0)
                b_dA = b_dA.to(b_v.dtype)
                b_dk += tl.dot(tl.trans(b_dA), b_q)

        for offset_outer in range(last_chunk_start, i_t * BT + S, -S):
            idx_j = (offset_outer // S) - 1
            b_k_accum = tl.zeros([BT, BK], dtype=tl.float32)
            b_k_accum += b_k_origin
            for i in range(idx_i+1, idx_j):
                p_h = tl.make_block_ptr(hc_whole + i * stride_h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
                b_h = tl.load(p_h, boundary_check=(0, 1))
                b_k_accum = (b_k_accum - tl.dot(b_k_accum.to(b_h.dtype), tl.trans(b_h)))
            b_k = b_k_accum.to(b_k_origin.dtype)

            p_h = tl.make_block_ptr(hc_whole + (idx_j) * stride_h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dk = b_dk - tl.dot(b_dk.to(b_h.dtype), b_h)

            for offset in range(offset_outer - BS, offset_outer-S-BS, -BS):
                p_delta = tl.make_block_ptr(D, (T, ), (HQ, ), (offset, ), (BS, ), (0, ))
                p_l = tl.make_block_ptr(L, (T, ), (HQ, ), (offset, ), (BS, ), (0, ))
                b_delta = tl.load(p_delta, boundary_check=(0, ))
                b_l = tl.load(p_l, boundary_check=(0, ))
                p_q = tl.make_block_ptr(q, (T, K), (HQ*K, 1), (offset, 0), (BS, BK), (1, 0))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_A = tl.dot(b_q, tl.trans(b_k))
                if USE_GATE:
                    p_g_cumsum_q = tl.make_block_ptr(g_cumsum, (T, ), (HQ, ), (offset, ), (BS, ), (0, ))
                    b_g_cumsum_q = tl.load(p_g_cumsum_q, boundary_check=(0, ))
                    b_A = b_A + b_g_cumsum_q[:, None] - b_g_cumsum_k[None, :]
                    b_A = tl.where((offset + tl.arange(0, BS) < T)[:, None], b_A, float("-inf"))  # avoid nan
                b_A_softmax = tl.math.exp2(b_A * sm_scale - b_l[:, None])
                p_do = tl.make_block_ptr(do, (T, V), (HQ*V, 1), (offset, 0), (BS, BV), (1, 0))
                b_do = tl.load(p_do, boundary_check=(0, 1))
                b_dv += tl.dot(tl.trans(b_A_softmax.to(b_do.dtype)), b_do)
                b_dp = tl.dot(b_do, tl.trans(b_v))

                b_dA = ((b_dp - b_delta[:, None]) * b_A_softmax * scale)
                if USE_GATE:
                    b_dg_cumsum_k -= tl.sum(b_dA, axis=0)
                b_dA = b_dA.to(b_v.dtype)
                b_dk += tl.dot(tl.trans(b_dA), b_q)

    p_dk = tl.make_block_ptr(dk, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dk, b_dk.to(dk.dtype.element_ty), boundary_check=(0, 1))
    tl.atomic_add(dv + (i_t * BT + tl.arange(0, BT))[:, None] * HQ*K + tl.arange(0, K)[None, :], b_dv, sem='relaxed')
    if USE_GATE:
        tl.atomic_add(dg_cumsum + (i_t * BT + tl.arange(0, BT)) * HQ, b_dg_cumsum_k, sem='relaxed')


def parallel_path_bwd_dkv_fn(
    q, k, v, g_cumsum, do, dv, dg_cumsum,
    hc_whole, scale, L, D,
    cu_seqlens,
    S, BT, BS
):
    B, T, HQ, K = q.shape
    V = v.shape[-1]
    H = k.shape[-2]
    G = HQ // H

    indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    split_offsets = prepare_chunk_offsets(cu_seqlens, S) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    # should be NS
    if cu_seqlens is not None:
        assert split_offsets[-1] == hc_whole.shape[0]
    dk = torch.empty_like(q, dtype=torch.float32)  # for later reduction use

    parallel_path_bwd_dkv_kernel[(NT, B*HQ)](
        q=q, k=k, v=v, g_cumsum=g_cumsum,
        hc_whole=hc_whole, scale=scale, L=L, D=D,
        dk=dk, dv=dv, do=do, dg_cumsum=dg_cumsum,
        cu_seqlens=cu_seqlens, indices=indices, split_offsets=split_offsets,
        T=T, S=S, BT=BT, BS=BS,
        G=G, HQ=HQ, H=H, K=K, V=V,
        BK=triton.next_power_of_2(K), BV=triton.next_power_of_2(V),
    )
    return dk, dv, dg_cumsum
