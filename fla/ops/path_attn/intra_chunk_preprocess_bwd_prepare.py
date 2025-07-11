import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets


@triton.heuristics({
    "USE_GATE": lambda args: args['g_cumsum'] is not None,
    "IS_VARLEN": lambda args: args['offsets'] is not None
})
@triton.jit(do_not_specialize=['T'])
def chunk_transform_qk_bwd_kernel_prepare(
    q,
    k,
    v,
    w,
    beta,
    g_cumsum,
    L,
    D,
    h,
    q_new,
    k_new,
    AT,
    dA_local,
    dv,
    do,
    dg_cumsum,
    scale,
    indices,  # varlen helper
    offsets,  # varlen helper
    chunk_offsets,  # varlen helper
    T,
    G: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATE: tl.constexpr,
    RETURN_H: tl.constexpr
):
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    sm_scale = scale * 1.44269504
    # offset calculations
    dA_local += (bos*HQ + i_hq) * BT
    AT += (bos*H + i_h) * BT
    q += (bos*HQ + i_hq) * K
    q_new += (bos*HQ + i_hq) * K
    k += (bos*H + i_h) * K
    k_new += (bos*H + i_h) * K
    w += (bos*H + i_h) * K
    v += (bos*H + i_h) * V
    do += (bos*HQ + i_hq) * V
    dv += (bos*HQ + i_hq) * V
    beta += (bos*H + i_h)
    if RETURN_H:
        h += ((boh + i_t) * H + i_h) * K * K
    else:
        h += (bos*H + i_h) * K
    if USE_GATE:
        g_cumsum += (bos*HQ + i_hq)
        dg_cumsum += (bos*HQ + i_hq)
    L += (bos*HQ + i_hq)
    D += (bos*HQ + i_hq)

    p_q = tl.make_block_ptr(q, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_t * BT), (BK, BT), (0, 1))
    p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_kt = tl.load(p_k, boundary_check=(0, 1))
    b_w = tl.load(p_w, boundary_check=(0, 1))
    p_T = tl.make_block_ptr(AT, (T, BT), (BT*H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_T = tl.load(p_T, boundary_check=(0, 1))

    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]
    p_beta = tl.make_block_ptr(beta, (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_w_beta = (b_w * b_beta[:, None]).to(b_w.dtype)

    b_Twb = tl.dot(b_T.to(b_w_beta.dtype), b_w_beta).to(b_w_beta.dtype)

    b_qw = tl.where(m_t, tl.dot(b_q, tl.trans(b_w)), 0).to(b_q.dtype)
    b_qwT = tl.dot(b_qw, b_T).to(b_q.dtype)
    b_wbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(b_w_beta, b_kt), 0).to(b_w.dtype)
    b_A = tl.where(m_t, tl.dot(b_q, b_kt) - tl.dot(b_qwT, b_wbk), 0)

    b_q = b_q - tl.dot(b_qwT, b_w_beta)
    p_q_new = tl.make_block_ptr(q_new, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, K), (1, 0))
    tl.store(p_q_new, b_q.to(p_q_new.dtype.element_ty), boundary_check=(0, 1))

    if i_hq % G == 0:
        if RETURN_H:
            b_h = tl.dot(tl.trans(b_w), b_Twb)
            p_h = tl.make_block_ptr(h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        else:
            p_h = tl.make_block_ptr(h, (T, K), (K * H, 1), (i_t * BT, 0), (BT, BK), (1, 0))
            tl.store(p_h, b_Twb.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        b_T_wbk = tl.dot(b_T, b_wbk).to(b_w.dtype)
        p_k_new = tl.make_block_ptr(k_new, (K, T), (1, K*H), (0, i_t * BT), (BK, BT), (0, 1))
        tl.store(p_k_new, (b_kt - tl.dot(tl.trans(b_w), b_T_wbk)).to(p_k_new.dtype.element_ty), boundary_check=(0, 1))

    if USE_GATE:
        p_g_cumsum = tl.make_block_ptr(g_cumsum, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
        b_g_cumsum = tl.load(p_g_cumsum, boundary_check=(0, ))
        b_A = b_A + (b_g_cumsum[:, None] - b_g_cumsum[None, :])
        b_A = tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], b_A, float("-inf"))  # avoid nan

    p_l = tl.make_block_ptr(L, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
    b_l = tl.load(p_l, boundary_check=(0, ))
    p_delta = tl.make_block_ptr(D, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
    delta = tl.load(p_delta, boundary_check=(0, ))

    b_A_softmax = tl.exp2(tl.where(o_i[:, None] >= o_i[None, :], b_A * sm_scale - b_l[:, None], float("-inf")))
    p_do = tl.make_block_ptr(do, (T, V), (HQ*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv = tl.dot(tl.trans(b_A_softmax.to(b_do.dtype)), b_do)
    p_dv = tl.make_block_ptr(dv, (T, V), (HQ*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    p_v = tl.make_block_ptr(v, (V, T), (1, H*V), (0, i_t * BT), (BV, BT), (0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dp = tl.dot(b_do, b_v)
    b_dA = ((b_dp - delta[:, None]) * b_A_softmax * scale)
    if USE_GATE:
        b_dgq = tl.sum(b_dA, axis=1) - tl.sum(b_dA, axis=0)
        p_dg = tl.make_block_ptr(dg_cumsum, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
        tl.store(p_dg, b_dgq.to(p_dg.dtype.element_ty), boundary_check=(0,))
    p_dA = tl.make_block_ptr(dA_local, (T, BT), (BT*HQ, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


def intra_chunk_preprocess_bwd_prepare_fn(q, k, v, w, beta, g_cumsum, A, L, D, do, scale, return_h=True, cu_seqlens=None):
    BT = A.shape[-1]
    HQ = q.shape[-2]
    B, T, H, K = k.shape
    G = HQ//H

    V = v.shape[-1]
    q_new = torch.empty_like(q)
    k_new = torch.empty_like(k)

    indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    grid = (NT, B*HQ)
    # better precision because h would be of norm smaller than 1 anyways
    if return_h:
        h = torch.empty(B, NT, H, K, K, dtype=w.dtype, device=q.device)
    else:
        h = torch.empty_like(w)
    dA_local = torch.empty(B, T, HQ, BT, dtype=w.dtype, device=q.device)
    dv = torch.empty(B, T, HQ, V, device=q.device, dtype=torch.float32)
    dg_cumsum = torch.empty_like(g_cumsum) if g_cumsum is not None else None

    chunk_transform_qk_bwd_kernel_prepare[grid](
        q=q,
        k=k,
        v=v,
        w=w,
        beta=beta,
        g_cumsum=g_cumsum,
        AT=A,
        dA_local=dA_local,
        dv=dv,
        dg_cumsum=dg_cumsum,
        do=do,
        L=L,
        D=D,
        h=h,
        q_new=q_new,
        k_new=k_new,
        scale=scale,
        offsets=cu_seqlens,
        indices=indices,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        G=G,
        HQ=HQ,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        BT=BT,
        RETURN_H=return_h
    )
    return q_new, k_new, h, dA_local, dv, dg_cumsum
