
import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices


@triton.heuristics({
    "USE_G": lambda args: args['g_cumsum'] is not None,
    "IS_VARLEN": lambda args: args['offsets'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def intra_chunk_preprocess_fwd_kernel(
    q,
    k,
    v,
    w,
    beta,
    g_cumsum,
    o,
    A,
    L,
    M,
    w2,
    q_new,
    k_new,
    scale,
    indices,  # varlen helper
    offsets,  # varlen helper
    T,
    H: tl.constexpr,
    G: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    sm_scale = scale * 1.44269504
    # offset calculations
    A += (bos*H + i_h) * BT
    q += (bos*HQ + i_hq) * K
    q_new += (bos*HQ + i_hq) * K
    k += (bos*H + i_h) * K
    k_new += (bos*H + i_h) * K
    w2 += (bos*H + i_h) * K
    w += (bos*H + i_h) * K
    v += (bos*H + i_h) * V
    o += (bos*HQ + i_hq) * V
    beta += (bos*H + i_h)
    if USE_G:
        g_cumsum += (bos*HQ + i_hq)
    L += (bos*HQ + i_hq)
    M += (bos*HQ + i_hq)

    p_q = tl.make_block_ptr(q, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_t * BT), (BK, BT), (0, 1))
    p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_kt = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_w = tl.load(p_w, boundary_check=(0, 1))
    p_T = tl.make_block_ptr(A, (T, BT), (BT*H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_T = tl.load(p_T, boundary_check=(0, 1))

    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]
    p_beta = tl.make_block_ptr(beta, (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_w_beta = (b_w * b_beta[:, None])

    b_qw = tl.where(m_t, tl.dot(b_q, tl.trans(b_w)), 0).to(b_q.dtype)
    b_qwT = tl.dot(b_qw, b_T).to(b_q.dtype)
    b_wbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(b_w_beta, b_kt), 0).to(b_q.dtype)
    b_A = tl.where(m_t, tl.dot(b_q, b_kt) - tl.dot(b_qwT, b_wbk), 0)

    b_q = b_q - tl.dot(b_qwT, b_w_beta)
    p_q_new = tl.make_block_ptr(q_new, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, K), (1, 0))
    tl.store(p_q_new, b_q.to(p_q_new.dtype.element_ty), boundary_check=(0, 1))

    if i_hq % G == 0:
        b_Twb = tl.dot(b_T, b_w_beta).to(b_w.dtype)
        p_w2 = tl.make_block_ptr(w2, (T, K), (K*H, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        tl.store(p_w2, b_Twb, boundary_check=(0, 1))
        b_T_wbk = tl.dot(b_T, b_wbk).to(b_w.dtype)
        p_k_new = tl.make_block_ptr(k_new, (K, T), (1, K*H), (0, i_t * BT), (BK, BT), (0, 1))
        tl.store(p_k_new, (b_kt - tl.dot(tl.trans(b_w), b_T_wbk)).to(p_k_new.dtype.element_ty), boundary_check=(0, 1))

    if USE_G:
        p_g_cumsum = tl.make_block_ptr(g_cumsum, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
        b_g_cumsum = tl.load(p_g_cumsum, boundary_check=(0, ))
        b_A = b_A + (b_g_cumsum[:, None] - b_g_cumsum[None, :])
        b_A = tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], b_A, float("-inf"))  # avoid nan

    b_qkT_softmax = tl.where(o_i[:, None] >= o_i[None, :], b_A * sm_scale, float("-inf"))
    m_i = tl.max(b_qkT_softmax, 1)
    b_qkT_softmax = tl.math.exp2(b_qkT_softmax - m_i[:, None])
    l_i = tl.sum(b_qkT_softmax, 1)
    b_o = tl.dot(b_qkT_softmax.to(b_v.dtype), b_v)
    p_o = tl.make_block_ptr(o, (T, V), (V*HQ, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    p_l = tl.make_block_ptr(L, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
    p_m = tl.make_block_ptr(M, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
    tl.store(p_m, m_i.to(p_m.dtype.element_ty), boundary_check=(0,))
    tl.store(p_l, l_i.to(p_l.dtype.element_ty), boundary_check=(0,))


def intra_chunk_preprocess_fwd_fn(q, k, v, w, beta, g_cumsum, A, scale, BT, cu_seqlens):
    HQ = q.shape[-2]
    B, T, H, K = k.shape
    V = v.shape[-1]
    q_new = torch.empty_like(q)
    k_new = torch.empty_like(k)
    o = torch.empty(B, T, HQ, V, device=q.device)

    indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    grid = (NT, B*HQ)
    L = torch.empty(B, T, HQ, dtype=torch.float32, device=q.device)
    M = torch.empty(B, T, HQ, dtype=torch.float32, device=q.device)
    w2 = torch.empty_like(w)
    G = HQ//H

    intra_chunk_preprocess_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        w=w,
        beta=beta,
        g_cumsum=g_cumsum,
        o=o,
        A=A,
        L=L,
        M=M,
        w2=w2,
        q_new=q_new,
        k_new=k_new,
        scale=scale,
        offsets=cu_seqlens,
        indices=indices,
        T=T,
        H=H,
        G=G,
        HQ=HQ,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        BT=BT,
    )
    return q_new, k_new, w2, o, L, M
