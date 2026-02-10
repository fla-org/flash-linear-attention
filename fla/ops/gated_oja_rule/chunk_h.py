
import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp
from fla.utils import check_shared_mem, is_nvidia_hopper, use_cuda_graph

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8, 16]


@triton.heuristics({
    'USE_GV': lambda args: args['gv'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_KEY': lambda args: args['k_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BK in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def chunk_oja_fwd_kernel_h_blockdim64(
    v,
    u,
    w,
    k_new,
    gv,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_KEY: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # (triton.cdiv(K, meta['BK']), N*H)
    i_k, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h1 = tl.zeros([BK, 64], dtype=tl.float32)
    if V > 64:
        b_h2 = tl.zeros([BK, 64], dtype=tl.float32)
    if V > 128:
        b_h3 = tl.zeros([BK, 64], dtype=tl.float32)
    if V > 192:
        b_h4 = tl.zeros([BK, 64], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * K*V).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    u += ((bos * H + i_h) * K).to(tl.int64)
    w += ((bos * H + i_h) * V).to(tl.int64)
    if SAVE_NEW_KEY:
        k_new += ((bos * H + i_h) * K).to(tl.int64)
    stride_v = H*V
    stride_h = H*K*V
    stride_k = H*K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (i_k * BK, 0), (BK, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if V > 64:
            p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (i_k * BK, 64), (BK, 64), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if V > 128:
            p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (i_k * BK, 128), (BK, 64), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if V > 192:
            p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (i_k * BK, 192), (BK, 64), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (i_k * BK, 0), (BK, 64), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if V > 64:
            p_h2 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (i_k * BK, 64), (BK, 64), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if V > 128:
            p_h3 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (i_k * BK, 128), (BK, 64), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if V > 192:
            p_h4 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (i_k * BK, 192), (BK, 64), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, V), (stride_v, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_k = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))  # BT BK
        if V > 64:
            p_w = tl.make_block_ptr(w, (T, V), (stride_v, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_k += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if V > 128:
            p_w = tl.make_block_ptr(w, (T, V), (stride_v, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_k += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if V > 192:
            p_w = tl.make_block_ptr(w, (T, V), (stride_v, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_k += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))

        p_u = tl.make_block_ptr(u, (T, K), (stride_k, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_u, boundary_check=(0, 1)) - b_k

        if SAVE_NEW_KEY:
            p_k = tl.make_block_ptr(k_new, (T, K), (stride_k, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            tl.store(p_k, b_k.to(p_k.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1

        if USE_GV:
            o_v1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gv + (bos + last_idx) * H*V + i_h * V + o_v1, mask=(o_v1 < V), other=0.)
            b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                o_v2 = 64 + o_v1
                b_gk_last2 = tl.load(gv + (bos + last_idx) * H*V + i_h * V + o_v2, mask=(o_v2 < V), other=0.)
                b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                o_v3 = 128 + o_v1
                b_gk_last3 = tl.load(gv + (bos + last_idx) * H*V + i_h * V + o_v3, mask=(o_v3 < V), other=0.)
                b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                o_v4 = 192 + o_v1
                b_gk_last4 = tl.load(gv + (bos + last_idx) * H*V + i_h * V + o_v4, mask=(o_v4 < K), other=0.)
                b_h4 *= exp(b_gk_last4)[None, :]

        b_k = b_k.to(v.dtype.element_ty)  # BT BK

        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))  # BT BV
        b_h1 += tl.dot(tl.trans(b_k), b_v)
        if V > 64:
            p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_h2 += tl.dot(tl.trans(b_k), b_v)
        if V > 128:
            p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_h3 += tl.dot(tl.trans(b_k), b_v)
        if V > 192:
            p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_h4 += tl.dot(tl.trans(b_k), b_v)
    # epilogue
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (i_k * BK, 0), (BK, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if V > 64:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (i_k * BK, 64), (BK, 64), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if V > 128:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (i_k * BK, 128), (BK, 64), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if V > 192:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (i_k * BK, 192), (BK, 64), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_oja_fwd_h(
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gv: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_key: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, V, K = *v.shape, u.shape[-1]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert V <= 256, "current kernel does not support head dimension larger than 256."

    h = v.new_empty(B, NT, H, K, V)
    final_state = v.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    k_new = torch.empty_like(u) if save_new_key else None
    def grid(meta): return (triton.cdiv(K, meta['BK']), N*H)
    chunk_oja_fwd_kernel_h_blockdim64[grid](
        v=v,
        u=u,
        w=w,
        k_new=k_new,
        gv=gv,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT
    )
    return h, k_new, final_state


@triton.heuristics({
    'USE_GV': lambda args: args['gv'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [4, 3, 2]
        for BK in [64, 32]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'USE_GV'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def chunk_oja_bwd_kernel_dhu_blockdim64(
    q,
    vg,
    w,
    gv,
    dht,
    dh0,
    do,
    dh,
    dk,
    dk2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_k, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_dh1 = tl.zeros([BK, 64], dtype=tl.float32)
    if V > 64:
        b_dh2 = tl.zeros([BK, 64], dtype=tl.float32)
    if V > 128:
        b_dh3 = tl.zeros([BK, 64], dtype=tl.float32)
    if V > 192:
        b_dh4 = tl.zeros([BK, 64], dtype=tl.float32)

    # calculate offset
    q += ((bos * H + i_h) * K).to(tl.int64)
    vg += ((bos * H + i_h) * V).to(tl.int64)
    w += ((bos * H + i_h) * V).to(tl.int64)
    do += ((bos * H + i_h) * V).to(tl.int64)
    dk += ((bos * H + i_h) * K).to(tl.int64)
    dk2 += ((bos * H + i_h) * K).to(tl.int64)
    dh += ((boh * H + i_h) * K*V).to(tl.int64)
    if USE_GV:
        gv += ((bos * H + i_h) * V).to(tl.int64)

    stride_v = H*V
    stride_h = H*K*V
    stride_k = H*K
    if USE_INITIAL_STATE:
        dh0 += i_nh * K*V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K*V

    if USE_FINAL_STATE_GRADIENT:
        p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (i_k * BK, 0), (BK, 64), (1, 0))  # [BK, BV]
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        if V > 64:
            p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (i_k * BK, 64), (BK, 64), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        if V > 128:
            p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (i_k * BK, 128), (BK, 64), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if V > 192:
            p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (i_k * BK, 192), (BK, 64), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        p_dh1 = tl.make_block_ptr(dh + i_t*stride_h, (K, V), (V, 1), (i_k * BK, 0), (BK, 64), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if V > 64:
            p_dh2 = tl.make_block_ptr(dh + i_t*stride_h, (K, V), (V, 1), (i_k * BK, 64), (BK, 64), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if V > 128:
            p_dh3 = tl.make_block_ptr(dh + i_t*stride_h, (K, V), (V, 1), (i_k * BK, 128), (BK, 64), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if V > 192:
            p_dh4 = tl.make_block_ptr(dh + i_t*stride_h, (K, V), (V, 1), (i_k * BK, 192), (BK, 64), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1

        # Update dk_new, 按K切分
        p_dk = tl.make_block_ptr(dk, (T, K), (stride_k, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))  # [BT, BK]
        p_dk2 = tl.make_block_ptr(dk2, (T, K), (stride_k, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))  # [BT, BK]

        if V > 0:
            p_v = tl.make_block_ptr(vg, (T, V), (stride_v, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))  # [BT, BV]
            b_dk = tl.dot(b_v, tl.trans(b_dh1).to(b_v.dtype))  # [BT, BV] @ [BV, BK] -> [BT, BK]

        if V > 64:
            p_v = tl.make_block_ptr(vg, (T, V), (stride_v, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_dk += tl.dot(b_v, tl.trans(b_dh2).to(b_v.dtype))

        if V > 128:
            p_v = tl.make_block_ptr(vg, (T, V), (stride_v, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_dk += tl.dot(b_v, tl.trans(b_dh3).to(b_v.dtype))

        if V > 192:
            p_v = tl.make_block_ptr(vg, (T, V), (stride_v, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_dk += tl.dot(b_v, tl.trans(b_dh4).to(b_v.dtype))

        b_dk += tl.load(p_dk, boundary_check=(0, 1))

        tl.store(p_dk2, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        # Update dh, 按照K切分，收集所有V维度，q一次就好，wdo要收集所有

        p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (i_k * BK, i_t * BT), (BK, BT), (0, 1))  # [BK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))

        if V > 0:
            p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, 0), (BT, 64), (1, 0))  # [BT, BV]
            b_do = tl.load(p_do, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, V), (stride_v, 1), (i_t * BT, 0), (BT, 64), (1, 0))  # [BT, BV]
            b_w = tl.load(p_w, boundary_check=(0, 1))
            p_gv = tl.make_block_ptr(gv, (T, V), (stride_v, 1), (i_t * BT, 0), (BT, 64), (1, 0))  # [BT, BV]
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            if USE_GV:
                o_v1 = tl.arange(0, 64)
                b_gv_last1 = tl.load(gv + last_idx * H*V + o_v1, mask=(o_v1 < V), other=0.)
                b_dh1 *= exp(b_gv_last1[None, :])
                b_do *= exp(b_gv)
            b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - \
                tl.dot(tl.trans(b_dk).to(b_w.dtype), b_w)  # [BK, BT] @ [BT, BV] - [BK, BT] @ [BT, BV]

        if V > 64:
            p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, V), (stride_v, 1), (i_t * BT, 64), (BT, 64), (1, 0))  # [BT, BV]
            b_w = tl.load(p_w, boundary_check=(0, 1))
            p_gv = tl.make_block_ptr(gv, (T, V), (stride_v, 1), (i_t * BT, 64), (BT, 64), (1, 0))  # [BT, BV]
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            if USE_GV:
                o_v2 = 64 + o_v1
                b_gv_last2 = tl.load(gv + last_idx * H*V + o_v2, mask=(o_v2 < V), other=0.)
                b_dh2 *= exp(b_gv_last2[None, :])
                b_do *= exp(b_gv)
            b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(tl.trans(b_dk).to(b_w.dtype), b_w)

        if V > 128:
            p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, V), (stride_v, 1), (i_t * BT, 128), (BT, 64), (1, 0))  # [BT, BV]
            b_w = tl.load(p_w, boundary_check=(0, 1))
            p_gv = tl.make_block_ptr(gv, (T, V), (stride_v, 1), (i_t * BT, 128), (BT, 64), (1, 0))  # [BT, BV]
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            if USE_GV:
                o_v3 = 128 + o_v1
                b_gv_last3 = tl.load(gv + last_idx * H*V + o_v3, mask=(o_v3 < V), other=0.)
                b_dh3 *= exp(b_gv_last3[None, :])
                b_do *= exp(b_gv)
            b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(tl.trans(b_dk).to(b_w.dtype), b_w)

        if V > 192:
            p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, V), (stride_v, 1), (i_t * BT, 192), (BT, 64), (1, 0))  # [BT, BV]
            b_w = tl.load(p_w, boundary_check=(0, 1))
            p_gv = tl.make_block_ptr(gv, (T, V), (stride_v, 1), (i_t * BT, 192), (BT, 64), (1, 0))  # [BT, BV]
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            if USE_GV:
                o_v4 = 192 + o_v1
                b_gv_last4 = tl.load(gv + last_idx * H*V + o_v4, mask=(o_v4 < V), other=0.)
                b_dh4 *= exp(b_gv_last4[None, :])
                b_do *= exp(b_gv)
            b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(tl.trans(b_dk).to(b_w.dtype), b_w)

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (i_k * BK, 0), (BK, 64), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if V > 64:
            p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (i_k * BK, 64), (BK, 64), (1, 0))
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if V > 128:
            p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (i_k * BK, 128), (BK, 64), (1, 0))
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if V > 192:
            p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (i_k * BK, 192), (BK, 64), (1, 0))
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))


def chunk_oja_bwd_dhu(
    q: torch.Tensor,
    vg: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dk: torch.Tensor,
    gv: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    states_in_fp32: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *q.shape, do.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = 64
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    dh = q.new_empty(B, NT, H, K, V, dtype=q.dtype if not states_in_fp32 else torch.float)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dk2 = torch.empty_like(dk)

    def grid(meta): return (triton.cdiv(K, meta['BK']), N*H)
    chunk_oja_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        vg=vg,
        w=w,
        gv=gv,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dk=dk,
        dk2=dk2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return dh, dh0, dk2


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['BT']
)
@triton.jit(do_not_specialize=['T'])
def chunk_gsa_bwd_k_kernel_dqkvg(
    q,
    k,
    v,
    h,
    g,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dg,
    dgv,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    o_i = tl.arange(0, BT)
    o_t = min(i_t * BT + BT, T)
    m_s = o_i[:, None] >= o_i[None, :]

    p_q = tl.make_block_ptr(q + (bos*HQ+i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos*H+i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_A = tl.make_block_ptr(A + ((i_k*all+bos)*HQ+i_hq)*BT, (T, BT), (HQ*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))

    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.dot((b_q * scale).to(b_q.dtype), tl.trans(b_k))
    b_A = tl.where(m_s, b_A, 0.)
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        p_v = tl.make_block_ptr(v + (bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_g = tl.make_block_ptr(g + (bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_gn = g + (bos + o_t - 1) * H*V + i_h * V + o_v
        p_do = tl.make_block_ptr(do + (bos*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + ((i_k*all+bos)*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dg = tl.make_block_ptr(dg + (bos*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dgv = tl.make_block_ptr(dgv+((i_k*all+bos)*HQ+i_hq)*V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K*V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh + (i_tg * HQ + i_hq) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        m_v = o_v < V

        # [BV,]
        b_gn = tl.load(p_gn, mask=m_v, other=0)
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_gv = exp(b_gn[None, :] - b_g)
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * exp(b_g) * scale).to(b_do.dtype)
        # [BK, BV]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        # [BV]
        b_dg = tl.sum(tl.trans(b_h) * b_dh, 0) * exp(b_gn)

        b_dh = b_dh.to(b_k.dtype)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_k.dtype))
        b_dk += tl.dot((b_v * b_gv).to(b_v.dtype), tl.trans(b_dh))
        # [BT, BV]
        b_dv = tl.dot(b_k, b_dh) * b_gv
        # [BV]
        b_dg += tl.sum(b_dv * b_v, 0)

        if i_k == 0:
            b_dgv = tl.load(p_dg, boundary_check=(0, 1)) + b_dg[None, :]
        else:
            b_dgv = tl.zeros([BT, BV], dtype=tl.float32) + b_dg[None, :]

        tl.store(p_dgv, b_dgv.to(p_dgv.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    p_dA = tl.make_block_ptr(dA + (bos*HQ + i_hq) * BT, (T, BT), (HQ*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_dq = tl.make_block_ptr(dq + (bos*HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + (bos*HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    # [BT, BT]
    b_dA = tl.load(p_dA, boundary_check=(0, 1))
    # [BT, BK]
    b_dq += tl.dot(b_dA, b_k)
    b_dk += tl.dot(tl.trans(b_dA).to(b_k.dtype), b_q)

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_GV': lambda args: args['gv'] is not None,
    'HAVE_GK': lambda args: args['dgk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_GV'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_oja_bwd_kernel_dvwg_h(
    k,
    v,
    gv,
    h,
    dh,
    dk,
    dw,
    dv,
    dgv_last,
    dgk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_GV: tl.constexpr,
    HAVE_GK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    gv += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K*V
    dh += (i_tg * H + i_h).to(tl.int64) * K*V
    dk += (bos * H + i_h) * K
    dw += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    dgv_last += (bos * H + i_h) * V

    b_dvg = tl.zeros([BT, BV], dtype=tl.float32)
    b_dw = tl.zeros([BT, BV], dtype=tl.float32)
    b_dgv_last = tl.zeros([BV,], dtype=tl.float32)

    if USE_GV:
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        p_gn = gv + (min(T, i_t * BT + BT) - 1) * H*V + o_v
        p_gv = tl.make_block_ptr(gv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_gn = tl.load(p_gn, mask=m_v, other=0)
        b_gv = tl.load(p_gv, boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))  # BT BK
        b_dk = tl.load(p_dk, boundary_check=(0, 1))  # BT BK
        b_h = tl.load(p_h, boundary_check=(0, 1))  # BK BV
        b_dh = tl.load(p_dh, boundary_check=(0, 1))  # BK BV

        b_dvg += tl.dot(b_k, b_dh.to(b_k.dtype))  # BT BK @ BK BV -> BT BV
        b_dw += tl.dot(b_dk.to(b_k.dtype), b_h.to(b_k.dtype))  # BT BK @ BK BV -> BT BV
        b_dgv_last += tl.sum((b_h * b_dh) * exp(b_gn), axis=0)

    if USE_GV:
        b_dv = b_dvg * exp(b_gn[None, :] - b_gv)

    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dw = tl.make_block_ptr(dw, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dgv_last = tl.make_block_ptr(dgv_last, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    b_dgv_last += tl.sum(b_dv * b_v, axis=0)

    # 留给GSA2的接口
    if HAVE_GK:
        dgk += (bos * H + i_h) * V
        p_dgk = tl.make_block_ptr(dgk, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_dgk = tl.load(p_dgk, boundary_check=(0, 1))
        b_dgv_last = b_dgk + b_dgv_last[None, :]
    else:
        b_dgv_last = tl.zeros([BT, BV], dtype=tl.float32) + b_dgv_last[None, :]

    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dgv_last, b_dgv_last.to(p_dgv_last.dtype.element_ty), boundary_check=(0, 1))


def chunk_oja_bwd_dvwg_h(
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    dk: torch.Tensor,
    gv: torch.Tensor | None = None,
    dgk: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NV = triton.cdiv(V, BV)
    dv = torch.empty_like(v, dtype=torch.float)
    dw = torch.empty_like(v)
    dgv_last = torch.empty_like(gv)

    grid = (NV, NT, B * H)
    chunk_oja_bwd_kernel_dvwg_h[grid](
        k=k,
        v=v,
        gv=gv,
        h=h,
        dh=dh,
        dw=dw,
        dk=dk,
        dv=dv,
        dgv_last=dgv_last,
        dgk=dgk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dv, dw, dgv_last
