# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

import torch
import triton
import triton.language as tl

from fla.ops.attn.parallel import parallel_attn_bwd_preprocess
from fla.ops.nsa.compression import parallel_nsa_compression
from fla.ops.nsa.utils import _bitonic_merge
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets, prepare_lens, prepare_token_indices
from fla.ops.utils.op import exp, log
from fla.ops.utils.pooling import mean_pooling
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, check_shared_mem, contiguous

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = flash_attn_varlen_func = None


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens_q'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK'],
    **autotune_cache_kwargs,
)
@triton.jit
def parallel_nsa_kernel_topk(
    q,
    k,
    lse,
    scale,
    block_indices,
    cu_seqlens_q,
    cu_seqlens_k,
    token_indices_q,
    chunk_offsets,
    TQ,
    TK,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices_q + i_t * 2).to(tl.int32), tl.load(token_indices_q + i_t * 2 + 1).to(tl.int32)
        bos_q, eos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32), tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
        bos_k, eos_k = tl.load(cu_seqlens_k + i_n).to(tl.int32), tl.load(cu_seqlens_k + i_n + 1).to(tl.int32)
        TQ = eos_q - bos_q
        TK = eos_k - bos_k
        TC = tl.cdiv(TK, BS)
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos_q, eos_q = i_b * TQ, i_b * TQ + TQ
        TC = tl.cdiv(TK, BS)
        boc = i_b * TC
    # boc is the start of the current sequence at [B, TC] dimensions
    p_q = tl.make_block_ptr(q + (bos_q + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    Q_OFFSET = TK - TQ

    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    # number of complete compression blocks visible to the query (q tokens are the last TQ of the sequence)
    NC = (i_t + Q_OFFSET + 1) // BS
    ################################
    # 1. lse computation
    ################################
    if lse is not None:
        b_lse = tl.load(lse + (bos_q + i_t) * HQ + i_h * G + tl.arange(0, G))
    else:
        # max scores for the current block
        b_m = tl.full([G], float('-inf'), dtype=tl.float32)
        # lse = log(acc) + m
        b_acc = tl.zeros([G], dtype=tl.float32)
        for i_c in range(0, NC, BC):
            o_c = i_c + tl.arange(0, BC)

            p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (K, TC), (1, H*K), (0, i_c), (BK, BC), (0, 1))
            # [BK, BC]
            b_k = tl.load(p_k, boundary_check=(0, 1))

            # [G, BC]
            b_s = tl.dot(b_q, b_k)
            b_s = tl.where((o_c < NC)[None, :], b_s, float('-inf'))

            # [G]
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = exp(b_mp - b_m)
            # [G, BC]
            b_p = exp(b_s - b_m[:, None])
            # [G]
            b_acc = b_acc * b_r + tl.sum(b_p, 1)

            b_mp = b_m
        if NC == 0:
            b_lse = tl.zeros([G], dtype=tl.float32)
        else:
            b_lse = b_m + log(b_acc)

    ################################
    # 2. topk selection
    ################################
    # [BC]
    b_i = tl.full([BC], -1, dtype=tl.float32)
    o_i = tl.zeros([BC], dtype=tl.int32)
    m_i = tl.arange(0, BC) < BC//2

    IC = (i_t + Q_OFFSET) // BS  # Idx of the current query block
    for i_c in range(0, IC + 1, BC):  # +1, because the current block might be also included
        o_c = i_c + tl.arange(0, BC)
        p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (K, TC), (1, H*K), (0, i_c), (BK, BC), (0, 1))
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [G, BC]
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where(o_c < IC, b_s, float('-inf'))
        # the 1st and the last 2 blocks are always selected, with normalized score set to 1.0
        b_p = tl.where((o_c == 0) | ((o_c == IC - 1) | (o_c == IC)), 1., exp(b_s - b_lse[:, None]))
        # [BC] importance score = sum over the group's heads
        b_i, b_ip = tl.sum(b_p, 0), b_i
        # blocks with index < 0 will be skipped
        o_i, o_ip = tl.where(o_c <= IC, o_c, -1), o_i

        n_dims: tl.constexpr = tl.standard._log2(b_i.shape[0])
        for i in tl.static_range(1, n_dims):
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), i, 2, n_dims)

        if i_c != 0:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, False, n_dims)
            b_i_new = b_ip * m_i + b_i * (1 - m_i)
            o_i_new = o_ip * m_i + o_i * (1 - m_i)
            b_i, o_i = _bitonic_merge(b_i_new, o_i_new.to(tl.int32), n_dims, True, n_dims)
        else:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, True, n_dims)

    m_top = tl.arange(0, BC // S) == 0
    b_top = tl.sum(m_top[:, None] * tl.reshape(o_i, [BC // S, S]), 0)

    p_b = tl.make_block_ptr(block_indices + (bos_q + i_t) * H*S, (H*S,), (1,), (i_h * S,), (S,), (0,))
    tl.store(p_b, b_top.to(p_b.dtype.element_ty))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens_q'] is not None,
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor),
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit
def parallel_nsa_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    block_indices,
    block_counts,
    cu_seqlens_q,
    cu_seqlens_k,
    token_indices_q,
    TQ,
    TK,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    # k/v: [B, TK, H, *], q: [B, TQ, HQ, K], block_indices: [B, TQ, H, S], lse: [B, TQ, HQ]; G = HQ // H

    if IS_VARLEN:
        # token_indices_q maps a flattened query to its (sequence index, in-sequence position)
        i_n, i_t = tl.load(token_indices_q + i_t * 2).to(tl.int32), tl.load(token_indices_q + i_t * 2 + 1).to(tl.int32)
        bos_q, eos_q = tl.load(cu_seqlens_q + i_n).to(tl.int32), tl.load(cu_seqlens_q + i_n + 1).to(tl.int32)
        bos_k, eos_k = tl.load(cu_seqlens_k + i_n).to(tl.int32), tl.load(cu_seqlens_k + i_n + 1).to(tl.int32)
        TQ = eos_q - bos_q
        TK = eos_k - bos_k
    else:
        bos_q, eos_q = i_b * TQ, i_b * TQ + TQ
        bos_k, eos_k = i_b * TK, i_b * TK + TK

    # q tokens are assumed to be the last TQ tokens of the sequence (cached decoding)
    Q_OFFSET = TK - TQ

    k += (bos_k * H + i_h) * K
    v += (bos_k * H + i_h) * V
    block_indices += (bos_q + i_t) * H * S + i_h * S

    # block_counts: [B, TQ, H]
    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos_q + i_t) * H + i_h)
    else:
        NS = S

    p_q = tl.make_block_ptr(q + (bos_q + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    # the Q block is kept in shared memory throughout the kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_o = tl.make_block_ptr(o + (bos_q + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + (bos_q + i_t) * HQ + i_h * G + tl.arange(0, G)

    # [G, BV]
    b_o = tl.zeros([G, BV], dtype=tl.float32)

    b_m = tl.full([G], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS  # start token index of the i-th selected KV block
        if i_s <= Q_OFFSET + i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, TK), (1, H*K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(v, (TK, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            # [BK, BS]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BS, BV]
            b_v = tl.load(p_v, boundary_check=(0, 1))
            # [G, BS]
            b_s = tl.dot(b_q, b_k)
            # causal mask against the absolute query position Q_OFFSET + i_t
            b_s = tl.where((Q_OFFSET + i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s, float('-inf'))

            # [G]
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = exp(b_mp - b_m)
            # [G, BS]
            b_p = exp(b_s - b_m[:, None])
            # [G]
            b_acc = b_acc * b_r + tl.sum(b_p, 1)
            # [G, BV]
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

    b_o = b_o / b_acc[:, None]
    b_m += log(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))


@triton.heuristics({
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor),
})
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_kernel_mask(
    block_indices,
    block_counts,
    block_mask,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    NS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t, i_b, i_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_s = i_hs // S, i_hs % S

    b_i = tl.load(block_indices + i_b * T * H * S + i_t * H * S + i_h * S + i_s)
    if USE_BLOCK_COUNTS:
        b_m = b_i * BS <= i_t and i_s < tl.load(block_counts + i_b * T * H + i_t * H + i_h)
    else:
        b_m = b_i * BS <= i_t

    if b_i < NS and b_i >= 0:
        tl.store(block_mask + i_b * T * H * NS + i_t * H * NS + i_h * NS + b_i, b_m.to(block_mask.dtype.element_ty))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor),
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    scale,
    block_indices,
    block_counts,
    cu_seqlens,
    token_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos + i_t) * HQ*K
    do += (bos + i_t) * HQ*V
    lse += (bos + i_t) * HQ
    delta += (bos + i_t) * HQ
    dq += (i_v * all + bos + i_t) * HQ*K
    block_indices += (bos + i_t) * H*S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V

    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_do = tl.make_block_ptr(do, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + i_h * G + tl.arange(0, G)
    p_delta = delta + i_h * G + tl.arange(0, G)

    # [G, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [G]
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)

    # [G, BK]
    b_dq = tl.zeros([G, BK], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(v, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
            # [BK, BS]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BV, BS]
            b_v = tl.load(p_v, boundary_check=(0, 1))

            # [G, BS]
            b_s = tl.dot(b_q, b_k)
            b_p = exp(b_s - b_lse[:, None])
            b_p = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_p, 0)

            # [G, BV] @ [BV, BS] -> [G, BS]
            b_dp = tl.dot(b_do, b_v)
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            # [G, BS] @ [BS, BK] -> [G, BK]
            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


def prepare_dkv_csr(block_mask, cu_seqlens, chunk_indices, block_size):
    # Inverse index for gather-based dkv: per KV block (keyed by the same id the
    # kernel decodes) the sorted list of *absolute* query positions selecting it,
    # as CSR (q_idx values + q_ptr offsets). Lets bwd_dkv gather selecting queries
    # and batch them into one MMA instead of scanning every query tile per block.
    # Handles both dense (key = (b*H+h)*NS + block) and varlen (key = global-chunk
    # * H + h, via cu_seqlens/chunk_offsets).
    B, T, H, NS = block_mask.shape
    nz = block_mask.nonzero(as_tuple=False)            # [P, 4] = (b, t, h, ns)
    b_, t_, h_, ns_ = nz[:, 0], nz[:, 1], nz[:, 2], nz[:, 3]
    if cu_seqlens is None:
        q_abs = (b_ * T + t_).to(torch.int32)
        gkey = (b_ * H + h_) * NS + ns_
        n_keys = B * H * NS
    else:
        n_ = torch.searchsorted(cu_seqlens[1:].contiguous(), t_, right=True)   # sequence index
        chunk_off = prepare_chunk_offsets(cu_seqlens, block_size)
        gkey = (chunk_off[n_] + ns_) * H + h_
        q_abs = t_.to(torch.int32)
        n_keys = chunk_indices.shape[0] * H
    order = (gkey.to(torch.int64) * (B * T) + q_abs).argsort()
    q_idx = q_abs[order].contiguous()
    counts = torch.bincount(gkey, minlength=n_keys)
    q_ptr = torch.zeros(n_keys + 1, dtype=torch.int32, device=block_mask.device)
    q_ptr[1:] = counts.cumsum(0)
    return q_idx, q_ptr.contiguous()


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BG': BG}, num_warps=nw, num_stages=ns)
        for BG in [1, 2, 4, 8]
        for nw in [4, 8]
        for ns in [1, 2]
    ],
    key=['BS', 'BK', 'BV', 'G'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_bwd_kernel_dkv_gather(
    q, k, v, lse, delta, do, dk, dv,
    q_idx, q_ptr, cu_seqlens, chunk_indices,
    scale,
    T, NTOK,
    B: tl.constexpr, H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, NS: tl.constexpr,
    BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_blk = tl.program_id(0), tl.program_id(1)
    # decode the program into this KV block's head and absolute key start; all
    # address arithmetic in int64 (NTOK*HQ*K overflows int32 at long seq / big heads).
    if IS_VARLEN:
        i_c, i_h = i_blk // H, i_blk % H
        i_n = tl.load(chunk_indices + i_c * 2).to(tl.int32)
        i_sl = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)
        kbos = tl.load(cu_seqlens + i_n).to(tl.int64)
        keos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        key_abs = kbos + i_sl * BS + tl.arange(0, BS)
    else:
        i_s = i_blk % NS
        bh = i_blk // NS
        i_b, i_h = bh // H, bh % H
        key_abs = (i_b * T).to(tl.int64) + i_s * BS + tl.arange(0, BS)
        keos = (i_b + 1) * T

    start = tl.load(q_ptr + i_blk).to(tl.int32)
    end = tl.load(q_ptr + i_blk + 1).to(tl.int32)
    cnt = end - start

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_g = i_h * G + tl.arange(0, G)
    r_m, k_m, v_m = key_abs < keos, o_k < K, o_v < V

    # K/V tile for this block: k/v[key_abs, i_h, :]
    b_k = tl.load(k + key_abs[:, None] * (H * K) + i_h * K + o_k[None, :], mask=r_m[:, None] & k_m[None, :], other=0.)
    b_v = tl.load(v + key_abs[:, None] * (H * V) + i_h * V + o_v[None, :], mask=r_m[:, None] & v_m[None, :], other=0.)
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    for it in range(0, tl.cdiv(cnt, BG)):
        offs = it * BG + tl.arange(0, BG)
        m_j = offs < cnt
        q_pos = tl.load(q_idx + start + offs, mask=m_j, other=0).to(tl.int64)         # [BG] absolute

        base = q_pos[:, None, None] * (HQ * K) + o_g[None, :, None] * K + o_k[None, None, :]
        b_q = tl.load(q + base, mask=m_j[:, None, None] & k_m[None, None, :], other=0.).to(tl.float32)
        b_q = tl.reshape(b_q * scale, [BG * G, BK]).to(b_k.dtype)

        based = q_pos[:, None, None] * (HQ * V) + o_g[None, :, None] * V + o_v[None, None, :]
        b_do = tl.load(do + based, mask=m_j[:, None, None] & v_m[None, None, :], other=0.)
        b_do = tl.reshape(b_do, [BG * G, BV])

        ld = q_pos[:, None] * HQ + o_g[None, :]
        b_lse = tl.reshape(tl.load(lse + ld, mask=m_j[:, None], other=0.), [BG * G])
        b_del = tl.reshape(tl.load(delta + ld, mask=m_j[:, None], other=0.), [BG * G])

        col_valid = tl.reshape(tl.broadcast_to(m_j[:, None], [BG, G]), [BG * G])
        qpos_rep = tl.reshape(tl.broadcast_to(q_pos[:, None], [BG, G]), [BG * G])

        b_s = tl.dot(b_k, tl.trans(b_q))                                              # [BS, BG*G]
        b_p = exp(b_s - b_lse[None, :])
        keep = (qpos_rep[None, :] >= key_abs[:, None]) & col_valid[None, :]
        b_p = tl.where(keep, b_p, 0.)

        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp.to(tl.float32) - b_del[None, :])
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    dk_off = (i_v * NTOK + key_abs)[:, None] * (H * K) + i_h * K + o_k[None, :]
    dv_off = key_abs[:, None] * (H * V) + i_h * V + o_v[None, :]
    tl.store(dk + dk_off, b_dk.to(dk.dtype.element_ty), mask=r_m[:, None] & k_m[None, :])
    tl.store(dv + dv_off, b_dv.to(dv.dtype.element_ty), mask=r_m[:, None] & v_m[None, :])


@contiguous
def parallel_nsa_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    TK: int,
    lse: torch.Tensor | None,
    block_counts: torch.LongTensor | int,
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: torch.LongTensor | tuple[torch.LongTensor, torch.LongTensor] | None = None,
) -> torch.LongTensor:
    B, TQ, HQ, K, H = *q.shape, k.shape[2]

    assert k.shape[0] == q.shape[0] and k.shape[-1] == q.shape[-1], "The last dimension of k and q must match"
    assert lse is None or lse.shape == (B, TQ, HQ), "The shape of lse must be (B, TQ, HQ)"

    if cu_seqlens is not None:
        if isinstance(cu_seqlens, tuple):
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
        else:
            cu_seqlens_q = cu_seqlens_k = cu_seqlens
        token_indices_q = prepare_token_indices(cu_seqlens_q)
    else:
        cu_seqlens_q = cu_seqlens_k = token_indices_q = None

    G = HQ // H
    # the number of selected blocks for each token
    S = block_counts if isinstance(block_counts, int) else block_counts.max().item()
    S = triton.next_power_of_2(S)
    # here we set BC = BS, but beware that they can be chosen separately if required
    BC = BS = block_size
    BK = max(triton.next_power_of_2(K), 16)
    assert BC >= 2 * S, f"BC ({BC}) must be greater than or equal to 2 * S ({S})"

    block_indices = torch.zeros(B, TQ, H, S, dtype=torch.int32, device=q.device)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens_k, BS) if cu_seqlens_k is not None else None
    grid = (TQ, B * H)
    # the 1st and the last 2 blocks are always selected
    parallel_nsa_kernel_topk[grid](
        q=q,
        k=k,
        lse=lse,
        scale=scale,
        block_indices=block_indices,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        token_indices_q=token_indices_q,
        chunk_offsets=chunk_offsets,
        TQ=TQ,
        TK=TK,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        S=S,
        BC=BC,
        BS=BS,
        BK=BK,
    )
    return block_indices


@contiguous
def parallel_nsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor | int,
    block_size: int,
    scale: float,
    cu_seqlens_q: torch.LongTensor | None = None,
    cu_seqlens_k: torch.LongTensor | None = None,
    token_indices_q: torch.LongTensor | None = None,
):
    B, T_kv, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    _, T_q, HQ, _ = q.shape
    G = HQ // H
    BS = block_size
    if check_shared_mem('hopper', q.device.index):
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    grid = (T_q, NV, B * H)
    o = torch.empty(B, T_q, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T_q, HQ, dtype=torch.float, device=q.device)

    parallel_nsa_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        scale=scale,
        block_indices=block_indices,
        block_counts=block_counts,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        token_indices_q=token_indices_q,
        TQ=T_q,
        TK=T_kv,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        S=S,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    return o, lse


def parallel_nsa_block_mask(
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor | int,
    cu_seqlens: torch.LongTensor,
    block_size: int,
):
    B, T, H, S = block_indices.shape
    BS = block_size
    if cu_seqlens is not None:
        NS = triton.cdiv(prepare_lens(cu_seqlens).max().item(), BS)
    else:
        NS = triton.cdiv(T, BS)
    block_mask = torch.zeros(B, T, H, NS, dtype=torch.bool, device=block_indices.device)

    parallel_nsa_kernel_mask[(T, B, H*S)](
        block_indices=block_indices,
        block_counts=block_counts,
        block_mask=block_mask,
        T=T,
        H=H,
        S=S,
        BS=BS,
        NS=NS,
    )
    return block_mask


def parallel_nsa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.LongTensor | int,
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    token_indices: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    BK = max(triton.next_power_of_2(K), 16)
    BV = min(128, max(triton.next_power_of_2(v.shape[-1]), 16))
    NV = triton.cdiv(V, BV)

    delta = parallel_attn_bwd_preprocess(o, do)

    dq = torch.empty(NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device)
    grid = (T, NV, B * H)
    parallel_nsa_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        block_indices=block_indices,
        block_counts=block_counts,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        S=S,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    dq = dq.sum(0)

    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BS)

    # [B, T, H, M] block_mask: which KV blocks each query selects (causal + count
    # valid); prepare_dkv_csr inverts it into the per-block list of selecting queries.
    block_mask = parallel_nsa_block_mask(block_indices, block_counts, cu_seqlens, block_size)
    M = block_mask.shape[-1]
    dk = torch.empty(NV, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)

    # gather the queries selecting each KV block (CSR inverse index) and batch them
    # into one MMA — far fewer, far larger matmuls than scanning every query tile.
    # Works for dense and varlen (the kernel decodes the block via chunk_indices).
    q_idx, q_ptr = prepare_dkv_csr(block_mask, cu_seqlens, chunk_indices, block_size)
    n_blk = chunk_indices.shape[0] * H if cu_seqlens is not None else B * H * M
    grid = (NV, n_blk)
    parallel_nsa_bwd_kernel_dkv_gather[grid](
        q, k, v, lse, delta, do, dk, dv, q_idx, q_ptr, cu_seqlens, chunk_indices,
        scale, T, B * T,
        B=B, H=H, HQ=HQ, G=G, K=K, V=V, NS=M, BS=BS, BK=BK, BV=BV,
    )
    dk = dk.sum(0)
    return dq, dk, dv


@torch.compile
class ParallelNSAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, block_indices, block_counts, block_size, scale, cu_seqlens):
        ctx.dtype = q.dtype

        # 2-d sequence indices denoting the cu_seqlens of tokens in each sequence
        # for example, if the passed `cu_seqlens` is [0, 2, 6],
        # then there are 2 and 4 tokens in the 1st and 2nd sequences respectively, and `token_indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        if cu_seqlens is not None:
            if isinstance(cu_seqlens, tuple):
                cu_seqlens_q, cu_seqlens_k = cu_seqlens
            else:
                cu_seqlens_q = cu_seqlens_k = cu_seqlens
            token_indices_q = prepare_token_indices(cu_seqlens_q)
        else:
            cu_seqlens_q = cu_seqlens_k = token_indices_q = None

        o, lse = parallel_nsa_fwd(
            q=q,
            k=k,
            v=v,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
            scale=scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            token_indices_q=token_indices_q
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.block_indices = block_indices
        ctx.block_counts = block_counts
        # Use cu_seqlens of q in backward, as cu_seqlens for q & k are different only for inference
        ctx.cu_seqlens = cu_seqlens_q
        ctx.token_indices = token_indices_q
        ctx.block_size = block_size
        ctx.scale = scale
        # q/k cu_seqlens differ only in cached inference (TQ != TK), where backward is not supported
        ctx.tq_ne_tk = isinstance(cu_seqlens, tuple)
        return o.to(q.dtype)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        if ctx.tq_ne_tk:
            raise NotImplementedError(
                "Backward is not supported when `cu_seqlens` differs for queries and keys (cached inference). "
                "Run the forward under `torch.no_grad()`."
            )
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = parallel_nsa_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            do=do,
            block_indices=ctx.block_indices,
            block_counts=ctx.block_counts,
            block_size=ctx.block_size,
            scale=ctx.scale,
            cu_seqlens=ctx.cu_seqlens,
            token_indices=ctx.token_indices,
        )
        return dq.to(q), dk.to(k), dv.to(v), None, None, None, None, None, None, None, None


@contiguous
def parallel_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor | None = None,
    g_slc: torch.Tensor | None = None,
    g_swa: torch.Tensor | None = None,
    block_indices: torch.LongTensor | None = None,
    block_counts: torch.LongTensor | int = 16,
    block_size: int = 64,
    window_size: int = 0,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | tuple[torch.LongTensor, torch.LongTensor] | None = None,
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, TQ, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16 (it is a kernel tile dimension; Triton requires power-of-2 block shapes).
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g_cmp (torch.Tensor):
            Gate score for compressed attention of shape `[B, TQ, HQ]`.
        g_slc (torch.Tensor):
            Gate score for selected attention of shape `[B, TQ, HQ]`.
        g_swa (torch.Tensor):
            Gate score for sliding attentionof shape `[B, TQ, HQ]`.
        block_indices (torch.LongTensor):
            Block indices of shape `[B, TQ, H, S]`.
            `S` is the number of selected blocks for each query token, which is set to 16 in the paper.
            Will override the computed block indices from compression if provided.
        block_counts (Optional[Union[torch.LongTensor, int]]):
            Number of selected blocks for each query.
            If a tensor is provided, with shape `[B, TQ, H]`,
            each query can select the same number of blocks.
            If not provided, it will default to 16.
        block_size (int):
            Selected block size. Default: 64.
        window_size (int):
            Sliding window size. Default: 0.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor] or None):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
            When a tuple is provided, it should contain two tensors: `(cu_seqlens_q, cu_seqlens_k)`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]`.
    """
    assert block_counts is not None, "block counts must be provided for selection"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
            f"Please flatten variable-length inputs before processing.",
        )
    assert q.shape[2] % (k.shape[2] * 16) == 0, "Group size must be a multiple of 16 in NSA"

    if cu_seqlens is not None:
        if isinstance(cu_seqlens, tuple):
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
        else:
            cu_seqlens_q = cu_seqlens_k = cu_seqlens
    else:
        cu_seqlens_q = cu_seqlens_k = None
    k_cmp, v_cmp = mean_pooling(k, block_size, cu_seqlens_k), mean_pooling(v, block_size, cu_seqlens_k)
    o_cmp, lse_cmp = None, None
    if g_cmp is not None:
        o_cmp, lse_cmp = parallel_nsa_compression(
            q=q,
            k=k_cmp,
            v=v_cmp,
            TK=k.shape[1],
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
        )
        if block_indices is None:
            block_indices = parallel_nsa_topk(
                q=q,
                k=k_cmp,
                lse=lse_cmp,
                TK=k.shape[1],
                block_counts=block_counts,
                block_size=block_size,
                scale=scale,
                cu_seqlens=cu_seqlens,
            )
        else:
            warnings.warn("`block_indices` is provided, overriding the selection computed from compression")
    o = o_slc = ParallelNSAFunction.apply(q, k, v, block_indices, block_counts, block_size, scale, cu_seqlens)
    if g_slc is not None:
        o = o_slc * g_slc.unsqueeze(-1)
    if o_cmp is not None:
        o = torch.addcmul(o, o_cmp, g_cmp.unsqueeze(-1))
    if window_size > 0:
        if cu_seqlens is not None:
            o_swa = flash_attn_varlen_func(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                causal=True,
                window_size=(window_size-1, 0),
            ).unsqueeze(0)
        else:
            o_swa = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(window_size-1, 0),
            )
        o = torch.addcmul(o, o_swa, g_swa.unsqueeze(-1))
    return o
