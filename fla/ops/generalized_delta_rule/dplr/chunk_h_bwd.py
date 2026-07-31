# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp2
from fla.utils import IS_AMD, autotune_cache_kwargs, check_shared_mem

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [2, 4, 8, 16, 32]


@triton.heuristics({
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV', 'V'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_bwd_kernel_dhu(
    qg,
    bg,
    w,
    gk,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_h = (o_k[:, None] < K) & (o_v[None, :] < V)
    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_dh += tl.load(p_dht, mask=m_h, other=0.0)

    mask_k = tl.arange(0, BK) < K
    for i_t in range(NT - 1, -1, -1):
        p_dh = dh + ((boh+i_t) * H + i_h) * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), mask=m_h)
        b_dh_tmp = tl.zeros([BK, BV], dtype=tl.float32)
        for i_c in range(tl.cdiv(BT, BC) - 1, -1, -1):
            o_c = i_t * BT + i_c * BC + tl.arange(0, BC)
            m_c = o_c < T
            m_kc = (o_k[:, None] < K) & m_c[None, :]
            m_bgc = m_c[:, None] & (o_k[None, :] < K)
            m_vc = m_c[:, None] & (o_v[None, :] < V)
            p_qg = qg+(bos*H+i_h)*K + o_k[:, None] + o_c[None, :] * (H*K)
            p_bg = bg+(bos*H+i_h)*K + o_c[:, None] * (H*K) + o_k[None, :]
            p_w = w+(bos*H+i_h)*K + o_k[:, None] + o_c[None, :] * (H*K)
            p_dv = dv+(bos*H+i_h)*V + o_c[:, None] * (H*V) + o_v[None, :]
            p_do = do+(bos*H+i_h)*V + o_c[:, None] * (H*V) + o_v[None, :]
            p_dv2 = dv2+(bos*H+i_h)*V + o_c[:, None] * (H*V) + o_v[None, :]
            # [BK, BT]
            b_qg = tl.load(p_qg, mask=m_kc, other=0.0)
            # [BT, BK]
            b_bg = tl.load(p_bg, mask=m_bgc, other=0.0)
            b_w = tl.load(p_w, mask=m_kc, other=0.0)
            # [BT, V]
            b_do = tl.load(p_do, mask=m_vc, other=0.0)
            b_dv = tl.load(p_dv, mask=m_vc, other=0.0)
            b_dv2 = b_dv + tl.dot(b_bg, b_dh.to(b_bg.dtype))
            tl.store(p_dv2, b_dv2.to(p_dv.dtype.element_ty), mask=m_vc)
            # [BK, BV]
            b_dh_tmp += tl.dot(b_qg, b_do.to(b_qg.dtype))
            b_dh_tmp += tl.dot(b_w, b_dv2.to(b_qg.dtype))
        last_idx = min((i_t + 1) * BT, T) - 1
        bg_last = tl.load(gk + ((bos + last_idx) * H + i_h) * K + tl.arange(0, BK), mask=mask_k)
        b_dh *= exp2(bg_last)[:, None]
        b_dh += b_dh_tmp

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=m_h)


def chunk_dplr_bwd_dhu(
    qg: torch.Tensor,
    bg: torch.Tensor,
    w: torch.Tensor,
    gk: torch.Tensor,
    h0: torch.Tensor,
    dht: torch.Tensor | None,
    do: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *qg.shape, do.shape[-1]
    BT = chunk_size
    BK = max(triton.next_power_of_2(K), 16)
    assert BK <= 256, "current kernel does not support head dimension being larger than 256."
    # H100
    if check_shared_mem('hopper', qg.device.index):
        BV = 64
        BC = 64 if K <= 128 else 32
    elif check_shared_mem('ampere', qg.device.index):  # A100
        BV = 32
        BC = 32
    else:  # Etc: 4090
        BV = 16
        BC = 16

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    BC = min(BT, BC)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    dh = qg.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.zeros_like(dv)

    grid = (NK, NV, N * H)
    chunk_dplr_bwd_kernel_dhu[grid](
        qg=qg,
        bg=bg,
        w=w,
        gk=gk,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
    )
    return dh, dh0, dv2
