# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""chunk_precond_kkt_fwd adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard, npu_leftover_mask
from fla.utils.ascend_ub_manager import compute_row_tile_block_size, get_npu_properties

# peak live fp32 tiles: b_A[BT,BT], b_k[BT,BK], b_kp[BT,BK], tl.dot buffer
_CHUNK_PRECOND_KKT_MEM_MULT = 5.0
_SAFETY_MARGIN = 0.85
_FALLBACK_BK = 16
_MAX_BK_FWD = 128


def _get_fwd_bk(BT: int, K: int) -> int:
    """UB-safe BK tile size for chunk_precond_kkt_fwd on NPU."""
    return compute_row_tile_block_size(
        BT,
        K,
        _CHUNK_PRECOND_KKT_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=_FALLBACK_BK,
        min_block=16,
        max_block=min(_MAX_BK_FWD, triton.next_power_of_2(K)),
    )


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'B', 'task_num', 'num_core'])
def chunk_precond_kkt_fwd_kernel_npu(
    k,
    k_precond,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    B,
    task_num: tl.int64,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MASK_LEFTOVER: tl.constexpr,
):
    T = T.to(tl.int64)
    bt_stride = B.to(tl.int64) * T
    core_id = tl.program_id(0)
    o_i = tl.arange(0, BT)
    m_causal = o_i[:, None] > o_i[None, :]

    for task_id in tl.range(core_id, task_num, num_core):
        bh = B.to(tl.int64) * HV
        i_t = task_id.to(tl.int64) // bh
        i_bh = task_id.to(tl.int64) % bh
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = eos - bos
        else:
            bos = tl.cast(i_b, tl.int64) * T
        o_t = i_t * BT + o_i
        m_t = o_t < T
        m_A = m_causal & (m_t[:, None] & m_t)
        t_off = (i_t * BT).to(tl.int32)
        # 1-token chunks: strictly-lower-tri kkt is 0; T=1 block_ptr misaligns UB.
        if i_t * BT + 1 < T:
            p_b = tl.make_block_ptr(beta + i_hv * bt_stride + bos, (T,), (1,), (t_off,), (BT,), (0,))
            b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
            if MASK_LEFTOVER:
                b_b = tl.where(m_t, b_b, 0)
            p_g = tl.make_block_ptr(g + i_hv * bt_stride + bos, (T,), (1,), (t_off,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)

            b_A = tl.zeros([BT, BT], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_k = tl.make_block_ptr(
                    k + (bos * H + i_h) * K, (T, K), (H * K, 1),
                    (t_off, i_k * BK), (BT, BK), (1, 0),
                )
                p_kp = tl.make_block_ptr(
                    k_precond + (bos * H + i_h) * K, (T, K), (H * K, 1),
                    (t_off, i_k * BK), (BT, BK), (1, 0),
                )
                b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
                b_kp = tl.load(p_kp, boundary_check=(0, 1)).to(tl.float32)
                if MASK_LEFTOVER:
                    o_k = i_k * BK + tl.arange(0, BK)
                    b_k = tl.where(m_t[:, None] & (o_k < K)[None, :], b_k, 0)
                    b_kp = tl.where(m_t[:, None] & (o_k < K)[None, :], b_kp, 0)
                # ascend tl.dot may clobber lhs; keep rhs on the original tile.
                b_k_lhs = b_k + 0.0
                b_A = tl.dot(b_k_lhs, tl.trans(b_kp), b_A, allow_tf32=False)

            # mask first so upper-triangle g_i-g_j cannot overflow to inf.
            b_g_diff = tl.where(m_A, b_g[:, None] - b_g[None, :], 0)
            b_A *= exp2(b_g_diff)
            b_A *= b_b[:, None]
            b_A = tl.where(m_A, b_A, 0)

            p_A = A + (bos * HV + i_hv) * BT + o_t[:, None] * (BT * HV) + o_i[None, :]
            tl.store(p_A, b_A.to(p_A.dtype.element_ty), mask=m_t[:, None])


@input_guard
def chunk_precond_kkt_fwd_npu(
    k: torch.Tensor,
    k_precond: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K = k.shape
    HV = beta.shape[-1]
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.zeros(B, T, HV, BT, device=k.device, dtype=output_dtype)
    BK = _get_fwd_bk(BT, K)
    mask_leftover = npu_leftover_mask(
        T=T, BT=BT, K=K, BK=BK, varlen=cu_seqlens is not None,
    )
    if mask_leftover:
        BK = min(64, BK)

    num_core = get_npu_properties()['num_aicore']
    g_arg = torch.permute(g, (2, 0, 1)).contiguous()
    beta_arg = torch.permute(beta, (2, 0, 1)).contiguous()
    chunk_precond_kkt_fwd_kernel_npu[(num_core,)](
        k=k,
        k_precond=k_precond,
        g=g_arg,
        beta=beta_arg,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        task_num=NT * B * HV,
        num_core=num_core,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BK=BK,
        MASK_LEFTOVER=mask_leftover,
    )
    return A
