# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""chunk_scaled_dot_kkt_fwd adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import compute_row_tile_block_size, get_npu_properties

# peak live fp32 tiles: b_A[BT,BT], b_k[BT,BK], tl.dot buffer; post-dot gate[BT,BT]
_CHUNK_SCALED_DOT_KKT_MEM_MULT = 4.0
_SAFETY_MARGIN = 0.85
_FALLBACK_BK = 16
_MAX_BK_FWD = 128


def _get_fwd_bk(BT: int, K: int) -> int:
    """UB-safe BK tile size for chunk_scaled_dot_kkt_fwd on NPU."""
    return compute_row_tile_block_size(
        BT,
        K,
        _CHUNK_SCALED_DOT_KKT_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=_FALLBACK_BK,
        min_block=16,
        max_block=min(_MAX_BK_FWD, triton.next_power_of_2(K)),
    )


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'B', 'task_num', 'num_core'])
def chunk_scaled_dot_kkt_fwd_kernel_npu(
    k,
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
    USE_G: tl.constexpr,
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
        i_b, i_h = i_bh // HV, i_bh % HV
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = eos - bos
        else:
            bos = tl.cast(i_b, tl.int64) * T
        o_t = i_t * BT + o_i
        m_t = o_t < T
        m_A = m_causal & (m_t[:, None] & m_t)
        # make_block_ptr offsets must be 32-bit; keep 64-bit o_t for regular indexing.
        t_off = (i_t * BT).to(tl.int32)
        # 1-token chunks: strictly-lower-tri kkt is 0; T=1 block_ptr misaligns UB.
        if i_t * BT + 1 < T:
            p_b = tl.make_block_ptr(beta + i_h * bt_stride + bos, (T,), (1,), (t_off,), (BT,), (0,))
            b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)

            if USE_G:
                p_g = tl.make_block_ptr(g + i_h * bt_stride + bos, (T,), (1,), (t_off,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)

            b_A = tl.zeros([BT, BT], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_k = tl.make_block_ptr(
                    k + (bos * H + i_h // (HV // H)) * K, (T, K), (H * K, 1),
                    (t_off, i_k * BK), (BT, BK), (1, 0),
                )
                b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
                # ascend tl.dot may clobber lhs; keep rhs on the original tile.
                b_k_lhs = b_k + 0.0
                b_A += tl.dot(b_k_lhs, tl.trans(b_k), allow_tf32=False)

            if USE_G:
                # mask first so upper-triangle g_i-g_j cannot overflow to inf.
                b_g_diff = tl.where(m_A, b_g[:, None] - b_g[None, :], 0)
                b_A *= exp2(b_g_diff)
            b_A *= b_b[:, None]
            b_A = tl.where(m_A, b_A, 0)

            p_A = A + (bos * HV + i_h) * BT + o_t[:, None] * (BT * HV) + o_i[None, :]
            tl.store(p_A, b_A.to(p_A.dtype.element_ty), mask=m_t[:, None])


@input_guard
def chunk_scaled_dot_kkt_fwd_npu(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]` where `H` is the number of query/key heads.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, HV]`. Default: `None`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, HV]` where `HV` is the number of value/output heads.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`
        chunk_indices (torch.LongTensor):
            The chunk indices of the input tensor. Default: None.

    Returns:
        beta * K * K^T of shape `[B, T, HV, BT]` where `BT` is the chunk size.
        For GVA, H < HV and HV % H == 0. For standard attention, H == HV.
    """
    B, T, H, K, HV = *k.shape, beta.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.zeros(B, T, HV, BT, device=k.device, dtype=output_dtype)
    BK = _get_fwd_bk(BT, K)

    num_core = get_npu_properties()['num_aicore']
    g_arg = torch.permute(g, (2, 0, 1)).contiguous() if g is not None else g
    beta_arg = torch.permute(beta, (2, 0, 1)).contiguous()
    chunk_scaled_dot_kkt_fwd_kernel_npu[(num_core,)](
        k=k,
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
    )
    return A
