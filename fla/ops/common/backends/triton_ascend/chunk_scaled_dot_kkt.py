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
import triton.runtime.driver as driver

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import compute_row_tile_block_size

# Peak live fp32 tiles: b_A[BT,BT], b_k[BT,BK], plus tl.dot intermediate buffer
_CHUNK_SCALED_DOT_KKT_MEM_MULT = 5.0
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


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_G": lambda args: args["g"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "B", "bh_step", "task_num", "num_core"])
def chunk_scaled_dot_kkt_fwd_kernel_npu(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    B,
    bh_step,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    bt_stride = B.to(tl.int64) * T.to(tl.int64)
    core_id = tl.program_id(0)
    T64 = T.to(tl.int64)

    for task_id in tl.range(core_id, task_num, num_core):
        i_t_i = task_id // bh_step
        i_bh = task_id % bh_step
        i_b, i_h = i_bh // HV, i_bh % HV
        if IS_VARLEN:
            i_n, i_t = (
                tl.load(chunk_indices + i_t_i * 2).to(tl.int32),
                tl.load(chunk_indices + i_t_i * 2 + 1).to(tl.int32),
            )
            bos = tl.load(cu_seqlens + i_n)
            eos = tl.load(cu_seqlens + i_n + 1)
            T = eos.to(tl.int32) - bos.to(tl.int32)
        else:
            bos = i_b.to(tl.int64) * T64
            eos = bos + T64
            i_t = i_t_i
        o_t = tl.arange(0, BT)
        o_t_fp32 = o_t.to(tl.float32)

        p_beta = tl.make_block_ptr(beta + i_h * bt_stride + bos, (T,), (1,), (i_t * BT,), (BT,), (0,))
        b_beta = tl.load(p_beta, boundary_check=(0,))

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(
                k + (bos * H + i_h // (HV // H)) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_A += tl.dot(b_k, tl.trans(b_k))

        if USE_G:
            p_g = tl.make_block_ptr(g + i_h * bt_stride + bos, (T,), (1,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_g_diff = b_g[:, None] - b_g[None, :]
            b_A *= exp2(b_g_diff)

        b_A *= b_beta[:, None]
        b_A = tl.where(o_t_fp32[:, None] > o_t_fp32[None, :], b_A, 0)
        p_A = tl.make_block_ptr(A + (bos * HV + i_h) * BT, (T, BT), (BT * HV, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


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

    num_core = get_npu_properties()["num_aicore"]
    bh_step = B * HV
    task_num = NT * bh_step
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
        bh_step=bh_step,
        task_num=task_num,
        num_core=num_core,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BK=BK,
    )
    return A
