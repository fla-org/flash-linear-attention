# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused KL divergence kernels adapted for triton-ascend on Huawei NPU."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.ops.utils.op import exp, log

MAX_FUSED_SIZE = 65536 // 2
STATIC_WARPS = 2


@triton.jit
def kl_div_kernel(
    logits,
    target_logits,
    loss,
    s_logits,
    s_loss,
    reduction: tl.constexpr,
    N: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)

    logits += i_n * s_logits
    target_logits += i_n * s_logits

    sm = float('-inf')
    tm = float('-inf')
    sd, td = 0.0, 0.0

    NV = tl.cdiv(V, BV)
    for iv in range(0, NV):
        o_x = iv * BV + tl.arange(0, BV)
        b_sl = tl.load(logits + o_x, mask=o_x < V, other=float('-inf'))
        b_sm = tl.max(b_sl)
        m_new = tl.maximum(sm, b_sm)
        sd = sd * exp(sm - m_new) + tl.sum(exp(b_sl - m_new))
        sm = m_new

        b_tl = tl.load(target_logits + o_x, mask=o_x < V, other=float('-inf'))
        b_tm = tl.max(b_tl)
        m_new = tl.maximum(tm, b_tm)
        td = td * exp(tm - m_new) + tl.sum(exp(b_tl - m_new))
        tm = m_new

    b_loss = 0.
    for iv in range(0, NV):
        o_x = iv * BV + tl.arange(0, BV)
        b_sl = tl.load(logits + o_x, mask=o_x < V, other=float('-inf'))
        b_tl = tl.load(target_logits + o_x, mask=o_x < V, other=float('-inf'))
        b_sp_log = b_sl - sm - log(sd)
        b_tp_log = b_tl - tm - log(td)
        b_sp = exp(b_sp_log)
        b_tp = exp(b_tp_log)
        b_kl = tl.where(o_x < V, b_tp * (b_tp_log - b_sp_log), 0)
        b_dl = -b_tp + b_sp
        b_loss += tl.sum(b_kl)
        if reduction == 'batchmean':
            b_dl = b_dl / N
        tl.store(logits + o_x, b_dl, mask=o_x < V)

    if reduction == 'batchmean':
        b_loss = b_loss / N

    tl.store(loss + i_n * s_loss, b_loss)


@triton.jit
def elementwise_mul_kernel(
    x,
    g,
    N: tl.constexpr,
    B: tl.constexpr,
):
    i_x = tl.program_id(0).to(tl.int64)
    o_x = i_x * B + tl.arange(0, B)

    b_g = tl.load(g)
    b_x = tl.load(x + o_x, mask=o_x < N)
    tl.store(x + o_x, b_x * b_g, mask=o_x < N)


def _npu_vocab_block_size(vocab_size: int, num_rows: int) -> int:
    block_size = 1024 if vocab_size <= 32768 else (2048 if vocab_size <= 131072 else 4096)
    max_splits = max(1, 65535 // max(num_rows, 1))
    return min(
        max(block_size, triton.next_power_of_2((vocab_size + max_splits - 1) // max_splits)),
        8192,
    )


def fused_kl_div_forward_npu(
    x: torch.Tensor,
    target_x: torch.Tensor,
    weight: torch.Tensor,
    target_weight: torch.Tensor,
    reduction: str = 'batchmean',
    accumulate_grad_in_fp32: bool = True,
):
    device = x.device

    N, H, V = *x.shape, weight.shape[0]
    BV = _npu_vocab_block_size(V, N)
    NC = min(8, triton.cdiv(V, H))
    C = triton.next_power_of_2(triton.cdiv(N, NC))
    NC = triton.cdiv(N, C)

    grad_dtype = torch.float32 if accumulate_grad_in_fp32 else weight.dtype

    dx = torch.zeros_like(x, device=device)
    dw = torch.zeros_like(weight, device=device, dtype=grad_dtype) if weight is not None else None
    loss = torch.zeros(N, dtype=torch.float32, device=device)

    for ic in range(NC):
        start, end = ic * C, min((ic + 1) * C, N)
        c_sx = x[start:end]
        c_tx = target_x[start:end]
        c_sl = F.linear(c_sx, weight)
        c_tl = F.linear(c_tx, target_weight)
        if weight is not None and c_sx.dtype != grad_dtype:
            c_sx = c_sx.to(dtype=grad_dtype)

        c_loss = loss[start:end]

        kl_div_kernel[(c_sx.shape[0],)](
            logits=c_sl,
            target_logits=c_tl,
            loss=c_loss,
            s_logits=c_sl.stride(-2),
            s_loss=c_loss.stride(-1),
            reduction=reduction,
            N=N,
            V=V,
            BV=BV,
            num_warps=STATIC_WARPS,
        )

        c_grad = c_sl if c_sl.is_contiguous() else c_sl.contiguous()
        dx[start:end] = torch.mm(c_grad, weight)

        if weight is not None:
            grad_w = c_grad.t().to(dtype=grad_dtype)
            grad_x = c_sx if c_sx.dtype == grad_dtype else c_sx.to(dtype=grad_dtype)
            dw.add_(grad_w @ grad_x)

    loss = loss.sum()
    if dw is not None:
        dw = dw.to(weight)
    return loss, dx, dw


def fused_kl_div_backward_npu(
    do: torch.Tensor,
    dx: torch.Tensor,
    dw: torch.Tensor,
):
    if torch.ne(do, torch.tensor(1.0, device=do.device)):
        N, H = dx.shape
        B = min(MAX_FUSED_SIZE, max(1024, triton.next_power_of_2((N * H + 65534) // 65535)))

        elementwise_mul_kernel[(triton.cdiv(N * H, B),)](
            x=dx,
            g=do,
            N=N*H,
            B=B,
            num_warps=STATIC_WARPS,
        )

        if dw is not None:
            V, H = dw.shape
            B_dw = min(MAX_FUSED_SIZE, max(1024, triton.next_power_of_2((V * H + 65534) // 65535)))
            elementwise_mul_kernel[(triton.cdiv(V * H, B_dw),)](
                x=dw,
                g=do,
                N=V*H,
                B=B_dw,
                num_warps=STATIC_WARPS,
            )

    return dx, dw
