# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused linear cross-entropy kernels adapted for triton-ascend on Huawei NPU."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.math import tanh

from fla.ops.utils.op import exp, log

MAX_FUSED_SIZE = 65536 // 2
STATIC_WARPS = 2


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
})
@triton.jit
def logsumexp_fwd_kernel(
    x,
    z,
    scale,
    softcapping: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
    ROWWISE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    if ROWWISE:
        row = x + i_n * D
        m = float('-inf')
        d = 0.0
        for start in range(0, D, B):
            o = start + tl.arange(0, B)
            b_x = tl.load(row + o, mask=o < D, other=float('-inf')).to(tl.float32)
            if HAS_SCALE:
                b_x = b_x * scale
            if HAS_SOFTCAPPING:
                b_x = softcapping * tanh(b_x / softcapping)
            blk_max = tl.max(b_x, 0)
            new_m = tl.maximum(m, blk_max)
            d = d * exp(m - new_m) + tl.sum(exp(b_x - new_m), 0)
            m = new_m
        tl.store(z + i_n, m + log(d))
    else:
        i_d = tl.program_id(1).to(tl.int64)
        o_d = i_d * B + tl.arange(0, B)
        m_d = o_d < D

        b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
        if HAS_SCALE:
            b_x = b_x * scale
        if HAS_SOFTCAPPING:
            b_x = softcapping * tanh(b_x / softcapping)
        b_m = tl.max(b_x, 0)
        b_z = log(tl.sum(exp(b_x - b_m), 0)) + b_m
        tl.store(z + i_n * tl.cdiv(D, B) + i_d, b_z)


@triton.jit
def cross_entropy_kernel(
    logits,
    lse,
    target,
    loss,
    total,
    ignore_index,
    label_smoothing: tl.constexpr,
    logit_scale: tl.constexpr,
    logit_softcapping: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
    reduction: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)

    b_y = tl.load(target + i_n)
    logits += i_n * V

    if b_y == ignore_index:
        for i in range(0, V, BV):
            o_v = i + tl.arange(0, BV)
            tl.store(logits + o_v, 0.0, mask=o_v < V)
        return

    b_l = tl.load(logits + b_y).to(tl.float32) * logit_scale
    if HAS_SOFTCAPPING:
        b_t_y = tanh(b_l / logit_softcapping)
        b_l = logit_softcapping * b_t_y
        b_softcap_deriv_y = 1.0 - b_t_y * b_t_y
    b_lse = tl.load(lse + i_n)

    b_loss = b_lse - b_l
    b_z = 0.0
    eps = label_smoothing / V

    for iv in range(0, NV):
        o_v = iv * BV + tl.arange(0, BV)
        b_logits = tl.load(logits + o_v, mask=o_v < V, other=float('-inf')).to(tl.float32) * logit_scale
        if HAS_SOFTCAPPING:
            b_t = tanh(b_logits / logit_softcapping)
            b_capped = logit_softcapping * b_t
        else:
            b_capped = b_logits
        if label_smoothing > 0:
            b_z += tl.sum(tl.where(o_v < V, -eps * b_capped, 0.0))
        b_p = (exp(b_capped - b_lse) - eps) * logit_scale
        if HAS_SOFTCAPPING:
            b_p = b_p * (1.0 - b_t * b_t)
        if reduction == "mean":
            b_p = b_p / total
        tl.store(logits + o_v, b_p, mask=o_v < V)

    if label_smoothing > 0:
        b_loss = b_loss * (1 - label_smoothing) + (b_z + label_smoothing * b_lse)

    b_l = tl.load(logits + b_y)

    if HAS_SOFTCAPPING:
        b_sc_factor = b_softcap_deriv_y
    else:
        b_sc_factor = 1.0

    if reduction == 'mean':
        b_loss = b_loss / total
        b_l += (label_smoothing - 1) / total * logit_scale * b_sc_factor
    else:
        b_l += (label_smoothing - 1) * logit_scale * b_sc_factor

    tl.store(loss + i_n, b_loss)
    tl.store(logits + b_y, b_l)


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


def logsumexp_fwd_npu(
    x,
    scale: float | None = None,
    softcapping: float | None = None,
    dtype: torch.dtype | None = None,
):
    shape = x.shape
    x = x.view(-1, shape[-1])
    N, D = x.shape
    B = _npu_vocab_block_size(D, N)
    has_softcapping = softcapping is not None
    softcap_val = float(softcapping) if has_softcapping else 0.0

    z = x.new_empty(N, dtype=torch.float)
    logsumexp_fwd_kernel[(N,)](
        x=x,
        z=z,
        scale=scale,
        softcapping=softcap_val,
        D=D,
        B=B,
        ROWWISE=True,
        HAS_SOFTCAPPING=has_softcapping,
    )
    z = z.view(*shape[:-1])
    if dtype is not None and dtype != torch.float:
        z = z.to(dtype)
    return z


def fused_linear_cross_entropy_forward_npu(
    x: torch.Tensor,
    target: torch.LongTensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    logit_softcapping: float = None,
    num_chunks: int = 8,
    reduction: str = "mean",
    use_l2warp: bool = False,
    l2_penalty_factor: float = 1e-4,
    accumulate_grad_in_fp32: bool = True,
):
    device = x.device
    N, H, V = *x.shape, weight.shape[0]
    BV = _npu_vocab_block_size(V, N)
    has_softcapping = logit_softcapping is not None
    softcap_val = float(logit_softcapping) if has_softcapping else 0.0
    NC = min(num_chunks, triton.cdiv(V, H))
    C = triton.next_power_of_2(triton.cdiv(N, NC))
    NC = triton.cdiv(N, C)

    dx = torch.zeros_like(x, device=device)
    grad_dtype = torch.float32 if accumulate_grad_in_fp32 else weight.dtype
    bias_grad_dtype = None
    if bias is not None:
        bias_grad_dtype = torch.float32 if accumulate_grad_in_fp32 else bias.dtype

    dw = torch.zeros_like(weight, device=device, dtype=grad_dtype) if weight is not None else None
    db = torch.zeros_like(bias, device=device, dtype=bias_grad_dtype) if bias is not None else None
    loss = torch.zeros(N, device=device, dtype=torch.float)

    total = target.ne(ignore_index).sum().item()

    for ic in range(NC):
        start, end = ic * C, min((ic + 1) * C, N)
        c_x = x[start:end]
        c_logits = F.linear(c_x, weight, bias)
        if weight is not None and c_x.dtype != grad_dtype:
            c_x = c_x.to(dtype=grad_dtype)
        c_target = target[start:end]
        c_lse = logsumexp_fwd_npu(c_logits, scale=logit_scale, softcapping=logit_softcapping, dtype=torch.float)

        c_loss = loss[start:end]
        if use_l2warp:
            c_maxx, c_ids = torch.max(c_logits, -1, keepdim=True)

        cross_entropy_kernel[(c_logits.shape[0],)](
            logits=c_logits,
            lse=c_lse,
            target=c_target,
            loss=c_loss,
            total=total,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            logit_scale=logit_scale,
            logit_softcapping=softcap_val,
            HAS_SOFTCAPPING=has_softcapping,
            reduction=reduction,
            V=V,
            BV=BV,
            num_warps=STATIC_WARPS,
        )
        if use_l2warp:
            g_logits_l2 = torch.zeros_like(c_logits)
            l2_factor = l2_penalty_factor / N
            penalty_grad = c_maxx * l2_factor
            g_logits_l2.scatter_(-1, c_ids, penalty_grad)

            if weight is not None:
                torch.addmm(
                    input=dw,
                    mat1=g_logits_l2.t().to(dtype=grad_dtype),
                    mat2=c_x,
                    out=dw,
                )
            if bias is not None:
                torch.add(input=db, other=g_logits_l2.sum(0, dtype=bias_grad_dtype), out=db)
            dx_l2_contribution = torch.mm(g_logits_l2, weight)
        else:
            dx_l2_contribution = 0.0

        c_grad = c_logits if c_logits.is_contiguous() else c_logits.contiguous()
        dx[start:end] = torch.mm(c_grad, weight) + dx_l2_contribution

        if weight is not None:
            grad_w = c_grad.t().to(dtype=grad_dtype)
            grad_x = c_x if c_x.dtype == grad_dtype else c_x.to(dtype=grad_dtype)
            dw.add_(grad_w @ grad_x)

        if bias is not None:
            torch.add(input=db, other=c_logits.sum(0, dtype=bias_grad_dtype), out=db)

    loss = loss.sum()
    if dw is not None:
        dw = dw.to(weight)
    if db is not None:
        db = db.to(bias)
    return loss, dx, dw, db


def fused_linear_cross_entropy_backward_npu(
    do: torch.Tensor,
    dx: torch.Tensor,
    dw: torch.Tensor,
    db: torch.Tensor,
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

        if db is not None:
            V = db.shape[0]
            B_db = min(MAX_FUSED_SIZE, max(1024, triton.next_power_of_2((V + 65534) // 65535)))
            elementwise_mul_kernel[(triton.cdiv(V, B_db),)](
                x=db,
                g=do,
                N=V,
                B=B_db,
                num_warps=STATIC_WARPS,
            )
    return dx, dw, db
