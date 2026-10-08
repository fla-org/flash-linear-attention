# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Code adapted from
# https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate, Shard, distribute_module
from torch.distributed.tensor.parallel import ParallelStyle

from fla.modules.backends import dispatch
from fla.modules.fused_cross_entropy import cross_entropy_bwd, cross_entropy_fwd
from fla.ops.utils.op import exp, log, tanh
from fla.utils import IS_AMD, input_guard

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

try:
    from torch.distributed.tensor import DTensor
except (ImportError, AttributeError):
    DTensor = None

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576
# https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2
STATIC_WARPS = 32 if not IS_AMD else 16


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'HAS_SOFTCAPPING': lambda args: args['softcapping'] is not None,
})
@triton.jit
def logsumexp_fwd_kernel(
    x,  # [N, D]
    z,  # [N, ND]
    scale,
    softcapping,
    D: tl.constexpr,
    BD: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
):
    i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    # [BD]
    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    if HAS_SCALE:
        b_x = b_x * scale
    if HAS_SOFTCAPPING:
        b_x = softcapping * tanh(b_x / softcapping)
    b_m = tl.max(b_x, 0)
    b_z = log(tl.sum(exp(b_x - b_m), 0)) + b_m
    tl.store(z + i_n * tl.cdiv(D, BD) + i_d, b_z)


@dispatch('modules')
def logsumexp_fwd(
    x,
    scale: float | None = None,
    softcapping: float | None = None,
    dtype: torch.dtype | None = None,
):
    shape = x.shape
    x = x.view(-1, shape[-1])
    N, D = x.shape
    BD = min(triton.next_power_of_2(D), 64 * 1024)
    ND = triton.cdiv(D, BD)

    z = x.new_empty(N, ND, dtype=torch.float)
    logsumexp_fwd_kernel[(N, ND)](
        x=x,
        z=z,
        scale=scale,
        softcapping=softcapping,
        D=D,
        BD=BD,
    )
    z = z.logsumexp(-1).view(*shape[:-1])
    if dtype is not None and dtype != torch.float:
        z = z.to(dtype)
    return z


@triton.jit
def elementwise_mul_kernel(
    x,  # [N]
    g,  # scalar
    N: tl.constexpr,
    BN: tl.constexpr,
):
    i_x = tl.program_id(0).to(tl.int64)
    o_x = i_x * BN + tl.arange(0, BN)

    b_g = tl.load(g)
    if b_g == 1.0:
        return
    # [BN]
    b_x = tl.load(x + o_x, mask=o_x < N)
    tl.store(x + o_x, b_x * b_g, mask=o_x < N)


@dispatch('modules')
def fused_linear_cross_entropy_fwd(
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
    process_group: ProcessGroup | None = None,
):
    device = x.device
    world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
    # inputs have shape: [N, H]
    # materialized activations will have shape: [N, V]
    # the increase in memory = [N, V]
    # reduction can be achieved by partitioning the number of tokens N into smaller chunks.

    # ideally, we would like to achieve the same memory consumption as [N, H],
    # so the expected chunk size should be:
    # NC = ceil(V / H)
    # C = ceil(N / NC)
    # for ex: N = 4096*4, V = 32000, H = 4096 ==> NC = 8, C = ceil(N / NC) = 2048
    N, H, V = *x.shape, weight.shape[0]
    NC = min(num_chunks, triton.cdiv(V, H))
    C = triton.next_power_of_2(triton.cdiv(N, NC))
    NC = triton.cdiv(N, C)

    # [N, H]
    dx = torch.zeros_like(x, device=device)
    grad_dtype = torch.float32 if accumulate_grad_in_fp32 else weight.dtype
    bias_grad_dtype = None
    if bias is not None:
        bias_grad_dtype = torch.float32 if accumulate_grad_in_fp32 else bias.dtype

    # [V, H]
    dw = torch.zeros_like(weight, device=device, dtype=grad_dtype) if weight is not None else None
    # [V]
    db = torch.zeros_like(bias, device=device, dtype=bias_grad_dtype) if bias is not None else None
    # [N]
    loss = torch.zeros(N, device=device, dtype=torch.float)

    total = target.ne(ignore_index).sum().clamp_min(1)
    dloss = total.float().reciprocal() if reduction == "mean" else x.new_ones((), dtype=torch.float)

    for ic in range(NC):
        start, end = ic * C, min((ic + 1) * C, N)
        # [C, H]
        c_x = x[start:end]
        # when doing matmul, use the original precision
        # [C, V]
        c_logits = F.linear(c_x, weight, bias)
        if weight is not None and c_x.dtype != grad_dtype:
            c_x = c_x.to(dtype=grad_dtype)
        c_target = target[start:end]
        c_loss, _, c_lse, total_classes, class_start_idx = cross_entropy_fwd(
            logits=c_logits,
            target=c_target,
            label_smoothing=label_smoothing,
            logit_scale=logit_scale,
            logit_softcapping=logit_softcapping,
            ignore_index=ignore_index,
            process_group=process_group,
        )
        loss[start:end] = c_loss
        if use_l2warp:
            c_maxx, c_ids = torch.max(c_logits, -1, keepdim=True)
            if world_size > 1:
                # only the shard owning the first global maximum contributes the L2 gradient
                local_max = c_maxx.clone()
                torch.distributed.all_reduce(c_maxx, op=torch.distributed.ReduceOp.MAX, group=process_group)
                c_ids = torch.where(local_max == c_maxx, c_ids + class_start_idx, total_classes)
                torch.distributed.all_reduce(c_ids, op=torch.distributed.ReduceOp.MIN, group=process_group)
                c_ids -= class_start_idx
                c_maxx = torch.where((c_ids >= 0) & (c_ids < V), c_maxx, 0)
                c_ids = c_ids.clamp(0, V - 1)

        cross_entropy_bwd(
            logits=c_logits,
            target=c_target,
            lse=c_lse,
            dloss=dloss.expand(end - start),
            label_smoothing=label_smoothing,
            logit_scale=logit_scale,
            logit_softcapping=logit_softcapping,
            ignore_index=ignore_index,
            total_classes=total_classes,
            class_start_idx=class_start_idx,
            inplace_backward=True,
        )
        if use_l2warp:
            # a. Calculate the L2 gradient w.r.t logits (g_logits_l2)
            g_logits_l2 = torch.zeros_like(c_logits)

            # Match L2Wrap: normalize by the full number of input tokens, not by non-ignored labels.
            l2_factor = l2_penalty_factor / N
            penalty_grad = c_maxx * l2_factor
            g_logits_l2.scatter_(-1, c_ids, penalty_grad)

            # b. Backpropagate g_logits_l2 to get its effect on dx, dw, db
            # and add it to the main gradients.
            # Total_dx = CE_dx + L2_dx
            # Total_dw = CE_dw + L2_dw
            # Total_db = CE_db + L2_db
            if weight is not None:
                torch.addmm(
                    input=dw,
                    mat1=g_logits_l2.t().to(dtype=grad_dtype),
                    mat2=c_x,
                    out=dw,
                )
            if bias is not None:
                torch.add(input=db, other=g_logits_l2.sum(0, dtype=bias_grad_dtype), out=db)
            # The dx contribution must be added to the final dx calculation
            dx_l2_contribution = torch.mm(g_logits_l2, weight)
        else:
            dx_l2_contribution = 0.0

        dx[start:end] = torch.mm(c_logits, weight) + dx_l2_contribution

        if weight is not None:
            torch.addmm(
                input=dw,
                mat1=c_logits.t().to(dtype=grad_dtype),
                mat2=c_x,
                out=dw,
            )

        if bias is not None:
            torch.add(input=db, other=c_logits.sum(0, dtype=bias_grad_dtype), out=db)

    if world_size > 1:
        torch.distributed.all_reduce(dx, group=process_group)
    loss = loss.sum()
    if reduction == "mean":
        loss = loss / total
    if dw is not None:
        dw = dw.to(weight)
    if db is not None:
        db = db.to(bias)
    return loss, dx, dw, db


@dispatch('modules')
def fused_linear_cross_entropy_bwd(
    do: torch.Tensor,
    dx: torch.Tensor,
    dw: torch.Tensor,
    db: torch.Tensor,
):
    # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
    # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
    N, H = dx.shape
    BN = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

    elementwise_mul_kernel[(triton.cdiv(N * H, BN),)](
        x=dx,
        g=do,
        N=N*H,
        BN=BN,
        num_warps=STATIC_WARPS,
    )

    # handle dw
    if dw is not None:
        V, H = dw.shape
        elementwise_mul_kernel[(triton.cdiv(V * H, BN),)](
            x=dw,
            g=do,
            N=V*H,
            BN=BN,
            num_warps=STATIC_WARPS,
        )

    if db is not None:
        V = db.shape[0]
        elementwise_mul_kernel[(triton.cdiv(V, BN),)](
            x=db,
            g=do,
            N=V,
            BN=BN,
            num_warps=STATIC_WARPS,
        )
    return dx, dw, db


fused_linear_cross_entropy_forward = fused_linear_cross_entropy_fwd
fused_linear_cross_entropy_backward = fused_linear_cross_entropy_bwd


class FusedLinearCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
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
        process_group: ProcessGroup | None = None,
    ):
        """Precompute gradients per token chunk so backward does not retain the logits."""
        loss, dx, dw, db = fused_linear_cross_entropy_fwd(
            x=x,
            target=target,
            weight=weight,
            bias=bias,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            logit_scale=logit_scale,
            logit_softcapping=logit_softcapping,
            num_chunks=num_chunks,
            reduction=reduction,
            use_l2warp=use_l2warp,
            l2_penalty_factor=l2_penalty_factor,
            accumulate_grad_in_fp32=accumulate_grad_in_fp32,
            process_group=process_group,
        )
        # downcast to dtype and store for backward
        ctx.save_for_backward(
            dx.detach(),
            dw.detach() if weight is not None else None,
            db.detach() if bias is not None else None,
        )
        return loss

    @staticmethod
    @input_guard
    def backward(ctx, do):
        dx, dw, db = ctx.saved_tensors
        dx, dw, db = fused_linear_cross_entropy_bwd(do=do, dx=dx, dw=dw, db=db)
        return dx, None, dw, db, None, None, None, None, None, None, None, None, None, None


def fused_linear_cross_entropy_loss(
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
    process_group: ProcessGroup | None = None,
) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): [batch_size * seq_len, hidden_size]
        target (torch.LongTensor): [batch_size * seq_len]
            where each value is in [0, vocab_size).
        weight (torch.Tensor): [vocab_size, hidden_size]
            where `vocab_size` is the number of classes.
        bias (Optional[torch.Tensor]): [vocab_size]
            where `vocab_size` is the number of classes.
        ignore_index: int.
            If target == ignore_index, the loss is set to 0.0.
        label_smoothing: float
        logit_scale: float
            A scaling factor applied to the logits. Default: 1.0
        logit_softcapping: float
            If not None, apply logit softcapping: logits = softcap * tanh(logits / softcap).
            Default: `None`.
        num_chunks: int
            The number of chunks to split the input tensor into for processing.
            This can help optimize memory usage and computation speed.
            Default: 8
        reduction:
            Specifies the reduction to apply to the output: 'mean' | 'sum'.
            'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.
            Default: 'mean'.
        use_l2warp:
            Whether to add the L2Warp logit regularization gradient. The penalty is normalized by
            the full number of input tokens, matching `fla.modules.l2warp.l2_warp`.
            Default: `False`.
        l2_penalty_factor:
            The L2Warp penalty factor. Default: `1e-4`.
        accumulate_grad_in_fp32:
            Whether to accumulate weight and bias gradients in fp32 before casting them
            back to the parameter dtype. Default: `True`.
        process_group (ProcessGroup, Optional):
            Group with equal contiguous vocabulary shards of weight and bias, and replicated inputs and targets.
            Loss and input gradients are replicated; weight and bias gradients remain sharded. Default: `None`.
            Multi-rank groups are not supported by the Ascend backend.
    Returns:
        Scalar loss in fp32.
    """
    return FusedLinearCrossEntropyFunction.apply(
        x,
        target,
        weight,
        bias,
        ignore_index,
        label_smoothing,
        logit_scale,
        logit_softcapping,
        num_chunks,
        reduction,
        use_l2warp,
        l2_penalty_factor,
        accumulate_grad_in_fp32,
        process_group,
    )


class FusedLinearCrossEntropyLoss(nn.Module):

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        logit_scale: float = 1.0,
        logit_softcapping: float = None,
        num_chunks: int = 8,
        reduction: str = "mean",
        use_l2warp: bool = False,
        l2_penalty_factor: float = 1e-4,
        accumulate_grad_in_fp32: bool = True,
        process_group: ProcessGroup | None = None,
    ):
        """
        Args:
            ignore_index: int.
                If target == ignore_index, the loss is set to 0.0.
            label_smoothing: float
            logit_scale: float
                A scaling factor applied to the logits. Default: 1.0
            logit_softcapping: float
                If not None, apply logit softcapping: logits = softcap * tanh(logits / softcap).
                Default: `None`.
            num_chunks: int
                The number of chunks to split the input tensor into for processing.
                This can help optimize memory usage and computation speed.
                Default: 8
            reduction:
                Specifies the reduction to apply to the output: 'mean' | 'sum'.
                'mean': the weighted mean of the output is taken,
                'sum': the output will be summed.
                Default: 'mean'.
            use_l2warp:
                Whether to add the L2Warp logit regularization gradient. The penalty is normalized by
                the full number of input tokens, matching `fla.modules.l2warp.l2_warp`.
                Default: `False`.
            l2_penalty_factor:
                The L2Warp penalty factor. Default: `1e-4`.
            accumulate_grad_in_fp32:
                Whether to accumulate weight and bias gradients in fp32 before casting them
                back to the parameter dtype. Default: `True`.
            process_group (ProcessGroup, Optional):
                Group with equal contiguous vocabulary shards of weight and bias, and replicated inputs and targets.
                Loss and input gradients are replicated; weight and bias gradients remain sharded. Default: `None`.
                Multi-rank groups are not supported by the Ascend backend.
        """
        super().__init__()

        assert reduction in ["mean", "sum"], f"reduction: {reduction} is not supported"

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.logit_scale = logit_scale
        self.logit_softcapping = logit_softcapping
        self.num_chunks = num_chunks
        self.reduction = reduction
        self.use_l2warp = use_l2warp
        self.l2_penalty_factor = l2_penalty_factor
        self.accumulate_grad_in_fp32 = accumulate_grad_in_fp32
        self.process_group = process_group

    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        target: torch.LongTensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        """
        Args:
            x (torch.Tensor): [batch_size, seq_len, hidden_size]
            target (torch.LongTensor): [batch_size, seq_len]
                where each value is in [0, V).
            weight (torch.Tensor): [vocab_size, hidden_size]
                where `vocab_size` is the number of classes.
            bias (Optional[torch.Tensor]): [vocab_size]
                where `vocab_size` is the number of classes.
        Returns:
            loss
        """
        loss = fused_linear_cross_entropy_loss(
            x=x.reshape(-1, x.shape[-1]),
            target=target.reshape(-1),
            weight=weight,
            bias=bias,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            logit_scale=self.logit_scale,
            logit_softcapping=self.logit_softcapping,
            num_chunks=self.num_chunks,
            reduction=self.reduction,
            use_l2warp=self.use_l2warp,
            l2_penalty_factor=self.l2_penalty_factor,
            accumulate_grad_in_fp32=self.accumulate_grad_in_fp32,
            process_group=self.process_group,
        )
        return loss


class LinearLossParallel(ParallelStyle):
    def __init__(
        self,
        *,
        sequence_dim: int = 1,
        use_local_output: bool = False,
    ):
        super().__init__()

        self.sequence_sharding = (Shard(sequence_dim),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
        x, target, weight, bias = inputs

        if not isinstance(x, DTensor):
            # assume the input passed in already sharded on the sequence dim and create the DTensor
            x = DTensor.from_local(x, device_mesh, sequence_sharding)
        if x.placements != sequence_sharding:
            x = x.redistribute(placements=sequence_sharding, async_op=True)
        if not isinstance(target, DTensor):
            target = DTensor.from_local(target, device_mesh, [Replicate()])
        if target.placements != sequence_sharding:
            target = target.redistribute(placements=sequence_sharding, async_op=True)

        if not isinstance(weight, DTensor):
            weight = DTensor.from_local(weight, device_mesh, [Replicate()])
        if weight.placements != [Replicate()]:
            # we replicate the weight/bias in FLCE
            weight = weight.redistribute(placements=[Replicate()], async_op=True)

        if bias is not None and not isinstance(bias, DTensor):
            bias = DTensor.from_local(bias, device_mesh, [Replicate()])
        if bias is not None and bias.placements != [Replicate()]:
            bias = bias.redistribute(placements=[Replicate()], async_op=True)

        return x.to_local(), target.to_local(), weight.to_local(), bias.to_local() if bias is not None else bias

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=None,
            input_fn=partial(self._prepare_input_fn, self.sequence_sharding),
            output_fn=partial(self._prepare_output_fn, self.use_local_output),
        )
