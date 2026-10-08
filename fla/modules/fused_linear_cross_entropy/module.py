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
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate, Shard, distribute_module
from torch.distributed.tensor.parallel import ParallelStyle

from fla.modules.fused_linear_cross_entropy.ops import fused_linear_cross_entropy_loss

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

try:
    from torch.distributed.tensor import DTensor
except (ImportError, AttributeError):
    DTensor = None


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
