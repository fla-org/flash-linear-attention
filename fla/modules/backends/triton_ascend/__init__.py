# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend (Huawei NPU) backend for FLA modules."""

from __future__ import annotations

from fla.ops.backends import BaseBackend


class TritonAscendBackend(BaseBackend):
    """Ascend NPU backend using triton-ascend kernels."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def rotary_embedding_fwdbwd(
        self,
        x,
        cos,
        sin,
        seqlen_offsets=0,
        cu_seqlens=None,
        interleaved=False,
        inplace=False,
        conjugate=False,
        chunk_indices=None,
    ):
        from fla.modules.backends.triton_ascend.rotary import rotary_embedding_fwdbwd_npu
        return rotary_embedding_fwdbwd_npu(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            interleaved=interleaved,
            inplace=inplace,
            conjugate=conjugate,
            chunk_indices=chunk_indices,
        )

    def fused_cross_entropy_forward(
        self,
        logits,
        target,
        label_smoothing=0.0,
        logit_scale=1.0,
        lse_square_scale=0.0,
        logit_softcapping=None,
        ignore_index=-100,
        process_group=None,
    ):
        from fla.modules.backends.triton_ascend.fused_cross_entropy import (
            fused_cross_entropy_forward_npu,
        )
        return fused_cross_entropy_forward_npu(
            logits,
            target,
            label_smoothing,
            logit_scale,
            lse_square_scale,
            logit_softcapping,
            ignore_index,
            process_group,
        )

    def fused_cross_entropy_backward(
        self,
        dlogits,
        grad_losses,
        logits,
        lse,
        target,
        label_smoothing,
        logit_scale,
        lse_square_scale,
        logit_softcapping,
        ignore_index,
        total_classes,
        class_start_idx,
    ):
        from fla.modules.backends.triton_ascend.fused_cross_entropy import (
            fused_cross_entropy_backward_npu,
        )
        return fused_cross_entropy_backward_npu(
            dlogits,
            grad_losses,
            logits,
            lse,
            target,
            label_smoothing,
            logit_scale,
            lse_square_scale,
            logit_softcapping,
            ignore_index,
            total_classes,
            class_start_idx,
        )

    def logsumexp_fwd(
        self,
        x,
        scale=None,
        softcapping=None,
        dtype=None,
    ):
        from fla.modules.backends.triton_ascend.fused_linear_cross_entropy import (
            logsumexp_fwd_npu,
        )
        return logsumexp_fwd_npu(x, scale=scale, softcapping=softcapping, dtype=dtype)

    def fused_linear_cross_entropy_forward(
        self,
        x,
        target,
        weight,
        bias=None,
        ignore_index=-100,
        label_smoothing=0.0,
        logit_scale=1.0,
        logit_softcapping=None,
        num_chunks=8,
        reduction="mean",
        use_l2warp=False,
        l2_penalty_factor=1e-4,
        accumulate_grad_in_fp32=True,
    ):
        from fla.modules.backends.triton_ascend.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_forward_npu,
        )
        return fused_linear_cross_entropy_forward_npu(
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
        )

    def fused_linear_cross_entropy_backward(
        self,
        do,
        dx,
        dw,
        db,
    ):
        from fla.modules.backends.triton_ascend.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_backward_npu,
        )
        return fused_linear_cross_entropy_backward_npu(do, dx, dw, db)
