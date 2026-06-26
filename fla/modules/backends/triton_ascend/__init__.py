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

    def cross_entropy_loss(
        self,
        logits,
        target,
        label_smoothing=0.0,
        logit_scale=1.0,
        lse_square_scale=0.0,
        logit_softcapping=None,
        ignore_index=-100,
        inplace_backward=False,
        process_group=None,
    ):
        from fla.modules.backends.triton_ascend.fused_cross_entropy import (
            cross_entropy_loss_npu,
        )
        return cross_entropy_loss_npu(
            logits,
            target,
            label_smoothing,
            logit_scale,
            lse_square_scale,
            logit_softcapping,
            ignore_index,
            inplace_backward,
            process_group,
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

    def sigmoid_fwd(self, x, output_contiguous=False):
        from fla.modules.backends.triton_ascend.activations import sigmoid_fwd_npu
        return sigmoid_fwd_npu(x, output_contiguous=output_contiguous)

    def sigmoid_bwd(self, x, dy, output_contiguous=False):
        from fla.modules.backends.triton_ascend.activations import sigmoid_bwd_npu
        return sigmoid_bwd_npu(x, dy, output_contiguous=output_contiguous)

    def logsigmoid_fwd(self, x, temperature=1., output_contiguous=False):
        from fla.modules.backends.triton_ascend.activations import logsigmoid_fwd_npu
        return logsigmoid_fwd_npu(x, temperature=temperature, output_contiguous=output_contiguous)

    def logsigmoid_bwd(self, x, dy, temperature=1., output_contiguous=False):
        from fla.modules.backends.triton_ascend.activations import logsigmoid_bwd_npu
        return logsigmoid_bwd_npu(x, dy, temperature=temperature, output_contiguous=output_contiguous)

    def swish_fwd(self, x, output_contiguous=False):
        from fla.modules.backends.triton_ascend.activations import swish_fwd_npu
        return swish_fwd_npu(x, output_contiguous=output_contiguous)

    def swish_bwd(self, x, dy, output_contiguous=False):
        from fla.modules.backends.triton_ascend.activations import swish_bwd_npu
        return swish_bwd_npu(x, dy, output_contiguous=output_contiguous)

    def swiglu_fwd(self, x, y, output_contiguous=False):
        from fla.modules.backends.triton_ascend.activations import swiglu_fwd_npu
        return swiglu_fwd_npu(x, y, output_contiguous=output_contiguous)

    def swiglu_fwdbwd(self, x, y, g, use_weight=False, output_contiguous=False):
        from fla.modules.backends.triton_ascend.activations import swiglu_fwdbwd_npu
        return swiglu_fwdbwd_npu(x, y, g, use_weight=use_weight, output_contiguous=output_contiguous)

    def swiglu_linear(self, x, y, weight, bias):
        from fla.modules.backends.triton_ascend.activations import swiglu_linear_npu
        return swiglu_linear_npu(x, y, weight, bias)

    def gelu_fwd(self, x):
        from fla.modules.backends.triton_ascend.activations import gelu_fwd_npu
        return gelu_fwd_npu(x)

    def gelu_bwd(self, g, x):
        from fla.modules.backends.triton_ascend.activations import gelu_bwd_npu
        return gelu_bwd_npu(g, x)

    def sqrelu_fwd(self, x):
        from fla.modules.backends.triton_ascend.activations import sqrelu_fwd_npu
        return sqrelu_fwd_npu(x)

    def sqrelu_bwd(self, g, x):
        from fla.modules.backends.triton_ascend.activations import sqrelu_bwd_npu
        return sqrelu_bwd_npu(g, x)

    def powglu_fwd(self, x, y, power=3.0, output_contiguous=False):
        from fla.modules.backends.triton_ascend.activations import powglu_fwd_npu
        return powglu_fwd_npu(x, y, power=power, output_contiguous=output_contiguous)

    def powglu_fwdbwd(self, x, y, g, power=3.0, use_weight=False, output_contiguous=False):
        from fla.modules.backends.triton_ascend.activations import powglu_fwdbwd_npu
        return powglu_fwdbwd_npu(
            x, y, g, power=power, use_weight=use_weight, output_contiguous=output_contiguous,
        )

    def powglu_linear(self, x, y, weight, bias, power=3.0):
        from fla.modules.backends.triton_ascend.activations import powglu_linear_npu
        return powglu_linear_npu(x, y, weight, bias, power)

    def fused_kl_div_forward(
        self,
        x,
        target_x,
        weight,
        target_weight,
        reduction='batchmean',
        accumulate_grad_in_fp32=True,
    ):
        from fla.modules.backends.triton_ascend.fused_kl_div import fused_kl_div_forward_npu
        return fused_kl_div_forward_npu(
            x,
            target_x,
            weight,
            target_weight,
            reduction,
            accumulate_grad_in_fp32,
        )

    def fused_kl_div_backward(self, do, dx, dw):
        from fla.modules.backends.triton_ascend.fused_kl_div import fused_kl_div_backward_npu
        return fused_kl_div_backward_npu(do, dx, dw)

    def layer_norm_fwd(
        self,
        x,
        weight,
        bias,
        eps=1e-5,
        residual=None,
        out_dtype=None,
        residual_dtype=None,
        is_rms_norm=False,
        num_groups=1,
    ):
        from fla.modules.backends.triton_ascend.layernorm import layer_norm_fwd_npu
        return layer_norm_fwd_npu(
            x,
            weight,
            bias,
            eps,
            residual,
            out_dtype,
            residual_dtype,
            is_rms_norm,
            num_groups,
        )

    def layer_norm_bwd(
        self,
        dy,
        x,
        weight,
        bias,
        mean=None,
        rstd=None,
        dres=None,
        has_residual=False,
        is_rms_norm=False,
        x_dtype=None,
        recompute_output=False,
        num_groups=1,
    ):
        from fla.modules.backends.triton_ascend.layernorm import layer_norm_bwd_npu
        return layer_norm_bwd_npu(
            dy,
            x,
            weight,
            bias,
            mean,
            rstd,
            dres,
            has_residual,
            is_rms_norm,
            x_dtype,
            recompute_output,
            num_groups,
        )

    def fused_grpo_loss(
        self,
        logits,
        ref_logp,
        input_ids,
        advantages,
        beta=0.1,
        completion_mask=None,
        save_kl=False,
        inplace=False,
    ):
        from fla.modules.backends.triton_ascend.grpo import fused_grpo_loss_npu
        return fused_grpo_loss_npu(
            logits,
            ref_logp,
            input_ids,
            advantages,
            beta,
            completion_mask,
            save_kl,
            inplace,
        )

    def causal_conv1d_fwd(
        self,
        x,
        weight,
        bias,
        residual,
        initial_state=None,
        output_final_state=False,
        activation=None,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        chunk_indices=None,
        BT=64,
        layout_fallback=False,
    ):
        from fla.modules.backends.triton_ascend.causal_conv1d import causal_conv1d_fwd_npu
        return causal_conv1d_fwd_npu(
            x,
            weight,
            bias,
            residual,
            initial_state,
            output_final_state,
            activation,
            cu_seqlens,
            cu_seqlens_cpu,
            chunk_indices,
            BT,
            layout_fallback,
        )

    def causal_conv1d_bwd(
        self,
        x,
        dy,
        dht,
        weight=None,
        bias=None,
        residual=None,
        initial_state=None,
        activation=None,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        chunk_indices=None,
        BT=64,
        layout_fallback=False,
    ):
        from fla.modules.backends.triton_ascend.causal_conv1d import causal_conv1d_bwd_npu
        return causal_conv1d_bwd_npu(
            x,
            dy,
            dht,
            weight,
            bias,
            residual,
            initial_state,
            activation,
            cu_seqlens,
            cu_seqlens_cpu,
            chunk_indices,
            BT,
            layout_fallback,
        )

    def compute_dh0_triton(
        self,
        dy,
        y,
        weight,
        initial_state,
        activation,
        cu_seqlens,
    ):
        from fla.modules.backends.triton_ascend.causal_conv1d import compute_dh0_npu
        return compute_dh0_npu(
            dy,
            y,
            weight,
            initial_state,
            activation,
            cu_seqlens,
        )

    def causal_conv1d_update_states(
        self,
        x,
        state_len,
        initial_state=None,
        cu_seqlens=None,
    ):
        from fla.modules.backends.triton_ascend.causal_conv1d import causal_conv1d_update_states_npu
        return causal_conv1d_update_states_npu(
            x,
            state_len,
            initial_state,
            cu_seqlens,
        )

    def causal_conv1d_update(
        self,
        x,
        cache,
        residual=None,
        weight=None,
        bias=None,
        activation=None,
    ):
        from fla.modules.backends.triton_ascend.causal_conv1d import causal_conv1d_update_npu
        return causal_conv1d_update_npu(
            x,
            cache,
            residual,
            weight,
            bias,
            activation,
        )
