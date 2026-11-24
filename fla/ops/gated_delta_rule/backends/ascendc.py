# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GDN adapters for the optional fla_npu AscendC wheel."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

import torch

from fla.backends import BaseBackend, register
from fla.modules.norm.l2norm import l2norm
from fla.ops.common.gate import fused_beta_sigmoid
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd
from fla.ops.gated_delta_rule.gate import fused_gdn_gate
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext


def _supports_qkv(q, k, v) -> bool:
    return (
        q.device.type == 'npu' and 'Ascend950' in torch.npu.get_device_name(q.device)
        and q.ndim == v.ndim == 4 and min(q.shape) > 0 and min(v.shape) > 0
        and q.shape == k.shape and q.shape[:2] == v.shape[:2]
        and q.shape[-1] == v.shape[-1] == 128
        and v.shape[2] % q.shape[2] == 0 and v.shape[2] // q.shape[2] in (1, 2, 3, 4)
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.device == k.device == v.device
    )


def _matches(tensor, shape, device, dtypes) -> bool:
    return (
        isinstance(tensor, torch.Tensor) and tensor.shape == shape
        and tensor.device == device and tensor.dtype in dtypes
    )


def _requires_backward(*tensors) -> bool:
    return torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in tensors
    )


def _chunk_fwd(q, k, v, g, beta, scale, initial_state, output_final_state, state_v_first, offsets, indices, save):
    from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd

    return chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=output_final_state,
        scale=scale, state_v_first=state_v_first, cu_seqlens=offsets, chunk_indices=indices,
        chunk_size=64, use_exp2=True, disable_recompute=save, layout='BSND',
    )


class _AscendCChunkGDNFunction(torch.autograd.Function):

    @staticmethod
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, beta, scale, offsets, indices):
        o, _, g_cumsum, A, *_ = _chunk_fwd(
            q, k, v, g, beta, scale, None, False, False, offsets, indices, True,
        )
        ctx.save_for_backward(q, k, v, g_cumsum, beta, A)
        ctx.scale = scale
        ctx.offsets = offsets
        ctx.indices = indices
        return o

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do):
        from fla_npu.ops.ascendc import chunk_gated_delta_rule_bwd

        q, k, v, g_cumsum, beta, A = ctx.saved_tensors
        dq, dk, dv, db, *_ = chunk_gated_delta_rule_bwd(
            q, k, v, g_cumsum, beta, A, do, ctx.scale,
            chunk_size=64, use_exp2=True, layout='BSND',
            cu_seqlens=ctx.offsets, chunk_indices=ctx.indices,
        )
        return dq, dk, dv, None, db, None, None, None


@register(chunk_gated_delta_rule, env_var='FLA_GDN_ASCENDC')
@register(fused_recurrent_gated_delta_rule_fwd, env_var='FLA_GDN_ASCENDC')
class AscendCGDNBackend(BaseBackend):
    backend_type = 'ascendc'
    package_name = 'fla_npu'
    default_enable = False
    priority = -1

    def chunk_gated_delta_rule_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if cp_context is not None or return_intermediate_states or chunk_indices is not None:
            return False, 'AscendC GDN uses canonical chunk order and does not support CP or intermediate states'
        if kwargs.keys() - {'chunk_size', 'use_gate_in_kernel', 'A_log', 'dt_bias'} or kwargs.get('chunk_size', 64) != 64:
            return False, 'AscendC GDN requires chunk_size=64 and supported optional arguments'
        if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
            return False, 'allow_neg_eigval requires beta sigmoid'
        if not _supports_qkv(q, k, v):
            return False, 'AscendC GDN requires Ascend950 with BF16 Q/K/V, K=V=128 and a head ratio of 1 through 4'
        gate_dtypes = (torch.bfloat16, torch.float32)
        if not all(_matches(x, v.shape[:-1], q.device, gate_dtypes) for x in (g, beta)):
            return False, 'AscendC GDN requires BF16 or FP32 gate and beta [B, T, HV]'
        a_log = kwargs.get('A_log') if kwargs.get('use_gate_in_kernel') else None
        dt_bias = kwargs.get('dt_bias') if kwargs.get('use_gate_in_kernel') else None
        if kwargs.get('use_gate_in_kernel') and not _matches(a_log, (v.shape[2],), q.device, gate_dtypes):
            return False, 'AscendC GDN requires A_log [HV] for raw gates'
        if dt_bias is not None and not _matches(dt_bias, (v.shape[2],), q.device, gate_dtypes):
            return False, 'AscendC GDN requires dt_bias [HV]'
        # The wheel's gate gradient exceeds the GDN tolerance on BF16 training inputs.
        if _requires_backward(g, a_log, dt_bias):
            return False, 'AscendC GDN training requires a frozen decay gate'
        sequences = q.shape[0]
        if cu_seqlens is not None:
            if sequences != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
                return False, 'AscendC GDN requires cumulative sequence lengths [N+1] and B=1 for packed inputs'
            if cu_seqlens.dtype != torch.long or cu_seqlens.device not in (q.device, torch.device('cpu')):
                return False, 'AscendC GDN requires int64 cu_seqlens on CPU or the input device'
            if cu_seqlens_cpu is not None and not _matches(
                cu_seqlens_cpu, cu_seqlens.shape, torch.device('cpu'), (torch.long,),
            ):
                return False, 'AscendC GDN requires an int64 CPU mirror of cu_seqlens'
            sequences = cu_seqlens.numel() - 1
        elif cu_seqlens_cpu is not None:
            return False, 'cu_seqlens_cpu requires cu_seqlens'
        if initial_state is not None and not _matches(
            initial_state, (sequences, v.shape[2], 128, 128), q.device, (torch.float32,),
        ):
            return False, 'AscendC GDN requires FP32 initial state'
        if (initial_state is not None or output_final_state) and _requires_backward(q, k, v, beta, initial_state):
            # The wheel backward requires BF16 state inputs, including dht.
            return False, 'AscendC GDN with recurrent state is inference-only'
        if scale is not None and (not isinstance(scale, (int, float)) or not isfinite(scale)):
            return False, 'AscendC GDN requires a finite scalar scale'
        return True, None

    @input_guard
    def chunk_gated_delta_rule(
        self,
        q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
        state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None,
        cp_context=None, return_intermediate_states=False, **kwargs,
    ):
        if kwargs.get('use_gate_in_kernel'):
            g = fused_gdn_gate(g, kwargs['A_log'], kwargs.get('dt_bias'))
        if use_qk_l2norm_in_kernel:
            q, k = l2norm(q), l2norm(k)
        if use_beta_sigmoid_in_kernel:
            beta = fused_beta_sigmoid(beta, scale=2.0 if allow_neg_eigval else 1.0)

        offsets = indices = None
        if cu_seqlens is not None:
            host_offsets = cu_seqlens.cpu() if cu_seqlens_cpu is None else cu_seqlens_cpu
            offsets = host_offsets.tolist()
            indices = prepare_chunk_indices(host_offsets, 64).flatten().tolist()
        if scale is None:
            scale = k.shape[-1] ** -0.5
        if _requires_backward(q, k, v, beta):
            return _AscendCChunkGDNFunction.apply(q, k, v, g, beta, scale, offsets, indices), None

        o, final_state, *_ = _chunk_fwd(
            q, k, v, g, beta, scale, initial_state, output_final_state, state_v_first, offsets, indices, False,
        )
        if output_final_state and offsets is not None:
            empty = [i for i, (start, end) in enumerate(zip(offsets, offsets[1:])) if start == end]
            if empty:
                # ChunkFwdH only writes state for sequences that have a chunk.
                final_state[empty] = 0 if initial_state is None else initial_state[empty]
        return o, final_state

    def fused_recurrent_gated_delta_rule_fwd_verifier(
        self, q, k, v, g=None, gk=None, gv=None, beta=None, A_log=None, dt_bias=None, scale=None,
        initial_state=None, output_final_state=False, inplace_final_state=False, use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, state_v_first=False, cu_seqlens=None,
    ) -> tuple[bool, str | None]:
        if inplace_final_state:
            return False, 'AscendC recurrent GDN returns a separate final state'
        if not _supports_qkv(q, k, v) or v.shape[2] > 256:
            return False, 'AscendC recurrent GDN requires Ascend950 with BF16 Q/K/V, K=V=128 and at most 256 heads'
        # The Ascend950 kernel returns early for sequences longer than MAX_MTP.
        if q.shape[1] > 8 or cu_seqlens is not None or gk is not None or gv is not None:
            return False, 'AscendC recurrent GDN supports dense scalar-gated decoding with at most 8 tokens'
        dtypes = (torch.bfloat16, torch.float32)
        if not _matches(g, v.shape[:-1], q.device, dtypes) or not _matches(beta, v.shape[:-1], q.device, dtypes):
            return False, 'AscendC recurrent GDN requires scalar gate and beta per value head'
        if A_log is not None and not _matches(A_log, (v.shape[2],), q.device, dtypes):
            return False, 'AscendC recurrent GDN requires A_log [HV]'
        if dt_bias is not None and not _matches(dt_bias, (v.shape[2],), q.device, dtypes):
            return False, 'AscendC recurrent GDN requires dt_bias [HV]'
        if initial_state is not None and not _matches(
            initial_state, (q.shape[0], v.shape[2], 128, 128), q.device, dtypes,
        ):
            return False, 'AscendC recurrent GDN requires BF16 or FP32 initial state'
        if not isinstance(scale, (int, float)) or not isfinite(scale):
            return False, 'AscendC recurrent GDN requires a finite scalar scale'
        return True, None

    @input_guard
    def fused_recurrent_gated_delta_rule_fwd(
        self, q, k, v, g=None, gk=None, gv=None, beta=None, A_log=None, dt_bias=None, scale=None,
        initial_state=None, output_final_state=False, inplace_final_state=False, use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, state_v_first=False, cu_seqlens=None,
    ):
        from fla_npu.ops.ascendc import recurrent_gated_delta_rule

        if A_log is not None:
            g = fused_gdn_gate(g, A_log, dt_bias)
        if use_qk_l2norm_in_kernel:
            q, k = l2norm(q), l2norm(k)
        if use_beta_sigmoid_in_kernel:
            beta = fused_beta_sigmoid(beta, scale=2.0 if allow_neg_eigval else 1.0)

        batch, tokens, heads, dim = v.shape
        if initial_state is None:
            state = v.new_zeros((batch, heads, dim, dim), dtype=torch.float32)
        else:
            state = initial_state if state_v_first else initial_state.transpose(-1, -2)
            # The decode kernel updates the V-first state in place.
            state = state.to(dtype=torch.float32, memory_format=torch.contiguous_format, copy=True)

        lengths = torch.full((batch + 1,), tokens, dtype=torch.int32, device=q.device)
        lengths[0] = 0
        slots = torch.arange(batch, dtype=torch.int32, device=q.device)
        if tokens > 1:
            slots = slots.repeat_interleave(tokens)
        out = recurrent_gated_delta_rule(
            q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), state,
            beta=beta.flatten(0, 1).to(v.dtype), g=g.flatten(0, 1).float(), scale=scale,
            actual_seq_lengths=lengths, ssm_state_indices=slots,
        )
        if output_final_state and not state_v_first:
            state = state.transpose(-1, -2).contiguous()
        return out.reshape_as(v), state if output_final_state else None
