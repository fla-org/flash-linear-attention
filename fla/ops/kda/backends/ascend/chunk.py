# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Ascend wheel backend for the complete dense KDA inference call."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

import torch

from fla.backends import BaseBackend, register
from fla.ops.kda.backends.ascend import is_ascendc_available
from fla.ops.kda.chunk import chunk_kda
from fla.utils import input_guard

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext


@register(chunk_kda, env_var='FLA_KDA_ASCENDC')
class AscendCKDABackend(BaseBackend):
    backend_type = 'ascendc'
    package_name = 'fla_npu'
    default_enable = False
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        return is_ascendc_available('FLA_KDA_ASCENDC')

    def chunk_kda_verifier(
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
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        use_graph: bool = False,
        max_num_seqs: int | None = None,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if torch.is_grad_enabled():
            return False, 'AscendC KDA currently supports inference only'
        if q.device.type != 'npu' or 'Ascend950' not in torch.npu.get_device_name(q.device):
            return False, 'AscendC KDA currently requires Ascend950'
        if cu_seqlens is not None or cu_seqlens_cpu is not None or cp_context is not None:
            return False, 'AscendC KDA currently requires dense inputs without context parallelism'
        if use_graph or max_num_seqs is not None or return_intermediate_states or disable_recompute:
            return False, 'AscendC KDA does not support graph mode or saved intermediate states'
        if use_qk_l2norm_in_kernel or use_gate_in_kernel or use_beta_sigmoid_in_kernel or allow_neg_eigval:
            return False, 'AscendC KDA currently requires preprocessed q/k, gate and beta'
        if lower_bound is not None or kwargs.keys() - {'chunk_size'}:
            return False, 'AscendC KDA received unsupported optional arguments'
        if kwargs.get('chunk_size', 64) != 64:
            return False, 'AscendC KDA requires chunk_size=64'
        if q.ndim != 4 or min(q.shape) <= 0 or q.shape[-1] != 128:
            return False, 'AscendC KDA requires nonempty [B, T, H, 128] inputs'
        if k.shape != q.shape or v.shape != q.shape or g.shape != q.shape or beta.shape != q.shape[:-1]:
            return False, 'AscendC KDA requires matching q/k/v/g shapes and beta [B, T, H], without GVA'
        if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
            return False, 'AscendC KDA requires bfloat16 q/k/v'
        if any(x.dtype not in (torch.bfloat16, torch.float32) for x in (g, beta)):
            return False, 'AscendC KDA requires bfloat16 or float32 gate and beta'
        tensors = [q, k, v, g, beta]
        if initial_state is not None:
            if initial_state.shape != (q.shape[0], q.shape[2], 128, 128) or initial_state.dtype != torch.float32:
                return False, 'AscendC KDA requires float32 initial_state [B, H, 128, 128]'
            tensors.append(initial_state)
        if any(x.device != q.device for x in tensors):
            return False, 'AscendC KDA requires all tensors on the same NPU'
        if scale is not None and (not isinstance(scale, (int, float)) or not isfinite(scale)):
            return False, 'AscendC KDA requires a finite scalar scale'
        return True, None

    @input_guard
    def chunk_kda(
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
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        use_graph: bool = False,
        max_num_seqs: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from fla_npu.ops.ascendc import chunk_kda_fwd

        # isolate the public initial state from the wheel's mutable state contract.
        state = initial_state.clone() if initial_state is not None else None
        result = chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=q.shape[-1] ** -0.5 if scale is None else scale,
            chunk_size=64,
            layout='BSND',
            initial_state=state,
            output_final_state=output_final_state,
            safe_gate=safe_gate,
            state_v_first=state_v_first,
        )
        return result[0], result[1] if output_final_state else None
