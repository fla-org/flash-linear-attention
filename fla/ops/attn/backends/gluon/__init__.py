# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch

from fla.backends import GluonBackend, register
from fla.utils import TRITON_ABOVE_3_5_1, get_device_capability


@register('attn')
class AttnGluonBackend(GluonBackend):
    env_var = "FLA_ATTN_GLUON"
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        return TRITON_ABOVE_3_5_1 and super().is_available()

    def parallel_attn_fwd_verifier(
        self,
        q,
        k,
        v,
        g_cumsum,
        sink_bias,
        scale,
        window_size=None,
        cu_seqlens=None,
        chunk_indices=None,
    ) -> tuple[bool, str | None]:
        if q.device.type != "cuda" or get_device_capability(q.device.index)[0] not in (9, 10):
            return False, "Gluon attention requires NVIDIA compute capability 9.x or 10.x"
        if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
            return False, "Gluon attention requires matching fp16 or bf16 q/k/v"
        if not (0 < q.shape[-1] <= 512 and 0 < v.shape[-1] <= 512):
            return False, "Gluon attention supports query/key and value dimensions from 1 through 512"
        return True, None

    def parallel_attn_fwd(self, q, k, v, g_cumsum, sink_bias, scale, window_size=None, cu_seqlens=None, chunk_indices=None):
        from fla.ops.attn.backends.gluon.parallel import parallel_attn_fwd_gluon
        return parallel_attn_fwd_gluon(
            q=q,
            k=k,
            v=v,
            g_cumsum=g_cumsum,
            sink_bias=sink_bias,
            scale=scale,
            window_size=window_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    def parallel_attn_bwd_verifier(
        self,
        q,
        k,
        v,
        o,
        g_cumsum,
        lse,
        do,
        sink_bias=None,
        scale=None,
        window_size=None,
        chunk_size=128,
        cu_seqlens=None,
        chunk_indices=None,
    ) -> tuple[bool, str | None]:
        return self.parallel_attn_fwd_verifier(
            q=q,
            k=k,
            v=v,
            g_cumsum=g_cumsum,
            sink_bias=sink_bias,
            scale=scale,
            window_size=window_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    def parallel_attn_bwd(
        self,
        q,
        k,
        v,
        o,
        g_cumsum,
        lse,
        do,
        sink_bias=None,
        scale=None,
        window_size=None,
        chunk_size=128,
        cu_seqlens=None,
        chunk_indices=None,
    ):
        from fla.ops.attn.backends.gluon.parallel import parallel_attn_bwd_gluon
        return parallel_attn_bwd_gluon(
            q=q,
            k=k,
            v=v,
            o=o,
            g_cumsum=g_cumsum,
            lse=lse,
            do=do,
            sink_bias=sink_bias,
            scale=scale,
            window_size=window_size,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    def attn_decoding_one_step_verifier(
        self,
        q,
        k,
        v,
        g=None,
        scale=None,
        cu_seqlens=None,
        do_gate_scale=False,
        *,
        window_size=None,
        sink_bias=None,
    ) -> tuple[bool, str | None]:
        supported, reason = self.parallel_attn_fwd_verifier(
            q=q,
            k=k,
            v=v,
            g_cumsum=None,
            sink_bias=sink_bias,
            scale=scale,
        )
        if not supported:
            return supported, reason
        if cu_seqlens is None:
            return False, "The cu_seqlens must be provided for varlen decoding"
        if window_size is not None and window_size < 0:
            return False, "window_size must be nonnegative"
        H, HQ = k.shape[2], q.shape[2]
        if H == 0 or HQ % H != 0:
            return False, "The number of query heads must be divisible by the number of key/value heads"
        if sink_bias is not None and sink_bias.shape != (HQ,):
            return False, "sink_bias must have shape [HQ]"
        return True, None

    def attn_decoding_one_step(
        self,
        q,
        k,
        v,
        g=None,
        scale=None,
        cu_seqlens=None,
        do_gate_scale=False,
        *,
        window_size=None,
        sink_bias=None,
    ):
        from fla.ops.attn.backends.gluon.decoding import attn_decoding_one_step
        return attn_decoding_one_step(
            q=q,
            k=k,
            v=v,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            do_gate_scale=do_gate_scale,
            window_size=window_size,
            sink_bias=sink_bias,
        )


__all__ = ['AttnGluonBackend']
