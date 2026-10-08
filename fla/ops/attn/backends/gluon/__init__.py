# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch

from fla.ops.backends import BaseBackend
from fla.utils import IS_NVIDIA, TRITON_ABOVE_3_5_1, find_spec_cached, get_device_capability


class AttnGluonBackend(BaseBackend):
    backend_type = "gluon"
    package_name = "triton.experimental.gluon"
    env_var = "FLA_ATTN_GLUON"
    default_enable = False
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        return IS_NVIDIA and TRITON_ABOVE_3_5_1 and find_spec_cached(cls.package_name) is not None

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

    def attn_decoding_fwd_verifier(
        self,
        q,
        k,
        v,
        g_cumsum,
        scale,
        cu_seqlens,
        window_size=None,
        sink_bias=None,
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
        )

    def attn_decoding_fwd(self, q, k, v, g_cumsum, scale, cu_seqlens, window_size=None, sink_bias=None):
        from fla.ops.attn.backends.gluon.decoding import attn_decoding_fwd_gluon
        return attn_decoding_fwd_gluon(
            q=q,
            k=k,
            v=v,
            g_cumsum=g_cumsum,
            scale=scale,
            cu_seqlens=cu_seqlens,
            window_size=window_size,
            sink_bias=sink_bias,
        )
