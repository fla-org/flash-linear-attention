# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Gluon backend for FLA modules."""

import torch

from fla.ops.backends import BaseBackend
from fla.utils import IS_NVIDIA, find_spec_cached


class GluonBackend(BaseBackend):
    """NVIDIA GPU backend using Gluon kernels."""

    backend_type = 'gluon'
    package_name = 'triton.experimental.gluon'
    env_var = 'FLA_CONV_GLUON'
    default_enable = False
    priority = 5

    @classmethod
    def is_available(cls) -> bool:
        return IS_NVIDIA and find_spec_cached('triton.experimental.gluon') is not None

    def causal_conv1d_fwd_verifier(
        self,
        x,
        weight,
        bias=None,
        residual=None,
        initial_state=None,
        output_final_state=False,
        activation=None,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        chunk_indices=None,
        BT=64,
        layout_fallback=False,
    ):
        if torch.distributed.is_initialized():
            return False, 'Gluon convolution uses the existing backend in distributed processes'
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False, 'Gluon convolution supports float16, bfloat16, and float32'
        if x.ndim != 3 or x.stride(-1) != 1:
            return False, 'Gluon convolution requires [B, T, D] with contiguous channels'
        if weight is None or weight.shape[0] != x.shape[-1] or weight.shape[1] not in (2, 3, 4):
            return False, 'Gluon convolution requires a width of 2, 3, or 4'
        if cu_seqlens is not None and x.shape[0] != 1:
            return False, 'Gluon packed convolution requires batch size 1'
        if BT != 64:
            return False, 'Gluon convolution requires 64-token chunk indices'
        return True, None

    def causal_conv1d_fwd(self, *args, **kwargs):
        from fla.modules.backends.gluon.causal_conv1d import causal_conv1d_fwd
        return causal_conv1d_fwd(*args, **kwargs)

    def causal_conv1d_bwd_verifier(
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
        if initial_state is not None or dht is not None:
            return False, 'Gluon convolution uses the existing backward for state gradients'
        return self.causal_conv1d_fwd_verifier(x=x, weight=weight, cu_seqlens=cu_seqlens, BT=BT)

    def causal_conv1d_bwd(self, *args, **kwargs):
        from fla.modules.backends.gluon.causal_conv1d import causal_conv1d_bwd
        return causal_conv1d_bwd(*args, **kwargs)
