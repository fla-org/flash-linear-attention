# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch


def chunk_local_cumsum_verifier(
    g, chunk_size, reverse=False, scale=None, cu_seqlens=None, output_dtype=torch.float,
    chunk_indices=None, use_graph=False, **kwargs,
):
    if kwargs:
        return False, 'AscendC does not accept extra arguments'
    if g.device.type != 'npu' or g.layout != torch.strided or g.ndim != 3 or min(g.shape) <= 0:
        return False, 'AscendC requires a nonempty NPU tensor in [B, T, H] layout'
    if not torch.npu.get_device_name(g.device).startswith('Ascend950'):
        return False, 'This AscendC build supports only Ascend950'
    if g.dtype not in (torch.float32, torch.float16, torch.bfloat16) or output_dtype != torch.float32:
        return False, 'AscendC requires FP32/FP16/BF16 input and FP32 output'
    if not isinstance(chunk_size, int) or chunk_size not in (64, 128) or reverse:
        return False, 'AscendC supports forward chunks of size 64 or 128'
    if cu_seqlens is not None or chunk_indices is not None or use_graph:
        return False, 'AscendC supports dense input without graph task metadata'
    if g.requires_grad and torch.is_grad_enabled():
        return False, 'The AscendC helper does not register autograd'
    if scale is not None and (
        not isinstance(scale, (int, float)) or not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38
    ):
        return False, 'AscendC requires a finite FP32 scalar scale'
    return True, None


def chunk_local_cumsum(
    g, chunk_size, reverse=False, scale=None, cu_seqlens=None, output_dtype=torch.float,
    chunk_indices=None, use_graph=False, **kwargs,
):
    from fla.utils.ascendc import chunk_local_cumsum as native_cumsum

    out = native_cumsum(g=g.transpose(1, 2).contiguous(), chunk_size=chunk_size, scale=1.0 if scale is None else scale)
    return out.transpose(1, 2).contiguous()
