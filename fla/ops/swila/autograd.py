# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Preserve the actual input device's autocast state for eager-capable helpers."""

from functools import wraps

import torch


def autocast_custom_fwd(function):
    @wraps(function)
    def forward(ctx, *args, **kwargs):
        tensor = next(value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor))
        ctx.swila_amp_device_type = tensor.device.type
        return torch.amp.custom_fwd(function, device_type=tensor.device.type)(ctx, *args, **kwargs)

    return forward


def autocast_custom_bwd(function):
    @wraps(function)
    def backward(ctx, *args, **kwargs):
        return torch.amp.custom_bwd(function, device_type=ctx.swila_amp_device_type)(ctx, *args, **kwargs)

    return backward
