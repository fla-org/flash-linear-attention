# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

import fla.layers.mamba as mamba_module
from fla.layers import Mamba


class UnfusedScanReached(Exception):
    pass


def test_mamba_training_path_honors_attention_mask(monkeypatch):
    torch.manual_seed(42)
    B, T, hidden_size = 2, 8, 32
    intermediate_size = 2 * hidden_size

    layer = Mamba(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        state_size=8,
        conv_kernel=4,
        layer_idx=0,
        backend='triton',
    ).to(device='cpu', dtype=torch.float32).train()

    fused_inputs = []
    conv_inputs = []

    def fused(projected_states, *args, **kwargs):
        fused_inputs.append(projected_states.detach().clone())
        raise AssertionError("the fused training kernel must not run when an attention_mask is given")

    def unfused_scan(*args, **kwargs):
        raise UnfusedScanReached

    def conv(**kwargs):
        # relay the tensor under test so no triton launch is needed
        conv_inputs.append(kwargs['x'].detach().clone())
        return kwargs['x'], None

    monkeypatch.setattr(mamba_module, "mamba_inner_fn", fused)
    monkeypatch.setattr(mamba_module, "selective_scan_fn", unfused_scan)
    layer.causal_conv1d_fn = conv

    hidden_states = torch.randn(B, T, hidden_size)
    attention_mask = torch.zeros(B, T, dtype=torch.long)
    attention_mask[0, 3:] = 1
    attention_mask[1, 1:] = 1

    with pytest.raises(UnfusedScanReached):
        layer.cuda_kernels_forward(hidden_states, None, False, attention_mask)

    assert fused_inputs == []

    # the depthwise conv is the first consumer of the padded projection
    padding = attention_mask == 0
    assert torch.count_nonzero(conv_inputs[0][padding]) == 0
