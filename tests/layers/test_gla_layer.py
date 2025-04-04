# -*- coding: utf-8 -*-

import os

import pytest
import torch

from fla.layers.gla import GatedLinearAttention
from fla.utils import device, device_platform

compiled_mode = os.getenv("COMPILER_MODE") == "1"
if compiled_mode:
    test_b_list = [1]
    test_t_list = [4096]
else:
    test_b_list = [2]
    test_t_list = [1, 15, 63, 300]


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", [2048])
@pytest.mark.parametrize("activation", ['swish'])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    device_platform == 'intel',
    reason="Intel Triton Failure"
)
def test_gla_layer(
    B: int,
    T: int,
    H: int,
    dtype: torch.dtype,
    activation: str
):
    naive = GatedLinearAttention(hidden_size=H, gate_fn=activation, fuse_norm=False).to(dtype).to(device)
    fused = GatedLinearAttention(hidden_size=H, gate_fn=activation, fuse_norm=True).to(dtype).to(device)
    fused.q_proj.weight.data.copy_(naive.q_proj.weight.data)
    fused.k_proj.weight.data.copy_(naive.k_proj.weight.data)
    fused.v_proj.weight.data.copy_(naive.v_proj.weight.data)
    fused.g_proj.weight.data.copy_(naive.g_proj.weight.data)
    fused.o_proj.weight.data.copy_(naive.o_proj.weight.data)
    fused.gk_proj[0].weight.data.copy_(naive.gk_proj[0].weight.data)
    fused.gk_proj[1].weight.data.copy_(naive.gk_proj[1].weight.data)
    fused.gk_proj[1].bias.data.copy_(naive.gk_proj[1].bias.data)

    x = torch.randn(B, T, H, dtype=dtype).to(device)
    naive_x = x.clone().requires_grad_(True)
    fused_x = x.clone().requires_grad_(True)
    naive_o, *_ = naive(naive_x)
    fused_o, *_ = fused(fused_x)
    naive_o.sum().backward()
    fused_o.sum().backward()
    print('Test passed, the gradients will be checked in op test')
