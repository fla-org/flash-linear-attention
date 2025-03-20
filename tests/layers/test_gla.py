# -*- coding: utf-8 -*-

import pytest
import torch

from fla.layers.gla import GatedLinearAttention


@pytest.mark.parametrize("B", [4, 8])
@pytest.mark.parametrize("T", [1024, 2048])
@pytest.mark.parametrize("H", [2048])
@pytest.mark.parametrize("activation", ['swish'])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gla(
    B: int,
    T: int,
    H: int,
    dtype: torch.dtype,
    activation: str
):
    naive = GatedLinearAttention(hidden_size=H, gate_fn=activation, fuse_norm=False).to(dtype).cuda()
    fused = GatedLinearAttention(hidden_size=H, gate_fn=activation, fuse_norm=True).to(dtype).cuda()
    fused.q_proj.weight.data.copy_(naive.q_proj.weight.data)
    fused.k_proj.weight.data.copy_(naive.k_proj.weight.data)
    fused.v_proj.weight.data.copy_(naive.v_proj.weight.data)
    fused.g_proj.weight.data.copy_(naive.g_proj.weight.data)
    fused.o_proj.weight.data.copy_(naive.o_proj.weight.data)
    fused.gk_proj[0].weight.data.copy_(naive.gk_proj[0].weight.data)
    fused.gk_proj[1].weight.data.copy_(naive.gk_proj[1].weight.data)
    fused.gk_proj[1].bias.data.copy_(naive.gk_proj[1].bias.data)

    x = torch.randn(B, T, H, dtype=dtype).cuda()
    naive_x = x.clone().requires_grad_(True)
    fused_x = x.clone().requires_grad_(True)
    naive_o, *_ = naive(naive_x)
    fused_o, *_ = fused(fused_x)
    naive_o.sum().backward()
    fused_o.sum().backward()
    assert naive_o.allclose(fused_o, 0, 1e-2)
    assert naive_x.grad.allclose(fused_x.grad, 0, 1e-2)
