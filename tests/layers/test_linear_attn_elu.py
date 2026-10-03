# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.layers.linear_attn import LinearAttention
from fla.modules.activations import elu_p1
from fla.utils import assert_close, device


@pytest.mark.parametrize('seq_len', [1, 65])
@pytest.mark.parametrize('autocast', [False, True])
@pytest.mark.parametrize('grad_enabled', [False, True])
@pytest.mark.parametrize('mode', ['chunk', 'fused_chunk', 'fused_recurrent'])
def test_elu_linear_attention_bf16(seq_len: int, autocast: bool, grad_enabled: bool, mode: str):
    torch.manual_seed(42)
    dtype = torch.float32 if autocast else torch.bfloat16
    layer = LinearAttention(
        mode=mode,
        hidden_size=128,
        num_heads=1,
        feature_map='elu',
        output_norm='identity',
        do_feature_map_norm=True,
    ).to(device=device, dtype=dtype)
    eye = torch.eye(128, device=device, dtype=dtype)
    with torch.no_grad():
        layer.q_proj.weight.copy_(-8 * eye)
        layer.k_proj.weight.zero_()
        layer.v_proj.weight.copy_(eye)
        layer.o_proj.weight.copy_(eye)

    assert layer.feature_map_q is layer.feature_map_k is elu_p1
    x = torch.ones(1, seq_len, 128, device=device, dtype=dtype, requires_grad=grad_enabled)
    with torch.set_grad_enabled(grad_enabled), torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=autocast):
        output = layer(x)[0]
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()

    ref_x = x.detach().double().requires_grad_(grad_enabled)
    ref_weights = [p.detach().double().requires_grad_(grad_enabled) for p in layer.parameters()]
    q, k, v = [F.linear(ref_x, w) for w in ref_weights[:3]]
    q = torch.where(q >= 0, q + 1, q.clamp_max(0).exp())
    k = torch.where(k >= 0, k + 1, k.clamp_max(0).exp())
    scores = (q @ k.mT).tril() * 128 ** -0.5
    ref_output = F.linear((scores @ v) / (scores.sum(-1, keepdim=True) + 1e-10), ref_weights[3])
    torch.testing.assert_close(output.double(), ref_output, rtol=1e-2, atol=1e-2)

    if grad_enabled:
        grads = torch.autograd.grad(output.float().mean(), [x, *layer.parameters()])
        ref_grads = torch.autograd.grad(ref_output.mean(), [ref_x, *ref_weights])
        for name, ref, tri in zip(['x', 'q_proj', 'k_proj', 'v_proj', 'o_proj'], ref_grads, grads):
            assert torch.isfinite(tri).all()
            assert_close(name, ref, tri.double(), 1e-2, err_atol=1e-4)
