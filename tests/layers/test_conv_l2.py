# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from unittest import mock

import pytest
import torch

from fla.layers import comba, delta_net, gated_deltanet, gated_deltaproduct, kda, mesa_net, mom
from fla.models.delta_net.configuration_delta_net import DeltaNetConfig
from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from fla.models.gated_deltaproduct.configuration_gated_deltaproduct import GatedDeltaProductConfig
from fla.models.kda.configuration_kda import KDAConfig
from fla.models.mom.configuration_mom import MomConfig
from fla.utils import assert_close, device


def _make_layer(kind, fuse_conv_l2):
    kwargs = dict(hidden_size=128, head_dim=64, num_heads=2, conv_bias=True)
    if fuse_conv_l2 is not None and kind not in ('comba', 'mesa'):
        kwargs['fuse_conv_l2'] = fuse_conv_l2
    if kind == 'gdn':
        return gated_deltanet.GatedDeltaNet(**kwargs)
    if kind.startswith('delta_'):
        return delta_net.DeltaNet(qk_activation=kind.removeprefix('delta_'), **kwargs)
    if kind == 'comba':
        return comba.Comba(**kwargs)
    if kind == 'gdp':
        return gated_deltaproduct.GatedDeltaProduct(**kwargs)
    if kind == 'mesa':
        return mesa_net.MesaNet(**kwargs)
    if kind.startswith('mom'):
        return mom.MomAttention(num_memories=2, topk=2, shared_mem=kind == 'mom_shared', **kwargs)
    return kda.KimiDeltaAttention(safe_gate=kind == 'kda_safe', lower_bound=-5 if kind == 'kda_safe' else None, **kwargs)


def _check_layer_gradients(reference, actual, ref_x, actual_x, ref_y, actual_y):
    assert torch.isfinite(actual_y).all()
    assert_close('y', ref_y, actual_y, 1e-3)
    dy = torch.randn_like(ref_y)
    ref_y.backward(dy)
    actual_y.backward(dy)
    assert torch.isfinite(actual_x.grad).all()
    assert_close('dx', ref_x.grad, actual_x.grad, 1e-3)
    ref_params = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        ref_grad = ref_params[name].grad
        if ref_grad is None:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert_close(f'd{name}', ref_grad, parameter.grad, 1e-3)


@pytest.mark.parametrize('kind', ['kda', 'kda_safe', 'gdn'])
@pytest.mark.parametrize('varlen', [False, True])
def test_conv_l2_chunk(kind, varlen):
    torch.manual_seed(42)
    reference = _make_layer(kind=kind, fuse_conv_l2=False).to(device, torch.bfloat16).train()
    actual = _make_layer(kind=kind, fuse_conv_l2=True).to(device, torch.bfloat16).train()
    actual.load_state_dict(reference.state_dict(), strict=True)
    B, T = (1, 128) if varlen else (2, 65)
    x = torch.randn(B, T, 128, device=device, dtype=torch.bfloat16)
    ref_x = x.detach().clone().requires_grad_()
    actual_x = x.detach().clone().requires_grad_()
    kwargs = {'cu_seqlens': torch.tensor([0, 63, 128], device=device)} if varlen else {}
    ref_y = reference(hidden_states=ref_x, **kwargs)[0]
    module = gated_deltanet if kind == 'gdn' else kda
    kernel_name = 'chunk_gated_delta_rule' if kind == 'gdn' else 'chunk_kda'
    with mock.patch.object(module, kernel_name, wraps=getattr(module, kernel_name)) as kernel:
        actual_y = actual(hidden_states=actual_x, **kwargs)[0]
    kernel.assert_called_once()
    # preserve FP32 attention gradients before the normalization backward
    assert kernel.call_args.kwargs['use_qk_l2norm_in_kernel']
    _check_layer_gradients(reference, actual, ref_x, actual_x, ref_y, actual_y)


@pytest.mark.parametrize(
    ('kind', 'module', 'kernel_name'),
    [
        ('delta_silu', delta_net, 'chunk_delta_rule'),
        ('delta_identity', delta_net, 'chunk_delta_rule'),
        ('delta_relu', delta_net, 'chunk_delta_rule'),
        ('delta_elu', delta_net, 'chunk_delta_rule'),
        ('comba', comba, 'chunk_comba'),
        ('gdp', gated_deltaproduct, 'chunk_gated_delta_product'),
        ('mesa', mesa_net, 'chunk_mesa_net'),
        ('mom', mom, 'chunk_gated_delta_rule'),
        ('mom_shared', mom, 'chunk_gated_delta_rule'),
    ],
    ids=['delta-silu', 'delta-identity', 'delta-relu', 'delta-elu', 'comba', 'gdp', 'mesa', 'mom', 'mom-shared'],
)
def test_conv_l2_other_layers(kind, module, kernel_name):
    torch.manual_seed(42)
    reference = _make_layer(kind=kind, fuse_conv_l2=None).to(device, torch.bfloat16).train()
    actual = _make_layer(kind=kind, fuse_conv_l2=True).to(device, torch.bfloat16).train()
    if kind not in ('comba', 'mesa'):
        assert reference.fuse_conv_l2 is False
    actual.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(1, 65, 128, device=device, dtype=torch.bfloat16)
    ref_x = x.detach().clone().requires_grad_()
    actual_x = x.detach().clone().requires_grad_()
    ref_y = reference(hidden_states=ref_x)[0]
    with mock.patch.object(module, kernel_name, wraps=getattr(module, kernel_name)) as kernel:
        actual_y = actual(hidden_states=actual_x)[0]
    assert kernel.call_count == (2 if kind == 'mom_shared' else 1)
    for call in kernel.call_args_list:
        assert call.kwargs['use_qk_l2norm_in_kernel']
    _check_layer_gradients(reference, actual, ref_x, actual_x, ref_y, actual_y)


@pytest.mark.parametrize(
    ('kind', 'module', 'kernel_name', 'varlen'),
    [
        ('kda', kda, 'chunk_kda', False),
        ('kda', kda, 'chunk_kda', True),
        ('kda_safe', kda, 'chunk_kda', False),
        ('gdn', gated_deltanet, 'chunk_gated_delta_rule', False),
        ('gdn', gated_deltanet, 'chunk_gated_delta_rule', True),
        ('delta_silu', delta_net, 'chunk_delta_rule', False),
        ('delta_identity', delta_net, 'chunk_delta_rule', False),
        ('delta_relu', delta_net, 'chunk_delta_rule', False),
        ('delta_elu', delta_net, 'chunk_delta_rule', False),
        ('gdp', gated_deltaproduct, 'chunk_gated_delta_product', False),
        ('mom', mom, 'chunk_gated_delta_rule', False),
        ('mom_shared', mom, 'chunk_gated_delta_rule', False),
    ],
    ids=['kda', 'kda-varlen', 'kda-safe', 'gdn-packed', 'gdn-varlen', 'delta-silu', 'delta-identity',
         'delta-relu', 'delta-elu', 'gdp', 'mom', 'mom-shared'],
)
@torch.no_grad()
def test_conv_l2_chunk_inference(kind, module, kernel_name, varlen):
    torch.manual_seed(42)
    reference = _make_layer(kind=kind, fuse_conv_l2=False).to(device, torch.bfloat16).eval()
    actual = _make_layer(kind=kind, fuse_conv_l2=True).to(device, torch.bfloat16).eval()
    actual.load_state_dict(reference.state_dict(), strict=True)
    B, T = (1, 128) if varlen else (2, 65)
    x = torch.randn(B, T, 128, device=device, dtype=torch.bfloat16)
    kwargs = {'cu_seqlens': torch.tensor([0, 63, 128], device=device)} if varlen else {}
    ref_y = reference(hidden_states=x, **kwargs)[0]
    with mock.patch.object(module, kernel_name, wraps=getattr(module, kernel_name)) as kernel:
        actual_y = actual(hidden_states=x, **kwargs)[0]
    assert kernel.call_count == (2 if kind == 'mom_shared' else 1)
    uses_conv_l2 = kind not in ('delta_relu', 'delta_elu') and (kind != 'gdn' or varlen)
    for call in kernel.call_args_list:
        assert call.kwargs['use_qk_l2norm_in_kernel'] == (not uses_conv_l2)
    assert torch.isfinite(actual_y).all()
    assert_close('y', ref_y, actual_y, 1e-3)


@pytest.mark.parametrize('kind', ['kda', 'gdn'])
@torch.no_grad()
def test_conv_l2_recurrent_fallback(kind):
    torch.manual_seed(42)
    reference = _make_layer(kind=kind, fuse_conv_l2=False).to(device, torch.bfloat16).eval()
    actual = _make_layer(kind=kind, fuse_conv_l2=True).to(device, torch.bfloat16).eval()
    actual.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 3, 128, device=device, dtype=torch.bfloat16)
    ref_y = reference(hidden_states=x)[0]
    module = gated_deltanet if kind == 'gdn' else kda
    kernel_name = 'fused_recurrent_gated_delta_rule' if kind == 'gdn' else 'fused_recurrent_kda'
    with mock.patch.object(module, kernel_name, wraps=getattr(module, kernel_name)) as kernel:
        actual_y = actual(hidden_states=x)[0]
    kernel.assert_called_once()
    assert kernel.call_args.kwargs['use_qk_l2norm_in_kernel']
    assert torch.isfinite(actual_y).all()
    assert_close('y', ref_y, actual_y, 0.0)


@pytest.mark.parametrize(
    'config_type',
    [DeltaNetConfig, GatedDeltaNetConfig, GatedDeltaProductConfig, KDAConfig, MomConfig],
)
def test_conv_l2_config_roundtrip(config_type):
    assert config_type().fuse_conv_l2 is False
    config = config_type(fuse_conv_l2=True)
    assert config_type.from_dict(config.to_dict()).fuse_conv_l2 is True
    legacy = config.to_dict()
    legacy.pop('fuse_conv_l2')
    assert config_type.from_dict(legacy).fuse_conv_l2 is False
