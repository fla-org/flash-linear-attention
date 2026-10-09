# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


import copy
import json

import pytest
import torch

from fla.models.gla.configuration_gla import GLAConfig
from fla.models.gla.modeling_gla import GLAModel
from fla.modules.residuals import StandardResidual, register_residual, registry

from .test_gla_external_norm import cpu_ops, make_config  # noqa: F401


@pytest.mark.parametrize(('mode', 'options'), [
    ('standard', {}),
    ('attnres', {'block_size': 3}),
    ('mhc', {'num_streams': 2, 'num_iters': 3, 'init_scale': 0.05}),
])
def test_nested_residual_options_roundtrip(mode, options, tmp_path):
    torch.manual_seed(42)
    options = dict(options, norm_eps=1e-4, fuse_norm=False)
    expected_options = copy.deepcopy(options)
    config = make_config(mode=mode, fuse_norm=True, residual_kwargs=options)
    options['norm_eps'] = 0.1
    assert config.residual_kwargs == expected_options
    config.get_residual_kwargs(0)['norm_eps'] = 0.2
    assert config.residual_kwargs == expected_options
    model = GLAModel(config)
    assert type(model.layers[0].residual_attn.norm) is torch.nn.RMSNorm
    assert model.layers[0].residual_attn.norm.eps == 1e-4
    if mode == 'attnres':
        assert model.layers[1].residual_mlp.is_boundary
    elif mode == 'mhc':
        assert model.layers[0].residual_attn.num_streams == 2
        assert model.layers[0].residual_attn.routing.num_iters == 3
        assert model.layers[-1].residual_mlp.is_last and model.layers[-1].residual_mlp.routing is None
    ids = torch.arange(6).reshape(2, 3)
    expected = model(ids).last_hidden_state
    model.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / 'config.json').read_text())
    assert saved['residual_kwargs'] == expected_options
    assert not {'attnres_block_size', 'mhc_num_streams', 'mhc_sinkhorn_iters', 'mhc_init_scale'} & saved.keys()
    restored = GLAModel.from_pretrained(tmp_path)
    torch.testing.assert_close(restored(ids).last_hidden_state, expected, atol=0, rtol=0)


@pytest.mark.parametrize('mode', ['standard', 'attnres', 'mhc'])
def test_residual_defaults_belong_to_modules(mode):
    model = GLAModel(make_config(mode=mode))
    assert model.config.residual_kwargs == {}
    assert not any(hasattr(model.config, name) for name in (
        'attnres_block_size', 'mhc_num_streams', 'mhc_sinkhorn_iters', 'mhc_init_scale',
    ))
    ids = torch.arange(6).reshape(2, 3)
    assert model(ids).last_hidden_state.shape == (2, 3, 32)


@pytest.mark.parametrize('options', [False, [], 4])
def test_residual_options_require_dict(options):
    with pytest.raises(TypeError, match='dict'):
        GLAConfig(residual_kwargs=options)


@pytest.mark.parametrize('name', ['hidden_size', 'sub_layer_idx', 'num_sublayers'])
def test_model_layout_cannot_be_overridden(name):
    with pytest.raises(ValueError, match='supplied by the model'):
        GLAConfig(residual_kwargs={name: 2})


@pytest.mark.parametrize(('mode', 'options'), [
    ('standard', {'block_size': 4}),
    ('standard', {'fuse_residual': False}),
    ('attnres', {'num_streams': 2}),
    ('mhc', {'sinkhorn_iter': 3}),
])
def test_unused_residual_options_are_rejected(mode, options):
    with pytest.raises(TypeError, match='Unexpected'):
        GLAModel(make_config(mode=mode, residual_kwargs=options))


def test_custom_residual_checkpoint_roundtrip(monkeypatch, tmp_path):
    class ScaledResidual(StandardResidual):
        def __init__(self, *args, branch_scale=1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.branch_scale = branch_scale

        def forward(self, branch_output, history):
            return super().forward(self.branch_scale * branch_output, history)

    torch.manual_seed(42)
    monkeypatch.setattr(registry, '_RESIDUAL_CLASSES', registry._RESIDUAL_CLASSES.copy())
    name = 'test.scaled'
    register_residual(name, ScaledResidual)
    model = GLAModel(make_config(mode=name, residual_kwargs={'branch_scale': 0.25}))
    for layer in model.layers:
        for residual in (layer.residual_attn, layer.residual_mlp):
            assert isinstance(residual, ScaledResidual) and residual.branch_scale == 0.25
    ids = torch.arange(6).reshape(2, 3)
    expected = model(ids).last_hidden_state
    expected.square().sum().backward()
    assert torch.isfinite(model.embeddings.weight.grad).all()
    model.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / 'config.json').read_text())
    assert saved['residual_mode'] == name and saved['residual_kwargs'] == {'branch_scale': 0.25}

    monkeypatch.delitem(registry._RESIDUAL_CLASSES, name)
    with pytest.raises(ValueError, match='register_residual'):
        GLAModel.from_pretrained(tmp_path)
    register_residual(name, ScaledResidual)
    restored = GLAModel.from_pretrained(tmp_path)
    actual = restored(ids).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    for (key, parameter), (_, restored_parameter) in zip(model.named_parameters(), restored.named_parameters()):
        torch.testing.assert_close(parameter.grad, restored_parameter.grad, atol=0, rtol=0, msg=key)
