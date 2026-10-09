# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from fla.models.gla.modeling_gla import GLAForCausalLM, GLAModel
from fla.modules.residuals import StandardResidual, register_residual, registry

from .test_gla_external_norm import cpu_ops, make_config  # noqa: F401


class ZeroLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        nn.init.constant_(self.bias, 0.25)


class ProjectedResidual(StandardResidual):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
        self.projection = nn.Sequential(ZeroLinear(hidden_size, hidden_size))

    def forward(self, branch_output, history):
        return super().forward(branch_output + self.projection(branch_output), history)


class InitializedResidual(ProjectedResidual):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
        self.gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = nn.Parameter(torch.empty(()))
        self.register_buffer('offset', torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.scale, 0.75)
        nn.init.constant_(self.offset, 0.5)
        nn.init.constant_(self.norm.weight, 1.25)

    def forward(self, branch_output, history):
        return super().forward(self.scale * branch_output + self.gate(branch_output) + self.offset, history)


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, '_RESIDUAL_CLASSES', registry._RESIDUAL_CLASSES.copy())


@pytest.mark.parametrize('model_class', [GLAModel, GLAForCausalLM])
def test_custom_child_initialization_is_preserved(model_class):
    register_residual('test.projected', ProjectedResidual)
    model = model_class(make_config(mode='test.projected'))
    backbone = model if isinstance(model, GLAModel) else model.model
    for layer in backbone.layers:
        for residual in (layer.residual_attn, layer.residual_mlp):
            projection = residual.projection[0]
            torch.testing.assert_close(projection.weight, torch.zeros_like(projection.weight), atol=0, rtol=0)
            torch.testing.assert_close(projection.bias, torch.full_like(projection.bias, 0.25), atol=0, rtol=0)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    model.init_weights()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


@pytest.mark.parametrize('model_class', [GLAModel, GLAForCausalLM])
def test_custom_residual_parameter_initialization(model_class):
    register_residual('test.initialized', InitializedResidual)
    model = model_class(make_config(mode='test.initialized'))
    standalone = InitializedResidual(hidden_size=32, fuse_norm=False)
    backbone = model if isinstance(model, GLAModel) else model.model
    for layer in backbone.layers:
        for residual in (layer.residual_attn, layer.residual_mlp):
            for name, value in residual.state_dict().items():
                torch.testing.assert_close(value, standalone.state_dict()[name], atol=0, rtol=0)
    ids = torch.arange(6).reshape(2, 3)
    output = backbone(ids).last_hidden_state
    output.square().mean().backward()
    for residual in (backbone.layers[0].residual_attn, backbone.layers[-1].residual_mlp):
        assert residual.scale.grad is not None and torch.isfinite(residual.scale.grad)
        assert residual.gate.weight.grad is not None and torch.isfinite(residual.gate.weight.grad).all()


@pytest.mark.parametrize('model_class', [GLAModel, GLAForCausalLM])
@pytest.mark.parametrize('missing', [None, 'scale', 'offset', 'gate.weight', 'projection.0.weight', 'norm.weight'])
def test_custom_residual_checkpoint_initialization(model_class, missing, tmp_path):
    register_residual('test.initialized', InitializedResidual)
    model = model_class(make_config(mode='test.initialized'))
    prefix = '' if model_class is GLAModel else 'model.'
    missing_key = f'{prefix}layers.0.residual_attn.{missing}' if missing is not None else None
    defaults = {name: value.clone() for name, value in model.state_dict().items()}
    state = {name: torch.randn_like(value) for name, value in defaults.items() if name != missing_key}
    model.config.save_pretrained(tmp_path)
    save_file(state, tmp_path / 'model.safetensors', metadata={'format': 'pt'})
    loaded, info = model_class.from_pretrained(tmp_path, output_loading_info=True)
    assert set(info['missing_keys']) == ({missing_key} if missing_key is not None else set())
    assert not info['unexpected_keys'] and not info['mismatched_keys']
    loaded.init_weights()
    for name, value in loaded.state_dict().items():
        expected = defaults[name] if name == missing_key else state[name]
        torch.testing.assert_close(value, expected, atol=0, rtol=0, msg=name)
