# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import copy

import pytest
import torch

from fla.models import GLAConfig
from fla.models.gla.modeling_gla import GLAForCausalLM, GLAModel
from fla.ops.attnres import fused_attnres
from fla.utils import assert_close, device, device_platform

from .utils import check_forward, check_generation, create_model_and_config


@pytest.mark.skipif(device_platform == 'cpu', reason='GLA kernels require an accelerator')
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'attnres_block_size', 'dtype', 'varlen'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-bs{}-{}-varlen{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64,  True,  None, torch.bfloat16, True),
            (4, 4, 1024, 4, 64,  False, None, torch.bfloat16, True),
            (4, 4, 1024, 4, 128, False, None, torch.bfloat16, True),
            (4, 4, 1024, 4, 64,  False, 1,    torch.bfloat16, True),
            (4, 4, 1024, 4, 64,  False, 4,    torch.bfloat16, True),
            (2, 2, 65,   2, 64,  False, None, torch.float16,  False),
            (3, 2, 129,  2, 64,  False, 4,    torch.bfloat16, False),
        ]
    ],
)
def test_modeling(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    use_l2warp: bool,
    attnres_block_size: int | None,
    dtype: torch.dtype,
    varlen: bool,
):
    check_forward(
        L,
        B,
        T,
        H,
        D,
        GLAConfig,
        use_l2warp=use_l2warp,
        attnres_block_size=attnres_block_size,
        dtype=dtype,
        varlen=varlen,
    )


@pytest.mark.skipif(device_platform == 'cpu', reason='GLA generation requires an accelerator')
@pytest.mark.parametrize(('L', 'B', 'T', 'H', 'D', 'dtype', 'block_size'), [
    pytest.param(2, 4, 2000, 8, 64, torch.float16, None, id='standard'),
    pytest.param(2, 2, 256, 2, 64, torch.float16, 1, id='full-attnres'),
    pytest.param(3, 2, 256, 2, 64, torch.float16, 4, id='block-attnres'),
])
def test_generation(L, B, T, H, D, dtype, block_size):
    torch.manual_seed(42)
    model, config = create_model_and_config(GLAConfig, L, H, D, dtype, attnres_block_size=block_size)
    check_generation(L, B, T, H, D, GLAConfig, dtype, model=model, config=config)


@pytest.mark.skipif(device_platform == 'cpu', reason='GLA kernels require an accelerator')
@pytest.mark.parametrize(('block_size', 'fuse_norm', 'dtype'), [
    pytest.param(None, True, torch.bfloat16, id='standard-fused'),
    pytest.param(None, False, torch.float16, id='standard-unfused'),
    pytest.param(1, True, torch.bfloat16, id='full-attnres'),
    pytest.param(4, True, torch.float16, id='block-attnres'),
])
def test_residual(block_size, fuse_norm, dtype):
    """Compare three GLA layers with explicit residual accumulation, including every layer output and gradient."""
    torch.manual_seed(42)
    model = GLAModel(GLAConfig(
        hidden_size=128,
        num_hidden_layers=3,
        num_heads=2,
        intermediate_size=256,
        vocab_size=128,
        attnres_block_size=block_size,
        fuse_norm=fuse_norm,
        use_cache=False,
    )).to(device=device, dtype=dtype)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('query.weight'):
                parameter.normal_(std=0.03)
    x = torch.randn(2, 65, 128, device=device, dtype=dtype, requires_grad=True)
    actual = model(inputs_embeds=x, output_hidden_states=True)

    expected_hidden_states = []
    if block_size is None:
        hidden_states = x
        for layer in model.layers:
            expected_hidden_states.append(hidden_states)
            branch_output = layer.attn(layer.residual_attn.norm(hidden_states))[0]
            if fuse_norm:
                mlp_input, residual = layer.residual_mlp.norm(branch_output, hidden_states, prenorm=True)
            else:
                residual = hidden_states + branch_output
                mlp_input = layer.residual_mlp.norm(residual)
            hidden_states = residual + layer.mlp(mlp_input)
        expected_hidden_states.append(model.residual_output.norm(hidden_states))
    else:
        blocks, partial = [x], None
        for layer_idx, layer in enumerate(model.layers):
            expected_hidden_states.append(x if partial is None else partial)
            for branch_idx, (residual, branch) in enumerate([
                (layer.residual_attn, layer.attn),
                (layer.residual_mlp, layer.mlp),
            ]):
                index = 2 * layer_idx + branch_idx
                if index == 0:
                    branch_input = residual.norm(x)
                else:
                    sources = [*blocks, partial]
                    branch_input = fused_attnres(
                        query=residual.query.weight,
                        residuals=sources,
                        rms_weight=residual.attnres_norm.weight,
                        output_rms_weight=residual.norm.weight,
                        rms_eps=residual.norm.eps,
                    )
                    if index % block_size == 0:
                        blocks, partial = sources, None
                branch_output = branch(branch_input)
                if branch_idx == 0:
                    branch_output = branch_output[0]
                partial = branch_output if partial is None else partial + branch_output
        output = model.residual_output
        expected_hidden_states.append(fused_attnres(
            query=output.query.weight,
            residuals=[*blocks, partial],
            rms_weight=output.attnres_norm.weight,
            output_rms_weight=output.norm.weight,
            rms_eps=output.norm.eps,
        ))
    for expected, hidden in zip(expected_hidden_states, actual.hidden_states, strict=True):
        assert_close('hidden_states', expected, hidden, 1e-3)
    parameters = (x, *model.parameters())
    gradient = torch.randn_like(actual.last_hidden_state)
    actual_grads = torch.autograd.grad(actual.last_hidden_state, parameters, gradient, allow_unused=True)
    expected_grads = torch.autograd.grad(expected_hidden_states[-1], parameters, gradient, allow_unused=True)
    for expected, actual in zip(expected_grads, actual_grads, strict=True):
        if expected is None:
            assert actual is None
        else:
            assert actual is not None and torch.isfinite(actual).all()
            assert_close('gradient', expected, actual, 1e-3)


@pytest.mark.skipif(device_platform == 'cpu', reason='GLA checkpointing requires an accelerator')
@pytest.mark.parametrize(('block_size', 'use_reentrant', 'dtype'), [
    pytest.param(None, True, torch.float16, id='standard-reentrant'),
    pytest.param(4, False, torch.bfloat16, id='attnres-non-reentrant'),
])
def test_checkpointing(block_size, use_reentrant, dtype):
    """Preserve loss and gradients when recomputing three layers from a residual state."""
    torch.manual_seed(42)
    model = GLAForCausalLM(GLAConfig(
        hidden_size=128,
        num_hidden_layers=3,
        num_heads=2,
        intermediate_size=256,
        vocab_size=128,
        attnres_block_size=block_size,
        fuse_cross_entropy=False,
        use_cache=False,
    )).to(device=device, dtype=dtype).train()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('query.weight'):
                parameter.normal_(std=0.03)
    reference = copy.deepcopy(model)
    model.gradient_checkpointing_enable({'use_reentrant': use_reentrant})
    labels = torch.randint(model.config.vocab_size, (2, 65), device=device)
    x = torch.randn(2, 65, 128, device=device, dtype=dtype, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    actual = model(inputs_embeds=x, labels=labels)
    expected = reference(inputs_embeds=ref_x, labels=labels)
    actual.loss.backward()
    expected.loss.backward()
    assert_close('loss', expected.loss, actual.loss, 1e-3)
    assert_close('dx', ref_x.grad, x.grad, 1e-3)
    for (name, parameter), (_, target) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        if target.grad is None:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert_close(name, target.grad, parameter.grad, 1e-3)
