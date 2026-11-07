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


def _check_residual_gradients(model, x, ref, tri):
    do = torch.randn_like(tri)
    parameters = {'x': x, **dict(model.named_parameters())}
    ref_grads = torch.autograd.grad(ref, tuple(parameters.values()), do, allow_unused=True)
    tri_grads = torch.autograd.grad(tri, tuple(parameters.values()), do, allow_unused=True)
    for name, ref_grad, tri_grad in zip(parameters, ref_grads, tri_grads, strict=True):
        if ref_grad is None:
            assert tri_grad is None, name
        else:
            assert tri_grad is not None and torch.isfinite(tri_grad).all(), name
            assert_close(name, ref_grad, tri_grad, 1e-3)


@pytest.mark.skipif(device_platform == 'cpu', reason='GLA kernels require an accelerator')
@pytest.mark.parametrize(('fuse_norm', 'dtype'), [
    pytest.param(True, torch.bfloat16, id='fused-bf16'),
    pytest.param(False, torch.float16, id='unfused-fp16'),
])
def test_standard_residual(fuse_norm: bool, dtype: torch.dtype):
    """Match three pre-norm GLA layers, including fused addition rounding and all training gradients."""
    torch.manual_seed(42)
    model = GLAModel(GLAConfig(
        hidden_size=128,
        num_hidden_layers=3,
        num_heads=2,
        intermediate_size=256,
        vocab_size=128,
        fuse_norm=fuse_norm,
        use_cache=False,
    )).to(device=device, dtype=dtype)
    x = torch.randn(2, 65, 128, device=device, dtype=dtype, requires_grad=True)
    tri = model(inputs_embeds=x, output_hidden_states=True)

    hidden_states, ref_hidden_states = x, []
    for layer in model.layers:
        ref_hidden_states.append(hidden_states)
        branch_output = layer.attn(layer.attn_residual.norm(hidden_states))[0]
        if fuse_norm:
            mlp_input, residual = layer.mlp_residual.norm(branch_output, hidden_states, prenorm=True)
        else:
            residual = hidden_states + branch_output
            mlp_input = layer.mlp_residual.norm(residual)
        hidden_states = residual + layer.mlp(mlp_input)
    ref_hidden_states.append(model.output_residual.norm(hidden_states))

    for i, (ref, out) in enumerate(zip(ref_hidden_states, tri.hidden_states, strict=True)):
        assert_close(f'layer_{i}', ref, out, 1e-3)
    _check_residual_gradients(model, x, ref_hidden_states[-1], tri.last_hidden_state)


@pytest.mark.skipif(device_platform == 'cpu', reason='GLA kernels require an accelerator')
@pytest.mark.parametrize(('block_size', 'dtype'), [
    pytest.param(1, torch.bfloat16, id='full-bf16'),
    pytest.param(4, torch.float16, id='block-fp16'),
])
def test_attention_residual(block_size: int, dtype: torch.dtype):
    """Compare three GLA layers against block sums formed directly from their six branch outputs."""
    torch.manual_seed(42)
    model = GLAModel(GLAConfig(
        hidden_size=128,
        num_hidden_layers=3,
        num_heads=2,
        intermediate_size=256,
        vocab_size=128,
        attnres_block_size=block_size,
        use_cache=False,
    )).to(device=device, dtype=dtype)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('query.weight'):
                parameter.normal_(std=0.03)
    x = torch.randn(2, 65, 128, device=device, dtype=dtype, requires_grad=True)
    tri = model(inputs_embeds=x, output_hidden_states=True)

    branches = [branch for layer in model.layers for branch in (layer.attn, layer.mlp)]
    residuals = [residual for layer in model.layers for residual in (layer.attn_residual, layer.mlp_residual)]
    outputs, ref_hidden_states = [], [x]
    for i, (branch, residual) in enumerate(zip(branches, residuals, strict=True)):
        sources = [x] + [sum(outputs[start:start + block_size]) for start in range(0, i, block_size)]
        branch_input = residual.norm(x) if i == 0 else fused_attnres(
            query=residual.query.weight,
            residuals=sources,
            rms_weight=residual.attnres_norm.weight,
            output_rms_weight=residual.norm.weight,
            rms_eps=residual.norm.eps,
        )
        branch_output = branch(branch_input)
        outputs.append(branch_output[0] if i % 2 == 0 else branch_output)
        if i % 2 == 1 and i + 1 < len(branches):
            ref_hidden_states.append(sum(outputs[(i // block_size) * block_size:]))
    output = model.output_residual
    ref_hidden_states.append(fused_attnres(
        query=output.query.weight,
        residuals=[x] + [sum(outputs[start:start + block_size]) for start in range(0, len(outputs), block_size)],
        rms_weight=output.attnres_norm.weight,
        output_rms_weight=output.norm.weight,
        rms_eps=output.norm.eps,
    ))

    for i, (ref, out) in enumerate(zip(ref_hidden_states, tri.hidden_states, strict=True)):
        assert_close(f'layer_{i}', ref, out, 1e-3)
    _check_residual_gradients(model, x, ref_hidden_states[-1], tri.last_hidden_state)


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
