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


@pytest.mark.skipif(device_platform == 'cpu', reason='GLA checkpointing requires an accelerator')
@pytest.mark.parametrize(('block_size', 'fuse_norm', 'dtype', 'checkpoint_kwargs', 'input_kind'), [
    pytest.param(None, False, torch.float32, None, 'input_ids', id='standard-default'),
    pytest.param(None, True, torch.float16, {'use_reentrant': True}, 'inputs_embeds', id='standard-reentrant'),
    pytest.param(1, True, torch.float16, None, 'input_ids', id='full-attnres-default'),
    pytest.param(4, True, torch.bfloat16, {'use_reentrant': False}, 'inputs_embeds', id='block-attnres'),
])
def test_checkpointing(block_size, fuse_norm, dtype, checkpoint_kwargs, input_kind):
    """Checkpoint three connected layers without changing their outputs or input/parameter gradients."""
    torch.manual_seed(42)
    config = GLAConfig(
        hidden_size=128,
        num_hidden_layers=3,
        num_heads=2,
        intermediate_size=256,
        vocab_size=128,
        attnres_block_size=block_size,
        attn_mode='chunk',
        fuse_norm=fuse_norm,
        fuse_swiglu=True,
        fuse_cross_entropy=False,
        use_cache=False,
    )
    model = GLAForCausalLM(config).to(device=device, dtype=dtype).train()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('query.weight'):
                parameter.normal_(std=0.03)
    reference = copy.deepcopy(model)
    model.gradient_checkpointing_enable(checkpoint_kwargs)
    assert model.is_gradient_checkpointing and not reference.is_gradient_checkpointing
    ids = torch.randint(config.vocab_size, (2, 65), device=device)
    if input_kind == 'inputs_embeds':
        embeddings = model.get_input_embeddings()(ids).detach().requires_grad_()
        ref_embeddings = embeddings.detach().clone().requires_grad_()
        actual_inputs, expected_inputs = {'inputs_embeds': embeddings}, {'inputs_embeds': ref_embeddings}
    else:
        actual_inputs = expected_inputs = {'input_ids': ids}
    actual = model(**actual_inputs, labels=ids, output_hidden_states=True)
    expected = reference(**expected_inputs, labels=ids, output_hidden_states=True)
    actual.loss.backward()
    expected.loss.backward()
    if input_kind == 'inputs_embeds':
        assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()
        assert_close('dinputs_embeds', ref_embeddings.grad, embeddings.grad, 1e-3)
    assert_close('loss', expected.loss, actual.loss, 1e-3)
    assert len(actual.hidden_states) == len(expected.hidden_states) == config.num_hidden_layers + 1
    assert actual.past_key_values is None
    for ref, tri in zip(expected.hidden_states, actual.hidden_states):
        assert_close('hidden_states', ref, tri, 1e-3)
    for (name, parameter), (_, target) in zip(model.named_parameters(), reference.named_parameters()):
        if target.grad is None:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert_close(name, target.grad, parameter.grad, 1e-3)
    model.gradient_checkpointing_disable()
    assert not model.is_gradient_checkpointing


@pytest.mark.parametrize(('dtype', 'fuse_norm'), [
    pytest.param(torch.bfloat16, True, id='fused-bf16'),
    pytest.param(torch.float16, False, id='unfused-fp16'),
])
@pytest.mark.skipif(device_platform == 'cpu', reason='Native GLA block kernels require an accelerator')
def test_block(dtype, fuse_norm):
    """Keep the attention-to-MLP fused sum and the materialized layer output numerically unchanged."""
    torch.manual_seed(42)
    config = GLAConfig(
        hidden_size=128,
        num_hidden_layers=1,
        num_heads=2,
        intermediate_size=256,
        vocab_size=128,
        fuse_norm=fuse_norm,
    )
    block = GLAModel(config).to(device=device, dtype=dtype).layers[0]
    x = torch.randn(2, 65, 128, device=device, dtype=dtype, requires_grad=True)
    state = block.residual_attn.initialize(x)
    state, _, _ = block(state)
    output = block.residual_mlp.readout(state)
    legacy_output, attentions, cache, attnres_states = block(x)
    assert attentions is None and cache is None and attnres_states is None
    assert_close('legacy_output', output, legacy_output, 1e-3)
    branch, _, _ = block.attn(block.residual_attn.norm(x))
    if fuse_norm:
        hidden, residual = block.residual_mlp.norm(branch, x, prenorm=True)
    else:
        residual = x + branch
        hidden = block.residual_mlp.norm(residual)
    expected = residual + block.mlp(hidden)
    grad = torch.randn_like(output)
    parameters = (x, *block.parameters())
    actual_grads = torch.autograd.grad(output, parameters, grad_outputs=grad)
    expected_grads = torch.autograd.grad(expected, parameters, grad_outputs=grad)
    assert_close('output', expected, output, 1e-3)
    for actual, target in zip(actual_grads, expected_grads):
        assert_close('gradient', target, actual, 1e-3)
