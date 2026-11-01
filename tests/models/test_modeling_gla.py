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
from fla.utils import IS_NVIDIA, assert_close, device, device_platform

from .utils import check_forward, check_generation, create_model_and_config

REQUIRES_NVIDIA = pytest.mark.skipif(not IS_NVIDIA, reason='mHC requires NVIDIA CUDA')


@pytest.mark.skipif(device_platform == 'cpu', reason='GLA kernels require an accelerator')
@pytest.mark.parametrize(
    ('L', 'B', 'T', 'H', 'D', 'use_l2warp', 'residual_mode', 'residual_kwargs', 'dtype', 'varlen'),
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-{}-{}-{}-varlen{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64,  True,  None,       {},                torch.bfloat16, True),
            (4, 4, 1024, 4, 64,  False, None,       {},                torch.bfloat16, True),
            (4, 4, 1024, 4, 128, False, None,       {},                torch.bfloat16, True),
            (4, 4, 1024, 4, 64,  False, 'standard', {},                torch.bfloat16, True),
            (4, 4, 1024, 4, 64,  False, 'attnres',  {'block_size': 1}, torch.bfloat16, True),
            (4, 4, 1024, 4, 64,  False, 'attnres',  {'block_size': 4}, torch.bfloat16, True),
            (2, 2, 65,   2, 64,  False, None,       {},                torch.float16,  False),
            (2, 2, 129,  2, 64,  False, 'attnres',  {'block_size': 4}, torch.bfloat16, False),
        ]
    ] + [
        pytest.param(4, 4, 1024, 4, 64, False, 'mhc', {'num_streams': 4}, torch.bfloat16, True,
                     id='mhc-varlen', marks=REQUIRES_NVIDIA),
        pytest.param(2, 2, 65, 2, 64, False, 'mhc', {'num_streams': 2}, torch.float16, False,
                     id='mhc-training', marks=REQUIRES_NVIDIA),
    ],
)
def test_modeling(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    use_l2warp: bool,
    residual_mode: str | None,
    residual_kwargs: dict,
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
        residual_mode=residual_mode,
        residual_kwargs=residual_kwargs,
        dtype=dtype,
        varlen=varlen,
    )


@pytest.mark.parametrize(
    ('L', 'B', 'T', 'H', 'D', 'dtype', 'residual_mode', 'residual_kwargs'),
    [
        (2, 4, 2000, 8, 64, torch.float16, None, {}),
        (2, 2, 256, 2, 64, torch.float16, 'standard', {}),
        (2, 2, 256, 2, 64, torch.float16, 'attnres', {'block_size': 1}),
        (2, 2, 256, 2, 64, torch.float16, 'attnres', {'block_size': 4}),
        pytest.param(2, 2, 256, 2, 64, torch.float16, 'mhc', {'num_streams': 4}, marks=REQUIRES_NVIDIA),
    ],
)
@pytest.mark.skipif(device_platform == 'cpu', reason='Native GLA generation requires an accelerator')
def test_generation(
    residual_mode: str | None,
    residual_kwargs: dict,
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    model, config = create_model_and_config(
        GLAConfig, L, H, D, dtype, residual_mode=residual_mode, residual_kwargs=residual_kwargs,
    )
    check_generation(L, B, T, H, D, GLAConfig, dtype, model=model, config=config)


@pytest.mark.skipif(device_platform != 'cuda', reason='GLA checkpointing tests require CUDA')
@pytest.mark.parametrize(
    ('residual_mode', 'residual_kwargs', 'param_dtype', 'dtype', 'checkpoint_kwargs', 'input_kind', 'num_layers'),
    [
        pytest.param('standard', {'fuse_norm': False}, torch.float32, torch.float32, None, 'input_ids', 1,
                     id='standard-default'),
        pytest.param('standard_defer', {}, torch.float16, torch.float16, {'use_reentrant': True}, 'inputs_embeds', 3,
                     id='deferred-reentrant-embeds'),
        pytest.param('attnres', {'block_size': 4}, torch.bfloat16, torch.bfloat16, {'use_reentrant': False}, 'input_ids', 3,
                     id='attnres-block'),
        pytest.param('mhc', {'num_streams': 4}, torch.bfloat16, torch.bfloat16, {'use_reentrant': True}, 'input_ids', 3,
                     id='mhc-reentrant', marks=REQUIRES_NVIDIA),
        pytest.param('mhc', {'num_streams': 2}, torch.float32, torch.bfloat16, None, 'input_ids', 1,
                     id='mhc-autocast', marks=REQUIRES_NVIDIA),
    ],
)
def test_checkpointing(
    residual_mode, residual_kwargs, param_dtype, dtype, checkpoint_kwargs, input_kind, num_layers, monkeypatch,
):
    torch.manual_seed(42)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    config = GLAConfig(
        hidden_size=128,
        num_hidden_layers=num_layers,
        num_heads=2,
        intermediate_size=256,
        vocab_size=128,
        residual_mode=residual_mode,
        residual_kwargs=residual_kwargs,
        attn_mode='chunk',
        fuse_norm=True,
        fuse_swiglu=True,
        fuse_cross_entropy=False,
        use_cache=False,
    )
    model = GLAForCausalLM(config).to(device=device, dtype=param_dtype).train()
    assert config.residual_mode == residual_mode
    assert model.model.residual_output.fuse_norm == residual_kwargs.get('fuse_norm', config.fuse_norm)
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
    with torch.autocast(device_type=torch.device(device).type, dtype=dtype, enabled=param_dtype != dtype):
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
        assert parameter.dtype == param_dtype
        if target.grad is None:
            assert parameter.grad is None, name
            assert residual_mode != 'mhc' or (input_kind == 'inputs_embeds' and name == 'model.embeddings.weight'), name
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert_close(name, target.grad, parameter.grad, 1e-3)
    model.gradient_checkpointing_disable()
    assert not model.is_gradient_checkpointing


@pytest.mark.parametrize(('dtype', 'fuse_norm'), [(torch.float16, True), (torch.bfloat16, False)])
@pytest.mark.skipif(device_platform == 'cpu', reason='Native GLA block kernels require an accelerator')
def test_block(dtype, fuse_norm):
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
    output = block.residual_mlp.get_hidden_state(state)
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
