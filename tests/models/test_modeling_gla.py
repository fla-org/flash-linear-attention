# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import copy
from unittest.mock import Mock

import pytest
import torch

from fla.models import GLAConfig
from fla.models.gla.modeling_gla import GLAForCausalLM, GLAModel
from fla.modules.residuals import ManifoldHyperConnection
from fla.utils import IS_NVIDIA, assert_close, device, device_platform

from .utils import check_forward, check_generation, create_model_and_config

REQUIRES_NVIDIA = pytest.mark.skipif(not IS_NVIDIA, reason='mHC requires NVIDIA CUDA')


@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
@pytest.mark.skipif(device_platform == 'cpu', reason='Native GLA kernels require an accelerator')
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'residual_mode', 'residual_kwargs', 'dtype'],
    [
        pytest.param(
            *test,
            id="L{}-B{}-T{}-H{}-D{}-l2{}-{}-{}-{}".format(*test),
            marks=REQUIRES_NVIDIA if test[6] == 'mhc' else (),
        )
        for test in [
            (4, 4, 1024, 4, 64,  True,  'standard', {}, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 'standard', {}, torch.bfloat16),
            (4, 4, 1024, 4, 128, False, 'standard', {}, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 'standard_defer', {}, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 'attnres', {'block_size': 1}, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 'attnres', {'block_size': 4}, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 'mhc', {'num_streams': 4}, torch.bfloat16),
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
    residual_mode: str,
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
        (2, 4, 2000, 8, 64, torch.float16, 'standard', {}),
        (2, 2, 256, 2, 64, torch.float16, 'standard_defer', {}),
        (2, 2, 256, 2, 64, torch.float16, 'attnres', {'block_size': 1}),
        (2, 2, 256, 2, 64, torch.float16, 'attnres', {'block_size': 4}),
        pytest.param(2, 2, 256, 2, 64, torch.float16, 'mhc', {'num_streams': 4}, marks=REQUIRES_NVIDIA),
    ],
)
@pytest.mark.skipif(device_platform == 'cpu', reason='Native GLA generation requires an accelerator')
def test_generation(
    residual_mode: str,
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


@pytest.mark.parametrize(('residual_mode', 'residual_kwargs'), [
    pytest.param('standard', {'fuse_norm': False}, id='standard-unfused'),
    pytest.param('standard', {'fuse_norm': True}, id='standard-fused'),
    pytest.param('standard_defer', {'fuse_norm': False}, id='standard_defer-unfused'),
    pytest.param('standard_defer', {'fuse_norm': True}, id='standard_defer-fused'),
    pytest.param('attnres', {'block_size': 1}, id='attnres-full'),
    pytest.param('attnres', {'block_size': 4}, id='attnres-block'),
    *[pytest.param('mhc', {'num_streams': n}, id=f'mhc-{n}') for n in [2, 4, 8]],
])
@pytest.mark.parametrize(('param_dtype', 'dtype', 'autocast'), [
    pytest.param(torch.float32, torch.float32, False, id='fp32'),
    pytest.param(torch.float32, torch.float16, True, id='fp16-amp'),
    pytest.param(torch.float32, torch.bfloat16, True, id='bf16-amp'),
    pytest.param(torch.float16, torch.float16, False, id='fp16-params'),
    pytest.param(torch.bfloat16, torch.bfloat16, False, id='bf16-params'),
])
@pytest.mark.parametrize(('checkpoint_kwargs', 'input_kind'), [
    (None, 'input_ids'), ({'use_reentrant': False}, 'input_ids'),
    ({'use_reentrant': None}, 'input_ids'),
    ({'use_reentrant': True}, 'input_ids'), ({'use_reentrant': True}, 'inputs_embeds'),
], ids=['default', 'non-reentrant', 'none-default', 'reentrant', 'reentrant-embeds'])
@pytest.mark.parametrize('backend', ['native', 'reference'])
@pytest.mark.parametrize('num_layers', [1, 3])
def test_checkpointing(
    residual_mode, residual_kwargs, param_dtype, dtype, autocast, checkpoint_kwargs, input_kind,
    backend, num_layers, request, monkeypatch,
):
    if backend == 'native' and device_platform != 'cuda':
        pytest.skip('Native autocast integration requires CUDA')
    if backend == 'native' and residual_mode == 'mhc' and not IS_NVIDIA:
        pytest.skip('mHC requires NVIDIA CUDA')
    if backend == 'reference':
        request.getfixturevalue('reference_gla_ops')
    test_device = 'cpu' if backend == 'reference' else device
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
        attn_mode='fused_recurrent' if backend == 'reference' else 'chunk',
        fuse_norm=backend == 'native',
        fuse_swiglu=backend == 'native',
        fuse_cross_entropy=False,
        use_cache=False,
    )
    model = GLAForCausalLM(config).to(device=test_device, dtype=param_dtype).train()
    assert config.residual_mode == residual_mode
    assert model.model.residual_output.fuse_norm == residual_kwargs.get('fuse_norm', config.fuse_norm)
    reentrant = checkpoint_kwargs is not None and checkpoint_kwargs['use_reentrant'] is True
    supports_reentrant = residual_mode != 'attnres'
    assert model.model.residual_output.supports_reentrant_checkpointing == supports_reentrant
    if reentrant and not supports_reentrant:
        with pytest.raises(ValueError, match='AttentionResidual.*use_reentrant=False'):
            model.gradient_checkpointing_enable(checkpoint_kwargs)
        assert not model.is_gradient_checkpointing
        return
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('query.weight'):
                parameter.normal_(std=0.03)
        if residual_mode == 'mhc' and reentrant:
            for module in model.modules():
                if isinstance(module, ManifoldHyperConnection):
                    module.norm.weight.uniform_(0.7, 1.3)
                    if not module.is_last_sublayer:
                        module.scale.fill_(0.2)
                        module.pre_bias.add_(0.1 * module.sub_layer_idx)
                        module.res_bias.normal_(std=0.3)
    reference = copy.deepcopy(model)
    if residual_mode == 'mhc' and reentrant:
        mhc_residuals = [module for module in model.modules() if isinstance(module, ManifoldHyperConnection)]
        for module in mhc_residuals:
            monkeypatch.setattr(module, 'prepare_coefficients', Mock(wraps=module.prepare_coefficients))
    model.gradient_checkpointing_enable(checkpoint_kwargs)
    assert model.is_gradient_checkpointing and not reference.is_gradient_checkpointing
    grad_modes = [[] for _ in model.model.layers]
    for i, layer in enumerate(model.model.layers):
        def observe(module, args, output, i=i):
            state = output[0]
            if residual_mode == 'mhc':
                assert state.streams.dtype == param_dtype
                assert tuple(value is None for value in state[1:]) == (i == num_layers - 1,) * 3
            grad_modes[i].append(torch.is_grad_enabled())
        layer.register_forward_hook(observe)
    ids = torch.randint(config.vocab_size, (2, 17 if backend == 'reference' else 65), device=test_device)
    if input_kind == 'inputs_embeds':
        embeddings = model.get_input_embeddings()(ids).detach().requires_grad_()
        ref_embeddings = embeddings.detach().clone().requires_grad_()
        actual_inputs, expected_inputs = {'inputs_embeds': embeddings}, {'inputs_embeds': ref_embeddings}
    else:
        actual_inputs = expected_inputs = {'input_ids': ids}
    with torch.autocast(device_type=torch.device(test_device).type, dtype=dtype, enabled=autocast):
        actual = model(**actual_inputs, labels=ids, output_hidden_states=True)
        expected = reference(**expected_inputs, labels=ids, output_hidden_states=True)
    actual.loss.backward()
    expected.loss.backward()
    if reentrant:
        assert grad_modes == [[False, True] for _ in model.model.layers]
        if residual_mode == 'mhc':
            assert [module.prepare_coefficients.call_count for module in mhc_residuals] == [2] * (2 * num_layers - 1) + [0, 1]
            assert model.model.residual_output.weight.grad.abs().max() > 0
            assert not hasattr(model.model.layers[-1].residual_mlp, 'weight')
    else:
        assert all(modes and all(modes) for modes in grad_modes)
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


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('fuse_norm', [False, True])
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
