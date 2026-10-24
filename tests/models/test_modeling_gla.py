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


@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
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
        (2, 2, 256, 2, 64, torch.float16, 'attnres', {'block_size': 1}),
        (2, 2, 256, 2, 64, torch.float16, 'attnres', {'block_size': 4}),
        pytest.param(2, 2, 256, 2, 64, torch.float16, 'mhc', {'num_streams': 4}, marks=REQUIRES_NVIDIA),
    ],
)
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
    ('standard', {}), ('attnres', {'block_size': 1}), ('attnres', {'block_size': 4}),
    pytest.param('mhc', {'num_streams': 4}, marks=REQUIRES_NVIDIA),
])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('use_reentrant', [False, True])
@pytest.mark.skipif(device_platform != 'cuda', reason='Autocast requires CUDA')
def test_checkpointing(residual_mode, residual_kwargs, dtype, use_reentrant, monkeypatch):
    torch.manual_seed(42)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    config = GLAConfig(
        hidden_size=128,
        num_hidden_layers=3,
        num_heads=2,
        intermediate_size=256,
        vocab_size=128,
        residual_mode=residual_mode,
        residual_kwargs=residual_kwargs,
        use_cache=False,
    )
    model = GLAForCausalLM(config).to(device).train()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('query.weight'):
                parameter.normal_(std=0.03)
    reference = copy.deepcopy(model)
    if use_reentrant and residual_mode == 'attnres':
        with pytest.raises(ValueError, match='use_reentrant=False'):
            model.gradient_checkpointing_enable({'use_reentrant': True})
        return
    model.gradient_checkpointing_enable({'use_reentrant': use_reentrant})
    ids = torch.randint(config.vocab_size, (2, 65), device=device)
    with torch.autocast(device_type=device_platform, dtype=dtype):
        actual = model(ids, labels=ids, output_hidden_states=True)
        expected = reference(ids, labels=ids, output_hidden_states=True)
    actual.loss.backward()
    expected.loss.backward()
    assert_close('loss', expected.loss, actual.loss, 1e-3)
    for ref, tri in zip(expected.hidden_states, actual.hidden_states):
        assert_close('hidden_states', ref, tri, 1e-3)
    for (name, parameter), (_, target) in zip(model.named_parameters(), reference.named_parameters()):
        assert parameter.dtype == torch.float32
        if target.grad is None:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert_close(name, target.grad, parameter.grad, 1e-3)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('fuse_norm', [False, True])
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
    output, _, _ = block(x)
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
