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
from fla.models.gla.modeling_gla import GLAForCausalLM
from fla.modules.residuals import attnres
from fla.ops.attnres import naive_attnres
from fla.utils import assert_close, device, device_platform

from .test_modeling_base import run_test_generation, run_test_model_forward_backward


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'attnres_block_size', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-bs{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64,  True,  None, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, None, torch.bfloat16),
            (4, 4, 1024, 4, 128, False, None, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 1,    torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 4,    torch.bfloat16),
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
):
    run_test_model_forward_backward(
        L,
        B,
        T,
        H,
        D,
        GLAConfig,
        use_l2warp=use_l2warp,
        residual_mode='standard' if attnres_block_size is None else 'attnres',
        residual_kwargs={} if attnres_block_size is None else {'block_size': attnres_block_size},
        dtype=dtype,
    )


# ===================================================================================
# Test for Generation
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 4, 2000, 8, 64, torch.float16),
        ]
    ],
)
def test_generation(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    run_test_generation(L, B, T, H, D, GLAConfig, dtype)


@pytest.mark.parametrize('block_size', [1, 4])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('checkpointing', [False, True])
@pytest.mark.skipif(device_platform != 'cuda', reason='AttnRes autocast regression requires CUDA')
def test_attnres_autocast(block_size, dtype, checkpointing, monkeypatch):
    torch.manual_seed(42)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    config = GLAConfig(
        hidden_size=128,
        num_hidden_layers=3,
        num_heads=2,
        intermediate_size=256,
        vocab_size=128,
        residual_mode='attnres',
        residual_kwargs={'block_size': block_size},
        use_cache=False,
        fuse_norm=True,
        fuse_swiglu=True,
        fuse_cross_entropy=True,
    )
    model = GLAForCausalLM(config).to(device).train()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('query.weight'):
                parameter.normal_(std=0.03)
    reference = copy.deepcopy(model)
    if checkpointing:
        model.gradient_checkpointing_enable()
    ids = torch.randint(config.vocab_size, (2, 65), device=device)
    with torch.autocast(device_type=device_platform, dtype=dtype):
        actual = model(ids, labels=ids)
    actual.loss.backward()

    with torch.autocast(device_type=device_platform, dtype=dtype):
        expected = reference(ids, labels=ids)
    expected.loss.backward()
    assert torch.isfinite(actual.loss)
    torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
    for (name, parameter), (_, target) in zip(model.named_parameters(), reference.named_parameters()):
        assert parameter.dtype == torch.float32
        if 'input_query' in name or 'input_key_norm' in name:
            assert parameter.grad is None and target.grad is None
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, target.grad, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize('block_size', [1, 4])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.skipif(device_platform != 'cuda', reason='AttnRes autocast regression requires CUDA')
def test_attnres_mixed_dtype_reference(block_size, dtype, monkeypatch):
    torch.manual_seed(42)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    modules = torch.nn.ModuleList([
        attnres.AttentionResidual(hidden_size=128, sub_layer_idx=i, block_size=block_size).to(device)
        for i in range(6)
    ])
    with torch.no_grad():
        for module in modules:
            module.query.weight.normal_(std=0.03)
    reference = copy.deepcopy(modules)
    x = torch.randn(2, 7, 128, device=device, requires_grad=True)
    branches = [torch.randn_like(x, dtype=dtype, requires_grad=True) for _ in modules]
    ref_x = x.detach().clone().requires_grad_()
    ref_branches = [branch.detach().clone().requires_grad_() for branch in branches]
    history, ref_history = [x], [ref_x.to(dtype)]
    loss, ref_loss = 0, 0
    for i, (module, ref_module) in enumerate(zip(modules, reference)):
        with torch.autocast(device_type=device_platform, dtype=dtype):
            output, history = module(branches[i], history)
        if i % block_size == 0:
            ref_history = [*ref_history, ref_branches[i]]
        else:
            ref_history = [*ref_history[:-1], ref_history[-1] + ref_branches[i]]
        assert all(state.dtype == dtype for state in history)
        assert output.dtype == dtype
        expected = naive_attnres(
            query=ref_module.query.weight,
            residuals=ref_history,
            rms_weight=ref_module.key_norm.weight,
            output_rms_weight=ref_module.norm.weight,
        )
        assert_close('output', expected, output, 0.005)
        grad = torch.randn_like(output)
        loss = loss + (output * grad).sum()
        ref_loss = ref_loss + (expected * grad).sum()
    loss.backward()
    ref_loss.backward()
    assert_close('embedding', ref_x.grad, x.grad, 0.005)
    for actual, expected in zip(branches, ref_branches):
        assert_close('branch', expected.grad, actual.grad, 0.005)
    for (name, actual), (_, expected) in zip(modules.named_parameters(), reference.named_parameters()):
        if expected.grad is None:
            assert actual.grad is None, name
        else:
            assert_close(name, expected.grad, actual.grad, 0.005)
