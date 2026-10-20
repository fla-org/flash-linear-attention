# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.nn import functional as F

from fla.layers.gsa2 import GatedSlotAttention2
from fla.models import GSA2Config, GSA2ForCausalLM
from fla.utils import IS_AMD, IS_NPU, IS_NVIDIA, assert_close, device
from tests.models.utils import check_forward, check_generate, check_generation, create_model_and_config

# GSA2 runs the GDN-2 kernels, which tests/ops/test_gdn2.py guards the same way
requires_accelerator = pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_NPU), reason="CUDA/ROCm or Ascend NPU required")

_MODELING_CASES = [
    pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-M{}-{}-{}-bs{}-{}".format(*test))
    for test in [
        (4, 4, 1024, 4, 64, True,  128, 'head', 'silu',    None, torch.bfloat16),
        (4, 4, 1024, 4, 64, False, 128, 'head', 'softmax', None, torch.bfloat16),
        (4, 4, 1024, 4, 64, False, 64,  'head', 'silu',    1,    torch.bfloat16),
        (4, 4, 1024, 4, 64, False, 256, 'slot', 'silu',    4,    torch.bfloat16),
    ]
]
_MODELING_ARGS = ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'num_slots',
                  'level2_axis', 'oja_out_act', 'attnres_block_size', 'dtype']


@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
@pytest.mark.parametrize(_MODELING_ARGS, _MODELING_CASES)
@requires_accelerator
def test_modeling(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    use_l2warp: bool,
    num_slots: int,
    level2_axis: str,
    oja_out_act: str,
    attnres_block_size: int | None,
    dtype: torch.dtype,
    varlen: bool,
):
    check_forward(
        L=L,
        B=B,
        T=T,
        H=H,
        D=D,
        config_class=GSA2Config,
        use_l2warp=use_l2warp,
        num_slots=num_slots,
        level2_axis=level2_axis,
        oja_out_act=oja_out_act,
        attnres_block_size=attnres_block_size,
        dtype=dtype,
        varlen=varlen,
    )


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 4, 2000, 8, 64, torch.float16),
        ]
    ],
)
@requires_accelerator
def test_generation(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    # two recurrences per layer use the existing GSA2 generation error budget
    check_generation(L, B, T, H, D, GSA2Config, dtype, tol=4e-3)


@requires_accelerator
def test_generation_with_slot_projections():
    L, B, T, H, D, dtype = 2, 4, 2000, 8, 64, torch.float16
    model, config = create_model_and_config(
        GSA2Config, L, H, D, dtype, head_dim=D, num_slots=128, level2_axis='head',
    )
    assert model.model.layers[0].attn.gdn_q_proj is not None
    check_generation(L, B, T, H, D, GSA2Config, dtype, model=model, config=config, tol=4e-3)


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 2, 64, 8, 64, torch.float32),
        ]
    ],
)
@requires_accelerator
def test_generate_prefill(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    check_generate(L, B, T, H, D, GSA2Config, dtype)


@pytest.mark.parametrize(
    ('fuse_cross_entropy', 'fuse_linear_cross_entropy'),
    [(False, False), (True, False), (False, True)],
    ids=['ce', 'fused-ce', 'fused-linear-ce'],
)
@pytest.mark.parametrize('num_items', [None, 10])
@requires_accelerator
def test_packed_loss(fuse_cross_entropy: bool, fuse_linear_cross_entropy: bool, num_items: int | None):
    """Packed loss and gradients match a batch without predicting across sequence boundaries."""
    torch.manual_seed(42)
    config = GSA2Config(
        hidden_size=64,
        num_hidden_layers=1,
        num_heads=1,
        head_dim=64,
        num_slots=64,
        vocab_size=64,
        use_cache=False,
        fuse_cross_entropy=fuse_cross_entropy,
        fuse_linear_cross_entropy=fuse_linear_cross_entropy,
    )
    model = GSA2ForCausalLM(config).to(device=device, dtype=torch.float32).train()
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(2, 7), device=device)
    labels = input_ids.clone()
    labels[:, 2] = -100

    expected = model(input_ids=input_ids, labels=labels, num_items_in_batch=num_items).loss
    expected.backward()
    expected_grads = {name: param.grad.clone() for name, param in model.named_parameters()}
    model.zero_grad(set_to_none=True)

    output = model(
        input_ids=input_ids.view(1, -1),
        labels=labels.view(1, -1),
        num_items_in_batch=num_items,
        output_hidden_states=True,
        cu_seqlens=torch.tensor([0, 7, 14], dtype=torch.int32, device=device),
    )
    output.hidden_states[-1].retain_grad()
    actual = output.loss
    actual.backward()
    has_gradient = output.hidden_states[-1].grad.reshape(2, 7, -1).ne(0).any(dim=-1)
    assert torch.equal(has_gradient[:, :-1], labels[:, 1:].ne(-100))
    assert not has_gradient[:, -1].any()
    assert_close('loss', expected, actual, 1e-4)
    for name, param in model.named_parameters():
        assert_close(name, expected_grads[name], param.grad, 1e-4)


@torch.no_grad()
@requires_accelerator
def test_logits_to_keep_with_labels():
    B, T, H, D, V = 2, 8, 2, 8, 32
    config = GSA2Config(
        hidden_size=H * D,
        num_hidden_layers=1,
        num_heads=H,
        head_dim=D,
        num_slots=64,
        level2_axis='slot',
        expand_v=1,
        vocab_size=V,
        fuse_cross_entropy=False,
    )
    model = GSA2ForCausalLM(config).eval().to(device)
    input_ids = torch.randint(V, (B, T), device=device)

    expected = model(input_ids, labels=input_ids)
    actual = model(input_ids, labels=input_ids, logits_to_keep=1)
    inference = model(input_ids, logits_to_keep=1)

    assert actual.logits.shape == expected.logits.shape == (B, T, V)
    assert inference.logits.shape == (B, 1, V)
    assert torch.isfinite(actual.loss)
    assert_close('loss', expected.loss, actual.loss, 1e-6)


def test_initialization(init_layout, make_model):
    torch.manual_seed(42)
    model = make_model(GSA2Config)
    with torch.device('meta'):
        layer = GatedSlotAttention2(hidden_size=64, head_dim=16, num_heads=4, num_slots=64, use_short_conv=False)
    model._init_weights(layer)
    assert all(parameter.is_meta for parameter in layer.parameters())
    layer.to_empty(device='cpu')
    names = ('A1_log', 'A2_log', 'dt1_bias', 'dt2_bias')
    for name in names:
        value = torch.full_like(getattr(layer, name), float('nan'))
        if init_layout is not None:
            mesh, placements = init_layout
            value = distribute_tensor(value, mesh, placements)
        setattr(layer, name, nn.Parameter(value))
    parameters = [getattr(layer, name) for name in names]
    model._init_weights(layer)
    values = []
    for name, parameter in zip(names, parameters, strict=True):
        actual = getattr(layer, name)
        assert actual is parameter
        assert actual._no_weight_decay
        if init_layout is not None:
            assert isinstance(actual, DTensor)
            assert actual.device_mesh == mesh
            assert actual.placements == placements
            actual = actual.to_local()
        assert torch.isfinite(actual).all()
        values.append(actual)
    for value in values[:2]:
        assert ((value.exp() >= 1) & (value.exp() <= 16)).all()
    assert values[2].count_nonzero() == 0
    dt = F.softplus(values[3])
    assert ((dt >= 0.001) & (dt <= 0.1)).all()
    for parameter in parameters:
        parameter._is_hf_initialized = True
    expected = [parameter.detach().clone() for parameter in parameters]
    model._init_weights(layer)
    for actual, reference in zip(parameters, expected, strict=True):
        if init_layout is not None:
            actual, reference = actual.to_local(), reference.to_local()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
