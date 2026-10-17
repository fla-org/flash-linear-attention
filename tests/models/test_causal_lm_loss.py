# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM

from fla.models import GLAConfig, KDAConfig, MomConfig, TransformerConfig
from fla.utils import assert_close, device_name


@pytest.fixture
def full_precision_matmul():
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize('config_class', [GLAConfig, KDAConfig, TransformerConfig])
@pytest.mark.parametrize('num_items', [None, 11, torch.tensor(11)])
@pytest.mark.parametrize('custom_criterion', [None, 'mean', 'sum'])
def test_causal_lm_loss_matches_shifted_reference(config_class, num_items, custom_criterion):
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(config_class(
        hidden_size=32, num_hidden_layers=0, vocab_size=32, fuse_norm=False, fuse_cross_entropy=False,
    ))
    if custom_criterion:
        model.criterion = nn.CrossEntropyLoss(ignore_index=-7, label_smoothing=0.1, reduction=custom_criterion)
    ignore_index = -7 if custom_criterion else -100
    labels = torch.randint(0, 32, (2, 7))
    labels[0, 3] = ignore_index
    output = model(input_ids=torch.randint(0, 32, (2, 7)), labels=labels, num_items_in_batch=num_items)
    reduction = 'sum' if num_items is not None else custom_criterion or 'mean'
    reference = F.cross_entropy(
        input=output.logits[:, :-1].reshape(-1, 32),
        target=labels[:, 1:].reshape(-1),
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=0.1 if custom_criterion else 0.0,
    )
    if num_items is not None:
        reference = reference / num_items
    torch.testing.assert_close(output.loss, reference)
    parameters = tuple(model.parameters())
    actual_gradients = torch.autograd.grad(output.loss, parameters, retain_graph=True)
    reference_gradients = torch.autograd.grad(reference, parameters)
    for actual, expected in zip(actual_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual, expected)
    if custom_criterion:
        assert model.criterion.reduction == custom_criterion


@pytest.mark.parametrize('config_class', [GLAConfig, pytest.param(
    MomConfig, marks=pytest.mark.skipif(device_name == 'cpu', reason='MoM normalization requires an accelerator.'),
)])
@pytest.mark.parametrize('selection', [None, 0, 2, torch.tensor([0, 2, 5])])
@pytest.mark.parametrize('return_dict', [False, True])
def test_causal_lm_logits_selection(config_class, selection, return_dict, full_precision_matmul):
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(config_class(
        hidden_size=32, num_hidden_layers=0, vocab_size=32, fuse_norm=False, fuse_cross_entropy=False,
    )).to(device_name).eval()
    input_ids = torch.randint(0, 32, (2, 7), device=device_name)
    with torch.no_grad():
        complete = model(input_ids=input_ids).logits
        selected = model(input_ids=input_ids, logits_to_keep=selection, return_dict=return_dict)
        if config_class is not MomConfig:
            with_labels = model(input_ids=input_ids, labels=input_ids, logits_to_keep=selection)
    logits = selected.logits if return_dict else selected[0]
    if isinstance(selection, torch.Tensor):
        expected = complete[:, selection]
    elif selection:
        expected = complete[:, -selection:]
    else:
        expected = complete
    torch.testing.assert_close(logits, expected)
    if config_class is not MomConfig:
        torch.testing.assert_close(with_labels.logits, complete)
    else:
        with torch.no_grad():
            legacy = model(input_ids=input_ids, num_logits_to_keep=selection).logits
        torch.testing.assert_close(legacy, expected)


@pytest.mark.skipif(device_name == 'cpu', reason='Fused cross entropy requires an accelerator.')
@pytest.mark.parametrize('linear', [False, True])
@pytest.mark.parametrize('use_l2warp', [False, True])
@pytest.mark.parametrize('num_items', [None, 11])
def test_fused_causal_lm_loss_matches_unfused(linear, use_l2warp, num_items):
    torch.manual_seed(42)
    config = GLAConfig(
        hidden_size=64,
        num_hidden_layers=0,
        vocab_size=128,
        fuse_norm=False,
        fuse_cross_entropy=False,
        use_l2warp=use_l2warp,
    )
    reference = AutoModelForCausalLM.from_config(config).to(device_name)
    fused = deepcopy(reference)
    fused.config.fuse_cross_entropy = not linear
    fused.config.fuse_linear_cross_entropy = linear
    input_ids = torch.randint(0, 128, (2, 7), device=device_name)
    labels = input_ids.clone()
    labels[0, 3] = -100
    expected = reference(input_ids=input_ids, labels=labels, num_items_in_batch=num_items)
    actual = fused(input_ids=input_ids, labels=labels, num_items_in_batch=num_items)
    assert (actual.logits is None) == linear
    assert_close('loss', expected.loss, actual.loss, 1e-3)
    expected.loss.backward()
    actual.loss.backward()
    for (name, parameter), fused_parameter in zip(reference.named_parameters(), fused.parameters(), strict=True):
        assert_close(name, parameter.grad, fused_parameter.grad, 1e-3)
