# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fla import models
from fla.models import GLAConfig, KDAConfig, MemoryAttnConfig, MomConfig, TransformerConfig
from fla.models.cache_utils import Cache
from fla.utils import assert_close, device_name

CAUSAL_LM_CONFIGS = [
    models.ABCConfig, models.BitNetConfig, models.CombaConfig, models.DeltaNetConfig,
    models.ForgettingTransformerConfig, models.GatedDeltaNetConfig, models.GatedDeltaProductConfig,
    models.GLAConfig, models.GSA2Config, models.GSAConfig, models.HGRNConfig, models.HGRN2Config, models.KDAConfig, models.LightNetConfig,
    models.MesaNetConfig, models.MLAConfig, models.MoBAConfig, models.NSAConfig, models.ParallaxConfig,
    models.PaTHAttentionConfig, models.PrecondGatedDeltaNetConfig, models.RavenConfig, models.RetNetConfig,
    models.RodimusConfig, models.RWKV6Config, models.TransformerConfig, models.WallTransformerConfig,
]

DECODER_CONFIGS = [
    models.ABCConfig, models.CombaConfig, models.DeltaNetConfig, models.GatedDeltaNetConfig,
    models.GatedDeltaProductConfig, models.GLAConfig, models.GSA2Config, models.GSAConfig, models.KDAConfig,
    models.MesaNetConfig, models.MLAConfig, models.MoBAConfig, models.NSAConfig, models.RavenConfig,
]


@pytest.fixture
def full_precision_matmul():
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize('config_class', [GLAConfig, models.GSA2Config, KDAConfig, MemoryAttnConfig, TransformerConfig])
@pytest.mark.parametrize('num_items', [None, 11, torch.tensor(11)])
@pytest.mark.parametrize('custom_criterion', [None, 'mean', 'sum'])
def test_causal_lm_loss(config_class, num_items, custom_criterion, make_model):
    torch.manual_seed(42)
    model = make_model(config_class, hidden_size=32, vocab_size=32)
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


@pytest.mark.parametrize('num_items', [None, 10])
@pytest.mark.parametrize('custom_criterion', [False, True])
@pytest.mark.parametrize('cu_seqlens', [[0, 7, 14], [0, 0, 7, 7, 14, 14]], ids=['packed', 'empty-sequences'])
def test_causal_lm_packed_loss(num_items, custom_criterion, cu_seqlens, make_model):
    torch.manual_seed(42)
    model = make_model(models.GSA2Config)
    if custom_criterion:
        model.criterion = nn.CrossEntropyLoss(ignore_index=-7, label_smoothing=0.1)
    ignore_index = -7 if custom_criterion else -100
    input_ids = torch.randint(0, 64, (1, 14))
    labels = input_ids.clone()
    labels[:, 2] = ignore_index
    original_labels = labels.clone()
    output = model(
        input_ids=input_ids,
        labels=labels,
        num_items_in_batch=num_items,
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32),
    )
    expected = F.cross_entropy(
        input=output.logits.reshape(2, 7, 64)[:, :-1].reshape(-1, 64),
        target=labels.reshape(2, 7)[:, 1:].reshape(-1),
        ignore_index=ignore_index,
        label_smoothing=0.1 if custom_criterion else 0.0,
        reduction='sum' if num_items is not None else 'mean',
    )
    if num_items is not None:
        expected = expected / num_items
    torch.testing.assert_close(output.loss, expected)
    assert torch.equal(labels, original_labels)
    parameters = tuple(model.parameters())
    actual_gradients = torch.autograd.grad(output.loss, parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, parameters)
    for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference)


@pytest.mark.skipif(device_name == 'cpu', reason='Fused cross entropy requires an accelerator.')
@pytest.mark.parametrize('linear', [False, True])
@pytest.mark.parametrize('use_l2warp', [False, True])
@pytest.mark.parametrize('num_items', [None, 11])
def test_causal_lm_loss_fused(linear, use_l2warp, num_items, make_model):
    torch.manual_seed(42)
    reference = make_model(GLAConfig, vocab_size=128, use_l2warp=use_l2warp).to(device_name)
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


@pytest.mark.parametrize('config_class', [GLAConfig, MemoryAttnConfig, pytest.param(
    MomConfig, marks=pytest.mark.skipif(device_name == 'cpu', reason='MoM normalization requires an accelerator.'),
)])
@pytest.mark.parametrize('selection', [None, 0, 2, torch.tensor([0, 2, 5])])
@pytest.mark.parametrize('return_dict', [False, True])
def test_causal_lm_logits(config_class, selection, return_dict, full_precision_matmul, make_model):
    torch.manual_seed(42)
    model = make_model(config_class, hidden_size=32, vocab_size=32).to(device_name).eval()
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


@pytest.mark.parametrize('config_class', CAUSAL_LM_CONFIGS, ids=lambda cls: cls.model_type)
@pytest.mark.parametrize('return_dict', [False, True])
def test_causal_lm_fwd(config_class, return_dict, make_model):
    torch.manual_seed(42)
    model = make_model(config_class, use_lower_bound=False)
    model.eval()
    model.config.return_dict = return_dict
    input_ids = torch.randint(0, 64, (2, 7))
    labels = input_ids.clone()
    labels[0, 3] = -100
    outputs = model(input_ids=input_ids, labels=labels, use_cache=False, output_hidden_states=True, return_dict=return_dict)
    decoder_outputs = model.model(input_ids=input_ids, use_cache=False, output_hidden_states=True, return_dict=True)
    reference_logits = model.lm_head(decoder_outputs.last_hidden_state)
    reference_loss = F.cross_entropy(reference_logits[:, :-1].reshape(-1, 64), labels[:, 1:].reshape(-1))
    loss, logits = (outputs.loss, outputs.logits) if return_dict else outputs[:2]
    torch.testing.assert_close(logits, reference_logits)
    torch.testing.assert_close(loss, reference_loss)
    parameters = tuple(model.parameters())
    actual_gradients = torch.autograd.grad(loss, parameters)
    reference_gradients = torch.autograd.grad(reference_loss, parameters)
    for actual, expected in zip(actual_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual, expected)
    hidden_states = outputs.hidden_states if return_dict else outputs[2]
    for actual, expected in zip(hidden_states, decoder_outputs.hidden_states, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('config_class', CAUSAL_LM_CONFIGS, ids=lambda cls: cls.model_type)
@pytest.mark.parametrize('return_dict', [False, True])
def test_causal_lm_fwd_args(config_class, return_dict, make_model):
    torch.manual_seed(42)
    model = make_model(config_class, use_lower_bound=False).eval()
    input_ids = torch.randint(0, 64, (2, 7))
    labels = input_ids.clone()
    with torch.no_grad():
        expected = model(input_ids=input_ids, labels=labels, use_cache=False)
        embeddings = model.get_input_embeddings()(input_ids)
        if config_class in (
            models.BitNetConfig, models.TransformerConfig, models.WallTransformerConfig,
            models.ForgettingTransformerConfig, models.ParallaxConfig, models.PaTHAttentionConfig,
        ):
            actual = model(None, None, None, embeddings, labels, False, False, True, return_dict)
        else:
            actual = model(None, None, embeddings, None, labels, False, False, True, return_dict)
        torch.testing.assert_close(actual.logits if return_dict else actual[1], expected.logits)
        selected = model(input_ids=input_ids, num_logits_to_keep=2, use_cache=False, return_dict=return_dict)
    torch.testing.assert_close(selected.logits if return_dict else selected[0], expected.logits[:, -2:])


@pytest.mark.parametrize('config_class', DECODER_CONFIGS, ids=lambda cls: cls.model_type)
@pytest.mark.parametrize(('training', 'use_cache', 'return_dict'), [
    pytest.param(True, None, False, id='train-default-tuple'),
    pytest.param(False, None, True, id='eval-default-dict'),
    pytest.param(True, True, True, id='train-cache-dict'),
    pytest.param(False, False, False, id='eval-no-cache-tuple'),
])
def test_decoder_fwd(config_class, training, use_cache, return_dict, make_model):
    torch.manual_seed(42)
    model = make_model(config_class).model.train(training)
    model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    for layer in model.layers:
        layer.forward = Mock(side_effect=lambda hidden_states, **kwargs: (
            hidden_states + hidden_states.tanh(), None, kwargs['past_key_values'], kwargs['attnres_states'],
        ))
    input_ids = torch.randint(0, 64, (2, 7))
    attention_mask = torch.ones_like(input_ids)
    cu_seqlens = torch.tensor([0, 7, 14], dtype=torch.int32)
    expected_cache = not training if use_cache is None else use_cache
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=use_cache,
        output_hidden_states=True,
        return_dict=return_dict,
        cu_seqlens=cu_seqlens,
    )
    cache = None
    if expected_cache:
        cache = outputs.past_key_values if return_dict else outputs[1]
        assert isinstance(cache, Cache)
    elif return_dict:
        assert outputs.past_key_values is None
    hidden_states = outputs.hidden_states if return_dict else outputs[-1]
    reference = model.embeddings(input_ids)
    for index, layer in enumerate(model.layers):
        torch.testing.assert_close(hidden_states[index], reference)
        layer.forward.assert_called_once()
        args, kwargs = layer.forward.call_args
        torch.testing.assert_close(args[0], reference)
        assert kwargs['attention_mask'] is attention_mask
        assert kwargs['past_key_values'] is cache
        assert kwargs['use_cache'] is expected_cache
        assert kwargs['output_attentions'] is False
        assert kwargs['attnres_states'] is None
        assert kwargs['cu_seqlens'] is cu_seqlens
        reference = reference + reference.tanh()
    reference = model.norm(reference)
    torch.testing.assert_close(outputs[0], reference)
    torch.testing.assert_close(hidden_states[-1], reference)
    gradient = torch.randn_like(reference)
    parameters = (model.embeddings.weight, model.norm.weight)
    actual_gradients = torch.autograd.grad(outputs[0], parameters, grad_outputs=gradient)
    reference_gradients = torch.autograd.grad(reference, parameters, grad_outputs=gradient)
    torch.testing.assert_close(actual_gradients, reference_gradients)


@pytest.mark.parametrize('config_class', DECODER_CONFIGS, ids=lambda cls: cls.model_type)
@pytest.mark.parametrize('return_dict', [False, True], ids=['tuple', 'dict'])
def test_decoder_inputs(config_class, return_dict, make_model):
    torch.manual_seed(42)
    model = make_model(config_class).model.eval()
    input_ids = torch.randint(0, 64, (2, 7))
    attention_mask = torch.ones_like(input_ids)
    cache = Cache()
    expected = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
    embeddings = model.embeddings(input_ids)
    outputs = model(None, attention_mask, embeddings, cache, True, False, True, return_dict)
    torch.testing.assert_close(outputs[0], expected.last_hidden_state)
    assert (outputs.past_key_values if return_dict else outputs[1]) is cache
    torch.testing.assert_close(outputs.hidden_states if return_dict else outputs[2], expected.hidden_states)
    with pytest.raises(ValueError, match='both input_ids and inputs_embeds'):
        model(input_ids=input_ids, inputs_embeds=embeddings)
    with pytest.raises(ValueError, match='either input_ids or inputs_embeds'):
        model()
