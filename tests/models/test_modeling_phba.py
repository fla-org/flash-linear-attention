# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from fla.models.phba import PHBAConfig, PHBAForCausalLM
from fla.ops.phba.naive import highest_precision
from fla.utils import IS_NVIDIA, assert_close, device, find_spec_cached


class _FloatCrossEntropy(nn.CrossEntropyLoss):
    def forward(self, logits, target):
        # fused losses accumulate in FP32; preserve that precision in the oracle
        return super().forward(logits.float(), target)


def make_config(**kwargs):
    options = dict(
        hidden_size=128, num_hidden_layers=2, num_heads=2, intermediate_size=256, vocab_size=128,
        max_position_embeddings=128, moba_chunk_size=32, moba_topk=2, gsa_num_slots=16,
        use_cache=False, use_output_gate=False, fuse_cross_entropy=False,
    )
    options.update(kwargs)
    return PHBAConfig(**options)


def test_registration_and_checkpoint(tmp_path):
    torch.manual_seed(42)
    config = make_config()
    model = AutoModelForCausalLM.from_config(config)
    assert isinstance(model, PHBAForCausalLM)
    model.save_pretrained(tmp_path)
    loaded_config = AutoConfig.from_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path)
    assert isinstance(loaded_config, PHBAConfig)
    assert model.state_dict().keys() == loaded.state_dict().keys()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name])


@pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None, reason='PHBA CUDA integration')
@pytest.mark.parametrize(
    ('qk_norm', 'use_output_gate', 'dtype'),
    [(False, False, torch.bfloat16), (True, True, torch.float16)],
    ids=['baseline-bf16', 'normalized-output-gate-fp16'],
)
def test_modeling(qk_norm, use_output_gate, dtype):
    torch.manual_seed(42)
    model = PHBAForCausalLM(make_config(qk_norm=qk_norm, use_output_gate=use_output_gate)).to(device).train()
    tokens = torch.randint(0, 128, (2, 67), device=device)
    with torch.autocast(device_type='cuda', dtype=dtype):
        result = model(input_ids=tokens, labels=tokens)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert result.logits.shape == (2, 67, 128)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_cache_is_explicitly_unsupported():
    with pytest.raises(ValueError, match='use_cache=False'):
        make_config(use_cache=True)
    model = PHBAForCausalLM(make_config(num_hidden_layers=0, fuse_norm=False))
    tokens = torch.ones(1, 4, dtype=torch.long)
    with pytest.raises(ValueError, match='use_cache=False'):
        model(input_ids=tokens, use_cache=True)
    with pytest.raises(ValueError, match='use_cache=False'):
        model(input_ids=tokens, past_key_values=object())


@pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None, reason='PHBA CUDA integration')
def test_checkpoint_gradients():
    torch.manual_seed(42)
    first = PHBAForCausalLM(make_config()).to(device).train()
    second = PHBAForCausalLM(make_config()).to(device).train()
    second.load_state_dict(first.state_dict())
    second.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    tokens = torch.randint(0, 128, (1, 67), device=device)
    for model in (first, second):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model(input_ids=tokens, labels=tokens).loss.backward()
    for (name, left), (_, right) in zip(first.named_parameters(), second.named_parameters()):
        assert torch.equal(left.grad, right.grad), name


@pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None, reason='PHBA CUDA integration')
@pytest.mark.parametrize(
    ('side', 'dtype'),
    [('right', torch.bfloat16), ('left', torch.bfloat16), ('left', torch.float16)],
    ids=['right-bf16', 'left-bf16', 'left-fp16'],
)
def test_padding(side, dtype):
    torch.manual_seed(42)
    model = PHBAForCausalLM(make_config()).to(device).eval()
    tokens = torch.randint(1, 128, (2, 67), device=device)
    mask = torch.ones_like(tokens)
    if side == 'right':
        mask[1, 49:] = 0
    else:
        mask[1, :18] = 0
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=dtype):
        padded = model(input_ids=tokens, attention_mask=mask).logits
        for index, length in enumerate((67, 49)):
            valid = mask[index].bool()
            expected = model(input_ids=tokens[index:index + 1, valid]).logits
            assert expected.shape[1] == length
            assert_close('padded_logits', expected.float(), padded[index:index + 1, valid].float(), .006)


@pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None, reason='PHBA CUDA integration')
@pytest.mark.parametrize('padded', [False, True], ids=['short-prompt', 'left-padded-batch-chunk-boundary'])
def test_cache_free_generation(padded):
    torch.manual_seed(42)
    model = PHBAForCausalLM(make_config()).to(device).eval()
    prompt = torch.randint(1, 128, (2, 63), device=device) if padded else torch.tensor([[1, 4, 7]], device=device)
    mask = torch.ones_like(prompt) if padded else None
    if padded:
        mask[0, :28] = 0
        prompt[0, :28] = 0
    steps = 3 if padded else 2
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        expected = prompt
        expected_mask = mask
        for _ in range(steps):
            token = model(input_ids=expected, attention_mask=expected_mask).logits[:, -1].argmax(-1, keepdim=True)
            expected = torch.cat((expected, token), dim=-1)
            if expected_mask is not None:
                expected_mask = torch.cat((expected_mask, torch.ones_like(token)), dim=-1)
        actual = model.generate(
            input_ids=prompt, attention_mask=mask, max_new_tokens=steps, do_sample=False, use_cache=False,
            pad_token_id=0, eos_token_id=None,
        )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('fuse_norm', [False, True], ids=['native-norm', 'fused-norm'])
def test_affine_configuration(fuse_norm):
    model = PHBAForCausalLM(make_config(elementwise_affine=False, fuse_norm=fuse_norm, qk_norm=True, use_output_gate=True))
    norms = [model.model.norm]
    for layer in model.model.layers:
        norms.extend((layer.attn_norm, layer.mlp_norm, layer.attn.q_norm, layer.attn.k_norm, layer.attn.o_norm))
    assert all(module.weight is None for module in norms)


@pytest.mark.parametrize('width', [8, 17, 136], ids=['below-minimum', 'unaligned', 'above-maximum'])
def test_configuration_head_dimensions(width):
    with pytest.raises(ValueError, match='head dimensions'):
        make_config(hidden_size=width * 2)


def test_central_model_exports():
    from fla.models import PHBAConfig as ExportedConfig
    from fla.models import PHBAForCausalLM as ExportedCausalLM
    from fla.models import PHBAModel

    assert ExportedConfig is PHBAConfig
    assert ExportedCausalLM is PHBAForCausalLM
    assert AutoConfig.for_model('phba').model_type == PHBAModel.config_class.model_type


@pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None, reason='PHBA CUDA integration')
def test_model_plan_boundaries_and_affine_gradients():
    from fla.ops.phba import SequenceLengthPlan

    torch.manual_seed(42)
    model = PHBAForCausalLM(make_config(elementwise_affine=False, qk_norm=True, use_output_gate=True)).to(device).train()
    tokens = torch.randint(0, 128, (2, 33), device=device)
    with pytest.raises(ValueError, match='dense batch boundary'):
        model(input_ids=tokens, phba_length_plan=SequenceLengthPlan.from_host((66,), device))
    packed = tokens.reshape(1, -1)
    plan = SequenceLengthPlan.from_host((33, 33), device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actual = model(input_ids=packed, labels=packed, phba_length_plan=plan)
        expected = model(input_ids=packed, labels=packed, cu_seqlens=plan.cu_seqlens)
    assert torch.equal(actual.logits, expected.logits)
    actual.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError, match='different boundaries'):
        model(input_ids=packed, cu_seqlens=torch.tensor([0, 66], device=device, dtype=torch.int32), phba_length_plan=plan)


@pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None, reason='PHBA CUDA integration')
def test_updated_cumulative_lengths_do_not_reuse_rotary_indices():
    torch.manual_seed(42)
    model = PHBAForCausalLM(make_config()).to(device).eval()
    tokens = torch.randint(0, 128, (1, 66), device=device)
    cu = torch.tensor([0, 33, 66], dtype=torch.int32, device=device)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        model(input_ids=tokens, cu_seqlens=cu)
        cu[1] = 32
        actual = model(input_ids=tokens, cu_seqlens=cu).logits
        expected = model(input_ids=tokens, cu_seqlens=cu.clone()).logits
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None, reason='PHBA CUDA integration')
@pytest.mark.parametrize(
    ('lengths', 'side', 'dtype'),
    [((67, 67), 'right', torch.bfloat16), ((65, 33), 'right', torch.bfloat16),
     ((65, 1, 33), 'left', torch.float16)],
    ids=['dense-bf16', 'ragged-bf16', 'left-padded-one-token-fp16'],
)
def test_dense_packed_equivalence(lengths, side, dtype):
    torch.manual_seed(42)
    dense = PHBAForCausalLM(make_config()).to(device).train()
    packed = deepcopy(dense)
    tokens = torch.randint(1, 128, (len(lengths), max(lengths)), device=device)
    positions = torch.arange(tokens.shape[1], device=device)
    sizes = torch.tensor(lengths, device=device)
    mask = positions[None] < sizes[:, None] if side == 'right' else positions[None] >= tokens.shape[1] - sizes[:, None]
    labels = tokens.masked_fill(~mask, -100)
    first = torch.zeros_like(sizes) if side == 'right' else tokens.shape[1] - sizes
    labels[torch.arange(len(lengths), device=device), first] = -100
    flat_tokens = tokens[mask].unsqueeze(0)
    flat_labels = labels[mask].unsqueeze(0)
    cu = torch.cat((sizes.new_zeros(1), sizes.cumsum(0))).to(torch.int32)
    with highest_precision(), torch.autocast('cuda', dtype=dtype):
        actual = packed(input_ids=flat_tokens, labels=flat_labels, cu_seqlens=cu, output_hidden_states=True)
        expected = dense(
            input_ids=tokens,
            labels=labels,
            attention_mask=None if len(set(lengths)) == 1 else mask,
            output_hidden_states=True,
        )
    assert len(actual.hidden_states) == len(expected.hidden_states) == dense.config.num_hidden_layers + 1
    for reference, result in zip(expected.hidden_states, actual.hidden_states, strict=True):
        assert_close('packed_hidden', reference[mask].float(), result.squeeze(0).float(), 1e-3)
    assert_close('packed_logits', expected.logits[mask].float(), actual.logits.squeeze(0).float(), 1e-3)
    assert_close('packed_loss', expected.loss, actual.loss, 1e-3)
    expected.loss.backward()
    actual.loss.backward()
    for (name, reference), (other_name, result) in zip(dense.named_parameters(), packed.named_parameters(), strict=True):
        assert name == other_name and reference.grad is not None and result.grad is not None
        assert torch.isfinite(result.grad).all() and torch.isfinite(reference.grad).all()
        assert_close(name, reference.grad, result.grad, .006)


@pytest.mark.parametrize(
    ('reduction', 'num_items', 'use_l2warp'),
    [(None, None, False), ('mean', None, True), ('sum', None, False),
     ('mean', 11, False), ('sum', torch.tensor(11), True)],
    ids=['default-mean', 'smoothed-l2warp', 'custom-sum', 'normalized-custom-mean', 'tensor-normalized-l2warp'],
)
def test_loss_contract(reduction, num_items, use_l2warp):
    torch.manual_seed(42)
    model = PHBAForCausalLM(make_config(num_hidden_layers=0, fuse_norm=False, use_l2warp=use_l2warp))
    if reduction is not None:
        model.criterion = nn.CrossEntropyLoss(ignore_index=-7, label_smoothing=.1, reduction=reduction)
    ignore_index = -100 if reduction is None else -7
    tokens = torch.randint(0, 128, (2, 7))
    labels = tokens.clone()
    labels[0, 3] = ignore_index
    original_labels = labels.clone()
    result = model(input_ids=tokens, labels=labels, num_items_in_batch=num_items)
    reference = F.cross_entropy(
        result.logits[:, :-1].reshape(-1, 128),
        labels[:, 1:].reshape(-1),
        ignore_index=ignore_index,
        label_smoothing=0. if reduction is None else .1,
        reduction='sum' if num_items is not None else reduction or 'mean',
    )
    if use_l2warp:
        maxima = result.logits.max(-1).values
        penalty = maxima.square().sum() * (1e-4 / (2 * maxima.numel()))
        reference = reference + (penalty - penalty.detach())
    if num_items is not None:
        reference = reference / num_items
    torch.testing.assert_close(result.loss, reference)
    parameters = tuple(model.parameters())
    actual_grads = torch.autograd.grad(result.loss, parameters, retain_graph=True)
    reference_grads = torch.autograd.grad(reference, parameters)
    for actual, expected in zip(actual_grads, reference_grads, strict=True):
        torch.testing.assert_close(actual, expected)
    assert torch.equal(labels, original_labels)
    if model.criterion is not None:
        assert model.criterion.reduction == reduction


@pytest.mark.parametrize(
    ('loss_kind', 'num_items'),
    [
        pytest.param('native', 7, id='native-integer-count'),
        pytest.param('native', torch.tensor(11), id='native-tensor-count'),
        pytest.param('ce', 7, id='fused-ce-integer-count',
                     marks=pytest.mark.skipif(not IS_NVIDIA, reason='Fused CE requires NVIDIA')),
        pytest.param('linear', torch.tensor(11), id='fused-linear-tensor-count',
                     marks=pytest.mark.skipif(not IS_NVIDIA, reason='Fused linear CE requires NVIDIA')),
    ],
)
def test_l2warp_gradient_scaling(loss_kind, num_items):
    torch.manual_seed(42)
    target_device = 'cpu' if loss_kind == 'native' else device
    model = PHBAForCausalLM(make_config(
        hidden_size=16,
        num_heads=1,
        num_hidden_layers=0,
        intermediate_size=32,
        vocab_size=16,
        fuse_norm=False,
        fuse_cross_entropy=loss_kind == 'ce',
        fuse_linear_cross_entropy=loss_kind == 'linear',
        use_l2warp=True,
    )).to(target_device)
    with torch.no_grad():
        model.lm_head.weight.copy_(torch.eye(16, device=target_device))
    inputs = torch.arange(1, 97, device=target_device, dtype=torch.float32).reshape(2, 3, 16) / 32
    inputs[0, 0].zero_()
    labels = torch.full((2, 3), -100, device=target_device, dtype=torch.long)
    labels[0, 1] = 0
    ignored = torch.ones((2, 3), device=target_device, dtype=torch.bool)
    ignored[0, 0] = False

    def evaluate(items):
        model.zero_grad(set_to_none=True)
        output = model(
            inputs_embeds=inputs, labels=labels, num_items_in_batch=items, output_hidden_states=True,
        )
        hidden = output.hidden_states[-1]
        hidden.retain_grad()
        saved_hidden = hidden.detach().clone()
        assert torch.count_nonzero(saved_hidden[0, 0]) == 0
        assert (output.logits is None) == (loss_kind == 'linear')
        output.loss.backward()
        return output.loss.detach(), saved_hidden, hidden.grad.detach().clone(), model.lm_head.weight.grad.detach().clone()

    # the only CE target uses a zero hidden row; ignored rows and the head weight isolate the L2 contribution.
    with highest_precision():
        base_loss, hidden, base_dh, base_dw = evaluate(None)
        loss, scaled_hidden, dh, dw = evaluate(num_items)
        torch.testing.assert_close(scaled_hidden, hidden, rtol=0, atol=0)
        oracle_hidden = hidden.double().requires_grad_()
        oracle_weight = model.lm_head.weight.detach().double().requires_grad_()
        maxima = F.linear(oracle_hidden, oracle_weight).max(-1).values
        penalty = maxima.square().sum() * (1e-4 / (2 * maxima.numel()))
        expected_dh, expected_dw = torch.autograd.grad(penalty, (oracle_hidden, oracle_weight))
    count = int(num_items)
    assert torch.count_nonzero(expected_dh[ignored]) > 0 and torch.count_nonzero(expected_dw) > 0
    assert torch.count_nonzero(dh[ignored]) > 0 and torch.count_nonzero(dw) > 0
    assert torch.isfinite(dh).all() and torch.isfinite(dw).all()
    # relative-only checks prevent a small normalized penalty from disappearing under an absolute tolerance.
    torch.testing.assert_close(base_dh[ignored].double(), expected_dh[ignored], rtol=1.3e-6, atol=0)
    torch.testing.assert_close(base_dw.double(), expected_dw, rtol=1.3e-6, atol=0)
    torch.testing.assert_close(dh[ignored].double(), expected_dh[ignored] / count, rtol=1.3e-6, atol=0)
    torch.testing.assert_close(dw.double(), expected_dw / count, rtol=1.3e-6, atol=0)
    torch.testing.assert_close(dh[ignored].double() * count, base_dh[ignored].double(), rtol=1.3e-6, atol=0)
    torch.testing.assert_close(dw.double() * count, base_dw.double(), rtol=1.3e-6, atol=0)
    torch.testing.assert_close(loss.double() * count, base_loss.double(), rtol=1.3e-6, atol=0)


@pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None, reason='PHBA CUDA integration')
@pytest.mark.parametrize(
    ('fuse_norm', 'fuse_swiglu', 'loss_kind', 'use_l2warp', 'num_items', 'packed', 'dtype', 'tied'),
    [
        (False, False, 'native', True, None, False, torch.bfloat16, False),
        (True, False, 'ce', False, 111, False, torch.float16, False),
        (False, True, 'linear', False, None, True, torch.float16, True),
        (True, True, 'linear', True, 111, False, torch.float16, False),
        (True, True, 'ce', True, torch.tensor(111), True, torch.bfloat16, False),
    ],
    ids=['native-l2warp', 'norm-ce', 'swiglu-linear-packed-tied', 'all-fused-linear-l2warp', 'all-fused-ce-packed-l2warp'],
)
def test_loss_fusions(fuse_norm, fuse_swiglu, loss_kind, use_l2warp, num_items, packed, dtype, tied):
    torch.manual_seed(42)
    reference = PHBAForCausalLM(make_config(
        fuse_norm=False, fuse_swiglu=False, use_l2warp=use_l2warp, tie_word_embeddings=tied,
    )).to(device).train()
    candidate = PHBAForCausalLM(make_config(
        fuse_norm=fuse_norm,
        fuse_swiglu=fuse_swiglu,
        fuse_cross_entropy=loss_kind == 'ce',
        fuse_linear_cross_entropy=loss_kind == 'linear',
        use_l2warp=use_l2warp,
        tie_word_embeddings=tied,
    )).to(device).train()
    candidate.load_state_dict(reference.state_dict())
    if loss_kind != 'native':
        reference.criterion = _FloatCrossEntropy()
    tokens = torch.randint(0, 128, (1 if packed else 2, 67), device=device)
    labels = tokens.clone()
    labels[:, 7] = -100
    kwargs = {}
    if packed:
        kwargs['cu_seqlens'] = torch.tensor([0, 33, 67], dtype=torch.int32, device=device)
        labels[:, 33] = -100
    original_labels = labels.clone()
    with highest_precision(), torch.autocast('cuda', dtype=dtype):
        expected = reference(input_ids=tokens, labels=labels, num_items_in_batch=num_items, **kwargs)
        actual = candidate(input_ids=tokens, labels=labels, num_items_in_batch=num_items, **kwargs)
    assert torch.equal(labels, original_labels)
    assert (actual.logits is None) == (loss_kind == 'linear')
    assert_close('fused_loss', expected.loss, actual.loss, 1e-3)
    if actual.logits is not None:
        assert_close('fused_logits', expected.logits.float(), actual.logits.float(), .006)
    expected.loss.backward()
    actual.loss.backward()
    for (name, original), (other_name, fused) in zip(reference.named_parameters(), candidate.named_parameters(), strict=True):
        assert name == other_name and original.grad is not None and fused.grad is not None
        assert torch.isfinite(original.grad).all() and torch.isfinite(fused.grad).all()
        assert_close(name, original.grad, fused.grad, .006)


@pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None, reason='PHBA CUDA integration')
@pytest.mark.parametrize(
    ('return_dict', 'inputs_embeds', 'training', 'selection'),
    [(True, False, True, None), (False, True, False, 2), (True, True, False, [0, 32, 66])],
    ids=['training-ids-dict', 'eval-embeds-tuple-tail', 'eval-embeds-dict-indices'],
)
def test_forward_decoder_contract(return_dict, inputs_embeds, training, selection):
    torch.manual_seed(42)
    model = PHBAForCausalLM(make_config()).to(device).train(training)
    tokens = torch.randint(0, 128, (2, 67), device=device)
    labels = tokens.clone()
    labels[0, 3] = -100
    selection = torch.tensor(selection, device=device) if isinstance(selection, list) else selection
    with highest_precision(), torch.autocast('cuda', dtype=torch.float16):
        full = model(input_ids=tokens, labels=labels, output_hidden_states=True)
        embeddings = model.get_input_embeddings()(tokens) if inputs_embeds else None
        ids = None if inputs_embeds else tokens
        decoder = model.model(ids, None, embeddings, use_cache=False, output_hidden_states=True, return_dict=return_dict)
        states = decoder.last_hidden_state if return_dict else decoder[0]
        hidden = decoder.hidden_states if return_dict else decoder[1]
        logits = model.lm_head(states)
        shifted_labels = F.pad(labels[:, 1:], (0, 1), value=-100)
        expected_loss = F.cross_entropy(logits.reshape(-1, 128), shifted_labels.reshape(-1))
        positional = model(ids, None, embeddings, None, labels, False, False, True, return_dict)
        complete = model(input_ids=tokens)
        selected = model(
            input_ids=tokens, use_cache=False, logits_to_keep=selection, return_dict=return_dict,
        )
        with_labels = model(input_ids=tokens, labels=labels, logits_to_keep=selection)
        legacy = model(input_ids=tokens, num_logits_to_keep=2)
    actual_loss, actual_logits = (positional.loss, positional.logits) if return_dict else positional[:2]
    torch.testing.assert_close(actual_loss, full.loss)
    torch.testing.assert_close(actual_logits, full.logits)
    torch.testing.assert_close(logits, full.logits)
    torch.testing.assert_close(expected_loss, full.loss)
    assert len(hidden) == len(full.hidden_states) == model.config.num_hidden_layers + 1
    for result, expected in zip(hidden, full.hidden_states, strict=True):
        torch.testing.assert_close(result, expected)
    if return_dict:
        assert decoder.past_key_values is None and positional.past_key_values is None
    else:
        assert len(decoder) == 2 and len(positional) == 3
    torch.testing.assert_close(complete.logits, full.logits)
    if return_dict:
        assert selected.loss is None and selected.hidden_states is None and selected.past_key_values is None
    else:
        assert len(selected) == 1
    chosen = selected.logits if return_dict else selected[0]
    if isinstance(selection, torch.Tensor):
        expected = full.logits[:, selection]
    else:
        expected = full.logits[:, -selection:] if selection else full.logits
    assert_close('selected_logits', expected.float(), chosen.float(), 1e-3)
    assert_close('legacy_logits', full.logits[:, -2:].float(), legacy.logits.float(), 1e-3)
    torch.testing.assert_close(with_labels.logits, full.logits)
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(actual_loss, parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected_loss, parameters)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match='both input_ids and inputs_embeds'):
        model.model(input_ids=tokens, inputs_embeds=model.get_input_embeddings()(tokens))
    with pytest.raises(ValueError, match='either input_ids or inputs_embeds'):
        model.model()


def _check_initialization(fuse_norm, affine, tied, on_meta):
    torch.manual_seed(42)
    config = make_config(
        fuse_norm=fuse_norm, elementwise_affine=affine, qkv_bias=True, qk_norm=True,
        use_output_gate=True, tie_word_embeddings=tied, pad_token_id=0,
    )
    with torch.device('meta' if on_meta else 'cpu'):
        model = PHBAForCausalLM(config)
    if on_meta:
        assert all(parameter.is_meta for parameter in model.parameters())
        model.apply(model._init_weights)
        assert all(parameter.is_meta for parameter in model.parameters())
        model.to_empty(device='cpu')
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(float('nan'))
            for buffer in model.buffers():
                if buffer.is_floating_point():
                    buffer.fill_(float('nan'))
        model.apply(model._init_weights)
    head_dim = config.hidden_size // config.num_heads
    frequencies = 1. / (config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    for layer in model.model.layers:
        torch.testing.assert_close(layer.attn.rotary.inv_freq, frequencies, rtol=0, atol=0)
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert model.get_input_embeddings().weight[0].count_nonzero() == config.hidden_size
    assert (model.get_input_embeddings().weight is model.get_output_embeddings().weight) == tied
    for module in model.modules():
        if isinstance(module, nn.Linear):
            assert module.weight.count_nonzero() > 0
            if module.bias is not None:
                assert module.bias.count_nonzero() == 0
        elif hasattr(module, 'eps') and getattr(module, 'weight', None) is not None:
            torch.testing.assert_close(module.weight, torch.ones_like(module.weight), rtol=0, atol=0)
    for module, weight in [(model.model.layers[0].attn, model.model.layers[0].attn.o_proj.weight),
                           (model.model.layers[0].mlp, model.model.layers[0].mlp.down_proj.weight)]:
        expected = torch.empty_like(weight)
        torch.manual_seed(42)
        nn.init.kaiming_uniform_(expected, a=math.sqrt(5))
        expected /= math.sqrt(2 * config.num_hidden_layers)
        torch.manual_seed(42)
        model._init_weights(module, prenorm_residual_strategy='rescale')
        torch.testing.assert_close(weight, expected, rtol=0, atol=0)
        torch.manual_seed(42)
        model._init_weights(module, prenorm_residual_strategy='rescale')
        torch.testing.assert_close(weight, expected, rtol=0, atol=0)
        model._init_weights(module, prenorm_residual_strategy='zero')
        assert weight.count_nonzero() == 0
        with pytest.raises(ValueError, match='Invalid prenorm_residual_strategy'):
            model._init_weights(module, prenorm_residual_strategy='invalid')
    assert identities == {name: id(parameter) for name, parameter in model.named_parameters()}


@pytest.mark.parametrize(
    ('fuse_norm', 'affine', 'tied', 'on_meta'),
    [(False, True, False, False), (True, False, True, False), (False, True, False, True)],
    ids=['native-biased', 'fused-affine-free-tied', 'meta-materialization'],
)
def test_initialization(fuse_norm, affine, tied, on_meta, run_python):
    if on_meta:
        # isolate meta factories from cached operator-test allocation wrappers
        run_python(f"""
            import importlib.util

            spec = importlib.util.spec_from_file_location('phba_model_tests', {__file__!r})
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module._check_initialization({fuse_norm!r}, {affine!r}, {tied!r}, True)
        """)
    else:
        _check_initialization(fuse_norm, affine, tied, False)
