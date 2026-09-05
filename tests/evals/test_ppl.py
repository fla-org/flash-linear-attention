# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import json
import math
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.modeling_outputs import CausalLMOutput

from evals import ppl
from evals.ppl import PerplexityEvaluator
from fla.models.gla import GLAConfig, GLAForCausalLM


class LookupLM(torch.nn.Module):
    def __init__(self, table: torch.Tensor, last_eos_logit: float | None = None):
        super().__init__()
        self.register_buffer('table', table)
        self.last_eos_logit = last_eos_logit
        self.config = SimpleNamespace(_commit_hash='model-sha')

    def forward(self, input_ids, labels=None, use_cache=None, return_dict=None):
        assert labels is None
        assert use_cache is False
        assert return_dict is True
        assert not self.training
        assert not torch.is_grad_enabled()
        logits = self.table[input_ids].clone()
        if self.last_eos_logit is not None:
            logits[:, -1, 0] = self.last_eos_logit
        return CausalLMOutput(logits=logits)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16], ids=['fp32', 'bf16'])
@pytest.mark.parametrize(
    ('block_size', 'bucket_size', 'batch_size', 'num_tokens'),
    [
        pytest.param(4, 2, 1, 3, id='short'),
        pytest.param(4, 2, 3, 3, id='short-large-batch'),
        pytest.param(4, 2, 1, 14, id='tail-single-batch'),
        pytest.param(4, 2, 3, 14, id='tail-multi-batch'),
        pytest.param(5, 2, 2, 11, id='singleton-tail'),
        pytest.param(5, 3, 2, 10, id='partial-bucket'),
        pytest.param(4, 1, 3, 13, id='single-token-buckets'),
        pytest.param(4, 8, 2, 10, id='oversized-bucket'),
    ],
)
def test_evaluate(block_size, bucket_size, batch_size, num_tokens, dtype):
    torch.manual_seed(42)
    table = torch.randn(7, 7, dtype=dtype)
    tokens = torch.randint(0, 7, (num_tokens,)).tolist()
    dataset = [{'input_ids': tokens[:2]}, {'input_ids': []}, {'input_ids': tokens[2:]}]
    evaluator = PerplexityEvaluator(
        model=LookupLM(table=table),
        tokenizer=SimpleNamespace(eos_token_id=None),
        device='cpu',
        block_size=block_size,
        bucket_size=bucket_size,
        batch_size=batch_size,
    )
    result = evaluator.evaluate(dataset=dataset)

    log_probs = table.double().log_softmax(dim=-1)
    reference_nlls = [[] for _ in range(math.ceil((block_size - 1) / bucket_size))]
    for start in range(0, len(tokens), block_size):
        sequence = tokens[start:start + block_size]
        for position, (context, target) in enumerate(zip(sequence[:-1], sequence[1:], strict=True)):
            reference_nlls[position // bucket_size].append(-log_probs[context, target].item())
    reference_counts = [len(values) for values in reference_nlls]
    reference_sums = [sum(values) for values in reference_nlls]
    reference_total = sum(reference_sums)
    reference_count = sum(reference_counts)

    assert result['total_tokens'] == reference_count
    assert result['total_input_tokens'] == num_tokens
    assert result['total_sentences'] == math.ceil(num_tokens / block_size)
    assert result['block_tokens'] == reference_counts
    assert result['block_nlls'] == pytest.approx(reference_sums, rel=1e-6)
    assert result['total_nll'] == pytest.approx(reference_total, rel=1e-6)
    assert result['perplexity'] == pytest.approx(math.exp(reference_total / reference_count), rel=1e-6)
    for actual, values in zip(result['block_perplexities'], reference_nlls, strict=True):
        if values:
            assert actual == pytest.approx(math.exp(sum(values) / len(values)), rel=1e-6)
        else:
            assert actual is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('last_eos_logit', [-20., 20.], ids=['unlikely-eos', 'likely-eos'])
def test_evaluate_last_logit(last_eos_logit):
    torch.manual_seed(42)
    evaluator = PerplexityEvaluator(
        model=LookupLM(table=torch.zeros(4, 4), last_eos_logit=last_eos_logit),
        tokenizer=SimpleNamespace(eos_token_id=0),
        device='cpu',
        block_size=4,
        bucket_size=2,
    )
    result = evaluator.evaluate(dataset=[{'input_ids': [1, 2, 0, 1]}])
    assert result['perplexity'] == pytest.approx(4.)
    assert result['block_perplexities'] == pytest.approx([4., 4.])
    assert result['block_tokens'] == [2, 1]


def test_evaluate_bf16_uniform_logits():
    torch.manual_seed(42)
    evaluator = PerplexityEvaluator(
        model=LookupLM(table=torch.zeros(1, 100000, dtype=torch.bfloat16)),
        tokenizer=SimpleNamespace(eos_token_id=None),
        device='cpu',
        block_size=4,
        bucket_size=2,
        batch_size=2,
    )
    result = evaluator.evaluate(dataset=[{'input_ids': [0] * 10}])
    assert result['perplexity'] == pytest.approx(100000., rel=1e-6)
    assert result['block_perplexities'] == pytest.approx([100000., 100000.], rel=1e-6)


def test_evaluate_no_scalar_reads():
    class RejectScalarReads(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._local_scalar_dense.default:
                raise AssertionError('Evaluation must not extract individual tensor scalars')
            return func(*args, **(kwargs or {}))

    torch.manual_seed(42)
    evaluator = PerplexityEvaluator(
        model=LookupLM(table=torch.zeros(4, 4)),
        tokenizer=None,
        device='cpu',
        block_size=4,
        bucket_size=1,
    )
    with RejectScalarReads():
        result = evaluator.evaluate(dataset=[{'input_ids': [1, 2, 3, 0] * 3}])
    assert result['perplexity'] == pytest.approx(4.)


def test_evaluate_fused_linear_cross_entropy():
    torch.manual_seed(42)
    config = GLAConfig(
        hidden_size=8,
        num_hidden_layers=0,
        num_heads=1,
        vocab_size=7,
        fuse_norm=False,
        fuse_cross_entropy=False,
        fuse_linear_cross_entropy=True,
        return_dict=False,
    )
    model = GLAForCausalLM(config=config).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        logits = model(input_ids=tokens, use_cache=False, return_dict=True).logits.double()
        expected = -logits[:, :-1].log_softmax(dim=-1).gather(-1, tokens[:, 1:, None]).mean().item()
    evaluator = PerplexityEvaluator(
        model=model,
        tokenizer=SimpleNamespace(eos_token_id=None),
        device='cpu',
        block_size=4,
        bucket_size=2,
    )
    result = evaluator.evaluate(dataset=[{'input_ids': tokens[0]}])
    assert result['perplexity'] == pytest.approx(math.exp(expected), rel=1e-6)
    assert model.config.fuse_linear_cross_entropy is True
    assert model.config.return_dict is False


@pytest.mark.parametrize('tokens', [[], [1]], ids=['empty', 'singleton'])
def test_evaluate_no_predictions(tokens):
    torch.manual_seed(42)
    evaluator = PerplexityEvaluator(
        model=LookupLM(table=torch.zeros(2, 2)),
        tokenizer=None,
        device='cpu',
        block_size=4,
        bucket_size=2,
    )
    with pytest.raises(ValueError, match='at least two tokens'):
        evaluator.evaluate(dataset=[{'input_ids': tokens}])


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({'block_size': 1}, id='short-block'),
        pytest.param({'bucket_size': 0}, id='empty-bucket'),
        pytest.param({'batch_size': -1}, id='negative-batch'),
    ],
)
def test_evaluator_invalid_sizes(kwargs):
    with pytest.raises(ValueError):
        PerplexityEvaluator(model=None, tokenizer=None, **kwargs)


def test_main(tmp_path, monkeypatch):
    datasets = pytest.importorskip('datasets')
    torch.manual_seed(42)
    model = LookupLM(table=torch.zeros(4, 4))
    tokenizer = SimpleNamespace(eos_token_id=None)
    dataset = datasets.Dataset.from_dict({'input_ids': [[1, 2, 3]]})
    monkeypatch.setattr(ppl.AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: tokenizer)

    def load_model(path, **kwargs):
        assert path == 'local-model'
        assert kwargs['revision'] == 'model-sha'
        assert kwargs['torch_dtype'] == torch.float32
        return model

    def load_dataset(path, **kwargs):
        assert kwargs == {'split': 'test', 'revision': 'data-sha'}
        return dataset

    monkeypatch.setattr(ppl.AutoModelForCausalLM, 'from_pretrained', load_model)
    monkeypatch.setattr(datasets, 'load_dataset', load_dataset)
    monkeypatch.setattr(datasets.Dataset, 'map', lambda self, *args, **kwargs: self)
    output = tmp_path / 'results' / 'ppl.json'
    monkeypatch.setattr(sys, 'argv', [
        'ppl', '--path', 'local-model', '--split', 'test', '--device', 'cpu', '--dtype', 'float32',
        '--revision', 'model-sha', '--data_revision', 'data-sha', '--num_proc', '1',
        '--block_size', '8', '--bucket_size', '2', '--output', str(output),
    ])
    ppl.main()
    result = json.loads(output.read_text())
    assert result['perplexity'] == pytest.approx(4.)
    assert result['block_perplexities'] == [pytest.approx(4.), None, None, None]
    assert result['total_tokens'] == 2
    assert result['config']['revision'] == 'model-sha'
    assert result['config']['data_revision'] == 'data-sha'
    assert result['config']['device'] == 'cpu'
    assert result['model_commit'] == 'model-sha'
    assert result['dataset_fingerprint'] == dataset._fingerprint
    assert result['versions']['torch'] == torch.__version__
