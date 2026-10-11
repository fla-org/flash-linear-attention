# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import argparse
import json
import math
from collections.abc import Iterator
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

import fla

if TYPE_CHECKING:
    from datasets import Dataset


class PerplexityEvaluator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        block_size: int = 32768,
        bucket_size: int = 2048,
        batch_size: int = 1,
    ):
        if block_size < 2:
            raise ValueError("block_size must be at least 2 to score next-token predictions")
        if bucket_size < 1 or batch_size < 1:
            raise ValueError("bucket_size and batch_size must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = block_size
        self.bucket_size = bucket_size
        self.batch_size = batch_size

    @staticmethod
    def preprocess(
        examples: dict[str, list[Any]],
        tokenizer: PreTrainedTokenizer,
        column_name: str = 'text',
    ) -> dict[str, list[list[int]]]:
        """Preprocess text data"""
        tokenized = tokenizer(examples[column_name])
        return {
            'input_ids': tokenized['input_ids'],
            'length': [len(ids) for ids in tokenized['input_ids']],
        }

    def batchify(self, dataset: 'Dataset', tokens_per_batch: int) -> Iterator[list[torch.Tensor]]:
        """Pack the token stream into fixed-length sequences, retaining the final partial sequence."""
        if tokens_per_batch < 2:
            raise ValueError("tokens_per_batch must be at least 2")
        current_tokens = []
        batch = []

        for sentence in dataset:
            tokens = sentence['input_ids'].tolist() if torch.is_tensor(sentence['input_ids']) else list(sentence['input_ids'])
            current_tokens.extend(tokens)
            start = 0
            while len(current_tokens) - start >= tokens_per_batch:
                batch.append(torch.tensor(current_tokens[start:start + tokens_per_batch], dtype=torch.long))
                start += tokens_per_batch
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            current_tokens = current_tokens[start:]

        if batch:
            yield batch
        if current_tokens:
            yield [torch.tensor(current_tokens, dtype=torch.long)]

    @torch.no_grad()
    def process_batch(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute FP32 next-token NLLs, excluding the unscored last logit of each sequence."""
        input_ids = torch.stack(batch).to(self.device)
        labels = input_ids[:, 1:]
        nlls = torch.empty(labels.shape, dtype=torch.float32, device=self.device)
        if labels.numel() > 0:
            outputs = self.model(input_ids=input_ids, use_cache=False, return_dict=True)
            logits = outputs['logits']
            # limit the FP32 softmax workspace to one positional bucket at a time
            for start in range(0, labels.shape[1], self.bucket_size):
                end = min(start + self.bucket_size, labels.shape[1])
                nlls[:, start:end] = F.cross_entropy(
                    input=logits[:, start:end].float().reshape(-1, logits.shape[-1]),
                    target=labels[:, start:end].reshape(-1),
                    reduction='none',
                ).view(input_ids.shape[0], -1)

        return {
            'input_ids': input_ids,
            'nlls': nlls,
            'labels': labels,
        }

    @torch.no_grad()
    def evaluate(self, dataset: 'Dataset') -> dict[str, Any]:
        """Score independent fixed-length contexts and the final tail using a shared token-weighted NLL."""
        self.model.eval()
        total_tokens = 0
        total_input_tokens = 0
        total_sentences = 0
        num_blocks = math.ceil((self.block_size - 1) / self.bucket_size)
        block_loss = torch.zeros(num_blocks, dtype=torch.float32, device=self.device)
        block_tokens = [0 for _ in range(num_blocks)]

        with tqdm(self.batchify(dataset, self.block_size), unit='batch') as bar:
            for batch in bar:
                batch_outputs = self.process_batch(batch)
                input_ids = batch_outputs['input_ids']
                nlls = batch_outputs['nlls']
                total_input_tokens += input_ids.numel()
                total_tokens += nlls.numel()
                total_sentences += input_ids.shape[0]

                for i, start in enumerate(range(0, nlls.shape[1], self.bucket_size)):
                    bucket = nlls[:, start:start + self.bucket_size]
                    block_loss[i] += bucket.sum()
                    block_tokens[i] += bucket.numel()

                bar.set_description_str(f"[{total_tokens:10} scored tokens, {total_sentences:8} sequences]")

        if total_tokens == 0:
            raise ValueError("The dataset must contain at least two tokens to evaluate perplexity")

        block_nlls = block_loss.cpu().tolist()
        total_nll = sum(block_nlls)
        block_ppls = [
            math.exp(loss / tokens) if tokens else None
            for loss, tokens in zip(block_nlls, block_tokens, strict=True)
        ]

        return {
            'perplexity': math.exp(total_nll / total_tokens),
            'block_perplexities': block_ppls,
            'block_nlls': block_nlls,
            'block_tokens': block_tokens,
            'total_nll': total_nll,
            'total_tokens': total_tokens,
            'total_input_tokens': total_input_tokens,
            'total_sentences': total_sentences,
        }


def main():
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description="Evaluate perplexity")
    parser.add_argument('-p', '--path', type=str, default='fla-hub/gla-1.3B-100B')
    parser.add_argument('-d', '--data', type=str, default='fla-hub/pg19')
    parser.add_argument('-s', '--split', type=str, default='train')
    parser.add_argument('-n', '--column_name', type=str, default='text')
    parser.add_argument('--block_size', type=int, default=28672)
    parser.add_argument('--bucket_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--num_proc', type=int, default=32)
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--data_revision', type=str, default='main')
    parser.add_argument('--output', type=Path, default=None, help='Write results and evaluation settings as JSON')
    args = parser.parse_args()
    if args.block_size < 2 or min(args.bucket_size, args.batch_size, args.num_proc) < 1:
        parser.error('block_size must be at least 2; bucket_size, batch_size and num_proc must be positive')

    # Set device and random seed
    if args.device is None:
        from fla.utils import device
    else:
        device = args.device
    torch.manual_seed(0)

    # Load model and tokenizer
    print(f"Loading model {args.path}")
    tokenizer = AutoTokenizer.from_pretrained(args.path, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        revision=args.revision,
        torch_dtype=getattr(torch, args.dtype),
        device_map={"": device},
    ).eval()
    print(f"{model}")

    # Load dataset
    print(f"Loading data {args.data}")
    dataset = load_dataset(args.data, split=args.split, revision=args.data_revision)
    dataset = dataset.map(
        partial(PerplexityEvaluator.preprocess, tokenizer=tokenizer, column_name=args.column_name),
        batched=True,
        num_proc=args.num_proc,
    )
    print(dataset)
    print("batch_size", args.batch_size,
          "block_size", args.block_size,
          "total_tokens_per_batch", args.batch_size * args.block_size)

    # Create evaluator and run evaluation
    evaluator = PerplexityEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        block_size=args.block_size,
        bucket_size=args.bucket_size,
        batch_size=args.batch_size,
    )

    results = evaluator.evaluate(dataset)
    if args.output is not None:
        results['config'] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        results['config']['device'] = str(device)
        results['model_commit'] = getattr(model.config, '_commit_hash', None)
        results['dataset_fingerprint'] = dataset._fingerprint
        results['versions'] = {
            'fla': fla.__version__,
            'torch': torch.__version__,
            'transformers': version('transformers'),
            'datasets': version('datasets'),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + '\n')

    # Print results
    print("\nEvaluation Results:")
    print(f"Final Perplexity: {results['perplexity']:.2f}")
    print(f"Scored Tokens: {results['total_tokens']}")
    print(f"Input Tokens: {results['total_input_tokens']}")
    print(f"Total Sequences: {results['total_sentences']}")
    print("\nBlock-wise Perplexities:")
    for i, ppl in enumerate(results['block_perplexities']):
        print(f"Block {i}: {ppl:.2f}" if ppl is not None else f"Block {i}: no scored tokens")


if __name__ == "__main__":
    main()
