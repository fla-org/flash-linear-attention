
import argparse
import math
from collections.abc import Iterator
from functools import partial
from typing import Any

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from fla.modules.fused_cross_entropy import FusedCrossEntropyLoss


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
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = block_size
        self.bucket_size = bucket_size
        self.batch_size = batch_size
        self.loss_fct = FusedCrossEntropyLoss(reduction='sum')

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

    def batchify(self, dataset: Dataset, tokens_per_batch: int) -> Iterator[list[torch.Tensor]]:
        """Split dataset into batches of exactly block_size length"""
        current_tokens = []  # Buffer to store all tokens

        for sentence in dataset:
            # Convert input_ids to list and add to buffer
            tokens = sentence['input_ids'].tolist() if torch.is_tensor(sentence['input_ids']) else list(sentence['input_ids'])
            if not tokens:
                continue
            current_tokens.extend(tokens)

            # When we have enough tokens, yield batches
            while len(current_tokens) >= self.block_size * self.batch_size:
                batch = []
                for _ in range(self.batch_size):
                    # Extract exactly block_size tokens
                    batch.append(torch.tensor(current_tokens[:self.block_size], dtype=torch.long))
                    current_tokens = current_tokens[self.block_size:]
                yield batch

        # Handle remaining tokens if they form complete blocks
        if len(current_tokens) >= self.block_size:
            remaining_batches = len(current_tokens) // self.block_size
            remaining_batches = min(remaining_batches, self.batch_size)
            if remaining_batches > 0:
                batch = []
                for _ in range(remaining_batches):
                    batch.append(torch.tensor(current_tokens[:self.block_size], dtype=torch.long))
                    current_tokens = current_tokens[self.block_size:]
                yield batch

    def process_batch(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Process a single batch of data"""
        # Stack the tensors - no need for padding since all sequences are block_size
        input_ids = torch.stack(batch).to(self.device)

        # Calculate number of blocks for each sequence
        blocks = [
            (self.block_size-1)//self.bucket_size
            for _ in range(input_ids.shape[0])
        ]

        # Prepare labels
        labels = input_ids.clone()

        # Forward pass
        outputs = self.model(input_ids, labels=labels)

        # Calculate next token prediction labels
        next_token_labels = torch.cat((
            input_ids[..., 1:],
            torch.full_like(input_ids[:, :1], self.tokenizer.eos_token_id),
        ), -1)

        # Calculate negative log likelihood
        nlls = (-outputs['logits'].log_softmax(-1)).gather(-1, next_token_labels.unsqueeze(-1)).squeeze(-1)

        return {
            'input_ids': input_ids,
            'loss': outputs['loss'],
            'nlls': nlls,
            'labels': next_token_labels,
            'blocks': blocks,
        }

    def evaluate(self, dataset: Dataset) -> dict[str, Any]:
        """Evaluate perplexity on the entire dataset"""
        total_loss = 0
        total_tokens = 0
        total_sentences = 0

        # Initialize block statistics
        num_blocks = (self.block_size - 1) // self.bucket_size + 1
        block_loss = [torch.tensor(0., dtype=torch.float, device=self.device) for _ in range(num_blocks)]
        block_tokens = [1e-10 for _ in range(num_blocks)]
        bucket_sizes = [0 for _ in range(num_blocks)]

        # Create progress bar
        bar = tqdm(self.batchify(dataset, self.block_size))

        for batch in bar:
            batch_outputs = self.process_batch(batch)
            input_ids = batch_outputs['input_ids']

            nlls = batch_outputs['nlls']
            labels = batch_outputs['labels']
            blocks = batch_outputs['blocks']

            # Update statistics
            total_tokens += input_ids.ne(self.loss_fct.ignore_index).sum()
            total_sentences += input_ids.shape[0]
            print(input_ids.shape[1])

            for i in blocks:
                bucket_sizes[i] += 1

            # Calculate block-level loss
            for i, j in enumerate(range(0, min(input_ids.shape[-1], self.block_size), self.bucket_size)):
                block_loss[i] += nlls[:, j:j+self.bucket_size].sum()
                block_tokens[i] += labels[:, j:j+self.bucket_size].ne(self.loss_fct.ignore_index).sum()

            # Update total loss
            total_loss += batch_outputs['loss'].item() * labels.ne(self.loss_fct.ignore_index).sum()

            # Update progress bar
            ppls = [f"{math.exp(loss / toks):6.2f}" for loss, toks in zip(block_loss, block_tokens, strict=False)]
            bar.set_description_str(f"[{total_tokens:10} tokens, {total_sentences:8} sentences] " + ' '.join(ppls))

        # Calculate final results
        final_ppl = math.exp(total_loss / total_tokens)
        block_ppls = [math.exp(loss / toks) for loss, toks in zip(block_loss, block_tokens, strict=False)]

        return {
            'perplexity': final_ppl,
            'block_perplexities': block_ppls,
            'total_tokens': total_tokens,
            'total_sentences': total_sentences,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity")
    parser.add_argument('-p', '--path', type=str, default='fla-hub/gla-1.3B-100B')
    parser.add_argument('-d', '--data', type=str, default='fla-hub/pg19')
    parser.add_argument('-s', '--split', type=str, default='train')
    parser.add_argument('-n', '--column_name', type=str, default='text')
    parser.add_argument('--block_size', type=int, default=28672)
    parser.add_argument('--bucket_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Set device and random seed
    if args.device is None:
        from fla.utils import device
    else:
        device = args.device
    torch.manual_seed(0)

    # Load model and tokenizer
    print(f"Loading model {args.path}")
    tokenizer = AutoTokenizer.from_pretrained(args.path)
    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        device_map={"": device},
    ).bfloat16().eval()
    print(f"{model}")

    # Load dataset
    print(f"Loading data {args.data}")
    dataset = load_dataset(args.data, split=args.split)
    dataset = dataset.map(
        partial(PerplexityEvaluator.preprocess, tokenizer=tokenizer, column_name=args.column_name),
        batched=True,
        num_proc=32,
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

    with torch.no_grad():
        results = evaluator.evaluate(dataset)

    # Print results
    print("\nEvaluation Results:")
    print(f"Final Perplexity: {results['perplexity']:.2f}")
    print(f"Total Tokens: {results['total_tokens']}")
    print(f"Total Sentences: {results['total_sentences']}")
    print("\nBlock-wise Perplexities:")
    for i, ppl in enumerate(results['block_perplexities']):
        print(f"Block {i}: {ppl:.2f}")


if __name__ == "__main__":
    main()
