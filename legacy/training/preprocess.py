
from __future__ import annotations

import argparse
from itertools import chain
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


def tokenize(
    examples: dict[str, list[Any]],
    tokenizer: AutoTokenizer,
    seq_len: int = 2048,
    ctx_len: int = None,
    return_offsets: bool = False,
) -> dict[str, list[list[int]]]:
    """
    Tokenize the input text and split into chunks of specified context length.

    Args:
        examples:
            Dictionary containing the input text.
        tokenizer:
            Initialized tokenizer.
        seq_len:
            Total sequence length for each training sample. Default: 2048.
        ctx_len:
            Max contiguous length to preserve (will not be split). Default: `None`.
        return_offsets:
            Return cumulative offsets for concatenated inputs. Default: `False`.

    Returns:
        Dictionary containing tokenized and chunked input ids, and optionally offsets.
    """
    text = examples['text']
    input_ids = tokenizer(text)['input_ids']
    # further split each input into chunks of length `ctx_len` if provided
    if ctx_len is not None:
        input_ids = [seq[i:i+ctx_len] for seq in input_ids for i in range(0, len(seq), ctx_len)]
    lens = torch.tensor([len(seq) for seq in input_ids]).cumsum(0)
    total_len = lens[-1] // seq_len * seq_len

    input_ids = list(chain(*input_ids))
    # each yielded sample is of length `seq_len`
    input_ids = [input_ids[i:i+seq_len] for i in range(0, total_len, seq_len)]

    if not return_offsets:
        return {'input_ids': input_ids}

    # insert boundaries into cumulative offsets
    offsets = torch.cat((lens, torch.arange(0, total_len, seq_len))).unique().sort()[0] % seq_len
    # split offsets according the start positions
    offsets = [i.tolist() + [seq_len] for i in offsets.tensor_split(torch.where(offsets.eq(0))[0][1:])][:len(input_ids)]
    return {'input_ids': input_ids, 'offsets': offsets}


def preprocess(
    dataset: str,
    name: str | None = None,
    split: str = 'train',
    seed: int = 42,
    output: str = 'data',
    tokenizer: str = 'fla-hub/gla-1.3B-100B',
    num_proc: int = 64,
    batch_size: int = 2048,
    seq_len: int = 2048,
    ctx_len: int = None,
    return_offsets: bool = False,
) -> None:
    """
    Load, tokenize, and save the processed dataset.

    Args:
        dataset:
            Path or name of the dataset. Default: 'HuggingFaceFW/fineweb-edu'.
        name:
            Name of the dataset configuration. Default: `None`.
        split:
            Dataset split to process. Default: 'train'.
        seed:
            Random seed for shuffling the dataset. Default: 42.
        output:
            Output directory. Default: 'data'.
        tokenizer:
            Tokenizer name. Default: 'fla-hub/gla-1.3B-100B'.
        num_proc:
            Number of processes for parallel processing. Default: 64.
        batch_size:
            Batch size for processing. Default: 2048.
        seq_len:
            Total sequence length for each training sample. Default: 2048.
        ctx_len:
            Max contiguous length to preserve (will not be split). Default: `None`.
        return_offsets:
            Return cumulative offsets for concatenated inputs. Default: `False`.
    """
    tokenized_path = f'{output}/{dataset}/{name}/{split}' if name is not None else f'{output}/{dataset}/{split}'

    if ctx_len is not None and ctx_len > seq_len:
        raise ValueError(f'ctx_len ({ctx_len}) must be less than or equal to seq_len ({seq_len})')

    logger.info(f'Loading tokenizer {tokenizer}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    logger.info(f'Tokenizer initialized:\n {tokenizer}')

    logger.info(f'Loading dataset: {dataset}')
    dataset = load_dataset(dataset, name=name, split=split)
    dataset = dataset.shuffle(seed=seed)
    logger.info(f'Dataset loaded: {dataset}')

    remove_columns = list(next(iter(dataset)).keys())
    logger.info(f'Tokenizing and processing the dataset with batch size {batch_size}')
    dataset = dataset.map(
        lambda examples: tokenize(examples, tokenizer, seq_len, ctx_len, return_offsets),
        batched=True,
        batch_size=batch_size,
        remove_columns=remove_columns,
        num_proc=num_proc,
        desc="Running tokenizer on dataset",
    )

    logger.info(f'Saving processed dataset to {tokenized_path}')
    dataset.save_to_disk(tokenized_path, num_proc=num_proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and tokenize dataset")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu", help="Path or name of the dataset")
    parser.add_argument("--name", default=None, help="Name of the dataset configuration")
    parser.add_argument("--split", default="train", help="Dataset split to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--tokenizer", default="fla-hub/gla-1.3B-100B", help="Tokenizer name")
    parser.add_argument("--num_proc", type=int, default=64, help="Number of processes for parallel processing")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for processing")
    parser.add_argument("--seq_len", type=int, default=2048, help="Total sequence length for each training sample")
    parser.add_argument("--ctx_len", type=int, default=None, help="Max contiguous length to preserve (will not be split)")
    parser.add_argument("--return_offsets", action="store_true", help="Return cumulative offsets for concatenated inputs")
    args = parser.parse_args()

    preprocess(
        dataset=args.dataset,
        name=args.name,
        split=args.split,
        seed=args.seed,
        output=args.output,
        tokenizer=args.tokenizer,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        ctx_len=args.ctx_len,
        return_offsets=args.return_offsets,
    )
