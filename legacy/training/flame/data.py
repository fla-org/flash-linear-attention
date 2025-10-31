
from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizer

from flame.logging import get_logger

logger = get_logger(__name__)


class HuggingfaceDataset(IterableDataset):

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        context_len: int = 2048,
        rank: int = 0,
        world_size: int = 1,
        buffer_size: int = 1024,
    ) -> HuggingfaceDataset:

        self.dataset = dataset
        self.tokenizer = tokenizer

        self.data = dataset.shard(world_size, rank)
        self.context_len = context_len
        self.rank = rank
        self.world_size = world_size
        self.buffer_size = buffer_size

        if tokenizer.vocab_size < torch.iinfo(torch.int16).max:
            self.dtype = torch.int16
        elif tokenizer.vocab_size < torch.iinfo(torch.int32).max:
            self.dtype = torch.int32
        else:
            self.dtype = torch.int64
        self.states = None
        self.buffer = torch.tensor([], dtype=self.dtype)
        self.tokens = []
        self.rand_id = 0
        self.token_id = 0
        self.rng_state = None
        self._epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._epoch + self.rank)
        if self.rng_state is not None:
            g.set_state(self.rng_state)

        rand_it = self.randint(0, self.buffer_size, g=g)
        if self.states is not None:
            self.data.load_state_dict(self.states)

        # max number of tokens allowed in the chunk buffer
        n_tokens = self.buffer_size * self.context_len

        while True:
            for sample in self.tokenize(self.data):
                # keep appending the samples to the token buffer
                self.tokens += sample
                # if the token buffer is full, start sampling
                # NOTE: we first convert the token ids to a tensor of shape [n_chunks, context_len] for efficiency
                if len(self.buffer) == 0 and len(self.tokens) >= n_tokens:
                    self.buffer = torch.tensor(self.tokens[:n_tokens], dtype=self.dtype).view(self.buffer_size, -1)
                    self.tokens = self.tokens[n_tokens:]
                if len(self.buffer) == self.buffer_size:
                    yield from self.sample(rand_it)

            n_chunks = len(self.tokens) // self.context_len
            # handle the left tokens in the buffer
            if n_chunks > 0:
                n_tokens = n_chunks * self.context_len
                indices = torch.randperm(n_chunks, generator=g).tolist()
                self.buffer = torch.tensor(self.tokens[:n_tokens], dtype=torch.long).view(n_chunks, -1)
                self.tokens = self.tokens[n_tokens:]
                for i in indices:
                    yield {'input_ids': self.buffer[i]}

    def tokenize(self, data, batch_size: int = 64):
        texts, states = [], []
        for sample in data:
            texts.append(sample['text'])
            states.append(self.data.state_dict())
            if len(texts) == batch_size:
                for s, tokenized in zip(states, self.tokenizer(texts, return_attention_mask=False)['input_ids'], strict=False):
                    self.states = s
                    yield tokenized
                texts, states = [], []
        if len(texts) > 0:
            for s, tokenized in zip(states, self.tokenizer(texts, return_attention_mask=False)['input_ids'], strict=False):
                self.states = s
                yield tokenized

    def sample(self, indices):
        n_tokens = (len(self.tokens) // self.context_len) * self.context_len
        while self.token_id < n_tokens:
            i = next(indices)
            start, end = self.token_id, self.token_id + self.context_len
            self.token_id += self.context_len
            yield {'input_ids': self.buffer[i].to(torch.long)}
            self.buffer[i] = torch.tensor(self.tokens[start:end], dtype=self.dtype)
        self.token_id = 0
        self.tokens = self.tokens[n_tokens:]

    def randint(
        self,
        low: int,
        high: int,
        batch_size: int = 1024,
        g: torch.Generator = torch.Generator(),
    ) -> Iterable[int]:
        indices = torch.empty(batch_size, dtype=torch.long)
        while True:
            # record the generator states before sampling
            self.rng_state = g.get_state()
            indices = torch.randint(low, high, (batch_size,), out=indices, generator=g)
            for i in indices[self.rand_id:].tolist():
                self.rand_id += 1
                yield i
            self.rand_id = 0

    def set_epoch(self, epoch):
        self._epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def state_dict(self):
        return {
            'states': self.states,
            'buffer': self.buffer.clone(),
            'tokens': deepcopy(self.tokens),
            'rand_id': self.rand_id,
            'token_id': self.token_id,
            'rng_state': self.rng_state,
            'epoch': self._epoch,
        }

    def load_state_dict(self, state_dict):
        self.states = state_dict['states']
        self.buffer = state_dict['buffer'].clone()
        self.tokens = deepcopy(state_dict['tokens'])
        self.rand_id = state_dict['rand_id']
        self.token_id = state_dict['token_id']
        self.rng_state = state_dict['rng_state'].clone() if state_dict['rng_state'] is not None else None
        self._epoch = state_dict['epoch']


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        varlen (`bool`):
            Whether to return sequences with variable lengths.
            If `True`, the offsets indicating the start and end of each sequence will be returned.
            For example, if the sequence lengths are `[4, 8, 12]`,
            the returned `input_ids` will be a long flattened tensor of shape `[1, 24]`, with `offsets` being `[0, 4, 12, 24]`.
            If `False`, the `input_ids` with shape `[batch_size, seq_len]` will be returned directly.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "pt".
    """

    tokenizer: PreTrainedTokenizer
    varlen: bool = False
    return_tensors: str = "pt"

    def __call__(
        self,
        examples: list[list[int] | dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(examples[0], dict):
            examples = [{'input_ids': example} for example in examples]

        def tensorize(example: dict[str, Any]) -> dict[str, Any]:
            tensorized = {}
            for key in ['input_ids', 'offsets']:
                if key not in example:
                    continue
                if isinstance(example[key], list):
                    tensorized[key] = torch.tensor(example[key], dtype=torch.long)
                elif isinstance(example[key], np.ndarray):
                    tensorized[key] = torch.from_numpy(example[key])
                else:
                    tensorized[key] = example[key]
            return tensorized

        examples = list(map(tensorize, examples))

        if not self.varlen:
            length_of_first = examples[0]['input_ids'].size(0)
            # Check if padding is necessary.
            if all(example['input_ids'].size(0) == length_of_first for example in examples):
                batch = {
                    'input_ids': torch.stack([example['input_ids'] for example in examples], dim=0),
                }
            else:
                # If yes, check if we have a `pad_token`.
                if self.tokenizer._pad_token is None:
                    raise ValueError(
                        f"You are attempting to pad samples but the tokenizer you are using "
                        f"({self.tokenizer.__class__.__name__}) does not have a pad token.",
                    )
                batch = self.tokenizer.pad(examples, return_tensors=self.return_tensors, return_attention_mask=False)
        else:
            if len(examples) > 1:
                raise ValueError("The batch size must be 1 for variable length inputs.")
            batch = {
                'input_ids': torch.cat([example['input_ids'] for example in examples], dim=0).unsqueeze(0),
            }
            if 'offsets' in examples[0]:
                batch['offsets'] = torch.cat([example['offsets'] for example in examples], dim=0).unsqueeze(0)
            else:
                # determine boundaries by bos/eos positions
                if self.tokenizer.add_bos_token:
                    offsets = []
                    if batch['input_ids'][0, 0] != self.tokenizer.bos_token_id:
                        offsets.append(torch.tensor([0], dtype=torch.long))
                    offsets.append(torch.where(batch['input_ids'].eq(self.tokenizer.bos_token_id))[1])
                    offsets.append(torch.tensor([len(batch['input_ids'][0])], dtype=torch.long))
                    batch['offsets'] = torch.cat(offsets, dim=0)
                elif self.tokenizer.add_eos_token:
                    offsets = [torch.tensor([0], dtype=torch.long)]
                    offsets.append(torch.where(batch['input_ids'].eq(self.tokenizer.eos_token_id))[1] + 1)
                    if batch['input_ids'][0, -1] != self.tokenizer.eos_token_id:
                        offsets.append(torch.tensor([len(batch['input_ids'][0])], dtype=torch.long))
                    batch['offsets'] = torch.cat(offsets, dim=0)
                else:
                    raise ValueError("You must allow the tokenizer to add either a bos or eos token as separators.")

        labels = batch['input_ids'].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch
