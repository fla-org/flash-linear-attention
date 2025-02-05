# -*- coding: utf-8 -*-

import argparse
import time
from typing import Optional, Tuple

import torch
from accelerate import Accelerator
from torch.cuda import max_memory_allocated, memory_allocated
from torch.optim import AdamW
from tqdm import trange
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.optimization import get_cosine_schedule_with_warmup

import fla

classes = [getattr(fla.models, i) for i in fla.models.__all__]
configs = {i.model_type: i() for i in classes if issubclass(i, PretrainedConfig)}


def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:.2f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.2f}Yi{suffix}'


def prepare_inputs(
    batch_size: int,
    seq_len: int,
    context_len: int,
    varlen: bool,
    vocab_size: int,
    device: torch.device
):
    if varlen:
        tokens = torch.randint(high=vocab_size, size=(1, batch_size * seq_len), device=device)
        cu_seqlens = torch.cat([
            torch.tensor([0]),
            torch.randperm(batch_size * seq_len - 16)[:torch.randint(8, 64, size=(1,))] + 16,
            torch.tensor([batch_size * seq_len])
        ], 0).sort()[0].to(dtype=torch.int32, device=device)
        if context_len is not None:
            cu_seqlens = torch.cat(
                [torch.arange(i, j, context_len) for i, j in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())] +
                [torch.tensor([len(tokens[0])])]
            ).to(dtype=torch.int32, device=device)
    else:
        tokens = torch.randint(high=vocab_size, size=(batch_size, seq_len), device=device)
        cu_seqlens = None
    return tokens, cu_seqlens


def profile(
    name: str,
    batch_size: int = 8,
    seq_len: int = 2048,
    context_len: int = 2048,
    varlen: bool = False,
    warmup_steps: int = 16,
    steps: int = 32,
    total_steps: int = 1024,
    lr: float = 3e-4,
    betas: Tuple[float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    mixed_precision: str = 'bf16',
    compile: bool = False
):
    device = torch.device('cuda')
    config = configs[name] if name in configs else AutoConfig.from_pretrained(name)
    model = AutoModelForCausalLM.from_config(config).cuda().to(dtype)
    if compile:
        print("Compiling the model")
        model = torch.compile(model)
    num_parameters = model.num_parameters()
    print(f"Initializing {name} model from the config:\n{config}\n{model}")
    print(f"Number of parameters in total: {num_parameters} ({sizeof_fmt(num_parameters)})")
    print(f"Allocated memory after initialization: {sizeof_fmt(memory_allocated(device))}")

    accelerator = Accelerator(mixed_precision=mixed_precision)
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        fused=True
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, total_steps)

    bar = trange(warmup_steps)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    torch.cuda.synchronize(device)
    for _ in bar:
        # forward pass
        tokens, cu_seqlens = prepare_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            context_len=context_len,
            varlen=varlen,
            vocab_size=config.vocab_size,
            device=device
        )
        outputs = model(tokens, labels=tokens, cu_seqlens=cu_seqlens)
        # backward pass
        accelerator.backward(outputs.loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        bar.set_description_str(f"Max memory allocated: {sizeof_fmt(max_memory_allocated(device))}")

    start, total_tokens = time.time(), 0
    bar = trange(steps)
    torch.cuda.synchronize(device)
    for _ in bar:
        # forward pass
        tokens, cu_seqlens = prepare_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            context_len=context_len,
            varlen=varlen,
            vocab_size=config.vocab_size,
            device=device
        )
        outputs = model(tokens, labels=tokens, cu_seqlens=cu_seqlens)
        # backward pass
        accelerator.backward(outputs.loss)
        optimizer.step()
        optimizer.zero_grad()

        total_tokens += batch_size * seq_len
        torch.cuda.synchronize(device)
        duration = time.time() - start
        bar.set_description_str(f"Thoughput: {total_tokens / duration:10.2f} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='retnet')
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--seq_len", default=2048, type=int)
    parser.add_argument("--context_len", default=None, type=int)
    parser.add_argument("--varlen", action='store_true')
    parser.add_argument("--warmup_steps", default=16, type=int)
    parser.add_argument("--steps", default=32, type=int)
    parser.add_argument("--compile", action='store_true')
    args = parser.parse_args()
    profile(
        name=args.name,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        context_len=args.context_len,
        varlen=args.varlen,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        compile=args.compile
    )
