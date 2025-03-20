# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024, Songlin Yang, Yu Zhang.

import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import fla  # noqa


def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation benchmarking")
    parser.add_argument("--path", type=str, default="fla-hub/transformer-1.3B-100B")
    parser.add_argument("--data", type=str, default="fla-hub/pg19")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--no-cache", action='store_true')
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--topp", type=float, default=0.2)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--compile", action='store_true')
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    print(f"Loading {args.path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.path,
        trust_remote_code=True,
        add_eos_token=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"{tokenizer}")

    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        device_map={"": device},
        torch_dtype=dtype,
        use_cache=not args.no_cache
    )
    if args.compile:
        print("Compiling the model")
        model = torch.compile(model)
    model.eval()
    print(f"{model.config}\n{model}\nNumber of parameters: {model.num_parameters()} ({sizeof_fmt(model.num_parameters())})\n")

    print(f"Loading {args.data}")
    dataset = load_dataset(args.data, split='train', trust_remote_code=True)
    print(f"{dataset}")

    prompt = dataset[0]['text']
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)[:, :args.length].contiguous()
    max_length = input_ids.shape[1] + args.maxlen

    torch.cuda.synchronize()
    start = time.time()
    with torch.inference_mode():
        text = model.generate(
            input_ids=input_ids,
            use_cache=not args.no_cache,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.bos_token_id,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.topp,
            repetition_penalty=args.repetition_penalty
        )
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Prompt:\n{tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0].strip()}\n")
    print(f"Generated:\n{tokenizer.batch_decode(text, skip_special_tokens=True)[0].strip()}\n")
    print(f"Prompt length: {len(input_ids[0])}, generation length: {len(text[0]) - len(input_ids[0])}")
    print(f"Total prompt processing + decoding time: {elapsed * 1000:.0f}ms")
    print(f"Max memory used: {sizeof_fmt(torch.cuda.max_memory_allocated())}")
