# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Benchmark one complete MoM LM configuration per process; select a revision with PYTHONPATH."""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

import fla
from fla.models.mom import MomConfig, MomForCausalLM
from fla.models.utils import Cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'forward', 'prefill', 'decode'], default='train')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--length', type=int, default=4096)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--memories', type=int, default=31)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--single-kv-proj', action='store_true')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--layout', choices=['dense', 'varlen'], default='dense')
    parser.add_argument('--routing', choices=['learned', 'skewed'], default='learned')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(8)
    config = MomConfig(
        hidden_size=args.hidden,
        head_dim=128,
        num_heads=args.hidden // 128,
        expand_v=1,
        num_hidden_layers=args.layers,
        vocab_size=8192,
        num_memories=args.memories,
        topk=args.topk,
        shared_mem=True,
        single_kv_proj=args.single_kv_proj,
        use_cache=False,
        fuse_cross_entropy=True,
    )
    model = MomForCausalLM(config).cuda().train(args.mode == 'train')
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.routing == 'skewed':
        for layer in model.model.layers:
            bias = torch.zeros(args.memories, device='cuda')
            bias[:args.topk] = 100
            layer.attn.gate.register_forward_hook(lambda module, inputs, output: output + bias)
    ids = torch.randint(0, config.vocab_size, (args.batch, args.length), device='cuda')
    kwargs = {}
    if args.layout == 'varlen':
        ids = ids.flatten().unsqueeze(0)
        lengths = [args.length - 7, args.length + 7] * (args.batch // 2)
        if args.batch % 2:
            lengths.append(args.length)
        kwargs['cu_seqlens'] = torch.tensor([0, *lengths], device='cuda', dtype=torch.int32).cumsum(0).int()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    context = torch.enable_grad if args.mode == 'train' else torch.no_grad
    cache = None
    if args.mode == 'decode':
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            cache = model(input_ids=ids, use_cache=True, past_key_values=Cache(), num_logits_to_keep=1).past_key_values
        ids = ids[:, :1]
    wall_times, event_times = [], []
    torch.cuda.reset_peak_memory_stats()
    with context():
        for i in range(args.warmup + args.samples):
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(
                    input_ids=ids,
                    labels=ids if args.mode == 'train' else None,
                    use_cache=args.mode in ('prefill', 'decode'),
                    past_key_values=Cache() if args.mode == 'prefill' else cache,
                    num_logits_to_keep=0 if args.mode == 'train' else 1,
                    **kwargs,
                )
            if args.mode == 'train':
                output.loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            end.record()
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000
            if i >= args.warmup:
                wall_times.append(wall_ms)
                event_times.append(start.elapsed_time(end))
            del output
    repo = Path(fla.__file__).resolve().parents[1]
    result = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        wall_ms=statistics.median(wall_times),
        event_ms=statistics.median(event_times),
        wall_samples_ms=wall_times,
        event_samples_ms=event_times,
        peak_gib=torch.cuda.max_memory_allocated() / 2**30,
        parameters=sum(p.numel() for p in model.parameters()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
