# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import time

import torch
from torch.cuda import max_memory_allocated, memory_allocated
from torch.optim import AdamW
from torch.profiler import ProfilerActivity, record_function
from torch.profiler import profile as torch_profile
from tqdm import trange
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

import fla
from benchmarks.utils.reporting import BenchmarkResult

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
    device: torch.device,
):
    if varlen:
        tokens = torch.randint(high=vocab_size, size=(1, batch_size * seq_len), device=device)
        cu_seqlens = torch.cat([
            torch.tensor([0]),
            torch.randperm(batch_size * seq_len - 16)[:torch.randint(8, 64, size=(1,))] + 16,
            torch.tensor([batch_size * seq_len]),
        ], 0).sort()[0].to(dtype=torch.int32, device=device)
        if context_len is not None:
            cu_seqlens = torch.cat(
                [torch.arange(i, j, context_len) for i, j in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())] +
                [torch.tensor([len(tokens[0])])],
            ).to(dtype=torch.int32, device=device)
    else:
        tokens = torch.randint(high=vocab_size, size=(batch_size, seq_len), device=device)
        cu_seqlens = None
    return tokens, cu_seqlens


def benchmark_training(
    name: str,
    batch_size: int = 4,
    seq_len: int = 4096,
    context_len: int = 4096,
    varlen: bool = False,
    num_heads: int | None = None,
    head_dim: int | None = None,
    num_hidden_layers: int | None = None,
    warmup_steps: int = 16,
    steps: int = 32,
    total_steps: int = 1024,
    lr: float = 3e-4,
    betas: tuple[float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    dtype: torch.dtype | None = torch.bfloat16,
    mixed_precision: str = 'bf16',
    compile: bool = False,
    enable_profile: bool = False,
    profile_steps: int = 64,
    profile_trace: str | None = None,
):
    from accelerate import Accelerator
    from transformers.optimization import get_cosine_schedule_with_warmup

    device = torch.device('cuda')
    config = configs[name] if name in configs else AutoConfig.from_pretrained(name)
    if num_heads is not None:
        if not hasattr(config, 'num_heads'):
            raise ValueError(
                f"`--num_heads` override is not supported for model '{name}': "
                f"its config ({type(config).__name__}) has no `num_heads` field."
            )
        config.num_heads = num_heads
    if head_dim is not None:
        if not hasattr(config, 'num_heads'):
            raise ValueError(
                f"`--head_dim` override requires `config.num_heads` to derive `hidden_size`, "
                f"but model '{name}' ({type(config).__name__}) has no `num_heads` field."
            )
        config.head_dim = head_dim
        config.hidden_size = config.num_heads * config.head_dim
    if hasattr(config, 'num_heads') and not hasattr(config, 'head_dim'):
        config.head_dim = config.hidden_size // config.num_heads
    elif hasattr(config, 'head_dim') and not hasattr(config, 'num_heads'):
        config.num_heads = config.hidden_size // config.head_dim
    if num_hidden_layers is not None:
        config.num_hidden_layers = num_hidden_layers

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
        fused=True,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, total_steps)

    # Warmup absorbs the one-time Triton autotune + torch.compile cost so the timed loop is steady-state.
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
            device=device,
        )
        outputs = model(tokens, labels=tokens, cu_seqlens=cu_seqlens)
        # backward pass
        accelerator.backward(outputs.loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        bar.set_description_str(f"Max memory allocated: {sizeof_fmt(max_memory_allocated(device))}")

    bar = trange(steps)
    torch.cuda.synchronize(device)
    start, total_tokens = time.time(), 0
    for _ in bar:
        # forward pass
        tokens, cu_seqlens = prepare_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            context_len=context_len,
            varlen=varlen,
            vocab_size=config.vocab_size,
            device=device,
        )
        outputs = model(tokens, labels=tokens, cu_seqlens=cu_seqlens)
        # backward pass
        accelerator.backward(outputs.loss)
        optimizer.step()
        optimizer.zero_grad()

        total_tokens += batch_size * seq_len
        torch.cuda.synchronize(device)
        duration = time.time() - start
        bar.set_description_str(f"Throughput: {total_tokens / duration:10.2f} tokens/s")

    bar.close()
    duration = time.time() - start
    result = BenchmarkResult(
        name=name,
        parameters={
            'B': batch_size, 'T': seq_len, 'context_len': context_len, 'varlen': varlen,
            'num_heads': getattr(config, 'num_heads', None), 'head_dim': getattr(config, 'head_dim', None),
            'num_hidden_layers': config.num_hidden_layers, 'dtype': str(dtype), 'compile': compile,
            'warmup_steps': warmup_steps, 'steps': steps,
        },
        metrics={
            'tokens_per_second': total_tokens / duration,
            'step_ms': duration / steps * 1000,
            'elapsed_s': duration,
            'peak_memory_bytes': max_memory_allocated(device),
        },
        primary_metric='tokens_per_second',
        higher_is_better=True,
    )

    if enable_profile:
        print(f"\nRunning torch.profiler over {profile_steps} steps "
              "(measurement loop above is unaffected; profiler adds overhead).")
        torch.cuda.synchronize(device)
        with torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for _ in range(profile_steps):
                with record_function("step"):
                    tokens, cu_seqlens = prepare_inputs(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        context_len=context_len,
                        varlen=varlen,
                        vocab_size=config.vocab_size,
                        device=device,
                    )
                    outputs = model(tokens, labels=tokens, cu_seqlens=cu_seqlens)
                    accelerator.backward(outputs.loss)
                    optimizer.step()
                    optimizer.zero_grad()
            torch.cuda.synchronize(device)

        print("=" * 100)
        print("Top by CUDA time (where the GPU is spending time)")
        print("=" * 100)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

        print("=" * 100)
        print("Top by CPU time (host stalls — H2D / launch overhead lives here)")
        print("=" * 100)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))

        if profile_trace:
            prof.export_chrome_trace(profile_trace)
            print(f"\nchrome trace written to {profile_trace}")
            print("  inspect via chrome://tracing or https://ui.perfetto.dev")

    return result


def _add_training_arguments(parser):
    parser.add_argument("--name", default='retnet')
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--seq_len", default=4096, type=int)
    parser.add_argument("--context_len", default=None, type=int)
    parser.add_argument("--varlen", action='store_true')
    parser.add_argument("--num_heads", default=None, type=int)
    parser.add_argument("--head_dim", default=None, type=int)
    parser.add_argument("--num_hidden_layers", default=None, type=int)
    parser.add_argument("--warmup_steps", default=64, type=int)
    parser.add_argument("--steps", default=256, type=int)
    parser.add_argument("--compile", action='store_true')
    parser.add_argument("--profile", action='store_true',
                        help="run torch.profiler over a few steps after the throughput loop")


def _run_training(args):
    return [benchmark_training(
        name=args.name,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        context_len=args.context_len,
        varlen=args.varlen,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_hidden_layers=args.num_hidden_layers,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        compile=args.compile,
        enable_profile=args.profile,
    )]


def _add_generation_arguments(parser):
    parser.add_argument("--path", type=str, default="fla-hub/transformer-1.3B-100B")
    parser.add_argument("--data", type=str, default="fla-hub/pg19")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--maxlen", type=int, default=256)
    parser.add_argument("--no-cache", action='store_true')
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--topp", type=float, default=0.2)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--output-generation", action='store_true')
    parser.add_argument("--compile", action='store_true')


def _run_generation(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    print(f"Loading {args.path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.path,
        trust_remote_code=True,
        add_eos_token=False,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"{tokenizer}")

    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        device_map={"": device},
        torch_dtype=dtype,
        use_cache=not args.no_cache,
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
            repetition_penalty=args.repetition_penalty,
        )
    torch.cuda.synchronize()
    elapsed = time.time() - start
    if args.output_generation:
        print(f"Prompt:\n{tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0].strip()}\n")
        print(f"Generated:\n{tokenizer.batch_decode(text, skip_special_tokens=True)[0].strip()}\n")
    print(f"Prompt length: {len(input_ids[0])}, generation length: {len(text[0]) - len(input_ids[0])}")
    return [BenchmarkResult(
        name=args.path,
        parameters={
            'data': args.data, 'dtype': str(dtype),
            'prompt_tokens': input_ids.shape[1], 'max_new_tokens': args.maxlen, 'compile': args.compile,
            'cache': not args.no_cache, 'temperature': args.temperature, 'top_p': args.topp,
            'repetition_penalty': args.repetition_penalty,
        },
        metrics={'elapsed_ms': elapsed * 1000, 'peak_memory_bytes': torch.cuda.max_memory_allocated()},
        primary_metric='elapsed_ms',
    )]


SCENARIOS = ('training', 'generation')


def add_arguments(parser, scenario):
    if scenario == 'training':
        _add_training_arguments(parser)
    else:
        _add_generation_arguments(parser)


def run(args):
    return _run_training(args) if args.scenario == 'training' else _run_generation(args)
