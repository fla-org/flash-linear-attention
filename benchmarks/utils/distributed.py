# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Process-group lifecycle and tensor gathering for distributed benchmarks."""

import os
import random
from contextlib import contextmanager

import torch
import torch.distributed as dist


@contextmanager
def distributed_context(enabled: bool = True, backend: str | None = None):
    if not enabled:
        yield
        return
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group(backend=backend)
    try:
        yield
    finally:
        dist.destroy_process_group()


def all_gather(x, group=None) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    y = torch.empty(world_size * x.size(0), *x.shape[1:], device=x.device, dtype=x.dtype)
    dist.all_gather_into_tensor(y, x, group=group)
    return y


def _all_to_all(x, stage=1, group=None, async_op=False):
    assert stage in [1, 2]
    x = x.contiguous()
    world_size = dist.get_world_size(group)
    t, h, d = x.shape
    if stage == 1:
        x = x.reshape(t, world_size, h//world_size, d).transpose(0, 1).contiguous()
    else:
        x = x.reshape(world_size, t//world_size, h, d).contiguous()
    out = torch.empty_like(x)
    handle = dist.all_to_all_single(out, x, group=group, async_op=async_op)
    if stage == 1:
        out = out.flatten(0, 1)
    else:
        out = out.transpose(0, 1).contiguous().flatten(1, 2)
    return out, handle


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, stages, group, async_op):
        fwd_stage, bwd_stage = stages
        out, handle = _all_to_all(x, fwd_stage, group, async_op=async_op)
        ctx.bwd_stage = bwd_stage
        ctx.group = group
        return out, handle

    @staticmethod
    def backward(ctx, do, *args):
        dx, handle = _all_to_all(do, stage=ctx.bwd_stage, group=ctx.group)
        return dx, None, None, None


def qkvo_all_to_all(x, is_qkv=True, group=None, async_op=False):
    if is_qkv:
        stages = [1, 2]
    else:
        stages = [2, 1]
    out, handle = _AllToAll.apply(x, stages, group, async_op)
    return out, handle


def measure_latency(fn, step=20, warm_up=10, grad_to_none=None):
    # triton.testing.do_bench have bug if there are some comms in kernels
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(warm_up):
        fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
    start_event.record()
    for i in range(step):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    t1 = start_event.elapsed_time(end_event)
    return t1 / step


def print_tensor_error(x, y, prefix=""):
    if x is None or y is None:
        return
    if any([x.dtype == torch.float32, y.dtype == torch.float32]):
        x, y = x.float(), y.float()
    diff = (x-y).abs()

    def print_synchronized(*args, **kwargs):
        dist.barrier()

        for r in range(dist.get_world_size()):
            if r == dist.get_rank():
                print(*args, **kwargs)
            dist.barrier()
    print_synchronized(
        prefix + f"max_diff: {diff.max().item()}, mean_diff: {diff.mean().item()}, absmax: {torch.maximum(x.abs().max(), y.abs().max()).item()}")


def generate_cu_seqlens(end=8192, mean=2048, var=512):
    random.seed(42)
    r = [0]
    while r[-1] < end:
        a = random.randint(max(mean-var, 1), mean+var)
        r.append(r[-1] + a)
    r[-1] = end
    cu_seqlens = torch.tensor(r, device=torch.cuda.current_device(), dtype=torch.int32)
    return cu_seqlens


def maybe_wait(handle):
    if handle is None:
        return
    handle.wait()


def profile_function(fn, path):

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
            schedule=torch.profiler.schedule(wait=0, warmup=2, active=8),
            with_stack=False,
            record_shapes=True,
            profile_memory=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(path)

    ) as prof:
        for step in range(10):
            torch.cuda.synchronize()
            fn()
            torch.cuda.synchronize()
            prof.step()
    return prof


def add_cp_arguments(parser):
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-path', default='')
    parser.add_argument('--seqlen', type=int, default=32768)
    parser.add_argument('--mean', type=int, default=32768)
    parser.add_argument('--std', type=int, default=0)


def run_distributed(args, measure):
    os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
    os.environ['FLA_CONV_BACKEND'] = 'cuda'
    with distributed_context():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        torch.manual_seed(rank + 42)
        torch.cuda.manual_seed(rank + 42)
        random.seed(42)
        args.group = dist.new_group(list(range(world_size)))
        return measure(args)
