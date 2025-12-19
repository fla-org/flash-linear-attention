import os
import random
import argparse
import torch
import torch.distributed as dist
import torch.nn.functional as F

os.environ['TRITON_PRINT_AUTOTUNING'] = '1'

DTYPE = torch.bfloat16
B = 1
H = 64
K = 128
V = 128

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", action="store_true")
    parser.add_argument("--kda", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-path", type=str, default="")
    parser.add_argument("--seqlen", type=int, default=1024*32)
    parser.add_argument("--mean", type=int, default=1024*32)
    parser.add_argument("--std", type=int, default=0)
    return parser.parse_args()

# torchrun --nproc-per-node=4 test_cp.py 
def a2a(x, stage=1, group=None, async_op=False):
    assert stage in [1, 2]
    x = x.contiguous()
    rank = dist.get_rank(group)
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

class _All2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, stages, group, async_op):
        fwd_stage, bwd_stage = stages
        out, handle = a2a(x, fwd_stage, group, async_op=async_op)
        ctx.bwd_stage = bwd_stage
        ctx.group = group
        return out, handle

    @staticmethod
    def backward(ctx, do, *args):
        dx, handle = a2a(do, stage=ctx.bwd_stage, group=ctx.group)
        return dx, None, None, None

def qkvo_all2ll(x, is_qkv=True, group=None, async_op=False):
    if is_qkv:
        stages = [1, 2]
    else:
        stages = [2, 1]
    out, handle = _All2All.apply(x, stages, group, async_op)
    return out, handle

def bench(fn, step=20, warm_up=10, grad_to_none=None):
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
    torch.cuda.synchronize()  # 等待CUDA操作完成
    t1 = start_event.elapsed_time(end_event)
    return t1 / step

def print_rank0(*args, **kwargs):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print(*args, **kwargs)
        torch.distributed.barrier()
    else:
        print(*args, **kwargs)

def compare(x, y, prefix=""):
    if x is None or y is None:
        return
    if any([x.dtype == torch.float32, y.dtype==torch.float32]):
        x,y = x.float(), y.float()
    diff = (x-y).abs()
    # diff = diff / (torch.max(x.abs(), y.abs()) + 1e-6)
    # if prefix:
    #     print(prefix, end=": ")
    print(prefix + f"max_diff: {diff.max().item()}, mean_diff: {diff.mean().item()}, absmax: {torch.maximum(x.abs().max(), y.abs().max()).item()}")

def get_ref_grad(*tensors):
    grads = []
    for t in tensors:
        grads.append(t.grad)
        t.grad = None
    if len(grads) == 1:
        grads = grads[0]
    return grads

def broadcast(x, group=None):
    dist.broadcast(x, src=0, group=group)

def all_gather(x, group=None) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    y = torch.empty(world_size * x.size(0), *x.shape[1:], device=x.device, dtype=x.dtype)
    dist.all_gather_into_tensor(y, x, group=group)
    return y

def generate_cu_seqlens(end=8192, mean=2048, var=512):
    random.seed(42)
    r = [0]
    while r[-1] < end:
        a = random.randint(max(mean-var, 1), mean+var)
        r.append(r[-1] + a)
    r[-1] = end
    cu_seqlens = torch.tensor(r, device=torch.cuda.current_device(), dtype=torch.int32)
    return cu_seqlens

def profile_func(fn, path):

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,  # 如果是GPU
        ],
            schedule=torch.profiler.schedule(wait=0, warmup=2, active=8),
            with_stack=False,
            record_shapes=True,
            profile_memory=False,
            # on_trace_ready=hook_fn,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(path)

    ) as prof:
        for step in range(10):  # 总步数需覆盖schedule的wait+warmup+active
            torch.cuda.synchronize()
            fn()
            torch.cuda.synchronize()
            prof.step()  # 通知profiler一个步骤完成
    return prof

def test_ops(args):
    from fla.ops.kda import chunk_kda
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.common.cp_chunk_delta_h import set_gdn_cp_context, get_gdn_cp_context

    device = torch.cuda.current_device()
    group = args.group
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    cu_seqlens = generate_cu_seqlens(args.seqlen, args.mean, args.std)
    T = args.seqlen // world_size

    q = torch.randn(B, T, H, K, dtype=DTYPE, device=device).requires_grad_(True)
    k = F.normalize(torch.randn(B, T, H, K, dtype=DTYPE, device=device), p=2, dim=-1).requires_grad_(True)
    v = torch.randn(B, T, H, V, dtype=DTYPE, device=device).requires_grad_(True)
    beta = torch.rand(B, T, H, dtype=DTYPE, device=device).sigmoid().requires_grad_(True)
    if not args.kda:
        # if / 10000., the max diff is not 0
        g = (F.logsigmoid(torch.rand(B, T, H, dtype=DTYPE, device=device)) / 1.).requires_grad_(True)
    else:
        g = (F.logsigmoid(torch.rand(B, T, H, K, dtype=DTYPE, device=device)) / 1.).requires_grad_(True)
    do = torch.randn(B, T, H, V, dtype=DTYPE, device=device)

    total_q = all_gather(q.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)
    total_k = all_gather(k.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)
    total_v = all_gather(v.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)
    total_beta = all_gather(beta.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)
    total_g = all_gather(g.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)
    total_do = all_gather(do.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)

    backward = args.backward
    op = chunk_gated_delta_rule if not args.kda else chunk_kda

    def gdn_no_cp():
        set_gdn_cp_context()
        total_out, total_ht = op(
        total_q, total_k, total_v, total_g, total_beta,
        cu_seqlens=cu_seqlens
        )
        dist.barrier()
        if backward:
            total_out.backward(total_do)
        dist.barrier()
        return total_out.chunk(world_size, 1)[rank]

    def gdn_with_custom_cp():
        set_gdn_cp_context(cu_seqlens, group)
        o, ht = op(
        q, k, v, g, beta,
        cu_seqlens=get_gdn_cp_context().cu_seqlens
        )
        dist.barrier()
        if backward:
            o.backward(do)
            dist.barrier()
        return o

    def gdn_with_a2a():
        set_gdn_cp_context()
        q2, k2, v2 = [qkvo_all2ll(t.squeeze(0), is_qkv=True, group=group)[0].unsqueeze(0) for t in [q, k, v]]
        if not args.kda:
            g2 = qkvo_all2ll(g.squeeze(0).unsqueeze(-1), is_qkv=True, group=group)[0].unsqueeze(0).squeeze(-1)
        else:
            g2 = qkvo_all2ll(g.squeeze(0), is_qkv=True, group=group)[0].unsqueeze(0)
        beta2 = qkvo_all2ll(beta.squeeze(0).unsqueeze(-1), is_qkv=True, group=group)[0].unsqueeze(0).squeeze(-1)
        o, ht = op(
        q2, k2, v2, g2, beta2,
        cu_seqlens=cu_seqlens
        )
        o = qkvo_all2ll(o.squeeze(0), is_qkv=False, group=group)[0].unsqueeze(0)
        dist.barrier()
        if backward:
            o.backward(do)
            dist.barrier()
        return o

    o = gdn_with_custom_cp()
    # o = gdn_no_cp()
    dq, dk, dv, dg, dbeta = get_ref_grad(q, k, v, g, beta)
    ref_o = gdn_with_a2a()
    ref_dq, ref_dk, ref_dv, ref_dg, ref_dbeta = get_ref_grad(q, k, v, g, beta)

    dist.barrier()
    compare(o, ref_o, f"rank:{rank}, out:")
    dist.barrier()
    if backward:
        compare(dq, ref_dq, f"rank:{rank}, dq:")
        dist.barrier()
        compare(dk, ref_dk, f"rank:{rank}, dk:")
        dist.barrier()
        compare(dv, ref_dv, f"rank:{rank}, dv:")
        dist.barrier()
        compare(dg, ref_dg, f"rank:{rank}, dg:")
        dist.barrier()
        compare(dbeta, ref_dbeta, f"rank:{rank}, dbeta:")
        dist.barrier()

    if args.bench:
        dist.barrier()
        t1 = bench(gdn_no_cp)
        dist.barrier()
        t2 = bench(gdn_with_custom_cp)
        dist.barrier()
        t3 = bench(gdn_with_a2a)
        if rank == world_size - 1:
            print(f"custom cp, non-cp time: {t1:.3f} ms, cp time: {t2:.3f} ms, rate: {((t1/world_size) / t2 * 100):.2f} %")
            print(f"all to all, non-cp time: {t1:.3f} ms, cp time: {t3:.3f} ms, rate: {((t1/world_size) / t3 * 100):.2f} %")

    if args.profile:
        def fn():
            gdn_no_cp()
            dist.barrier()
            gdn_with_custom_cp()
            dist.barrier()
            gdn_with_a2a()
            dist.barrier()
        profile_func(fn, args.profile_path)

def test_layer(args):
    from fla.layers.kda import KimiDeltaAttention, KimiDeltaAttentionWithCP
    from fla.layers.gated_deltanet import GatedDeltaNet, GatedDeltaNetWithCP
    from fla.ops.common.cp_chunk_delta_h import set_gdn_cp_context, get_gdn_cp_context

    device = torch.cuda.current_device()
    group = args.group
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    cu_seqlens = generate_cu_seqlens(args.seqlen, args.mean, args.std)
    T = args.seqlen // world_size
    hidden_size = 2304
    n_head = 32
    head_dim = 128
    use_conv = True

    x = torch.randn(B, T, hidden_size, dtype=DTYPE, device=device).requires_grad_(True)
    do = torch.randn(B, T, hidden_size, dtype=DTYPE, device=device)
    total_x = all_gather(x.squeeze(0), group=group).unsqueeze(0).detach().requires_grad_(True)
    total_do = all_gather(do.squeeze(0), group=group).unsqueeze(0)
    if not args.kda:
        layer = GatedDeltaNet(hidden_size, expand_v=1, head_dim=head_dim, num_heads=n_head, use_short_conv=use_conv)
        cp_layer = GatedDeltaNetWithCP(hidden_size, expand_v=1, head_dim=head_dim, num_heads=n_head, use_short_conv=use_conv, group=group)
    else:
        layer = KimiDeltaAttention(hidden_size, expand_v=1, head_dim=head_dim, num_heads=n_head, use_short_conv=use_conv)
        cp_layer = KimiDeltaAttentionWithCP(hidden_size, expand_v=1, head_dim=head_dim, num_heads=n_head, use_short_conv=use_conv, group=group)
    layer = layer.to(x)
    cp_layer = cp_layer.to(x)
    for p in layer.parameters():
        broadcast(p, group)
    cp_layer.load_state_dict(layer.state_dict())

    backward = args.backward
    def gdn_no_cp():
        set_gdn_cp_context()
        total_o = layer(total_x, cu_seqlens=cu_seqlens)[0]
        dist.barrier()
        if backward:
            total_o.backward(total_do)
        dist.barrier()
        return total_o.chunk(world_size, 1)[rank]

    def gdn_with_custom_cp():
        o = cp_layer(x, cu_seqlens=cu_seqlens)[0]
        dist.barrier()
        if backward:
            o.backward(do)
        dist.barrier()
        return o

    ref_o = gdn_no_cp()
    if backward:
        ref_dx = total_x.grad.chunk(world_size, 1)[rank]
    else:
        ref_dx = None
    o = gdn_with_custom_cp()
    dx = x.grad

    dist.barrier()
    compare(o, ref_o, f"rank:{rank}, out:")
    if backward:
        compare(ref_dx, dx, f"rank:{rank}, input grad:")
        dist.barrier()
        for (name1, p1), (name2, p2) in zip(layer.named_parameters(), cp_layer.named_parameters()):
            assert name1 == name2
            grad1 = p1.grad.float()
            grad2 = p2.grad.float()
            torch.distributed.all_reduce(grad2, group=group)
            compare(grad1, grad2, f"rank:{rank}, {name1} grad:")
            dist.barrier()

    if args.bench:
        dist.barrier()
        t1 = bench(gdn_no_cp)
        dist.barrier()
        t2 = bench(gdn_with_custom_cp)
        dist.barrier()
        if rank == world_size - 1:
            print(f"custom cp, non-cp time: {t1:.3f} ms, cp time: {t2:.3f} ms, rate: {((t1/world_size) / t2 * 100):.2f} %")

    if args.profile:
        def fn():
            gdn_no_cp()
            dist.barrier()
            gdn_with_custom_cp()
            dist.barrier()
        profile_func(fn, args.profile_path)

def main():
    dist.init_process_group()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    group = dist.new_group(list(range(world_size)))
    args = get_args()
    args.group = group
    if args.ops:
        test_ops(args)
    else:
        test_layer(args)

if __name__ == '__main__':
    main()
'''
torchrun --nproc-per-node=4 test_gdn_with_cp.py
'''


