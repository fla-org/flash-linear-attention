# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Context-parallel layer comparison shared by GDN and KDA."""

from functools import partial

import torch
import torch.distributed as dist

from benchmarks.utils.distributed import all_gather, generate_cu_seqlens, measure_latency, print_tensor_error, profile_function
from benchmarks.utils.reporting import BenchmarkResult

try:
    import fused_weight_gradient_mlp_cuda
    APEX_WEIGHT_GRADIENT = None
    if hasattr(fused_weight_gradient_mlp_cuda, "wgrad_gemm_accum_fp32"):
        APEX_WEIGHT_GRADIENT = fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32
    if hasattr(fused_weight_gradient_mlp_cuda, "fused_wgrad_gemm_accum_fp32"):
        APEX_WEIGHT_GRADIENT = fused_weight_gradient_mlp_cuda.fused_wgrad_gemm_accum_fp32
    if APEX_WEIGHT_GRADIENT is None:
        HAVE_APEX = False
    else:
        HAVE_APEX = True
except Exception:
    HAVE_APEX = False

DTYPE = torch.bfloat16
B = 1


class _LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias):
        inp_shape = inp.shape
        out = torch.matmul(inp.view(-1, inp_shape[-1]), weight.t())
        if bias is not None:
            out += bias
        ctx.save_for_backward(inp, weight, bias)
        return out.view(*inp_shape[:-1], -1)

    @staticmethod
    def backward(ctx, dout):
        inp, weight, bias = ctx.saved_tensors
        inp_shape = inp.shape
        inp = inp.view(-1, inp_shape[-1])
        dout = dout.view(-1, dout.size(-1))
        dgrad = torch.matmul(dout, weight).view(inp_shape)
        main_grad = getattr(weight, "main_grad", None)
        if main_grad is not None:
            if HAVE_APEX:
                wgrad = APEX_WEIGHT_GRADIENT(inp, dout, main_grad)
            else:
                wgrad = torch.matmul(dout.t(), inp)
                main_grad += wgrad
            if hasattr(weight, 'grad_added_to_main_grad'):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                wgrad = torch.empty(
                    main_grad.shape,
                    dtype=inp.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
                weight.grad_added_to_main_grad = True
            else:
                wgrad = None
        else:
            wgrad = torch.matmul(dout.t(), inp)
        if bias is not None:
            dbias = dout.sum(0)
        else:
            dbias = None
        return dgrad, wgrad, dbias


def fp32_grad_linear_forward(input, self):
    out = _LinearFunction.apply(input, self.weight, self.bias)
    return out


def patch_fp32_grad_linear_forward(model: torch.nn.Module):
    if HAVE_APEX:
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                m.forward = partial(fp32_grad_linear_forward, self=m)
                m.weight.main_grad = torch.zeros_like(m.weight, dtype=torch.float32)


def benchmark_layer(args):
    from benchmarks.layers.gated_deltanet import gdn_forward
    from benchmarks.layers.kda import kda_forward
    from benchmarks.modules.conv import short_conv_forward
    from fla.layers.gated_deltanet import GatedDeltaNet
    from fla.layers.kda import KimiDeltaAttention

    results = []

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
        cp_layer = GatedDeltaNet(hidden_size, expand_v=1, head_dim=head_dim,
                                 num_heads=n_head, use_short_conv=use_conv)
        if args.use_cp2hp:
            cp_layer.forward = partial(gdn_forward, self=cp_layer, cp_size=world_size, cp_rank=rank, cp_group=group)
    else:
        layer = KimiDeltaAttention(hidden_size, expand_v=1, head_dim=head_dim, num_heads=n_head, use_short_conv=use_conv)
        cp_layer = KimiDeltaAttention(hidden_size, expand_v=1, head_dim=head_dim,
                                      num_heads=n_head, use_short_conv=use_conv)
        if args.use_cp2hp:
            cp_layer.forward = partial(kda_forward, self=cp_layer, cp_size=world_size, cp_rank=rank, cp_group=group)
    if args.use_cp2hp and use_conv:
        cp_layer.q_conv1d.forward = partial(short_conv_forward, self=cp_layer.q_conv1d, cp_size=world_size, cp_rank=rank)
        cp_layer.k_conv1d.forward = partial(short_conv_forward, self=cp_layer.k_conv1d, cp_size=world_size, cp_rank=rank)
        cp_layer.v_conv1d.forward = partial(short_conv_forward, self=cp_layer.v_conv1d, cp_size=world_size, cp_rank=rank)

    layer = layer.to(x)
    cp_layer = cp_layer.to(x)
    for p in layer.parameters():
        dist.broadcast(p, src=0, group=group)
    cp_layer.load_state_dict(layer.state_dict())

    patch_fp32_grad_linear_forward(layer)
    patch_fp32_grad_linear_forward(cp_layer)
    layer.dt_bias.to(torch.float)
    layer.A_log.to(torch.float)
    cp_layer.dt_bias.to(torch.float)
    cp_layer.A_log.to(torch.float)

    backward = args.backward

    def single_device():
        total_o = layer(total_x, cu_seqlens=cu_seqlens)[0]
        dist.barrier()
        if backward:
            total_o.backward(total_do)
        dist.barrier()
        return total_o.chunk(world_size, 1)[rank]

    def context_parallel():
        o = cp_layer(x, cu_seqlens=cu_seqlens)[0]
        dist.barrier()
        if backward:
            o.backward(do)
        dist.barrier()
        return o

    ref_o = single_device()
    if backward:
        ref_dx = total_x.grad.chunk(world_size, 1)[rank]
    else:
        ref_dx = None
    o = context_parallel()
    dx = x.grad

    if rank == 0:
        print(cu_seqlens)
    dist.barrier()
    print_tensor_error(o, ref_o, f"rank:{rank}, out:")
    if backward:
        print_tensor_error(ref_dx, dx, f"rank:{rank}, input grad:")
        dist.barrier()
        for (name1, p1), (name2, p2) in zip(layer.named_parameters(), cp_layer.named_parameters()):
            assert name1 == name2
            if HAVE_APEX and hasattr(p1, "main_grad"):
                grad1 = p1.main_grad
                grad2 = p2.main_grad
            else:
                grad1 = p1.grad.float()
                grad2 = p2.grad.float()
            torch.distributed.all_reduce(grad2, group=group)
            print_tensor_error(grad1, grad2, f"rank:{rank}, {name1} grad:")
            dist.barrier()

    dist.barrier()
    t1 = measure_latency(single_device)
    dist.barrier()
    t2 = measure_latency(context_parallel)
    dist.barrier()
    for provider, elapsed in (('non_cp', t1), ('custom_cp', t2)):
        results.append(BenchmarkResult(
            name='kda' if args.kda else 'gdn',
            parameters={'scope': 'layer', 'provider': provider, 'B': B, 'T': args.seqlen,
                        'hidden_size': hidden_size, 'world_size': world_size, 'backward': backward},
            metrics={'mean_ms': elapsed},
            primary_metric='mean_ms',
        ))

    if args.profile:
        def fn():
            single_device()
            dist.barrier()
            context_parallel()
            dist.barrier()
        profile_function(fn, args.profile_path)
    return results
