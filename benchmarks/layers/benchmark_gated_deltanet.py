# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import argparse

import torch
import triton
from einops import rearrange

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.modules.convolution import causal_conv1d
from fla.utils import device

def _separate_conv(layer, q, k, v):
    q = layer.q_conv1d(q)[0]
    k = layer.k_conv1d(k)[0]
    v = layer.v_conv1d(v)[0]
    return q, k, v


def _fused_conv(layer, q, k, v):
    qkv = torch.cat([q, k, v], dim=-1)
    qkv_weight = torch.cat(
        [
            rearrange(layer.q_conv1d.weight, 'd 1 w -> d w'),
            rearrange(layer.k_conv1d.weight, 'd 1 w -> d w'),
            rearrange(layer.v_conv1d.weight, 'd 1 w -> d w'),
        ],
        dim=0,
    )
    bias = None
    if layer.conv_bias:
        bias = torch.cat([layer.q_conv1d.bias, layer.k_conv1d.bias, layer.v_conv1d.bias], dim=0)
    qkv = causal_conv1d(
        x=qkv,
        weight=qkv_weight,
        bias=bias,
        activation=layer.q_conv1d.activation,
        backend=layer.q_conv1d.backend,
    )[0]
    return torch.split(qkv, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)


def benchmark(args):
    torch.manual_seed(args.seed)

    layer = GatedDeltaNet(
        hidden_size=args.hidden_size,
        expand_v=args.expand_v,
        head_dim=args.head_dim,
        num_heads=args.num_heads,
        num_v_heads=args.num_v_heads,
        mode='chunk',
        use_gate=True,
        use_short_conv=True,
        conv_size=args.conv_size,
    ).to(device=device, dtype=args.dtype).train()

    x = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device=device, dtype=args.dtype)
    q = layer.q_proj(x)
    k = layer.k_proj(x)
    v = layer.v_proj(x)

    providers = args.providers.split(',')
    fns = {}
    if 'full' in providers:
        fns['full'] = lambda: layer(x)[0]
    if 'separate_conv' in providers:
        fns['separate_conv'] = lambda: _separate_conv(layer, q, k, v)
    if 'fused_conv' in providers:
        fns['fused_conv'] = lambda: _fused_conv(layer, q, k, v)

    for fn in fns.values():
        for _ in range(args.warmup_iters):
            fn()
    torch.cuda.synchronize()

    print(
        f"shape B={args.batch_size} T={args.seq_len} hidden={args.hidden_size} "
        f"heads={args.num_heads} head_dim={args.head_dim} expand_v={args.expand_v} dtype={args.dtype}"
    )
    for name, fn in fns.items():
        ms = triton.testing.do_bench(fn, warmup=args.warmup_ms, rep=args.rep_ms, quantiles=[0.5, 0.2, 0.8])
        print(f"{name:>14s}: median={ms[0]:.6f}ms p20={ms[1]:.6f}ms p80={ms[2]:.6f}ms")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the GatedDeltaNet layer and q/k/v short-conv section.")
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--hidden-size', type=int, default=2048)
    parser.add_argument('--head-dim', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=6)
    parser.add_argument('--num-v-heads', type=int, default=None)
    parser.add_argument('--expand-v', type=float, default=2.0)
    parser.add_argument('--conv-size', type=int, default=4)
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16'])
    parser.add_argument('--providers', type=str, default='full,separate_conv,fused_conv')
    parser.add_argument('--warmup-iters', type=int, default=5)
    parser.add_argument('--warmup-ms', type=int, default=25)
    parser.add_argument('--rep-ms', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)
    return args


if __name__ == '__main__':
    benchmark(parse_args())
