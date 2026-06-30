# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
from einops import rearrange

from fla.modules.convolution import causal_conv1d
from fla.ops.utils.index import prepare_sequence_ids

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=["T", "D"],
        # different possible values for `x_name`
        x_vals=[(128 * 2**i, d) for d in [256, 512, 1024, 2048, 4096] for i in range(1, 10)],
        # argument name whose value corresponds to a different line in the plot
        line_arg="provider",
        # possible values for `line_arg``
        line_vals=["causal_conv1d_fwd", "causal_conv1d_cuda_fwd", "causal_conv1d_fwdbwd", "causal_conv1d_cuda_fwdbwd"],
        # label name for the lines
        line_names=["causal_conv1d_fwd", "causal_conv1d_cuda_fwd", "causal_conv1d_fwdbwd", "causal_conv1d_cuda_fwdbwd"],
        # line styles
        styles=[
            ("green", "-"),
            ("blue", "--"),
            ("red", "-."),
            ("cyan", ":"),
            ("yellow", "dotted"),
            ("cyan", "--"),
            ("cyan", "-"),
            ("black", ":"),
        ],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    ),
)
def benchmark(T, D, provider):
    from fla.utils import device

    dtype = torch.bfloat16
    requires_grad = True
    B, N, W = 1, 16, 4
    if T < 2048:
        N = 4

    x = torch.randn(B, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    weight = torch.randn(D, W).to(device)
    bias = torch.randn(D).to(device)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0

    cu_seqlens = (
        torch.cat(
            [
                torch.tensor([0], dtype=torch.long),
                torch.arange(16, T)[torch.randperm(T - 16)[: N - 1]],
                torch.tensor([T], dtype=torch.long),
            ],
            0,
        )
        .to(device)
        .sort()[0]
    )
    if provider == "causal_conv1d_fwd":
        results = triton.testing.do_bench(
            lambda: causal_conv1d(x, weight, bias, activation="swish", cu_seqlens=cu_seqlens)[0],
            quantiles=quantiles,
        )
    elif provider == "causal_conv1d_cuda_fwd":
        results = triton.testing.do_bench(
            lambda: rearrange(
                causal_conv1d_fn(
                    x=rearrange(x, "b t d -> b d t"),
                    weight=weight,
                    bias=bias,
                    activation="swish",
                    seq_idx=prepare_sequence_ids(cu_seqlens).to(torch.int32).unsqueeze(0),
                ),
                "b d t -> b t d",
            ),
            quantiles=quantiles,
        )
    elif provider == "causal_conv1d_fwdbwd":
        results = triton.testing.do_bench(
            lambda: causal_conv1d(x, weight, bias, activation="swish", cu_seqlens=cu_seqlens)[0].backward(x),
            quantiles=quantiles,
        )
    elif provider == "causal_conv1d_cuda_fwdbwd":
        results = triton.testing.do_bench(
            lambda: rearrange(
                causal_conv1d_fn(
                    x=rearrange(x, "b t d -> b d t"),
                    weight=weight,
                    bias=bias,
                    activation="swish",
                    seq_idx=prepare_sequence_ids(cu_seqlens).to(torch.int32).unsqueeze(0),
                ),
                "b d t -> b t d",
            ).backward(x),
            quantiles=quantiles,
        )
    return results


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # Sweep the number of conv streams together with sequence length / per-stream channels.
        x_names=["N", "T", "D"],
        x_vals=[(n, t, d) for n in [2, 3, 4] for t in [2048, 4096] for d in [512, 1024]],
        line_arg="provider",
        line_vals=["separate_fwd", "fused_fwd", "separate_fwdbwd", "fused_fwdbwd"],
        line_names=["separate_fwd", "fused_fwd", "separate_fwdbwd", "fused_fwdbwd"],
        styles=[("green", "-"), ("blue", "--"), ("red", "-."), ("cyan", ":")],
        ylabel="Execution Time (ms)",
        plot_name="GroupedShortConvFusion",
        args={},
    ),
)
def benchmark_grouped_conv_fusion(N, T, D, provider):
    """Compare ``N`` separate grouped short convs vs ``concat -> one causal_conv1d -> split``.

    This is the generic, model-agnostic version of fusing several independent depthwise
    short convs over ``[B, T, D]`` inputs into a single grouped convolution. It is
    parameterized purely by the number of streams ``N``; GatedDeltaNet's q/k/v short
    convs are the ``N = 3`` instance.
    """
    from fla.utils import device

    dtype = torch.bfloat16
    B, W = 1, 4
    quantiles = [0.5, 0.2, 0.8]

    # Per-stream inputs / depthwise weights / biases. Only the inputs are differentiated,
    # so the timed backward matches the work a module does for its conv inputs.
    xs = [torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True) for _ in range(N)]
    weights = [torch.randn(D, W, device=device, dtype=dtype) for _ in range(N)]
    biases = [torch.randn(D, device=device, dtype=dtype) for _ in range(N)]
    sizes = [D] * N

    def separate():
        return [causal_conv1d(x, w, b, activation="swish")[0] for x, w, b in zip(xs, weights, biases)]

    def fused():
        # Depthwise conv is per-channel, so concatenating channels + concatenating the
        # [D, W] weights is exactly equivalent to running the convs independently.
        x = torch.cat(xs, dim=-1)
        weight = torch.cat(weights, dim=0)
        bias = torch.cat(biases, dim=0)
        y = causal_conv1d(x, weight, bias, activation="swish")[0]
        return list(torch.split(y, sizes, dim=-1))

    # Validate the fused path matches the separate path before timing it, so a wrong
    # kernel can never masquerade as a faster one.
    if provider.startswith("fused"):
        with torch.no_grad():
            for ref, got in zip(separate(), fused()):
                torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)

    grads = [torch.randn(B, T, D, device=device, dtype=dtype) for _ in range(N)]

    def _reset_grads():
        for x in xs:
            x.grad = None

    if provider == "separate_fwd":
        fn = separate
    elif provider == "fused_fwd":
        fn = fused
    elif provider == "separate_fwdbwd":

        def fn():
            _reset_grads()
            torch.autograd.backward(separate(), grads)
    elif provider == "fused_fwdbwd":

        def fn():
            _reset_grads()
            torch.autograd.backward(fused(), grads)
    else:
        raise ValueError(provider)

    return triton.testing.do_bench(fn, quantiles=quantiles)


if __name__ == "__main__":
    benchmark.run(print_data=True)
    benchmark_grouped_conv_fusion.run(print_data=True)
