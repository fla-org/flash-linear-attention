
import torch
import triton

from fla.modules.activations import fast_gelu_impl as gelu
from fla.modules.activations import logsigmoid, sigmoid, sqrelu, swiglu, swish
from fla.utils import device

DTYPE = torch.bfloat16


def fwd(fn, *args):
    return fn(*args)


def fwdbwd(fn, *args):
    y = fn(*args)
    g = torch.randn_like(y)
    y.backward(g)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['B', 'T', 'D'],
        x_vals=[
            (b, t, d)
            for b in [4]
            for t in [512, 1024, 2048, 4096, 8192]
            for d in [1024, 2048, 4096]
        ],
        line_arg='provider',
        line_vals=[
            'sigmoid_fwd', 'sigmoid_fwdbwd',
            'logsigmoid_fwd', 'logsigmoid_fwdbwd',
            'swish_fwd', 'swish_fwdbwd',
            'gelu_fwd', 'gelu_fwdbwd',
            'sqrelu_fwd', 'sqrelu_fwdbwd',
            'swiglu_fwd', 'swiglu_fwdbwd',
        ],
        line_names=[
            'sigmoid_fwd', 'sigmoid_fwdbwd',
            'logsigmoid_fwd', 'logsigmoid_fwdbwd',
            'swish_fwd', 'swish_fwdbwd',
            'gelu_fwd', 'gelu_fwdbwd',
            'sqrelu_fwd', 'sqrelu_fwdbwd',
            'swiglu_fwd', 'swiglu_fwdbwd',
        ],
        styles=[('green', '-'), ('green', '--'),
                ('blue', '-'), ('blue', '--'),
                ('red', '-'), ('red', '--'),
                ('cyan', '-'), ('cyan', '--'),
                ('magenta', '-'), ('magenta', '--'),
                ('yellow', '-'), ('yellow', '--')],
        ylabel="Time (ms)",
        plot_name="activation_performance",
        args={},
    ),
)
def benchmark(B, T, D, provider):
    requires_grad = True
    x = torch.randn(B, T, D, device=device, dtype=DTYPE, requires_grad=requires_grad)

    if 'swiglu' in provider:
        y = torch.randn_like(x)
        inputs = (x, y)
    elif 'bias_gelu' in provider:
        bias = torch.randn(D, device=device, dtype=DTYPE, requires_grad=True)
        inputs = (x, bias)
    else:
        inputs = (x,)

    if provider.startswith('sigmoid'):
        fn = sigmoid
    elif provider.startswith('logsigmoid'):
        fn = logsigmoid
    elif provider.startswith('swish'):
        fn = swish
    elif provider.startswith('gelu'):
        fn = gelu
    elif provider.startswith('sqrelu'):
        fn = sqrelu
    elif provider.startswith('swiglu'):
        fn = swiglu
    else:
        raise ValueError(provider)

    if provider.endswith('fwd'):
        fn_to_call = lambda: fwd(fn, *inputs)  # noqa: E731
    elif provider.endswith('fwdbwd'):
        fn_to_call = lambda: fwdbwd(fn, *inputs)  # noqa: E731
    else:
        raise ValueError(provider)

    ms, min_ms, max_ms = triton.testing.do_bench(
        fn_to_call,
        quantiles=[0.5, 0.2, 0.8],
    )
    return ms, min_ms, max_ms


if __name__ == '__main__':
    benchmark.run(print_data=True, save_path='./activation_benchmark')
