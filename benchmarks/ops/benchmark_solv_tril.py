
import torch
import triton


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['B', 'T', 'H', 'chunk_size'],
        x_vals=[
            (b, t, h, c)
            for b in [8]
            for t in [2048, 4096, 8192]
            for h in [16, 64]
            for c in [16, 32, 64]
        ],
        line_arg='provider',
        line_vals=[
            'solve_tril_tma',
        ],
        line_names=[
            'solve_tril_tma',
        ],
        styles=[('green', '-'), ('green', '--')],
        ylabel="Time (ms)",
        plot_name="solve_tril_performance",
        args={},
    ),
)
def benchmark(B, T, H, chunk_size, provider):
    from fla.ops.utils.solve_tril import solve_tril
    from fla.utils import device

    requires_grad = True
    dtype = torch.float32

    k = torch.randn((B, H, T, 64), dtype=dtype, device=device, requires_grad=requires_grad)
    k = torch.nn.functional.normalize(k, dim=-1)

    padding_size = (chunk_size - T % chunk_size) % chunk_size
    T_padded = T + padding_size
    k_padded = torch.nn.functional.pad(k, (0, 0, 0, padding_size, 0, 0, 0, 0))
    k_padded = k_padded.reshape(B, H, T_padded // chunk_size, chunk_size, 64)

    A = (k_padded @ k_padded.transpose(-1, -2)).tril(-1)
    A = A.permute(0, 2, 1, 3, 4).contiguous()
    A = A.view(B, T_padded, H, chunk_size)
    A = A[:, :T, :, :]

    results = triton.testing.do_bench(
        lambda:  solve_tril(A),
        quantiles=[0.5, 0.2, 0.8],
    )
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True, save_path=".")
