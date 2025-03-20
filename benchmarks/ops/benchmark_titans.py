# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"

import torch
from benchmark import benchmark_backward, benchmark_combined, benchmark_forward
from torch.nn import functional as F

from fla.ops.titans.naive import chunk_titans_linear_ref

# from flash_attn import flash_attn_func


def time_fwd(func, *args, **kwargs):
    time_fb = benchmark_forward(func, *args, **kwargs)
    return time_fb[1].mean


def time_fwd_bwd(func, *args, **kwargs):
    time_fb = benchmark_combined(func, *args, **kwargs)
    return time_fb[1].mean


def time_bwd(func, *args, **kwargs):
    time_fb = benchmark_backward(func, *args, **kwargs)
    return time_fb[1].mean


repeats = 256
device = "cuda"
dtype = torch.bfloat16

bs_seqlen_vals = [(2, 1024), (2, 2048)]
causal_vals = [True]
headdim_vals = [4, 8]
dim = 16
dropout_p = 0.0

methods = ["naive_titans", "chunk_titans"]
time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for causal in causal_vals:
    for headdim in headdim_vals:
        for B, seqlen in bs_seqlen_vals:
            config = (causal, headdim, B, seqlen)
            H = dim // headdim

            q = torch.randn(
                B, H, seqlen, headdim, device=device, requires_grad=True, dtype=dtype
            )
            k = F.normalize(
                torch.randn(B, H, seqlen, headdim, device=device, dtype=dtype),
                p=2,
                dim=-1,
            ).requires_grad_(True)
            v = torch.randn(
                B, H, seqlen, headdim, device=device, requires_grad=True, dtype=dtype
            )
            w = torch.randn(H, headdim, device=device, requires_grad=True, dtype=dtype)
            b = torch.randn(H, headdim, device=device, requires_grad=True, dtype=dtype)
            theta = torch.rand(
                B, H, seqlen, 1, dtype=dtype, device=device, requires_grad=True
            )
            alpha = torch.rand(
                B, H, seqlen, 1, dtype=dtype, device=device, requires_grad=True
            )
            eta = torch.rand(
                B, H, seqlen, 1, dtype=dtype, device=device, requires_grad=True
            )
            o2, _ = chunk_titans_linear_ref(
                q, k, v, w, b, theta, alpha, eta, chunk_size=16, use_chunk=False
            )
            o2.sum().backward(retain_graph=True)
            f_b = time_fwd_bwd(
                chunk_titans_linear_ref,
                q,
                k,
                v,
                w,
                b,
                theta,
                alpha,
                eta,
                use_chunk=False,
                verbose=False,
            )
            time_f_b[config, "naive_titans"] = f_b

            o3, _ = chunk_titans_linear_ref(
                q, k, v, w, b, theta, alpha, eta, chunk_size=16, use_chunk=True
            )
            o3.sum().backward(retain_graph=True)
            f_b = time_fwd_bwd(
                chunk_titans_linear_ref,
                q,
                k,
                v,
                w,
                b,
                theta,
                alpha,
                eta,
                chunk_size=16,
                use_chunk=True,
                verbose=False,
            )
            time_f_b[config, "chunk_titans"] = f_b

            print(f"### causal={causal}, headdim={headdim}, B={B}, seqlen={seqlen} ###")
            for method in methods:
                # time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                print(
                    f"{method:>50} fwd + bwd:\t {time_f_b[config, method] * 1000:>6.4f} ms "
                )

                # speed_f[config, method] = efficiency(
                #     flops(B, seqlen, headdim, H, causal, mode="fwd"),
                #     time_f[config, method]
                # )
                # speed_b[config, method] = efficiency(
                #     flops(B, seqlen, headdim, H, causal, mode="bwd"),
                #     time_b[config, method]
                # )
                # speed_f_b[config, method] = efficiency(
                #     flops(B, seqlen, headdim, H, causal, mode="fwd_bwd"),
                #     time_f_b[config, method]
                # )
                # print(
                #     f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
                #     f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, "
                #     f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s"
                # )

# with open('flash2_attn_time.plk', 'wb') as fp:
#     pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
