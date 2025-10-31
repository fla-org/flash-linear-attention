# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"

import torch
from benchmark import benchmark_backward, benchmark_combined, benchmark_forward
from torch.nn import functional as F

from fla.ops.delta_rule import chunk_delta_rule
from fla.utils import device


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
dtype = torch.bfloat16


bs_seqlen_vals = [(8, 2048), (4, 4096), (2, 8192)]
causal_vals = [True]
headdim_vals = [64, 128, 256]
dim = 2048
dropout_p = 0.0


methods = (["chunk_delta_rule"])
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
            q = torch.randn(B, seqlen, H, headdim, device=device, requires_grad=True, dtype=dtype)
            k = F.normalize(torch.randn(B, seqlen, H, headdim, device=device, dtype=dtype), p=2, dim=-1).requires_grad_(True)
            v = torch.randn(B, seqlen, H, headdim, device=device, requires_grad=True, dtype=dtype)
            beta = torch.rand(B, seqlen, H, device=device, dtype=dtype).sigmoid().requires_grad_(True)
            o1, _ = chunk_delta_rule(q, k, v, beta)
            o1.sum().backward(retain_graph=True)
            f_b = time_fwd_bwd(
                chunk_delta_rule, q, k, v, beta, verbose=False,
            )
            time_f_b[config, "chunk_delta_rule"] = f_b

            # q = torch.randn(B, seqlen, H, headdim, device=device, requires_grad=True, dtype=dtype)
            # k = torch.randn(B, seqlen, H, headdim, device=device, requires_grad=True, dtype=dtype)
            # v = torch.randn(B, seqlen, H, headdim, device=device, requires_grad=True, dtype=dtype)
#            f_b = time_fwd_bwd(
#                fused_chunk_delta_rule, q, k, v, beta, verbose=False
#            )
#            time_f_b[config, "fused_chunk_delta_rule"] = f_b

            print(f"### causal={causal}, headdim={headdim}, B={B}, seqlen={seqlen} ###")
            for method in methods:
                # time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                print(f"{method:>50} fwd + bwd:\t {time_f_b[config, method]*1000:>6.4f} ms ")

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
