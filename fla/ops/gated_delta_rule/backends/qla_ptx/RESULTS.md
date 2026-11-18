# B300 PTX forward measurements

The public `chunk_gated_delta_rule` entry achieved a **1.497495x geometric-mean
speedup** over FlashQLA across the 20 exported prefill configurations. Seventeen
configurations were faster; three were slower. These measurements do not imply
support or gains for other shapes, decoding, training, or fused preprocessing.

FlashQLA revision: `a14dbf3474514e1674526431e960325f72081e9f`.
FLA base revision: `cc4648a3b35c27c6bc6b3cfb0079a14d57bcbc5b`.
Both processes ran sequentially on the same NVIDIA B300 SXM6 AC GPU, using
Python 3.12, PyTorch 2.14.1+cu130, CUDA 13.0, Triton 3.6.0, TVM FFI
0.1.13.dev57, and NVIDIA driver 615.71.10. Inputs use seed 42, normalized Q/K,
nonpositive log decays, and post-sigmoid beta.

Each measurement uses five warmups, three complete operator calls per CUDA graph
replay, and the median of 20 CUDA event samples. Initial compilation and PTX
loading are excluded. Timing includes every GPU stage. The CPU dispatch and
metadata checks execute during capture and are outside replay timing.

Reproduce with FlashQLA installed, in separate processes:

```bash
python -m benchmarks.ops.benchmark_qla_ptx --backend flash-qla --output baseline.json
python -m benchmarks.ops.benchmark_qla_ptx --backend ptx --output candidate.json
```

| Workload | Initial state | FlashQLA (us) | PTX (us) | Speedup |
| --- | --- | ---: | ---: | ---: |
| small | No | 15.653 | 5.221 | 2.998x |
| small | Yes | 15.995 | 6.613 | 2.419x |
| dense-H4 | No | 111.851 | 62.720 | 1.783x |
| dense-H4 | Yes | 112.912 | 65.451 | 1.725x |
| dense-G4 | No | 140.539 | 75.781 | 1.855x |
| dense-G4 | Yes | 140.971 | 79.669 | 1.769x |
| dense-H16 | No | 147.376 | 107.845 | 1.367x |
| dense-H16 | Yes | 147.909 | 106.176 | 1.393x |
| packed-small | No | 18.144 | 10.048 | 1.806x |
| packed-small | Yes | 18.549 | 10.027 | 1.850x |
| packed-disparate | No | 101.488 | 81.147 | 1.251x |
| packed-disparate | Yes | 102.144 | 85.168 | 1.199x |
| packed-G2 | No | 134.800 | 134.944 | 0.999x |
| packed-G2 | Yes | 137.195 | 134.789 | 1.018x |
| packed-partial | No | 18.133 | 10.661 | 1.701x |
| packed-partial | Yes | 18.304 | 10.672 | 1.715x |
| packed-long | No | 149.813 | 118.069 | 1.269x |
| packed-long | Yes | 150.203 | 113.728 | 1.321x |
| dense-long | No | 480.869 | 515.749 | 0.932x |
| dense-long | Yes | 481.856 | 484.859 | 0.994x |

The slower configurations are packed-G2 without an initial state and both
dense-long configurations. The complete table retains these regressions.
