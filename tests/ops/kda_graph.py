# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F

from fla.utils import device

# static capture capacity
T = 1024
H = 4
D = 64
MAX_NUM_SEQS = 4

# (tag, cu_seqlens) -- all cover [0, T]; trailing repeats are zero-length padding sequences
_VARLEN_CONFIGS = [
    ("4seq", [0, 300, 600, 900, 1024]),
    ("2seq_zerotail", [0, 512, 1024, 1024, 1024]),
    ("4seq_alt", [0, 100, 400, 700, 1024]),
    ("1seq_zerotail", [0, 1024, 1024, 1024, 1024]),
]


def _rand_inputs(seed, gate, safe_gate, hv=H):
    # q/k carry H heads; v/g/beta/state carry HV heads (HV>H exercises GVA).
    g = torch.Generator(device).manual_seed(seed)
    dt = torch.bfloat16
    q = torch.randn(1, T, H, D, dtype=dt, device=device, generator=g)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32, device=device, generator=g), p=2, dim=-1).to(dt)
    v = torch.randn(1, T, hv, D, dtype=dt, device=device, generator=g)
    if gate:
        gg = torch.randn(1, T, hv, D, dtype=dt, device=device, generator=g)
        A_log = torch.log(torch.empty(1, 1, hv, 1, device=device).uniform_(1, 16, generator=g))
        dt_bias = torch.randn(hv * D, dtype=torch.float32, device=device, generator=g)
    else:
        gg = F.logsigmoid(torch.randn(1, T, hv, D, dtype=torch.float32, device=device, generator=g))
        if safe_gate:
            gg = gg.clamp(-5, 0)
        A_log = dt_bias = None
    beta = torch.rand(1, T, hv, dtype=dt, device=device, generator=g).sigmoid()
    h0 = torch.randn(MAX_NUM_SEQS, hv, D, D, dtype=torch.float32, device=device, generator=g)
    do = torch.randn(1, T, hv, D, dtype=dt, device=device, generator=g)
    dht = torch.randn(MAX_NUM_SEQS, hv, D, D, dtype=torch.float32, device=device, generator=g)
    return dict(q=q, k=k, v=v, g=gg, beta=beta, h0=h0, A_log=A_log, dt_bias=dt_bias, do=do, dht=dht)
