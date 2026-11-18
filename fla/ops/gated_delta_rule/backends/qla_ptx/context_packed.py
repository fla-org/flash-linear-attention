# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
from functools import cache

import torch

from fla.ops.gated_delta_rule.backends.qla_ptx._runtime import load_kernel


@cache
def modules_context(t, hk, hv, initial, length, workers, n):
    if hv >= 16:
        edge_map, edge_prefix, edge_correct = (True, True, True)
    else:
        edge_map, edge_prefix, edge_correct = (True, False, True)
    nc = (t + 15) // 16
    nx = (t + length - 1) // length
    prep = load_kernel(T=t, NC=nc, HK=hk, HV=hv, PACKED=True, FILL_ZERO=0, NSEQ=n)
    mapper = load_kernel(
        T=t,
        NC=nc,
        HV=hv,
        PACKED=True,
        HAS_INITIAL=initial,
        NSEQ=n,
        VSTEP=32,
        NX=nx,
        LENGTH=length,
        PERSISTENT=True,
        use_pdl=edge_map,
    )
    prefix = load_kernel(PG=1, T=t, HV=hv, NX=nx, LENGTH=length, PACKED=True, HAS_INITIAL=initial, use_pdl=edge_prefix)
    correct = load_kernel(
        NSEQ=n,
        PACKED=True,
        WORKERS=workers,
        PERSISTENT=True,
        T=t,
        HV=hv,
        NX=nx,
        LENGTH=length,
        HAS_INITIAL=initial,
        use_pdl=edge_correct,
    )
    return (prep, mapper, prefix, correct)


def run(q, k, v, g, beta, scale, initial_state=None, cu_seqlens=None, *, length=512, workers=256):
    b, t, hk, _ = q.shape
    hv = v.shape[2]
    n = cu_seqlens.numel() - 1
    nc = (t + 15) // 16
    nx = (t + length - 1) // length
    prepared = torch.empty((n * nc * hv, 3, 16, 128), device=q.device, dtype=torch.bfloat16)
    mixing = torch.empty((n * nc * hv, 16, 16), device=q.device, dtype=torch.bfloat16)
    updates = torch.empty((n * nc * hv, 16, 128), device=q.device, dtype=torch.bfloat16)
    decays = torch.empty(n * nc * hv, device=q.device, dtype=torch.float32)
    maps = torch.empty((n * nx * hv, 128, 128), device=q.device, dtype=torch.float32)
    map_a = torch.empty_like(maps, dtype=torch.bfloat16)
    entries = torch.empty_like(maps, dtype=torch.bfloat16)
    sensitivity = torch.empty((b, t, hv, 128), device=q.device, dtype=torch.bfloat16)
    out = torch.empty_like(sensitivity)
    zero_after = torch.empty(n * nx * hv, device=q.device, dtype=torch.int32)
    final = torch.empty((n, hv, 128, 128), device=q.device, dtype=torch.float32)
    tasks = torch.empty(n * nx * (length // 128) * hv, device=q.device, dtype=torch.int32)
    task_count = torch.empty(1, device=q.device, dtype=torch.int32)
    state = initial_state if initial_state is not None else final
    prep, mapper, prefix, correct = modules_context(t, hk, hv, initial_state is not None, length, workers, n)
    prep.launch(
        grid=(nc + n - 1, hv, 1),
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu=cu_seqlens,
        zero=task_count,
        prepared=prepared,
        updates=updates,
        mixing=mixing,
        decays=decays,
        scale=scale,
    )
    mapper.launch(
        grid=(2, hv, nx + n - 1),
        initial=state,
        cu=cu_seqlens,
        prepared=prepared,
        updates=updates,
        mixing=mixing,
        decays=decays,
        out=out,
        final=maps,
        terminal=final,
        map_a=map_a,
        sensitivity=sensitivity,
        zero_after=zero_after,
        tasks=tasks,
        task_count=task_count,
    )
    prefix.launch(
        grid=(1 if hv >= 16 else 16, hv, n),
        initial=state,
        cu=cu_seqlens,
        maps=maps,
        map_a=map_a,
        zero_after=zero_after,
        entries=entries,
        final=final,
    )
    correct.launch(
        grid=(workers, 1, 1),
        cu=cu_seqlens,
        tasks=tasks,
        task_count=task_count,
        sensitivity=sensitivity,
        entries=entries,
        out=out,
        zero_after=zero_after,
    )
    return (out, final)
