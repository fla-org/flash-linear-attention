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
def reset_count_module():
    return load_kernel()


@cache
def modules_context(t, hk, hv, initial, packed, length, groups, warps, workers):
    if t >= 32768:
        edge_map, edge_prefix, edge_correct = (False, False, True) if initial else (True, True, True)
    else:
        edge_map, edge_prefix, edge_correct = (False, True, True)
    parts = 2 if groups * warps == 8 else 1
    assert not packed and groups * warps * parts == 16 and (t % 128 == 0) and (length % 128 == 0)
    nc = (t + 15) // 16
    nx = (t + length - 1) // length
    prep = load_kernel(T=t, NC=nc, HK=hk, HV=hv, PACKED=False, FILL_ZERO=0)
    mapper = load_kernel(
        T=t,
        NC=nc,
        HV=hv,
        PACKED=False,
        HAS_INITIAL=initial,
        VSTEP=8 * groups,
        NX=nx,
        LENGTH=length,
        PERSISTENT=t >= 32768,
        use_pdl=edge_map,
    )
    prefix = load_kernel(PG=1, T=t, HV=hv, NX=nx, LENGTH=length, PACKED=False, HAS_INITIAL=initial, use_pdl=edge_prefix)
    apply = load_kernel(
        PERSISTENT=t >= 32768, WORKERS=workers, T=t, HV=hv, NX=nx, LENGTH=length, HAS_INITIAL=initial, use_pdl=edge_correct
    )
    return (prep, mapper, prefix, apply)


def run(q, k, v, g, beta, scale, initial_state=None, cu_seqlens=None, *, length=2048, groups=4, warps=4, workers=256):
    b, t, hk, _ = q.shape
    hv = v.shape[2]
    nc = (t + 15) // 16
    nx = (t + length - 1) // length
    prepared = torch.empty((b * nc * hv, 3, 16, 128), device=q.device, dtype=torch.bfloat16)
    mixing = torch.empty((b * nc * hv, 16, 16), device=q.device, dtype=torch.bfloat16)
    updates = torch.empty((b * nc * hv, 16, 128), device=q.device, dtype=torch.bfloat16 if t < 32768 else torch.float32)
    decays = torch.empty(b * nc * hv, device=q.device, dtype=torch.float32)
    maps = torch.empty((b * nx * hv, 128, 128), device=q.device, dtype=torch.float32)
    map_a = torch.empty_like(maps, dtype=torch.bfloat16)
    entries = torch.empty_like(maps, dtype=torch.bfloat16)
    sensitivity = torch.empty((b, t, hv, 128), device=q.device, dtype=torch.bfloat16)
    out = torch.empty_like(sensitivity)
    zero_after = torch.empty(b * nx * hv * (2 if groups * warps == 8 else 1), device=q.device, dtype=torch.int32)
    final = torch.empty((b, hv, 128, 128), device=q.device, dtype=torch.float32)
    cu = torch.empty(1, device=q.device, dtype=torch.int64)
    state = initial_state if initial_state is not None else final
    persistent = t >= 32768
    tasks = torch.empty(b * (t // 128) * hv, device=q.device, dtype=torch.int32) if persistent else zero_after
    task_count = torch.empty(1, device=q.device, dtype=torch.int32) if persistent else zero_after
    prep, mapper, prefix, apply = modules_context(
        t, hk, hv, initial_state is not None, cu_seqlens is not None, length, groups, warps, workers
    )
    prep.launch(
        grid=(nc, hv, b),
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu=cu,
        zero=task_count if persistent else final,
        prepared=prepared,
        updates=updates,
        mixing=mixing,
        decays=decays,
        scale=scale,
    )
    mapper.launch(
        grid=(32 // (groups * warps), hv, b * nx),
        initial=state,
        cu=cu,
        prepared=prepared,
        updates=updates,
        mixing=mixing,
        decays=decays,
        out=out,
        final=maps,
        map_a=map_a,
        sensitivity=sensitivity,
        zero_after=zero_after,
        tasks=tasks,
        task_count=task_count,
    )
    prefix.launch(
        grid=(2 if t >= 32768 and initial_state is None else 16, hv, b),
        initial=state,
        cu=cu,
        maps=maps,
        map_a=map_a,
        entries=entries,
        final=final,
        zero_after=zero_after,
    )
    apply.launch(
        grid=(workers, 1, 1) if persistent else (t // 128, hv, b),
        tasks=tasks,
        task_count=task_count,
        sensitivity=sensitivity,
        entries=entries,
        out=out,
        zero_after=zero_after,
    )
    return (out, final)
