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
def modules_context(t, hk, hv, initial, packed, length, groups, warps, n=1):
    nc = (t + 15) // 16
    nx = (t + length - 1) // length
    prep = load_kernel(T=t, NC=nc, HK=hk, HV=hv, PACKED=packed, FILL_ZERO=0, NSEQ=n if packed else 0)
    common = {
        "T": t,
        "NC": nc,
        "HV": hv,
        "PACKED": packed,
        "HAS_INITIAL": True,
        "VSTEP": 8 * groups,
        "NX": nx,
        "LENGTH": length,
        "NSEQ": n,
    }
    mapper = load_kernel(EARLY_EXIT=t >= 16384, **{**common, "HAS_INITIAL": initial})
    apply = load_kernel(**common)
    prefix = load_kernel(T=t, HV=hv, NX=nx, LENGTH=length, PACKED=packed, HAS_INITIAL=initial)
    return (prep, mapper, prefix, apply)


def run(q, k, v, g, beta, scale, initial_state=None, cu_seqlens=None, *, length=None, groups=None, warps=None):
    b, t, hk, _ = q.shape
    if length is None:
        length = 2048 if t >= 16384 and cu_seqlens is None else 512
    if groups is None:
        groups = 4 if t >= 16384 and cu_seqlens is None else 2
    if warps is None:
        warps = 4 if t >= 16384 and cu_seqlens is None else 1
    hv = v.shape[2]
    n = b if cu_seqlens is None else cu_seqlens.numel() - 1
    nc = (t + 15) // 16
    nx = (t + length - 1) // length
    prepared = torch.empty((n * nc * hv, 3, 16, 128), device=q.device, dtype=torch.bfloat16)
    mixing = torch.empty((n * nc * hv, 16, 16), device=q.device, dtype=torch.bfloat16)
    updates = torch.empty((n * nc * hv, 16, 128), device=q.device, dtype=torch.float32)
    decays = torch.empty(n * nc * hv, device=q.device, dtype=torch.float32)
    maps = torch.empty((n * nx * hv, 128, 128), device=q.device, dtype=torch.float32)
    map_a = torch.empty((n * nx * hv, 128, 128), device=q.device, dtype=torch.bfloat16)
    entries = torch.empty((n * nx * hv, 128, 128), device=q.device, dtype=torch.float32)
    out = torch.empty((b, t, hv, 128), device=q.device, dtype=torch.bfloat16)
    final = torch.empty((n, hv, 128, 128), device=q.device, dtype=torch.float32)
    cu = cu_seqlens if cu_seqlens is not None else torch.empty(1, device=q.device, dtype=torch.int64)
    state = initial_state if initial_state is not None else final
    prep, mapper, prefix, apply = modules_context(
        t, hk, hv, initial_state is not None, cu_seqlens is not None, length, groups, warps, n
    )
    prep.launch(
        grid=(nc + n - 1, hv, 1) if cu_seqlens is not None else (nc, hv, n),
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu=cu,
        zero=final,
        prepared=prepared,
        updates=updates,
        mixing=mixing,
        decays=decays,
        scale=scale,
    )
    common = {"cu": cu, "prepared": prepared, "updates": updates, "mixing": mixing, "decays": decays, "out": out}
    mapper.launch(
        grid=(32 // (groups * warps), hv, nx + n - 1 if cu_seqlens is not None else n * nx),
        initial=state,
        final=maps,
        map_a=map_a,
        **common,
    )
    prefix.launch(grid=(8, hv, n), initial=state, cu=cu, maps=maps, map_a=map_a, entries=entries)
    apply.launch(
        grid=(16 // (groups * warps), hv, nx + n - 1 if cu_seqlens is not None else n * nx),
        initial=entries,
        final=final,
        **common,
    )
    return (out, final)
