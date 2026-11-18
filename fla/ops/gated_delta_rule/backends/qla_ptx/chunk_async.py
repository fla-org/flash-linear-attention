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
def modules(t, hk, hv, initial, packed, vstep):
    nc = (t + 15) // 16
    prep = load_kernel(T=t, NC=nc, HK=hk, HV=hv, PACKED=packed)
    apply = load_kernel(
        T=t, NC=nc, HV=hv, PACKED=packed, HAS_INITIAL=initial, SKIP_FIRST=int(not initial), VSTEP=vstep, use_pdl=True
    )
    return (prep, apply)


def run(q, k, v, g, beta, scale, initial_state=None, cu_seqlens=None, *, vstep=8):
    if q.shape[1] <= 32 and cu_seqlens is None and (initial_state is None):
        return fused_run(q, k, v, g, beta, scale, initial_state, cu_seqlens)
    b, t, hk, _ = q.shape
    hv = v.shape[2]
    n = b if cu_seqlens is None else cu_seqlens.numel() - 1
    nc = (t + 15) // 16
    prepared = torch.empty((n * nc * hv, 3, 16, 128), device=q.device, dtype=torch.bfloat16)
    mixing = torch.empty((n * nc * hv, 16, 16), device=q.device, dtype=torch.bfloat16)
    updates = torch.empty((n * nc * hv, 16, 128), device=q.device, dtype=torch.float32)
    decays = torch.empty(n * nc * hv, device=q.device, dtype=torch.float32)
    out = torch.empty((b, t, hv, 128), device=q.device, dtype=torch.bfloat16)
    final = torch.empty((n, hv, 128, 128), device=q.device, dtype=torch.float32)
    cu = cu_seqlens if cu_seqlens is not None else torch.empty(1, device=q.device, dtype=torch.int64)
    prep, apply = modules(t, hk, hv, initial_state is not None, cu_seqlens is not None, vstep)
    prep.launch(
        grid=(nc, hv, n),
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
    apply.launch(
        grid=(128 // vstep, hv, n),
        initial=initial_state if initial_state is not None else final,
        cu=cu,
        prepared=prepared,
        updates=updates,
        mixing=mixing,
        decays=decays,
        out=out,
        final=final,
    )
    return (out, final)


@cache
def fused_module(t, hk, hv, packed, initial):
    return load_kernel(T=t, HK=hk, HV=hv, PACKED=packed, HAS_INITIAL=initial, VSTEP=8)


def fused_run(q, k, v, g, beta, scale, initial_state=None, cu_seqlens=None):
    b, t, hk, _ = q.shape
    hv = v.shape[2]
    n = b if cu_seqlens is None else cu_seqlens.numel() - 1
    out = torch.empty((b, t, hv, 128), device=q.device, dtype=torch.bfloat16)
    final = torch.empty((n, hv, 128, 128), device=q.device, dtype=torch.float32)
    cu = cu_seqlens if cu_seqlens is not None else torch.empty(1, device=q.device, dtype=torch.int64)
    fused_module(t, hk, hv, cu_seqlens is not None, initial_state is not None).launch(
        grid=(8, hv, n),
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu=cu,
        initial=initial_state if initial_state is not None else final,
        out=out,
        final=final,
        scale=scale,
    )
    return (out, final)
