# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# Copyright 2026 Co-developed by KDA (Kernel Design Agents)

from functools import cache

import torch

from fla.ops.gated_delta_rule.backends.qla_ptx._runtime import KernelRef
from fla.ops.gated_delta_rule.backends.qla_ptx._runtime import load_module as build_kernel_module

C = 64
D = 128
TILE = C * C
LP = C + 4
TP = 36
TPW = 40
FACTORS = 4 * C
STAGE = 67584
FUSED = 1e-20


def item_heads(one, hq, hv):
    return 1 if one else hv // hq


prep_item = KernelRef()
gdn_prep = KernelRef()
gdn_prep1 = KernelRef()
PREP1 = KernelRef()
PREP1_LIMIT = 1184
PREP = {blocks: KernelRef() for blocks in (0, 5, 6)}
bf16_copy = KernelRef()
decay_blocks = KernelRef()
identity = KernelRef()
residual = KernelRef()
mqk_tile = KernelRef()
handoff = KernelRef()
QK_TMA = None
O_TMA = None
CHUNK_BYTES = 3 * C * D * 2 + TILE * 2 + FACTORS * 4
FAC_OFF = 3 * C * D * 2 + 2 * TILE * 2
SE_BYTES = 8 * C * 4
gdn_recur = KernelRef()


@cache
def prep_module(hq, hv, large):
    blocks = 0
    if hq < hv and hq >= 4 and large:
        blocks = 6 if hv == 2 * hq and hq > 4 else 5
    return build_kernel_module(PREP[blocks], HQ=hq, HV=hv, _ordinal=0)


@cache
def prep1_module(hq, hv):
    return build_kernel_module(PREP1, HQ=hq, HV=hv, _ordinal=0)


def launch_prep(hq, hv, slots, **args):
    if hv >= 3 * hq and slots * hv <= PREP1_LIMIT:
        prep1_module(hq, hv).launch(grid=(slots, hv, 1), slots=slots, **args)
    else:
        prep_module(hq, hv, slots * hq >= 2048).launch(grid=(slots, hq, 1), slots=slots, **args)


@cache
def modules(hq, hv, flag=0):
    return build_kernel_module(gdn_recur, tma_abi="grid_constant", use_pdl=flag != 2, HQ=hq, HV=hv, FLAG=flag, _ordinal=0)


def run(q, k, v, g, beta, scale, initial_state, cu_seqlens):
    t, hq = (q.shape[1], q.shape[2])
    hv = v.shape[2]
    n = cu_seqlens.numel() - 1
    slots = t // C + n
    dev = q.device
    inv = torch.empty((hv * slots, TILE), device=dev, dtype=torch.bfloat16)
    factors = torch.empty((hv * slots, FACTORS), device=dev, dtype=torch.float32)
    out = torch.empty((1, t, hv, D), device=dev, dtype=torch.bfloat16)
    final = torch.empty((n, hv, D, D), device=dev, dtype=torch.float32)
    recur = modules(hq, hv)
    launch_prep(hq, hv, slots, k=k, g=g, beta=beta, cu=cu_seqlens, inv=inv, factors=factors, nseq=n)
    recur.launch(
        grid=(hv, n, 1),
        q_tma=q,
        k_tma=k,
        v_tma=v,
        initial=initial_state,
        cu=cu_seqlens,
        inv=inv,
        factors=factors,
        out=out,
        out_tma=out,
        final=final,
        flags=factors.view(torch.uint32),
        flag_mem=factors.view(torch.uint32),
        slots=slots,
        scale=scale,
    )
    return (out, final)
