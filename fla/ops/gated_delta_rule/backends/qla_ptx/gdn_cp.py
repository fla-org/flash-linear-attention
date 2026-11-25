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
from fla.ops.gated_delta_rule.backends.qla_ptx.gdn import FACTORS, TILE, C, D, launch_prep
from fla.ops.gated_delta_rule.backends.qla_ptx.gdn import modules as recur_modules
from fla.ops.gated_delta_rule.backends.qla_ptx.gdn import run as plain_run

STAGE1 = 2 * C * D * 2 + TILE * 2 + FACTORS * 4
STATE_BYTES = STAGE1
NS1 = 3
LOG_TINY = -27.631021
PREP_PER_SM = 4
decayed = KernelRef()
decayed_g = KernelRef()
contexts = KernelRef()
plan = KernelRef()
unit_ctx = KernelRef()
cp_counts = KernelRef()
cp_item = KernelRef()
cp_prep_one = KernelRef()
gdn_prep_cp = KernelRef()
gdn_prep_cp1 = KernelRef()
PREP_CP = KernelRef()
PREP_CP1 = {blocks: KernelRef() for blocks in (0, 5, 6)}
plain_item = KernelRef()
gdn_prep_plain = KernelRef()
PREP_PLAIN = {blocks: KernelRef() for blocks in (0, 5, 6)}
gdn_prep_plain1 = KernelRef()
PREP_PLAIN1 = KernelRef()
bf16_half = KernelRef()
cp_state = KernelRef()
comb_body = KernelRef()


def comb_kernel(vw):
    cp_comb = KernelRef()
    return cp_comb


COMB = {8: comb_kernel(8), 16: comb_kernel(16), 32: comb_kernel(32)}


def comb_width(hv, n, lcmin):
    if hv < 48 and n <= 2 and (lcmin >= 10):
        return 8
    heads = hv if n <= 2 else hv * n
    return 8 if heads <= 8 else 16 if heads <= 16 else 32


cp_out = KernelRef()


@cache
def modules(hq, hv, vw, flag):
    state = build_kernel_module(cp_state, tma_abi="grid_constant", use_pdl=flag != 2, HQ=hq, HV=hv, FLAG=flag, _ordinal=0)
    comb = build_kernel_module(COMB[vw], use_pdl=True, HV=hv, _ordinal=1)
    recur = build_kernel_module(cp_out, tma_abi="grid_constant", use_pdl=True, HQ=hq, HV=hv, FLAG=flag, _ordinal=2)
    return (state, comb, recur)


@cache
def prep_cp_module(hq, hv):
    return build_kernel_module(PREP_CP, HQ=hq, HV=hv, _ordinal=0)


@cache
def prep_cp1_module(hq, hv, large):
    blocks = 0
    if hq < hv and hq >= 4 and large:
        blocks = 6 if hv == 2 * hq and hq > 4 else 5
    return build_kernel_module(PREP_CP1[blocks], use_pdl=True, HQ=hq, HV=hv, _ordinal=0)


@cache
def sm_count(dev):
    return torch.cuda.get_device_properties(dev).multi_processor_count


FLAGS = {}
RETIRED = []
LOADED = set()


def flag_buffer(dev, count):
    buf = FLAGS.get(dev)
    if buf is None or buf.numel() < count:
        if torch.cuda.is_current_stream_capturing():
            return None
        if buf is not None:
            RETIRED.append(buf)
        buf = torch.zeros(max(count, 1 << 16), device=dev, dtype=torch.uint32)
        FLAGS[dev] = buf
    return buf


def run(q, k, v, g, beta, scale, initial_state, cu_seqlens, jmax, overlap=0):
    t, hq = (q.shape[1], q.shape[2])
    hv = v.shape[2]
    n = cu_seqlens.numel() - 1
    slots = t // C + n
    lcmin = -(-(t // C) // jmax)
    cslots = t // C // lcmin + n
    dev = q.device
    inv = torch.empty((hv * slots, TILE), device=dev, dtype=torch.bfloat16)
    factors = torch.empty((hv * slots, FACTORS), device=dev, dtype=torch.float32)
    xbuf = torch.empty((2, cslots, hv, D, D), device=dev, dtype=torch.float32)
    xflag = torch.empty((cslots, hv), device=dev, dtype=torch.int32)
    out = torch.empty((1, t, hv, D), device=dev, dtype=torch.bfloat16)
    final = torch.empty((n, hv, D, D), device=dev, dtype=torch.float32)
    vw = comb_width(hv, n, lcmin)
    units = jmax - n + max(jmax - n - 1, 0)
    buf = flag_buffer(dev, hv * slots + 1) if overlap else None
    flag = 0 if buf is None else overlap
    flags = None if buf is None else buf[: hv * slots]
    state, comb, recur = modules(hq, hv, vw, flag)
    state_args = {
        "k_tma": k,
        "v_tma": v,
        "initial": initial_state,
        "cu": cu_seqlens,
        "inv": inv,
        "factors": factors,
        "xbuf": xbuf,
        "xflag": xflag,
        "g": g,
        "nseq": n,
        "slots": slots,
        "cslots": cslots,
        "lcmin": lcmin,
        "jmax": jmax,
    }
    if flag == 2:
        prep = prep_cp1_module(hq, hv, slots * hq >= 2048)
        if prep not in LOADED:
            prep.launch(
                grid=(1, 1, 1),
                k=k,
                g=g,
                beta=beta,
                cu=cu_seqlens,
                inv=inv,
                factors=factors,
                flags=flags,
                done=buf[-1:],
                nseq=0,
                slots=slots,
                lcmin=lcmin,
                jmax=jmax,
            )
            LOADED.add(prep)
        state.launch(grid=(hv, units, 1), flags=flags, **state_args)
        prep.launch(
            grid=(slots * hq, 1, 1),
            k=k,
            g=g,
            beta=beta,
            cu=cu_seqlens,
            inv=inv,
            factors=factors,
            flags=flags,
            done=buf[-1:],
            nseq=n,
            slots=slots,
            lcmin=lcmin,
            jmax=jmax,
        )
    elif flag:
        prep_cp_module(hq, hv).launch(
            grid=(PREP_PER_SM * sm_count(dev), 1, 1),
            k=k,
            g=g,
            beta=beta,
            cu=cu_seqlens,
            inv=inv,
            factors=factors,
            flags=flags,
            nseq=n,
            slots=slots,
            lcmin=lcmin,
            jmax=jmax,
        )
    else:
        flags = xflag.view(torch.uint32)
        launch_prep(hq, hv, slots, k=k, g=g, beta=beta, cu=cu_seqlens, inv=inv, factors=factors, nseq=n)
    if flag != 2:
        state.launch(grid=(hv, units, 1), flags=flags, **state_args)
    if jmax - n >= 2:
        comb.launch(
            grid=(D // vw, hv, n), cu=cu_seqlens, xbuf=xbuf, xflag=xflag, nseq=n, lcmin=lcmin, jmax=jmax, cslots=cslots
        )
    recur.launch(
        grid=(hv, cslots, 1),
        q_tma=q,
        k_tma=k,
        v_tma=v,
        initial=initial_state,
        cu=cu_seqlens,
        inv=inv,
        factors=factors,
        xbuf=xbuf,
        out=out,
        out_tma=out,
        final=final,
        flags=flags,
        nseq=n,
        slots=slots,
        lcmin=lcmin,
        jmax=jmax,
        scale=scale,
    )
    return (out, final)


@cache
def prep_plain1_module(hq, hv):
    return build_kernel_module(PREP_PLAIN1, use_pdl=True, HQ=hq, HV=hv, _ordinal=0)


@cache
def prep_plain_module(hq, hv, large):
    blocks = 0
    if hq < hv and hq >= 4 and large:
        blocks = 5 if hq == 4 else 6
    return build_kernel_module(PREP_PLAIN[blocks], use_pdl=True, HQ=hq, HV=hv, _ordinal=0)


def run_plain(q, k, v, g, beta, scale, initial_state, cu_seqlens):
    t, hq = (q.shape[1], q.shape[2])
    hv = v.shape[2]
    n = cu_seqlens.numel() - 1
    slots = t // C + n
    dev = q.device
    buf = flag_buffer(dev, hv * slots + 1)
    if buf is None:
        return plain_run(q, k, v, g, beta, scale, initial_state, cu_seqlens)
    inv = torch.empty((hv * slots, TILE), device=dev, dtype=torch.bfloat16)
    factors = torch.empty((hv * slots, FACTORS), device=dev, dtype=torch.float32)
    out = torch.empty((1, t, hv, D), device=dev, dtype=torch.bfloat16)
    final = torch.empty((n, hv, D, D), device=dev, dtype=torch.float32)
    flags = buf[: hv * slots]
    items = slots * hq
    if hq < hv and n * hv <= 64:
        prep = prep_plain1_module(hq, hv)
        items = slots * hv
    else:
        prep = prep_plain_module(hq, hv, slots * hq >= 2048)
    prep_args = {
        "k": k,
        "g": g,
        "beta": beta,
        "cu": cu_seqlens,
        "inv": inv,
        "factors": factors,
        "flags": flags,
        "done": buf[-1:],
        "slots": slots,
    }
    if prep not in LOADED:
        prep.launch(grid=(1, 1, 1), nseq=0, **prep_args)
        LOADED.add(prep)
    recur_modules(hq, hv, 2).launch(
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
        flags=flags,
        flag_mem=flags,
        slots=slots,
        scale=scale,
    )
    prep.launch(grid=(items, 1, 1), nseq=n, **prep_args)
    return (out, final)
