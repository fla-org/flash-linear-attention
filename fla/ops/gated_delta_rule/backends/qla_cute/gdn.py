# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# Copyright 2026 Co-developed by KDA (Kernel Design Agents)

"""Host entry for the chunked CuTe DSL Gated DeltaNet forward."""

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from fla.ops.gated_delta_rule.backends.qla_cute.gdn_prep import CTRI, prep_launch
from fla.ops.gated_delta_rule.backends.qla_cute.gdn_recur import comb_launch, recur_launch
from fla.ops.gated_delta_rule.backends.qla_cute.gdn_tc import tc_launch

C = 64
TC_STAGES = 3
_COMPILED = {}
_DUMMY = {}


def _ptr(t, dtype):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def _dummy(device, dtype):
    key = (device, dtype)
    if key not in _DUMMY:
        _DUMMY[key] = torch.zeros(16, dtype=dtype, device=device)
    return _DUMMY[key]


def _sms(device):
    key = ("sms", device)
    if key not in _DUMMY:
        _DUMMY[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return _DUMMY[key]


def _call(key, fn, args, consts, stream):
    if key not in _COMPILED:
        _COMPILED[key] = cute.compile(fn, *args, *consts, stream)
    _COMPILED[key](*args, stream)


def run(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state=None,
    cu_seqlens=None,
    W=4,
    STAGES=2,
    lc=0,
    W1=None,
    W2=None,
    tc=False,
    tccp=False,
    cpick=0,
):
    """lc > 0: context parallelism with lc chunks per context (tccp: on the tcgen05
    kernel, else mma.sync); lc <= 0: whole sequences (tc: tcgen05, else mma.sync).
    Two packed sequences with tccp and cpick > 0: device-side choice by the split
    (near-equal iff 2 max len <= cpick * min len) between the CP passes and the
    whole-sequence tcgen05 kernel."""
    B, T, HQ, DK = q.shape
    HV, DV = v.shape[-2:]
    assert DK == 128 and DV == 128 and (HV % HQ == 0)
    N = B if cu_seqlens is None else cu_seqlens.numel() - 1
    ntok = B * T
    dense_t = T if cu_seqlens is None else 0
    nslot = ntok // C + N
    dev = q.device
    bf, f32, i32 = (cutlass.BFloat16, cutlass.Float32, cutlass.Int32)
    cmp = tccp or (lc <= 0 and tc)
    per = CTRI if cmp else C * C
    a1 = torch.empty(nslot * HV * per, dtype=torch.bfloat16, device=dev)
    mqk = torch.empty(nslot * HV * per, dtype=torch.bfloat16, device=dev)
    fac = torch.empty(nslot * HV * 192, dtype=torch.float32, device=dev)
    out = torch.empty((B, T, HV, DV), dtype=torch.bfloat16, device=dev)
    final = torch.empty((N, HV, DK, DV), dtype=torch.float32, device=dev)
    cu = cu_seqlens if cu_seqlens is not None else _dummy(dev, torch.int32)
    if cu.dtype != torch.int32:
        raise TypeError("cu_seqlens must be int32")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    has_h = initial_state is not None
    h0 = initial_state if has_h else final
    scale = float(scale)
    pargs = (
        _ptr(k, bf),
        _ptr(q, bf),
        _ptr(g, f32),
        _ptr(beta, f32),
        _ptr(cu, i32),
        _ptr(a1, bf),
        _ptr(mqk, bf),
        _ptr(fac, f32),
        ntok,
        nslot,
        N,
        HQ,
        HV,
        dense_t,
        scale,
    )
    minb = 4
    gqa = HV > HQ
    _call(("prep", minb, gqa, cmp), prep_launch, pargs, (minb, gqa, cmp), stream)
    dz = _dummy(dev, torch.float32)
    di = _dummy(dev, torch.int32)
    head = (
        _ptr(q, bf),
        _ptr(k, bf),
        _ptr(v, bf),
        _ptr(a1, bf),
        _ptr(mqk, bf),
        _ptr(fac, f32),
        _ptr(h0, f32),
        _ptr(cu, i32),
        _ptr(out, bf),
        _ptr(final, f32),
    )
    two = N == 2 and cu_seqlens is not None

    def whole_tc(pk, ratio):
        targs = head + (
            _ptr(dz, f32),
            _ptr(dz, f32),
            _ptr(dz, f32),
            _ptr(di, i32),
            ntok,
            nslot,
            N,
            1,
            1,
            HQ,
            HV,
            dense_t,
            scale,
            ratio,
        )
        _call(("tc", TC_STAGES, has_h, pk, 0), tc_launch, targs, (TC_STAGES, has_h, pk, 0), stream)

    if lc <= 0:
        if tc:
            whole_tc(0, 0)
            return (out, final)
        rargs = head + (
            _ptr(dz, f32),
            _ptr(dz, f32),
            _ptr(dz, f32),
            _ptr(di, i32),
            ntok,
            nslot,
            N,
            1,
            1,
            HQ,
            HV,
            dense_t,
            scale,
        )
        lpt = cu_seqlens is not None and N > 1 and (_sms(dev) < 128 // (16 * W) * HV * N)
        _call(("recur", W, STAGES, has_h, 0, 0, lpt), recur_launch, rargs, (W, STAGES, has_h, 0, 0, lpt), stream)
        return (out, final)
    ncs = ntok // C // lc + N + 1
    hb = torch.empty(ncs * HV * DK * DV, dtype=torch.float32, device=dev)
    xb = torch.empty_like(hb)
    sb = torch.empty_like(hb)
    xf = torch.empty(ncs * HV, dtype=torch.int32, device=dev)
    base = head + (_ptr(hb, f32), _ptr(xb, f32), _ptr(sb, f32), _ptr(xf, i32), ntok, nslot, N, ncs, lc, HQ, HV, dense_t, scale)
    use_cpick = tccp and cpick > 0 and two
    pk = 1 if use_cpick else 0
    ratio = cpick if use_cpick else 0
    cargs = (_ptr(hb, f32), _ptr(xb, f32), _ptr(sb, f32), _ptr(xf, i32), _ptr(cu, i32), N, ncs, HV, dense_t, lc, ratio)
    crb = 16 if pk or (HV * N >= 48 and (N != 2 or HV >= 32)) else 8 if HV * N >= 24 else 4
    if tccp:
        tbase = base + (ratio,)
        if use_cpick:
            whole_tc(2, cpick)
        _call(("tc", TC_STAGES, has_h, pk, 1), tc_launch, tbase, (TC_STAGES, has_h, pk, 1), stream)
        _call(("comb", pk, crb), comb_launch, cargs, (pk, crb), stream)
        _call(("tc", TC_STAGES, has_h, pk, 2), tc_launch, tbase, (TC_STAGES, has_h, pk, 2), stream)
        return (out, final)
    w1 = W1 or W
    w2 = W2 or W
    _call(("recur", w1, STAGES, has_h, 1, 0, False), recur_launch, base, (w1, STAGES, has_h, 1, 0, False), stream)
    _call(("comb", 0, crb), comb_launch, cargs, (0, crb), stream)
    _call(("recur", w2, STAGES, has_h, 2, 0, False), recur_launch, base, (w2, STAGES, has_h, 2, 0, False), stream)
    return (out, final)
