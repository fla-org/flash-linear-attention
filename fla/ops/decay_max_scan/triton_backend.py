# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton backend for the decay-max scan: chunked forward with block-carry
composition, and a reverse BPTT backward over coefficients recomputed in torch
(so no coefficient tensors are saved for backward).

Kernels index tensors through explicit strides; the public layout is
``[B, T, D, H]`` with per-lane ``[B, D, H]`` state. Decay parameters are stored
flat as ``[D * H]`` (element ``(d, h)`` at offset ``d * H + h``).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softplus(pre):
    return tl.where(pre > 0, pre + tl.log(1 + tl.exp(-pre)), tl.log(1 + tl.exp(pre)))


@triton.jit(do_not_specialize=['S'])
def decay_max_scan_chunk_local_fwd_kernel(
        c_ptr, v_ptr, b_ptr, cm_ptr, cg_ptr, ca_ptr, cs_ptr,
        S, C, DH, H, CHUNK,
        sc0, sc1, sc2, sc3,
        USE_DECAY: tl.constexpr,
        BLOCK_DH: tl.constexpr):
    """Per-chunk reverse scan producing the carry (decay-weighted chunk max m,
    suffix decay product g, owner index, owner-to-chunk-end suffix product cs).
    Read-only over the input; e/idx are written by the carry kernel."""
    pid = tl.program_id(0)
    ck = tl.program_id(1)
    b = pid // H
    h = pid % H
    offs = tl.arange(0, BLOCK_DH)
    lmask = offs < DH
    pv = tl.load(v_ptr + offs * H + h, mask=lmask, other=0.0).to(tl.float32)
    pb = tl.load(b_ptr + offs * H + h, mask=lmask, other=0.0).to(tl.float32)
    P = tl.full([BLOCK_DH], 1.0, tl.float32)     # suffix gamma product
    M = tl.full([BLOCK_DH], float('-inf'), tl.float32)
    own = tl.full([BLOCK_DH], -1, tl.int32)
    cs = tl.full([BLOCK_DH], 1.0, tl.float32)    # owner -> chunk-end gamma product
    base = ck * CHUNK
    for jj in range(0, CHUNK):
        j = base + CHUNK - 1 - jj                # reverse order -> suffix product
        msk = lmask & (j < S)
        x = tl.load(c_ptr + b * sc0 + j * sc2 + h * sc3 + offs * sc1,
                    mask=msk, other=float('-inf')).to(tl.float32)
        x_s = tl.where(msk, x, 0.0)
        if USE_DECAY:
            pre = x_s * pv + pb
            g = tl.where(msk, tl.exp(-_softplus(pre)), 1.0)
        else:
            g = tl.full([BLOCK_DH], 1.0, tl.float32)
        cand = P * x                             # prod_{k>j} g_k * x_j (own g excluded!)
        upd = msk & (cand >= M)                  # descending scan + >= -> tie to earliest
        own = tl.where(upd, j, own)
        cs = tl.where(upd, P, cs)
        M = tl.maximum(M, cand)
        P = P * g
    m_off = (b * DH + offs) * (C * H) + ck * H + h
    tl.store(cm_ptr + m_off, M, mask=lmask)
    tl.store(cg_ptr + m_off, P, mask=lmask)
    tl.store(ca_ptr + m_off, own, mask=lmask)
    tl.store(cs_ptr + m_off, cs, mask=lmask)


@triton.jit(do_not_specialize=['S'])
def decay_max_scan_carry_fwd_kernel(
        c_ptr, v_ptr, b_ptr, cm_ptr, cg_ptr, ca_ptr,
        e_ptr, idx_ptr, st_ptr,
        S, C, DH, H, CHUNK,
        sc0, sc1, sc2, sc3,
        se0, se1, se2, se3,
        sst0, sst1, sst2, sst3,
        HAS_INIT: tl.constexpr,
        USE_DECAY: tl.constexpr,
        BLOCK_DH: tl.constexpr):
    """Carry composition across chunks followed by a single forward sweep that
    writes the scan output e and the argmax owner chain idx."""
    pid = tl.program_id(0)
    ck = tl.program_id(1)
    b = pid // H
    h = pid % H
    offs = tl.arange(0, BLOCK_DH)
    lmask = offs < DH
    pv = tl.load(v_ptr + offs * H + h, mask=lmask, other=0.0).to(tl.float32)
    pb = tl.load(b_ptr + offs * H + h, mask=lmask, other=0.0).to(tl.float32)
    run = tl.full([BLOCK_DH], float('-inf'), tl.float32)
    cur = tl.full([BLOCK_DH], -1, tl.int32)
    if HAS_INIT:
        run = tl.load(st_ptr + b * sst0 + 0 * sst2 + h * sst3 + offs * sst1,
                      mask=lmask, other=float('-inf')).to(tl.float32)
    for cc in range(0, ck):                      # carry composition (C is small)
        m_off = (b * DH + offs) * (C * H) + cc * H + h
        Gm = tl.load(cg_ptr + m_off, mask=lmask, other=1.0).to(tl.float32)
        M = tl.load(cm_ptr + m_off, mask=lmask, other=float('-inf')).to(tl.float32)
        p = tl.load(ca_ptr + m_off, mask=lmask, other=-1)
        win = Gm * run < M
        cur = tl.where(win, p, cur)
        run = tl.maximum(Gm * run, M)
    base = ck * CHUNK
    for j in range(0, CHUNK):
        pos = base + j
        msk = lmask & (pos < S)
        x = tl.load(c_ptr + b * sc0 + pos * sc2 + h * sc3 + offs * sc1,
                    mask=msk, other=float('-inf')).to(tl.float32)
        x_s = tl.where(msk, x, 0.0)
        if USE_DECAY:
            pre = x_s * pv + pb
            g = tl.exp(-_softplus(pre))
        else:
            g = 1.0
        cand = g * run
        win = x > cand                           # tie -> carry/init wins (first-peak semantics)
        cur = tl.where(msk & win, pos, cur)
        run = tl.maximum(cand, tl.where(msk, x, float('-inf')))
        tl.store(e_ptr + b * se0 + pos * se2 + h * se3 + offs * se1,
                 run.to(e_ptr.dtype.element_ty), mask=msk)
        tl.store(idx_ptr + (b * DH + offs) * (S * H) + pos * H + h, cur, mask=msk)


@triton.jit(do_not_specialize=['S'])
def decay_max_scan_bwd_kernel(
        ge_ptr, rp_ptr, mx_ptr, rho_ptr, kap_ptr, kapx_ptr, v_ptr, b_ptr,
        gc_ptr, gd_ptr, gdx_ptr, r0_ptr,
        S, DH, H,
        sg0, sg1, sg2, sg3,
        BLOCK_DH: tl.constexpr):
    """Reverse BPTT over precomputed per-position coefficients:
    R_j = ge_j + rho_{j+1} * R_{j+1}; dx_j = mx_j * R_j + dpre_j * v_d,
    dpre_j = -R_j * rp_j * kap_j; dv accumulates -R_j * rp_j * kapx_j."""
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    offs = tl.arange(0, BLOCK_DH)
    lmask = offs < DH
    pv = tl.load(v_ptr + offs * H + h, mask=lmask, other=0.0).to(tl.float32)
    base = b * sg0 + h * sg3 + offs * sg1
    R = tl.zeros([BLOCK_DH], tl.float32)
    rho_c = tl.zeros([BLOCK_DH], tl.float32)   # rho_{j+1} loaded on the previous step
    for jj in range(0, S):
        j = S - 1 - jj
        off_j = base + j * sg2
        Rj = tl.load(ge_ptr + off_j, mask=lmask, other=0.0).to(tl.float32) \
            + rho_c * R                          # R_j = ge_j + rho_{j+1} * R_{j+1}
        mx = tl.load(mx_ptr + off_j, mask=lmask, other=0.0).to(tl.float32)
        kp = tl.load(kap_ptr + off_j, mask=lmask, other=0.0).to(tl.float32)
        kpx = tl.load(kapx_ptr + off_j, mask=lmask, other=0.0).to(tl.float32)
        rp = tl.load(rp_ptr + off_j, mask=lmask, other=0.0).to(tl.float32)
        dpre = -Rj * rp * kp                     # first step w/o init: rp finite, kap = 0 -> 0
        tl.store(gc_ptr + off_j, mx * Rj + dpre * pv, mask=lmask)
        tl.store(gdx_ptr + off_j, -Rj * rp * kpx, mask=lmask)
        tl.store(gd_ptr + off_j, dpre, mask=lmask)
        if j == 0:
            tl.store(r0_ptr + b * (DH * H) + offs * H + h, Rj, mask=lmask)
        rho_c = tl.load(rho_ptr + off_j, mask=lmask, other=0.0).to(tl.float32)
        R = Rj


def _flat_params(v, b, D, H):
    """Broadcast `[D]` decay parameters to the flat `[D * H]` lane layout."""
    if v is None:
        return None, None
    vf = v.reshape(-1).float()
    bf = b.reshape(-1).float()
    if vf.numel() == D:
        vf = vf.unsqueeze(-1).expand(D, H).reshape(D * H)
        bf = bf.unsqueeze(-1).expand(D, H).reshape(D * H)
    return vf.contiguous(), bf.contiguous()


class DecayMaxScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, state, v, b):
        B, T, D, H = x.shape
        vf, bf = _flat_params(v, b, D, H)
        use_decay = vf is not None
        CHUNK = 128
        C = triton.cdiv(T, CHUNK)
        BDH = triton.next_power_of_2(D)
        cc = x.contiguous()
        # stride maps: sc1 multiplies the channel offset, sc2 the position offset
        sc = (cc.stride(0), cc.stride(2), cc.stride(1), cc.stride(3))
        e = torch.empty(cc.shape, dtype=torch.float32, device=cc.device)
        se = (e.stride(0), e.stride(2), e.stride(1), e.stride(3))
        idx = torch.empty((B, D, T, H), dtype=torch.int32, device=cc.device)
        cm = torch.empty((B, D, C, H), dtype=torch.float32, device=cc.device)
        cg = torch.empty((B, D, C, H), dtype=torch.float32, device=cc.device)
        ca = torch.empty((B, D, C, H), dtype=torch.int32, device=cc.device)
        cs = torch.empty((B, D, C, H), dtype=torch.float32, device=cc.device)
        grid = (B * H, C)
        if state is not None:
            st = state.float().contiguous()
            sst = (st.stride(0), st.stride(1), 0, st.stride(2))
        else:
            st = cc
            sst = (0, 0, 0, 0)
        decay_max_scan_chunk_local_fwd_kernel[grid](
            cc, vf if use_decay else cc, bf if use_decay else cc,
            cm, cg, ca, cs, T, C, D, H, CHUNK, *sc,
            USE_DECAY=use_decay, BLOCK_DH=BDH, num_warps=4)
        decay_max_scan_carry_fwd_kernel[grid](
            cc, vf if use_decay else cc, bf if use_decay else cc,
            cm, cg, ca, e, idx, st,
            T, C, D, H, CHUNK, *sc, *se, *sst,
            HAS_INIT=state is not None, USE_DECAY=use_decay,
            BLOCK_DH=BDH, num_warps=4)
        # backward coefficients are recomputed in backward from (x, e, state, v, b)
        ctx.save_for_backward(cc, e, state if state is not None else None, vf, bf)
        ctx.has_init = state is not None
        ctx.vw = vf.view(D, H) if use_decay else None
        ctx.bw = bf.view(D, H) if use_decay else None
        out = e.to(x.dtype)
        return out, e[:, -1].to(torch.float32)

    @staticmethod
    def backward(ctx, ge, gs=None):
        # the returned final state is e[:, -1]; fold its gradient into the last position
        ge = ge.float()
        if gs is not None:
            ge = ge.clone()
            ge[:, -1] += gs.float()
        x, e, state, vf, bf = ctx.saved_tensors
        B, T, D, H = x.shape
        gef = ge.float().contiguous()
        xf = x.float()
        if ctx.vw is not None:
            pre = xf * ctx.vw.view(1, 1, D, H) + ctx.bw.view(1, 1, D, H)
            g = torch.exp(-torch.nn.functional.softplus(pre))
            sig = torch.sigmoid(pre)
        else:
            g = torch.ones_like(xf)
            sig = torch.zeros_like(xf)
        # -1e30 (not -inf): keeps the coefficient products finite
        rp0 = torch.full((B, 1, D, H), -1e30, dtype=torch.float32, device=x.device) \
            if state is None else state.float().view(B, 1, D, H)
        rp = torch.cat([rp0, e.float()[:, :T - 1]], dim=1)
        xwin = (xf > g * rp).float()
        if state is None:
            xwin[:, 0] = 1.0
        rho = ((1.0 - xwin) * g).contiguous()
        kap = (rho * sig).contiguous()
        kapx = (kap * xf).contiguous()

        def tl_layout(t):
            return t.permute(0, 2, 1, 3).contiguous()   # [B, D, T, H]

        rp, mx, rho, kap, kapx = map(tl_layout, (rp, xwin.contiguous(), rho, kap, kapx))
        ef = tl_layout(e.float())
        gc = torch.empty(B, D, T, H, dtype=torch.float32, device=x.device)
        gd = torch.empty_like(gc)
        gdx = torch.empty_like(gc)
        r0 = torch.empty(B * D * H, dtype=torch.float32, device=x.device)
        BDH = triton.next_power_of_2(D)
        decay_max_scan_bwd_kernel[(B * H,)](
            tl_layout(gef), rp, mx, rho, kap, kapx, vf, bf,
            gc, gd, gdx, r0, T, D, H, *ef.stride(),
            BLOCK_DH=BDH, num_warps=4)
        gv = gdx.sum(dim=(0, 2)).reshape(-1) if ctx.needs_input_grad[2] else None
        gb = gd.sum(dim=(0, 2)).reshape(-1) if ctx.needs_input_grad[3] else None
        gx = gc.transpose(1, 2).contiguous().to(x.dtype) if ctx.needs_input_grad[0] else None
        gs = None
        if ctx.has_init and ctx.needs_input_grad[1]:
            gs = (tl_layout(rho)[:, :, 0, :] * r0.view(B, D, H)).to(x.dtype)
        return gx, gs, gv, gb


TRITON_AVAILABLE = True


def triton_decay_max_scan(x, decay_v, decay_b, initial_state):
    B, T, D, H = x.shape
    vf, bf = _flat_params(decay_v, decay_b, D, H)
    return DecayMaxScanFn.apply(x, initial_state, vf, bf)
