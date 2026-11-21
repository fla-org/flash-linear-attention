# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"FP64 mathematical, storage-aware and metadata references for PHBA."

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from fla.ops.phba.routing import RoutingLayout


@dataclass(frozen=True)
class HostLayout:
    """Host-side block boundaries for the independent PHBA references."""

    chunk_size: int
    block_ptr: tuple[int, ...]
    seq_block_ptr: tuple[int, ...]

    @property
    def Nc(self) -> int:  # noqa: N802
        return len(self.block_ptr) - 1

    @property
    def Nseq(self) -> int:  # noqa: N802
        return len(self.seq_block_ptr) - 1

    @classmethod
    def from_lengths(cls, lengths, chunk_size: int) -> HostLayout:
        block_ptr, seq_block_ptr, t = [0], [0], 0
        for length in lengths:
            n = -(-length // chunk_size)
            for j in range(n):
                t += min(chunk_size, length - j * chunk_size)
                block_ptr.append(t)
            seq_block_ptr.append(seq_block_ptr[-1] + n)
        return cls(chunk_size, tuple(block_ptr), tuple(seq_block_ptr))

    def blocks_of_sequence(self, s: int) -> range:
        return range(self.seq_block_ptr[s], self.seq_block_ptr[s + 1])

    def token_block(self, device=None) -> torch.Tensor:
        lens = [self.block_ptr[i + 1] - self.block_ptr[i] for i in range(self.Nc)]
        return torch.repeat_interleave(torch.arange(self.Nc, device=device), torch.tensor(lens, device=device))


def compute_dtype(storage: torch.dtype) -> torch.dtype:
    return torch.float64 if storage == torch.float64 else torch.float32


def _st(x: torch.Tensor, storage: torch.dtype) -> torch.Tensor:
    """Round through storage dtype before returning to the accumulation dtype."""
    return x.to(storage).to(x.dtype)


def local_cumsum(x: torch.Tensor, layout: HostLayout, *, reverse: bool = False, dtype: torch.dtype | None = None):
    cd = dtype or compute_dtype(x.dtype)
    out = torch.empty(x.shape, dtype=cd, device=x.device)
    for b in range(layout.Nc):
        r0, r1 = layout.block_ptr[b], layout.block_ptr[b + 1]
        seg = x[r0:r1].to(cd)
        out[r0:r1] = seg.flip(0).cumsum(0).flip(0) if reverse else seg.cumsum(0)
    return out


def prefix_forward(s, z, gk, layout: HostLayout):
    """Return storage-dtype prefixes [Nc, M, H, D] and accumulation-dtype final states [Nseq, H, M, D]."""
    storage, cd = z.dtype, gk.dtype
    T, H, M = s.shape
    D = z.shape[-1]
    out = torch.empty((layout.Nc, M, H, D), dtype=storage, device=z.device)
    finals = []
    for seq in range(layout.Nseq):
        S = torch.zeros((H, M, D), dtype=cd, device=z.device)
        for b in layout.blocks_of_sequence(seq):
            out[b] = S.permute(1, 0, 2).to(storage)
            r0, r1 = layout.block_ptr[b], layout.block_ptr[b + 1]
            g = gk[r0:r1]
            g_last = g[-1]
            S = S * torch.exp(g_last)[..., None]
            W = _st(s[r0:r1].to(cd) * torch.exp(g_last[None] - g), storage)
            S = S + torch.einsum("lhm,lhd->hmd", W, z[r0:r1].to(cd))
        finals.append(S)
    return out, torch.stack(finals, 0)


def prefix_backward_branch(s, z, gk, prefix, dprefix, layout: HostLayout):
    """Differentiate one prefix branch while preserving storage rounding of dot operands."""
    storage, cd = z.dtype, gk.dtype
    T, H, M = s.shape
    D = z.shape[-1]
    ds = torch.zeros((T, H, M), dtype=cd, device=z.device)
    dz = torch.zeros((T, H, D), dtype=cd, device=z.device)
    dg = torch.zeros((T, H, M), dtype=cd, device=z.device)
    for seq in range(layout.Nseq):
        grad_next = torch.zeros((H, M, D), dtype=cd, device=z.device)
        for b in reversed(layout.blocks_of_sequence(seq)):
            r0, r1 = layout.block_ptr[b], layout.block_ptr[b + 1]
            g = gk[r0:r1]
            g_last = g[-1]
            eg = torch.exp(g_last)
            k_f = s[r0:r1].to(cd)
            exp_diff = torch.exp(g_last[None] - g)
            gn_st = _st(grad_next, storage)
            v_val = z[r0:r1].to(cd)
            dot_lk = torch.einsum("lhd,hmd->lhm", v_val, gn_st)
            grad_k = exp_diff * dot_lk
            k_scaled = _st(k_f * exp_diff, storage)
            grad_v = torch.einsum("lhm,hmd->lhd", k_scaled, gn_st)
            grad_g = -(k_f * grad_k)
            sum_k_gradk = (k_f * grad_k).sum(0)
            h_val = prefix[b].to(cd).permute(1, 0, 2)
            dot_state = (h_val * grad_next).sum(-1)
            grad_g[-1] = grad_g[-1] + (eg * dot_state + sum_k_gradk)
            ds[r0:r1], dz[r0:r1], dg[r0:r1] = grad_k, grad_v, grad_g
            grad_next = dprefix[b].to(cd).permute(1, 0, 2) + eg[..., None] * grad_next
    return ds, dz, dg


class RefChunkwisePrefixFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s_val, k_raw, v, g_log, layout: HostLayout):
        cd = compute_dtype(k_raw.dtype)
        gk = local_cumsum(g_log, layout, dtype=cd)
        pk, _ = prefix_forward(s_val, k_raw, gk, layout)
        pv, _ = prefix_forward(s_val, v, gk, layout)
        ctx.layout = layout
        ctx.save_for_backward(s_val, k_raw, v, g_log, pk, pv)
        return pk, pv

    @staticmethod
    def backward(ctx, dpk, dpv):
        s_val, k_raw, v, g_log, pk, pv = ctx.saved_tensors
        layout = ctx.layout
        cd = compute_dtype(k_raw.dtype)
        gk = local_cumsum(g_log, layout, dtype=cd)
        dpk = torch.zeros_like(pk) if dpk is None else dpk
        dpv = torch.zeros_like(pv) if dpv is None else dpv
        ds_k, dk, dg_k = prefix_backward_branch(s_val, k_raw, gk, pk, dpk, layout)
        ds_v, dv, dg_v = prefix_backward_branch(s_val, v, gk, pv, dpv, layout)
        d_g_log = local_cumsum(dg_k + dg_v, layout, reverse=True, dtype=cd)
        return (ds_k + ds_v).to(s_val.dtype), dk.to(k_raw.dtype), dv.to(v.dtype), d_g_log.to(g_log.dtype), None


def route_masks(rtok: torch.Tensor, rstate: torch.Tensor, layout: HostLayout, h: int, M: int):
    """Build local [T, T], historical-token [T, T] and prefix-state [T, Nc * M] masks for one head."""
    T = rtok.shape[1]
    dev = rtok.device
    tb = layout.token_block(dev)
    idx = torch.arange(T, device=dev)
    local = (tb[:, None] == tb[None, :]) & (idx[None, :] <= idx[:, None])

    def block_mask(routes: torch.Tensor) -> torch.Tensor:
        # the final column collects invalid route sentinels
        m = torch.zeros((T, layout.Nc + 1), dtype=torch.bool, device=dev)
        ids = routes[h].long().clamp(min=-1)
        ids = torch.where(ids < 0, torch.full_like(ids, layout.Nc), ids)
        m.scatter_(1, ids, True)
        return m[:, : layout.Nc]

    token = block_mask(rtok)[:, tb] if rtok.shape[-1] > 0 else torch.zeros((T, T), dtype=torch.bool, device=dev)
    state = block_mask(rstate).repeat_interleave(M, dim=1)
    return local, token, state


def _masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x.masked_fill(~mask, float("-inf"))


def _rowmax(x: torch.Tensor) -> torch.Tensor:
    return x.max(dim=-1).values if x.shape[-1] else torch.full(x.shape[:-1], float("-inf"), dtype=x.dtype, device=x.device)


def _joint_forward_head(q, k, vv, qr, pk, pv, masks, scale, storage, local_seed=None):
    local, token, state = masks
    s_qk = (q @ k.T) * scale
    s_loc = _masked(s_qk, local)
    if local_seed is None:
        m_loc = _rowmax(s_loc)
        p_loc = torch.exp(s_loc - m_loc[:, None])
        l_loc = p_loc.sum(-1)
        o_loc = _st((_st(p_loc, storage) @ vv) / l_loc[:, None], storage)
        lse_loc = m_loc + torch.log(l_loc)
    else:
        o_loc, lse_loc = local_seed
    s_tok = _masked(s_qk, token)
    s_st = _masked((qr @ pk.T) * scale, state)
    m = torch.maximum(lse_loc, torch.maximum(_rowmax(s_tok), _rowmax(s_st)))
    p_tok = torch.exp(s_tok - m[:, None])
    p_st = torch.exp(s_st - m[:, None])
    z = torch.exp(lse_loc - m) + p_tok.sum(-1) + p_st.sum(-1)
    u = o_loc * torch.exp(lse_loc - m)[:, None] + _st(p_tok, storage) @ vv + _st(p_st, storage) @ pv
    return (u / z[:, None]).to(storage), m + torch.log(z), (s_loc, s_tok, s_st)


class RefJointAttentionFn(torch.autograd.Function):
    """Joint attention reference with storage-dtype inputs/output and FP32 log-normalizers."""

    @staticmethod
    def forward(
        ctx,
        q_rope,
        k_rope,
        v,
        q_raw,
        prefix_k,
        prefix_v,
        rtok,
        rstate,
        layout: HostLayout,
        scale: float,
        local_impl="torch",
        fa_segment_batch_max=65535,
        fa_rows_per_call_max=1 << 20,
        fa_deterministic=False,
    ):
        if local_impl not in ("torch", "flash_attn"):
            raise ValueError(f"unknown Reference A local_impl: {local_impl}")
        storage, cd = q_rope.dtype, compute_dtype(q_rope.dtype)
        T, H, D = q_rope.shape
        M = prefix_k.shape[1]
        ctx.local_impl, ctx.fa_deterministic = local_impl, fa_deterministic
        oloc = lloc = None
        extra = []
        if local_impl == "flash_attn":
            from types import SimpleNamespace

            from fla.ops.phba.flash_attention import ensure_ready, local_attention
            from fla.ops.phba.plans import _local_fa_batches

            if q_rope.device.type != "cuda" or storage not in (torch.float16, torch.bfloat16):
                raise ValueError("FlashAttention Reference A requires CUDA fp16/bf16 inputs")
            if any(
                (
                    x.device != q_rope.device
                    or x.dtype != storage
                    or x.shape != q_rope.shape
                    or (not x.is_contiguous())
                    or x.data_ptr() % 16
                    for x in (q_rope, k_rope, v)
                ),
            ):
                raise ValueError("FlashAttention Reference A requires matching aligned contiguous Q/K/V")
            if fa_segment_batch_max < 1 or fa_rows_per_call_max < 1:
                raise ValueError("FlashAttention batch limits must be positive")
            ensure_ready(q_rope.device, storage, D)
            block_ptr = torch.tensor(layout.block_ptr, device=q_rope.device, dtype=torch.int32)
            fa_layout = SimpleNamespace(chunk_size=layout.chunk_size, block_ptr_host=layout.block_ptr, block_ptr=block_ptr)
            ctx.batches = _local_fa_batches(layout.block_ptr, layout.chunk_size, fa_segment_batch_max, fa_rows_per_call_max)
            oloc, lloc = local_attention(
                q_rope,
                k_rope,
                v,
                fa_layout,
                scale=scale,
                fa_segment_batch_max=fa_segment_batch_max,
                fa_rows_per_call_max=fa_rows_per_call_max,
            )
            extra.append(block_ptr)
        out = torch.empty((T, H, D), dtype=storage, device=q_rope.device)
        lse = torch.empty((H, T), dtype=cd, device=q_rope.device)
        for h in range(H):
            masks = route_masks(rtok, rstate, layout, h, M)
            seed = None if oloc is None else (oloc[:, h].to(cd), lloc[h].to(cd))
            o, l, _ = _joint_forward_head(
                q_rope[:, h].to(cd),
                k_rope[:, h].to(cd),
                v[:, h].to(cd),
                q_raw[:, h].to(cd),
                prefix_k[:, :, h].reshape(-1, D).to(cd),
                prefix_v[:, :, h].reshape(-1, D).to(cd),
                masks,
                scale,
                storage,
                local_seed=seed,
            )
            out[:, h], lse[h] = o, l
        ctx.layout, ctx.scale = layout, scale
        ctx.save_for_backward(q_rope, k_rope, v, q_raw, prefix_k, prefix_v, rtok, rstate, out, lse, *extra)
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, d_out, _d_lse):
        q_rope, k_rope, v, q_raw, prefix_k, prefix_v, rtok, rstate, out, lse, *extra = ctx.saved_tensors
        layout, scale = ctx.layout, ctx.scale
        storage, cd = q_rope.dtype, compute_dtype(q_rope.dtype)
        T, H, D = q_rope.shape
        nc, M = prefix_k.shape[0], prefix_k.shape[1]
        grads = {n: torch.zeros((T, H, D), dtype=cd, device=q_rope.device) for n in ("dq", "dk", "dv", "dqr")}
        dpk = torch.zeros((nc, M, H, D), dtype=cd, device=q_rope.device)
        dpv = torch.zeros_like(dpk)
        d_out = d_out.to(storage).contiguous()
        local_grads = None
        if ctx.local_impl == "flash_attn":
            from fla.ops.phba.flash_attention import local_backward

            local_grads = local_backward(
                d_out,
                q_rope,
                k_rope,
                v,
                out,
                lse,
                extra[0],
                ctx.batches,
                deterministic=ctx.fa_deterministic,
                softmax_scale=scale,
            )
        for h in range(H):
            masks = route_masks(rtok, rstate, layout, h, M)
            q, k, vv, qr = (x[:, h].to(cd) for x in (q_rope, k_rope, v, q_raw))
            pk = prefix_k[:, :, h].reshape(-1, D).to(cd)
            pv = prefix_v[:, :, h].reshape(-1, D).to(cd)
            s_qk = (q @ k.T) * scale
            s_loc, s_tok = _masked(s_qk, masks[0]), _masked(s_qk, masks[1])
            s_st = _masked((qr @ pk.T) * scale, masks[2])
            L = lse[h]
            do = d_out[:, h].to(cd)
            delta = (do * out[:, h].to(cd)).sum(-1)
            do_st = _st(do, storage)

            def branch(s, q_in, k_in, v_in):
                p = torch.exp(s - L[:, None])
                dp = do_st @ v_in.T
                ds = p * (dp - delta[:, None])
                dq = (_st(ds, storage) @ k_in) * scale
                dk = (_st(ds, storage).T @ q_in) * scale
                dv = _st(p, storage).T @ do_st
                return dq, dk, dv

            if local_grads is None:
                dq_l, dk_l, dv_l = (_st(x, storage) for x in branch(s_loc, q, k, vv))
            else:
                dq_l, dk_l, dv_l = (x[:, h].to(cd) for x in local_grads)
            dq_t, dk_t, dv_t = branch(s_tok, q, k, vv)
            dqr, dpk_h, dpv_h = branch(s_st, qr, pk, pv)
            grads["dq"][:, h] = dq_l + dq_t
            grads["dk"][:, h] = dk_l + dk_t
            grads["dv"][:, h] = dv_l + dv_t
            grads["dqr"][:, h] = dqr
            dpk[:, :, h] = dpk_h.reshape(nc, M, D)
            dpv[:, :, h] = dpv_h.reshape(nc, M, D)
        result = (
            grads["dq"].to(storage),
            grads["dk"].to(storage),
            grads["dv"].to(storage),
            grads["dqr"].to(storage),
            dpk.to(storage),
            dpv.to(storage),
        )
        return result + (None,) * (len(ctx.needs_input_grad) - len(result))


def gate_from_storage(a_storage: torch.Tensor, tau: float, storage: torch.dtype):
    """Match the production gate formula and storage rounding while retaining autograd connectivity."""
    cd = compute_dtype(storage)
    g_log = F.logsigmoid(a_storage.to(cd)) / tau
    return g_log, (-torch.expm1(g_log)).to(storage)


def reference_a_chunkwise(
    q_raw,
    q_rope,
    k_rope,
    k_raw,
    v,
    a_storage,
    *,
    tau: float,
    layout: HostLayout,
    rtok: torch.Tensor,
    rstate: torch.Tensor,
    scale: float | None = None,
    local_impl="torch",
    fa_segment_batch_max=65535,
    fa_rows_per_call_max=1 << 20,
    fa_deterministic=False,
):
    """Return output, log-normalizers and key/value prefixes using the production storage boundaries.

    Autograd combines attention and prefix value gradients in storage dtype, matching the production path.
    """
    storage = v.dtype
    scale = 1.0 / math.sqrt(q_rope.shape[-1]) if scale is None else scale
    g_log, s_val = gate_from_storage(a_storage, tau, storage)
    pk, pv = RefChunkwisePrefixFn.apply(s_val, k_raw, v, g_log, layout)
    out, lse = RefJointAttentionFn.apply(
        q_rope,
        k_rope,
        v,
        q_raw,
        pk,
        pv,
        rtok,
        rstate,
        layout,
        scale,
        local_impl,
        fa_segment_batch_max,
        fa_rows_per_call_max,
        fa_deterministic,
    )
    return out, lse, pk, pv


@contextlib.contextmanager
def highest_precision():
    """Disable TF32 for the mathematical and routing references."""
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[0], old[1]
        torch.set_float32_matmul_precision(old[2])


def prefix_states_recurrent(k_raw, v, g_log, layout: HostLayout):
    """Compute differentiable prefixes [Nc, M, H, D] by token recurrence in the gate dtype."""
    T, H, M = g_log.shape
    D = k_raw.shape[-1]
    A = torch.exp(g_log)
    I = -torch.expm1(g_log)  # noqa: E741
    pk, pv = [None] * layout.Nc, [None] * layout.Nc
    for seq in range(layout.Nseq):
        Sk = g_log.new_zeros((H, M, D))
        Sv = g_log.new_zeros((H, M, D))
        for b in layout.blocks_of_sequence(seq):
            pk[b], pv[b] = Sk.permute(1, 0, 2), Sv.permute(1, 0, 2)
            for t in range(layout.block_ptr[b], layout.block_ptr[b + 1]):
                a_t, i_t = A[t][..., None], I[t][..., None]  # [H, M, 1]
                Sk = a_t * Sk + i_t * k_raw[t][:, None, :]
                Sv = a_t * Sv + i_t * v[t][:, None, :]
    return torch.stack(pk, 0), torch.stack(pv, 0)


def reference_b(
    q_raw,
    q_rope,
    k_rope,
    k_raw,
    v,
    a_storage,
    *,
    tau: float,
    layout: HostLayout,
    rtok: torch.Tensor,
    rstate: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    scale: float | None = None,
):
    """Return output, log-normalizers and key/value prefixes in the requested dtype, retaining autograd links."""
    with highest_precision():
        q_raw, q_rope, k_rope, k_raw, v, a = (x.to(dtype) for x in (q_raw, q_rope, k_rope, k_raw, v, a_storage))
        T, H, D = q_rope.shape
        scale = 1.0 / math.sqrt(D) if scale is None else scale
        g_log = F.logsigmoid(a) / tau
        pk, pv = prefix_states_recurrent(k_raw, v, g_log, layout)
        M = pk.shape[1]
        outs, lses = [], []
        for h in range(H):
            local, token, state = route_masks(rtok, rstate, layout, h, M)
            s_tt = (q_rope[:, h] @ k_rope[:, h].T) * scale
            s_tt = s_tt.masked_fill(~(local | token), float("-inf"))
            pk_h = pk[:, :, h].reshape(-1, D)
            pv_h = pv[:, :, h].reshape(-1, D)
            s_st = ((q_raw[:, h] @ pk_h.T) * scale).masked_fill(~state, float("-inf"))
            s = torch.cat([s_tt, s_st], dim=-1)
            lse = torch.logsumexp(s, dim=-1)
            p = torch.exp(s - lse[:, None])
            outs.append(p @ torch.cat([v[:, h], pv_h], dim=0))
            lses.append(lse)
        return torch.stack(outs, 1), torch.stack(lses, 0), pk, pv


def naive_phba(q, k, v, q_rope, k_rope, gate, lengths, token_routes, state_routes, chunk_size=128, tau=8.0):
    """Differentiable FP64 PHBA formula for frozen routes and small reference cases."""
    shape = q.shape
    heads, width = shape[-2:]
    q, k, v, q_rope, k_rope = (x.reshape(-1, heads, width) for x in (q, k, v, q_rope, k_rope))
    out, _, _, _ = reference_b(
        q_raw=q,
        q_rope=q_rope,
        k_rope=k_rope,
        k_raw=k,
        v=v,
        a_storage=gate.reshape(-1, heads, gate.shape[-1]),
        tau=tau,
        layout=HostLayout.from_lengths(lengths, chunk_size),
        rtok=token_routes,
        rstate=state_routes,
        dtype=torch.float64,
    )
    return out.reshape(shape)


def metadata_reference(route: torch.Tensor, nc: int) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    """Return counts, offsets and sorted row indices from an independent route-slot traversal."""
    H, T, K = route.shape
    count = torch.zeros(nc * H, dtype=torch.int64)
    rows: list[list[int]] = [[] for _ in range(nc * H)]
    r = route.cpu()
    for h in range(H):
        for t in range(T):
            for k in range(K):
                b = int(r[h, t, k])
                if b >= 0:
                    count[b * H + h] += 1
                    rows[b * H + h].append(t * H + h)
    offset = torch.zeros(nc * H + 1, dtype=torch.int64)
    offset[1:] = torch.cumsum(count, 0)
    return count.to(torch.int32), offset.to(torch.int32), [sorted(x) for x in rows]


def centroid_reference(k_rope: torch.Tensor, block_ptr_host) -> torch.Tensor:
    """Average each block in FP32; its reduction order can differ from the production kernel."""
    T, H, D = k_rope.shape
    nc = len(block_ptr_host) - 1
    out = torch.zeros((nc, H, D), dtype=torch.float32, device=k_rope.device)
    for b in range(nc):
        a, e = block_ptr_host[b], block_ptr_host[b + 1]
        if e > a:
            out[b] = k_rope[a:e].float().sum(0) / (e - a)
    return out


def _history_mask(layout: RoutingLayout, device) -> torch.Tensor:
    """Mask blocks outside the query sequence or at/after the current block."""
    cb = layout.current_block.to(device).long()
    bs = layout.block_seq.to(device).long()
    b = torch.arange(layout.Nc, device=device)
    return (bs[None, :] == bs[cb][:, None]) & (b[None, :] < cb[:, None])


def routing_scores(q_rope: torch.Tensor, centroids: torch.Tensor, layout: RoutingLayout):
    """Return scores [H, T, Nc], masking invalid candidates with negative infinity.

    Round centroids to query dtype before an FP64 dot and FP32 output conversion.
    FP16/BF16 products are exact in FP32/FP64; reduction order can still differ from the kernel.
    """
    dev = q_rope.device
    mask = _history_mask(layout, dev)
    q = q_rope.detach().to(torch.float64)
    c = centroids.to(q_rope.dtype).to(torch.float64)
    s = torch.einsum("thd,bhd->htb", q, c).to(torch.float32)
    return s.masked_fill(~mask[None], float("-inf"))


def topk_from_scores(scores: torch.Tensor, kh: int) -> torch.Tensor:
    """Sort scores descending with lower block IDs breaking ties; map invalid scores to -1."""
    H, T, nc = scores.shape
    if kh == 0 or nc == 0:
        return torch.full((H, T, kh), -1, dtype=torch.int32, device=scores.device)
    vals, idx = torch.sort(scores, dim=-1, descending=True, stable=True)
    k = min(kh, nc)
    ids = torch.where(vals[..., :k] > float("-inf"), idx[..., :k], torch.full_like(idx[..., :k], -1))
    out = torch.full((H, T, kh), -1, dtype=torch.int32, device=scores.device)
    out[..., :k] = ids.to(torch.int32)
    return out


def rstate_from_rtok(rtok: torch.Tensor, layout: RoutingLayout) -> torch.Tensor:
    H, T, kh = rtok.shape
    pv = layout.prefix_valid.to(rtok.device)
    cb = layout.current_block.to(rtok.device).long()
    cur = torch.where(pv[cb], cb, torch.full_like(cb, -1)).to(torch.int32).view(1, T, 1).expand(H, T, 1)
    hist = torch.where((rtok >= 0) & pv[rtok.long().clamp(min=0)], rtok, torch.full_like(rtok, -1))
    return torch.cat([cur, hist], dim=-1).contiguous()


def route_reference(q_rope, centroids, layout: RoutingLayout, kh: int):
    """Return token routes, state routes and scores using production-matched operand rounding."""
    scores = routing_scores(q_rope, centroids, layout)
    rtok = topk_from_scores(scores, kh)
    return rtok, rstate_from_rtok(rtok, layout), scores


__all__ = [
    "HostLayout",
    "RefChunkwisePrefixFn",
    "RefJointAttentionFn",
    "centroid_reference",
    "compute_dtype",
    "gate_from_storage",
    "highest_precision",
    "local_cumsum",
    "metadata_reference",
    "naive_phba",
    "prefix_backward_branch",
    "prefix_forward",
    "prefix_states_recurrent",
    "reference_a_chunkwise",
    "reference_b",
    "route_masks",
    "route_reference",
    "routing_scores",
    "rstate_from_rtok",
    "topk_from_scores",
]
