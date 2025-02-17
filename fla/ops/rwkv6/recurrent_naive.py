# -*- coding: utf-8 -*-

from typing import Tuple, Optional

import torch
from fla.utils import autocast_custom_fwd, autocast_custom_bwd


def naive_recurrent_rwkv6(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False
):
    orig_dtype = q.dtype
    B, H, T, K, V = *q.shape, v.shape[-1]
    q, k, v, w, u = map(lambda x: x.float(), (q, k, v, w, u))
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    o = torch.zeros_like(v)

    if scale is None:
        scale = K ** -0.5

    if initial_state is not None:
        h += initial_state

    for i in range(T):
        q_i = q[:, :, i, :] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        w_i = w[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u[None, ..., None] * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * w_i[..., None] + kv_i
    ht = h if output_final_state else None
    return o.to(orig_dtype), ht


@torch.no_grad
@torch.jit.script
def naive_recurrent_rwkv6_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None
):
    q, k, v, w, u, o, do = (x.to(dtype=torch.float32) for x in (q, k, v, w, u, o, do))
    B, H, T, K, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1]
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    dq = torch.zeros_like(q)
    dq_aux = torch.zeros_like(q)

    if initial_state is not None:
        h += initial_state

    for i in range(T):
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        w_i = w[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        h_i = (h + u[None, ..., None] * kv_i)
        dq_i = (do[:, :, i, None, :] * h_i).sum(-1)
        dq_aux_i = (do[:, :, i, None, :] * h).sum(-1)
        dq[:, :, i] = dq_i
        dq_aux[:, :, i] = dq_aux_i
        h = h * w_i[..., None] + kv_i

    du = torch.zeros_like(u)
    dh = torch.zeros_like(h)
    dk = torch.zeros_like(k)
    dk_aux = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    for i in range(T - 1, -1, -1):
        d_kv_i = do[:, :, i, None, :] * q[:, :, i, :, None]
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        du_i = (d_kv_i * k_i[..., None] * v_i[..., None, :]).sum(-1)
        du += du_i.sum(0)
        dk_i = (dh * v_i[..., None, :]).sum(-1)
        dk_aux[:, :, i] = dk_i
        dk_i += (d_kv_i * u[None, ..., None] * v_i[..., None, :]).sum(-1)
        dv_i = (d_kv_i * u[None, ..., None] * k_i[..., None]).sum(-2)
        dv_i += (dh * k_i[..., None]).sum(-2)

        dk[:, :, i] = dk_i
        dv[:, :, i] = dv_i
        dh = dh * w[:, :, i, :, None].exp() + d_kv_i

    # dw = q * dq_aux - k * dk_aux
    dw = torch.zeros_like(w)
    for i in range(T - 2, -1, -1):
        dw[:, :, i] = dw[:, :, i+1] + dq_aux[:, :, i+1] * q[:, :, i+1] - dk_aux[:, :, i] * k[:, :, i]

    return dq, dk, dv, dw, du, dh


class NativeRecurrentRWKV6Function(torch.autograd.Function):
    @staticmethod
    @autocast_custom_fwd
    def forward(ctx, q, k, v, w, u, scale, initial_state, output_final_state: bool = False):
        o, ht = naive_recurrent_rwkv6(q, k, v, w, u, scale, initial_state, output_final_state)
        if initial_state is not None:
            initial_state = initial_state.clone()

        ctx.save_for_backward(q, k, v, w, u, o, initial_state)
        ctx.scale = scale
        return o, ht

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, w, u, o, initial_state = ctx.saved_tensors
        dq, dk, dv, dw, du, dh = naive_recurrent_rwkv6_bwd(q, k, v, w, u, o, do, dht, initial_state, ctx.scale)
        dh = dh if initial_state is not None else None
        return dq, dk, dv, dw, du, None, dh, None


def native_recurrent_rwkv6(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        r (torch.Tensor):
            reception of shape `(B, H, T, K)`. Alias: q, query in linear attention.
        k (torch.Tensor):
            keys of shape `(B, H, T, K)`
        v (torch.Tensor):
            values of shape `(B, H, T, V)`
        w (torch.Tensor):
            data-dependent decays of shape `(B, H, T, K)` in log space! Alias: g.
        u (torch.Tensor):
            bonus of shape `(H, K)` or `(B, H, K)` for each head.
        scale (Optional[int]):
            Scale factor for the RWKV6 attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `(B, H, K, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `(B, H, K, V)`. Default: `False`.
    """
    if scale == -1.0:
        scale = r.shape[-1] ** -0.5

    assert cu_seqlens is None, "cu_seqlens is not supported in the native implementation."
    assert head_first, "head_first=False is not supported in the native implementation."

    o, final_state = NativeRecurrentRWKV6Function.apply(r, k, v, w, u, scale, initial_state, output_final_state)

    return o, final_state
