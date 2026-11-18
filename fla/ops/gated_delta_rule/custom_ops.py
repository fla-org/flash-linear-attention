# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

r"""
torch.library custom operators for the chunked gated delta rule, callable inside
``torch.compile(fullgraph=True)`` without a graph break.

The public :func:`fla.ops.gated_delta_rule.chunk_gated_delta_rule` is ``torch.compiler.disable``'d and is a
``torch.autograd.Function``; Dynamo graph-breaks on it, and it has no forward-mode rule. The operators here are
opaque to Dynamo/Inductor (each runs the library's Triton kernels eagerly) and carry fake (meta) implementations,
so a compiled model traces through them:

- ``fla::chunk_gated_delta_rule``      forward; its backward is registered with ``torch.library.register_autograd``
                                       and is itself the operator ``fla::chunk_gated_delta_rule_bwd``
- ``fla::chunk_gated_delta_rule_jvp``  forward-mode derivative along D directions, returning (o, o_dot)

Restrictions (shared by both): no initial / final state, no variable lengths, no GVA, no in-kernel l2 norm,
gate or beta activation; ``g`` is the per-step log decay (not cumulative).
"""

import torch

from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd, chunk_gated_delta_rule_fwd
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.gated_delta_rule.chunk_jvp import chunk_gated_delta_rule_jvp as _chunk_gated_delta_rule_jvp
from fla.ops.gated_delta_rule.chunk_jvp import chunk_gated_delta_rule_jvp_ref
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2

__all__ = ['chunk_gated_delta_rule_jvp_op', 'chunk_gated_delta_rule_op']


def _check(q, k, v, g, beta, chunk_size):
    B, T, H, K = q.shape
    torch._check(k.shape == q.shape, lambda: "k must have the shape of q")
    torch._check(v.dim() == 4 and tuple(v.shape[:3]) == (B, T, H), lambda: "v must be [B, T, H, V] (no GVA)")
    torch._check(tuple(g.shape) == (B, T, H) and tuple(beta.shape) == (B, T, H), lambda: "g, beta must be [B, T, H]")
    torch._check(chunk_size in (16, 32, 64), lambda: "chunk_size must be 16, 32 or 64")


#####################################################################################################
# forward + backward
#####################################################################################################

@torch.library.custom_op('fla::chunk_gated_delta_rule', mutates_args=(), device_types='cuda')
def _gdn_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    scale: float, chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Returns (o [B, T, H, V] in q.dtype, A [B, T, H, BT], g_cum [B, T, H] fp32); A and g_cum feed the backward."""
    _check(q, k, v, g, beta, chunk_size)
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    g_cum, o, A, _, _, _ = chunk_gated_delta_rule_fwd(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=None, output_final_state=False, chunk_size=chunk_size,
    )
    return o.to(q.dtype), A, g_cum


@_gdn_fwd.register_fake
def _(q, k, v, g, beta, scale, chunk_size):
    B, T, H, _ = q.shape
    o = v.new_empty(v.shape, dtype=q.dtype)
    A = k.new_empty(B, T, H, chunk_size)
    g_cum = g.new_empty(B, T, H, dtype=torch.float32)
    return o, A, g_cum


@torch.library.custom_op('fla::chunk_gated_delta_rule_bwd', mutates_args=(), device_types='cuda')
def _gdn_bwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor,
    do: torch.Tensor, scale: float, chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Gradients (dq, dk, dv, dg, dbeta) of the forward from the saved A and g_cum (log2-unit cumulative gate)."""
    do = do.contiguous()
    dq, dk, dv, db, dg, _, _, _ = chunk_gated_delta_rule_bwd(
        q=q, k=k, v=v, g=g_cum, beta=beta, A=A, scale=scale,
        initial_state=None, do=do, dht=None, chunk_size=chunk_size,
    )
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg.to(g_cum.dtype), db.to(beta.dtype)


@_gdn_bwd.register_fake
def _(q, k, v, g_cum, beta, A, do, scale, chunk_size):
    return tuple(x.new_empty(x.shape) for x in (q, k, v, g_cum, beta))


def _gdn_setup_context(ctx, inputs, output):
    q, k, v, g, beta, scale, chunk_size = inputs
    _, A, g_cum = output
    # the backward kernels take contiguous inputs; the forward made the same copies, if any
    q, k, v, beta = (x.contiguous() for x in (q, k, v, beta))
    ctx.save_for_backward(q, k, v, beta, A, g_cum)
    ctx.g_dtype = g.dtype
    ctx.scale = scale
    ctx.chunk_size = chunk_size
    ctx.mark_non_differentiable(A, g_cum)


def _gdn_backward(ctx, do, dA, dg_cum):
    q, k, v, beta, A, g_cum = ctx.saved_tensors
    dq, dk, dv, dg, db = torch.ops.fla.chunk_gated_delta_rule_bwd(q, k, v, g_cum, beta, A, do, ctx.scale, ctx.chunk_size)
    return dq, dk, dv, dg.to(ctx.g_dtype), db, None, None


torch.library.register_autograd('fla::chunk_gated_delta_rule', _gdn_backward, setup_context=_gdn_setup_context)


def chunk_gated_delta_rule_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    scale: float | None = None, chunk_size: int = 64,
) -> torch.Tensor:
    r"""
    Compilable ``chunk_gated_delta_rule(q, k, v, g, beta, scale)[0]`` with autograd.

    Args: q, k [B, T, H, K]; v [B, T, H, V]; g [B, T, H] per-step log decay; beta [B, T, H].
    Returns: o [B, T, H, V] in q.dtype.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    o, _, _ = torch.ops.fla.chunk_gated_delta_rule(q, k, v, g, beta, float(scale), int(chunk_size))
    return o


#####################################################################################################
# forward-mode derivative
#####################################################################################################

@torch.library.custom_op('fla::chunk_gated_delta_rule_jvp', mutates_args=(), device_types='cuda')
def _gdn_jvp(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    q_dot: torch.Tensor, k_dot: torch.Tensor, v_dot: torch.Tensor, g_dot: torch.Tensor, beta_dot: torch.Tensor,
    scale: float, chunk_size: int, input_precision: str | None, fuse_output: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""(o [B, T, H, V], o_dot [D, B, T, H, V]); see :func:`fla.ops.gated_delta_rule.chunk_jvp.chunk_gated_delta_rule_jvp`."""
    _check(q, k, v, g, beta, chunk_size)
    return _chunk_gated_delta_rule_jvp(
        q, k, v, g, beta, q_dot, k_dot, v_dot, g_dot, beta_dot,
        scale=scale, chunk_size=chunk_size, input_precision=input_precision, fuse_output=fuse_output,
    )


@_gdn_jvp.register_fake
def _(q, k, v, g, beta, q_dot, k_dot, v_dot, g_dot, beta_dot, scale, chunk_size, input_precision, fuse_output=True):
    D = q_dot.shape[0]
    o = v.new_empty(v.shape, dtype=q.dtype)
    o_dot = v.new_empty((D, *v.shape))
    return o, o_dot


#####################################################################################################
# reverse mode through the forward-mode derivative
#
# o_dot[d] = J(x) x_dot[d] is linear in the tangents, so the gradient of <lam_dot, o_dot> with respect to x_dot[d]
# is J(x)^T lam_dot[d]: the library's backward with do = lam_dot[d] (exact and fast; the D directions are folded
# into the batch). The gradient with respect to the primal x is J(x)^T lam (from o's cotangent lam: the library's
# backward) plus the second-order term d/dx <lam_dot[d], J(x) x_dot[d]>, which this file computes by reverse-mode
# autograd through the fp32 PyTorch chunked reference (exact up to fp32 rounding, but slow and memory hungry).
#####################################################################################################

@torch.library.custom_op('fla::chunk_gated_delta_rule_vjp', mutates_args=(), device_types='cuda')
def _gdn_vjp(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, do: torch.Tensor,
    scale: float, chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    J^T do for each leading slice of ``do``: q, k, v, g, beta are [B, T, H, *]; do is [B, T, H, V] or [D, B, T, H, V]
    (then the D directions are folded into the batch and the gradients are [D, B, T, H, *]).
    """
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    lead = do.shape[:-4]
    D = do.shape[0] if lead else 1
    B, T, H, K = q.shape
    g_cum = chunk_local_cumsum(g, chunk_size=chunk_size, scale=RCP_LN2)
    _, _, A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g_cum, beta=beta, chunk_size=chunk_size)
    if D > 1 or lead:
        def rep(x):
            # a materialized copy: reshape of an expand can return a stride-0 view, and the kernels ignore strides
            return x.unsqueeze(0).expand(D, *x.shape).reshape(D * B, *x.shape[1:]).contiguous()
        q, k, v, beta, g_cum, A = (rep(x) for x in (q, k, v, beta, g_cum, A))
    do = do.reshape(D * B, T, H, do.shape[-1]).contiguous()
    dq, dk, dv, db, dg, _, _, _ = chunk_gated_delta_rule_bwd(
        q=q, k=k, v=v, g=g_cum, beta=beta, A=A, scale=scale,
        initial_state=None, do=do, dht=None, chunk_size=chunk_size,
    )
    out = (dq, dk, dv, dg, db)
    return tuple(x.reshape(*lead, B, *x.shape[1:]).to(dt) for x, dt in zip(out, (q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype)))


@_gdn_vjp.register_fake
def _(q, k, v, g, beta, do, scale, chunk_size):
    lead = tuple(do.shape[:-4])
    return tuple(x.new_empty((*lead, *x.shape)) for x in (q, k, v, g, beta))


@torch.library.custom_op('fla::chunk_gated_delta_rule_jvp_primal_vjp_ref', mutates_args=(), device_types='cuda')
def _gdn_jvp_primal_vjp_ref(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    q_dot: torch.Tensor, k_dot: torch.Tensor, v_dot: torch.Tensor, g_dot: torch.Tensor, beta_dot: torch.Tensor,
    do_dot: torch.Tensor, scale: float, chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Second-order term d/dx sum_d <do_dot[d], J(x) x_dot[d]> by autograd through the fp32 PyTorch reference."""
    # an operator's kernel runs below the autograd dispatch keys (and with grad mode off): re-enable both so that
    # torch.autograd.grad can differentiate the reference here
    exclude = torch._C._dispatch_tls_local_exclude_set()
    for key in (torch._C.DispatchKey.AutogradFunctionality, torch._C.DispatchKey.AutogradOther,
                torch._C.DispatchKey.AutogradNestedTensor):
        exclude = exclude.remove(key)
    with torch._C._ForceDispatchKeyGuard(torch._C._dispatch_tls_local_include_set(), exclude), torch.enable_grad():
        xs = [x.detach().float().requires_grad_() for x in (q, k, v, g, beta)]
        tans = [t.detach().float() for t in (q_dot, k_dot, v_dot, g_dot, beta_dot)]
        _, o_dot = chunk_gated_delta_rule_jvp_ref(*xs, *tans, scale=scale, chunk_size=chunk_size)
        grads = torch.autograd.grad(o_dot, xs, do_dot.float())
    return tuple(gr.to(x.dtype).contiguous() for gr, x in zip(grads, (q, k, v, g, beta)))


@_gdn_jvp_primal_vjp_ref.register_fake
def _(q, k, v, g, beta, q_dot, k_dot, v_dot, g_dot, beta_dot, do_dot, scale, chunk_size):
    # contiguous, like the real outputs (empty_like would copy the strides of possibly strided inputs)
    return tuple(x.new_empty(x.shape) for x in (q, k, v, g, beta))


def _jvp_setup_context(ctx, inputs, output):
    q, k, v, g, beta, q_dot, k_dot, v_dot, g_dot, beta_dot, scale, chunk_size = inputs[:12]
    ctx.save_for_backward(q, k, v, g, beta, q_dot, k_dot, v_dot, g_dot, beta_dot)
    ctx.scale = scale
    ctx.chunk_size = chunk_size


def _jvp_backward(ctx, do, do_dot):
    q, k, v, g, beta, q_dot, k_dot, v_dot, g_dot, beta_dot = ctx.saved_tensors
    needs = ctx.needs_input_grad
    grads = [None] * 14
    if do_dot is not None and any(needs[5:10]):
        grads[5:10] = torch.ops.fla.chunk_gated_delta_rule_vjp(q, k, v, g, beta, do_dot, ctx.scale, ctx.chunk_size)
    if any(needs[:5]):
        primal = [torch.zeros_like(x) for x in (q, k, v, g, beta)]
        if do is not None:
            primal = list(torch.ops.fla.chunk_gated_delta_rule_vjp(q, k, v, g, beta, do, ctx.scale, ctx.chunk_size))
        if do_dot is not None:
            second = torch.ops.fla.chunk_gated_delta_rule_jvp_primal_vjp_ref(
                q, k, v, g, beta, q_dot, k_dot, v_dot, g_dot, beta_dot, do_dot, ctx.scale, ctx.chunk_size)
            primal = [a + b for a, b in zip(primal, second)]
        grads[:5] = primal
    return tuple(grads)


torch.library.register_autograd('fla::chunk_gated_delta_rule_jvp', _jvp_backward, setup_context=_jvp_setup_context)


def chunk_gated_delta_rule_jvp_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    q_dot: torch.Tensor, k_dot: torch.Tensor, v_dot: torch.Tensor, g_dot: torch.Tensor, beta_dot: torch.Tensor,
    scale: float | None = None, chunk_size: int = 64, input_precision: str | None = None, fuse_output: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compilable forward-mode derivative of ``chunk_gated_delta_rule`` along D directions, with a registered
    reverse mode (see the section comment above for its cost).

    Args: primal q, k [B, T, H, K]; v [B, T, H, V]; g, beta [B, T, H]; tangents with a leading D:
    q_dot [D, B, T, H, K], ..., g_dot [D, B, T, H]. ``input_precision``: 'ieee' / 'tf32' / 'tf32x3' / None.
    ``fuse_output``: see :func:`fla.ops.gated_delta_rule.chunk_jvp.chunk_gated_delta_rule_jvp`.
    Returns: (o [B, T, H, V], o_dot [D, B, T, H, V]).
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return torch.ops.fla.chunk_gated_delta_rule_jvp(
        q, k, v, g, beta, q_dot, k_dot, v_dot, g_dot, beta_dot, float(scale), int(chunk_size), input_precision,
        bool(fuse_output),
    )
