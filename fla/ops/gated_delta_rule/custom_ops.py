# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

r"""
torch.library custom operators for the chunked gated delta rule, callable inside `torch.compile(fullgraph=True)`
without a graph break.

:func:`fla.ops.gated_delta_rule.chunk_gated_delta_rule` is an autograd Function under `torch.compiler.disable`,
so Dynamo graph-breaks on it, and it has no forward-mode rule. These operators run the library's Triton kernels
eagerly and carry fake implementations, so a compiled model traces through them:

- `fla::chunk_gated_delta_rule`: the forward, whose backward is the operator `fla::chunk_gated_delta_rule_bwd`;
- `fla::chunk_gated_delta_rule_jvp`: the forward-mode derivative along D directions, with the backward described
  in README.md.

Neither supports initial or final states, variable lengths, grouped value heads, or in-kernel l2 norm, gate or beta
activation, and `g` is the per-step log decay.
"""

import torch

from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd, chunk_gated_delta_rule_fwd
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.gated_delta_rule.chunk_jvp import chunk_gated_delta_rule_jvp, chunk_gated_delta_rule_jvp_ref
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.utils import input_guard

__all__ = ['chunk_gated_delta_rule_jvp_op', 'chunk_gated_delta_rule_op']

DEVICE_TYPES = ('cuda', 'xpu')


def _check(q, k, v, g, beta, chunk_size):
    B, T, H, K = q.shape
    torch._check(k.shape == q.shape, lambda: "k must have the shape of q")
    torch._check(v.dim() == 4 and tuple(v.shape[:3]) == (B, T, H), lambda: "v must be [B, T, H, V] (no GVA)")
    torch._check(tuple(g.shape) == (B, T, H) and tuple(beta.shape) == (B, T, H), lambda: "g, beta must be [B, T, H]")
    torch._check(chunk_size in (16, 32, 64), lambda: "chunk_size must be 16, 32 or 64")


@torch.library.custom_op('fla::chunk_gated_delta_rule', mutates_args=(), device_types=DEVICE_TYPES)
@input_guard
def _gdn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Returns o [B, T, H, V] in q.dtype, and A [B, T, H, BT] and g_cum [B, T, H] in fp32 for the backward."""
    _check(q, k, v, g, beta, chunk_size)
    g_cum, o, A, *_ = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        chunk_size=chunk_size,
    )
    return o.to(q.dtype), A, g_cum


@_gdn_fwd.register_fake
def _(q, k, v, g, beta, scale, chunk_size):
    B, T, H, _ = q.shape
    o = v.new_empty(v.shape, dtype=q.dtype)
    A = k.new_empty(B, T, H, chunk_size)
    g_cum = g.new_empty(B, T, H, dtype=torch.float32)
    return o, A, g_cum


@torch.library.custom_op('fla::chunk_gated_delta_rule_bwd', mutates_args=(), device_types=DEVICE_TYPES)
@input_guard
def _gdn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cum: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Gradients (dq, dk, dv, dg, dbeta) of the forward from its saved A and g_cum (log2-unit cumulative gate)."""
    dq, dk, dv, db, dg, _, _, _ = chunk_gated_delta_rule_bwd(
        q=q,
        k=k,
        v=v,
        g=g_cum,
        beta=beta,
        A=A,
        scale=scale,
        initial_state=None,
        do=do,
        dht=None,
        chunk_size=chunk_size,
    )
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg.to(g_cum.dtype), db.to(beta.dtype)


@_gdn_bwd.register_fake
def _(q, k, v, g_cum, beta, A, do, scale, chunk_size):
    return tuple(x.new_empty(x.shape) for x in (q, k, v, g_cum, beta))


def _gdn_setup_context(ctx, inputs, output):
    q, k, v, g, beta, scale, chunk_size = inputs
    _, A, g_cum = output
    ctx.save_for_backward(q, k, v, beta, A, g_cum)
    ctx.g_dtype = g.dtype
    ctx.scale = scale
    ctx.chunk_size = chunk_size
    ctx.mark_non_differentiable(A, g_cum)


def _gdn_backward(ctx, do, dA, dg_cum):
    q, k, v, beta, A, g_cum = ctx.saved_tensors
    dq, dk, dv, dg, db = torch.ops.fla.chunk_gated_delta_rule_bwd(
        q=q,
        k=k,
        v=v,
        g_cum=g_cum,
        beta=beta,
        A=A,
        do=do,
        scale=ctx.scale,
        chunk_size=ctx.chunk_size,
    )
    return dq, dk, dv, dg.to(ctx.g_dtype), db, None, None


torch.library.register_autograd('fla::chunk_gated_delta_rule', _gdn_backward, setup_context=_gdn_setup_context)


def chunk_gated_delta_rule_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    r"""
    Compilable `chunk_gated_delta_rule(q, k, v, g, beta, scale)[0]` with autograd.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            Per-step log decay of shape `[B, T, H]`.
        beta (torch.Tensor):
            Betas of shape `[B, T, H]`.
        scale (float, Optional):
            Scale of the attention scores. Default: `K ** -0.5`.
        chunk_size (int, Optional):
            Chunk size, one of 16, 32 and 64. Default: 64.

    Returns:
        o (torch.Tensor):
            Output of shape `[B, T, H, V]` in the dtype of `q`.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    # contiguous before the operator, so that the backward saves the copies the forward used rather than new ones
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    o, _, _ = torch.ops.fla.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=float(scale),
        chunk_size=int(chunk_size),
    )
    return o


@torch.library.custom_op('fla::chunk_gated_delta_rule_jvp', mutates_args=(), device_types=DEVICE_TYPES)
def _gdn_jvp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    q_dot: torch.Tensor,
    k_dot: torch.Tensor,
    v_dot: torch.Tensor,
    g_dot: torch.Tensor,
    beta_dot: torch.Tensor,
    scale: float,
    chunk_size: int,
    input_precision: str | None,
    fuse_output: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""(o [B, T, H, V], o_dot [D, B, T, H, V]) of :func:`fla.ops.gated_delta_rule.chunk_gated_delta_rule_jvp`."""
    _check(q, k, v, g, beta, chunk_size)
    return chunk_gated_delta_rule_jvp(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        q_dot=q_dot,
        k_dot=k_dot,
        v_dot=v_dot,
        g_dot=g_dot,
        beta_dot=beta_dot,
        scale=scale,
        chunk_size=chunk_size,
        input_precision=input_precision,
        fuse_output=fuse_output,
    )


@_gdn_jvp.register_fake
def _(q, k, v, g, beta, q_dot, k_dot, v_dot, g_dot, beta_dot, scale, chunk_size, input_precision, fuse_output=True):
    D = q_dot.shape[0]
    o = v.new_empty(v.shape, dtype=q.dtype)
    o_dot = v.new_empty((D, *v.shape))
    return o, o_dot


@torch.library.custom_op('fla::chunk_gated_delta_rule_vjp', mutates_args=(), device_types=DEVICE_TYPES)
@input_guard
def _gdn_vjp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    J^T do for each leading slice of `do`, which is [B, T, H, V] or [D, B, T, H, V].
    The D directions are folded into the batch, and the gradients have the leading dims of `do`.
    """
    lead = do.shape[:-4]
    D = do.shape[0] if lead else 1
    B, T, H, K = q.shape
    g_cum = chunk_local_cumsum(g=g, chunk_size=chunk_size, scale=RCP_LN2)
    _, _, A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g_cum, beta=beta, chunk_size=chunk_size)
    if lead:
        def rep(x):
            # a materialized copy: the reshape of an expand can return a stride-0 view, and the kernels ignore strides
            return x.unsqueeze(0).expand(D, *x.shape).reshape(D * B, *x.shape[1:]).contiguous()
        q, k, v, beta, g_cum, A = (rep(x) for x in (q, k, v, beta, g_cum, A))
    dq, dk, dv, db, dg, _, _, _ = chunk_gated_delta_rule_bwd(
        q=q,
        k=k,
        v=v,
        g=g_cum,
        beta=beta,
        A=A,
        scale=scale,
        initial_state=None,
        do=do.reshape(D * B, T, H, do.shape[-1]),
        dht=None,
        chunk_size=chunk_size,
    )
    dtypes = (q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype)
    return tuple(x.reshape(*lead, B, *x.shape[1:]).to(dt) for x, dt in zip((dq, dk, dv, dg, db), dtypes))


@_gdn_vjp.register_fake
def _(q, k, v, g, beta, do, scale, chunk_size):
    lead = tuple(do.shape[:-4])
    return tuple(x.new_empty((*lead, *x.shape)) for x in (q, k, v, g, beta))


@torch.library.custom_op('fla::chunk_gated_delta_rule_jvp_primal_vjp_ref', mutates_args=(), device_types=DEVICE_TYPES)
def _gdn_jvp_primal_vjp_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    q_dot: torch.Tensor,
    k_dot: torch.Tensor,
    v_dot: torch.Tensor,
    g_dot: torch.Tensor,
    beta_dot: torch.Tensor,
    do_dot: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Second-order term d/dx sum_d <do_dot[d], J(x) x_dot[d]> by autograd through the fp32 PyTorch reference."""
    # an operator's kernel runs below the autograd dispatch keys and with grad mode off: re-enable both so that
    # torch.autograd.grad can differentiate the reference here
    exclude = torch._C._dispatch_tls_local_exclude_set()
    for key in (torch._C.DispatchKey.AutogradFunctionality, torch._C.DispatchKey.AutogradOther,
                torch._C.DispatchKey.AutogradNestedTensor):
        exclude = exclude.remove(key)
    with torch._C._ForceDispatchKeyGuard(torch._C._dispatch_tls_local_include_set(), exclude), torch.enable_grad():
        xs = [x.detach().float().requires_grad_() for x in (q, k, v, g, beta)]
        tangents = [t.detach().float() for t in (q_dot, k_dot, v_dot, g_dot, beta_dot)]
        _, o_dot = chunk_gated_delta_rule_jvp_ref(*xs, *tangents, scale=scale, chunk_size=chunk_size)
        grads = torch.autograd.grad(o_dot, xs, do_dot.float())
    return tuple(gr.to(x.dtype).contiguous() for gr, x in zip(grads, (q, k, v, g, beta)))


@_gdn_jvp_primal_vjp_ref.register_fake
def _(q, k, v, g, beta, q_dot, k_dot, v_dot, g_dot, beta_dot, do_dot, scale, chunk_size):
    # contiguous, like the real outputs, whatever the strides of the inputs
    return tuple(x.new_empty(x.shape) for x in (q, k, v, g, beta))


def _jvp_setup_context(ctx, inputs, output):
    q, k, v, g, beta, q_dot, k_dot, v_dot, g_dot, beta_dot, scale, chunk_size = inputs[:12]
    ctx.save_for_backward(q, k, v, g, beta, q_dot, k_dot, v_dot, g_dot, beta_dot)
    ctx.scale = scale
    ctx.chunk_size = chunk_size
    # an unused output gets None rather than zeros, which skips its backward work, including the reference
    ctx.set_materialize_grads(False)


def _jvp_backward(ctx, do, do_dot):
    q, k, v, g, beta, q_dot, k_dot, v_dot, g_dot, beta_dot = ctx.saved_tensors
    needs = ctx.needs_input_grad
    grads = [None] * 14
    use_do = do is not None and any(needs[:5])
    use_do_dot = do_dot is not None and any(needs[5:10])
    if use_do or use_do_dot:
        # J^T do for the primal and J^T do_dot for the tangents in one backward call, as 1 + D directions
        if use_do and use_do_dot:
            cotangent = torch.cat((do.unsqueeze(0), do_dot))
        else:
            cotangent = do.unsqueeze(0) if use_do else do_dot
        vjps = torch.ops.fla.chunk_gated_delta_rule_vjp(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            do=cotangent,
            scale=ctx.scale,
            chunk_size=ctx.chunk_size,
        )
        if use_do:
            grads[:5] = [x[0] for x in vjps]
        if use_do_dot:
            grads[5:10] = [x[int(use_do):] for x in vjps]
    if do_dot is not None and any(needs[:5]):
        second = torch.ops.fla.chunk_gated_delta_rule_jvp_primal_vjp_ref(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            q_dot=q_dot,
            k_dot=k_dot,
            v_dot=v_dot,
            g_dot=g_dot,
            beta_dot=beta_dot,
            do_dot=do_dot,
            scale=ctx.scale,
            chunk_size=ctx.chunk_size,
        )
        grads[:5] = [a + b for a, b in zip(grads[:5], second)] if use_do else second
    return tuple(grads)


torch.library.register_autograd('fla::chunk_gated_delta_rule_jvp', _jvp_backward, setup_context=_jvp_setup_context)


def chunk_gated_delta_rule_jvp_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    q_dot: torch.Tensor,
    k_dot: torch.Tensor,
    v_dot: torch.Tensor,
    g_dot: torch.Tensor,
    beta_dot: torch.Tensor,
    scale: float | None = None,
    chunk_size: int = 64,
    input_precision: str | None = None,
    fuse_output: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compilable :func:`fla.ops.gated_delta_rule.chunk_gated_delta_rule_jvp` with a registered backward.
    Its gradient with respect to the primal inputs goes through the fp32 PyTorch reference (see README.md),
    which is slower and uses more memory than the kernels.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            Per-step log decay of shape `[B, T, H]`.
        beta (torch.Tensor):
            Betas of shape `[B, T, H]`.
        q_dot (torch.Tensor):
            Tangent of `q` of shape `[D, B, T, H, K]`.
        k_dot (torch.Tensor):
            Tangent of `k` of shape `[D, B, T, H, K]`.
        v_dot (torch.Tensor):
            Tangent of `v` of shape `[D, B, T, H, V]`.
        g_dot (torch.Tensor):
            Tangent of `g` of shape `[D, B, T, H]`.
        beta_dot (torch.Tensor):
            Tangent of `beta` of shape `[D, B, T, H]`.
        scale (float, Optional):
            Scale of the attention scores. Default: `K ** -0.5`.
        chunk_size (int, Optional):
            Chunk size, one of 16, 32 and 64. Default: 64.
        input_precision (str, Optional):
            Precision of the fp32 `tl.dot` in the JVP kernels: `'ieee'`, `'tf32'`, `'tf32x3'` or `None`,
            as in :func:`fla.ops.gated_delta_rule.chunk_gated_delta_rule_jvp`. Default: `None`.
        fuse_output (bool, Optional):
            As in :func:`fla.ops.gated_delta_rule.chunk_gated_delta_rule_jvp`. Default: `True`.

    Returns:
        o (torch.Tensor):
            Output of shape `[B, T, H, V]`.
        o_dot (torch.Tensor):
            Derivative of `o` along each direction, of shape `[D, B, T, H, V]`.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    # contiguous before the operator, so that the backward saves the copies the forward used rather than new ones
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    return torch.ops.fla.chunk_gated_delta_rule_jvp(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        q_dot=q_dot,
        k_dot=k_dot,
        v_dot=v_dot,
        g_dot=g_dot,
        beta_dot=beta_dot,
        scale=float(scale),
        chunk_size=int(chunk_size),
        input_precision=input_precision,
        fuse_output=bool(fuse_output),
    )
