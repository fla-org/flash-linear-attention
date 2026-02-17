from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.utils import autotune_cache_kwargs, input_guard


def _zero_output(nargs):
    nargs['o'].zero_()


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BP': BP}, num_warps=num_warps, pre_hook=_zero_output)
        for BP in [32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['P'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_linoss_fwd_kernel(
    x,          # input: (B, T, H)
    B_re,       # B real part: (P, H)
    B_im,       # B imag part: (P, H)
    C_re,       # C real part: (H, P)
    C_im,       # C imag part: (H, P)
    a_diag,     # diagonal A: (P,)
    dt,         # discretization steps: (P,)
    d_skip,     # skip connection: (H,)
    o,          # output: (B, T, H)
    h0,         # initial state: (N, 2, P) or None
    ht,         # final state: (N, 2, P) or None
    T,
    H: tl.constexpr,
    P: tl.constexpr,
    BP: tl.constexpr,
    IS_IMEX: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    """Fused recurrent forward for LinOSS.

    Each program handles one batch element and a block of P state dimensions.
    We iterate over time steps sequentially, computing:
      Bu_t = B @ u_t  (complex, shape P)
      Then the 2x2 recurrence per state dim, and finally y_t = Re(C @ h2_t) + D * u_t.

    Since B and C are complex but stored as separate real/imag, we compute
    Bu_re and Bu_im separately.
    """
    i_p, i_b = tl.program_id(0), tl.program_id(1)

    o_p = i_p * BP + tl.arange(0, BP)
    mask_p = o_p < P

    b_a = tl.load(a_diag + o_p, mask=mask_p, other=0.).to(tl.float32)
    b_a = tl.maximum(b_a, tl.zeros_like(b_a))  # relu(A)
    b_dt = tl.load(dt + o_p, mask=mask_p, other=0.).to(tl.float32)
    b_dt = tl.sigmoid(b_dt)  # sigmoid(steps)

    if IS_IMEX:
        b_M11 = tl.full([BP], 1.0, dtype=tl.float32)
        b_M12 = -b_dt * b_a
        b_M21 = b_dt + tl.zeros([BP], dtype=tl.float32)
        b_M22 = 1.0 - b_dt * b_dt * b_a
    else:
        b_schur = 1.0 / (1.0 + b_dt * b_dt * b_a)
        b_M11 = 1.0 - b_dt * b_dt * b_a * b_schur
        b_M12 = -b_dt * b_a * b_schur
        b_M21 = b_dt * b_schur
        b_M22 = b_schur

    b_h1_re = tl.zeros([BP], dtype=tl.float32)
    b_h1_im = tl.zeros([BP], dtype=tl.float32)
    b_h2_re = tl.zeros([BP], dtype=tl.float32)
    b_h2_im = tl.zeros([BP], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h1_re += tl.load(h0 + i_b * 2 * P + o_p, mask=mask_p, other=0.).to(tl.float32)
        b_h2_re += tl.load(h0 + i_b * 2 * P + P + o_p, mask=mask_p, other=0.).to(tl.float32)

    for t in range(0, T):
        p_x = x + i_b * T * H + t * H

        b_bu_re = tl.zeros([BP], dtype=tl.float32)
        b_bu_im = tl.zeros([BP], dtype=tl.float32)
        for h in range(H):
            b_u_h = tl.load(p_x + h).to(tl.float32)
            b_bre_h = tl.load(B_re + o_p * H + h, mask=mask_p, other=0.).to(tl.float32)
            b_bim_h = tl.load(B_im + o_p * H + h, mask=mask_p, other=0.).to(tl.float32)
            b_bu_re += b_bre_h * b_u_h
            b_bu_im += b_bim_h * b_u_h

        if IS_IMEX:
            b_f1_re = b_bu_re * b_dt
            b_f1_im = b_bu_im * b_dt
            b_f2_re = b_bu_re * b_dt * b_dt
            b_f2_im = b_bu_im * b_dt * b_dt
        else:
            b_f1_re = b_M11 * b_bu_re * b_dt
            b_f1_im = b_M11 * b_bu_im * b_dt
            b_f2_re = b_M21 * b_bu_re * b_dt
            b_f2_im = b_M21 * b_bu_im * b_dt

        b_h1_re_new = b_M11 * tl.where(mask_p, b_h1_re, 0.) + b_M12 * tl.where(mask_p, b_h2_re, 0.) + b_f1_re
        b_h1_im_new = b_M11 * tl.where(mask_p, b_h1_im, 0.) + b_M12 * tl.where(mask_p, b_h2_im, 0.) + b_f1_im
        b_h2_re_new = b_M21 * tl.where(mask_p, b_h1_re, 0.) + b_M22 * tl.where(mask_p, b_h2_re, 0.) + b_f2_re
        b_h2_im_new = b_M21 * tl.where(mask_p, b_h1_im, 0.) + b_M22 * tl.where(mask_p, b_h2_im, 0.) + b_f2_im

        b_h1_re = b_h1_re_new
        b_h1_im = b_h1_im_new
        b_h2_re = b_h2_re_new
        b_h2_im = b_h2_im_new

        p_o = o + i_b * T * H + t * H
        for h in range(H):
            b_cre = tl.load(C_re + h * P + o_p, mask=mask_p, other=0.).to(tl.float32)
            b_cim = tl.load(C_im + h * P + o_p, mask=mask_p, other=0.).to(tl.float32)
            b_y_h = tl.sum(b_cre * b_h2_re - b_cim * b_h2_im, axis=0)
            if i_p == 0:
                b_d_h = tl.load(d_skip + h).to(tl.float32)
                b_u_h = tl.load(p_x + h).to(tl.float32)
                b_y_h += b_d_h * b_u_h
            tl.atomic_add(p_o + h, b_y_h)

    if STORE_FINAL_STATE:
        tl.store(ht + i_b * 2 * P + o_p, b_h1_re.to(ht.dtype.element_ty), mask=mask_p)
        tl.store(ht + i_b * 2 * P + P + o_p, b_h2_re.to(ht.dtype.element_ty), mask=mask_p)


def fused_recurrent_linoss_fwd(
    x: torch.Tensor,
    B_re: torch.Tensor,
    B_im: torch.Tensor,
    C_re: torch.Tensor,
    C_im: torch.Tensor,
    a_diag: torch.Tensor,
    dt: torch.Tensor,
    d_skip: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    discretization: str = 'IM',
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H = x.shape
    P = a_diag.shape[0]

    o = x.new_zeros(B, T, H)
    final_state = x.new_empty(B, 2, P) if output_final_state else None

    def grid(meta):
        return (triton.cdiv(P, meta['BP']), B)

    fused_recurrent_linoss_fwd_kernel[grid](
        x=x,
        B_re=B_re,
        B_im=B_im,
        C_re=C_re,
        C_im=C_im,
        a_diag=a_diag,
        dt=dt,
        d_skip=d_skip,
        o=o,
        h0=initial_state,
        ht=final_state,
        T=T,
        H=H,
        P=P,
        IS_IMEX=(discretization == 'IMEX'),
    )
    return o, final_state


class FusedRecurrentLinOSSFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        B_re: torch.Tensor,
        B_im: torch.Tensor,
        C_re: torch.Tensor,
        C_im: torch.Tensor,
        a_diag: torch.Tensor,
        dt: torch.Tensor,
        d_skip: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        discretization: str = 'IM',
    ):
        o, ht = fused_recurrent_linoss_fwd(
            x=x,
            B_re=B_re,
            B_im=B_im,
            C_re=C_re,
            C_im=C_im,
            a_diag=a_diag,
            dt=dt,
            d_skip=d_skip,
            initial_state=initial_state,
            output_final_state=output_final_state,
            discretization=discretization,
        )
        ctx.save_for_backward(x, B_re, B_im, C_re, C_im, a_diag, dt, d_skip, initial_state)
        ctx.discretization = discretization
        return o, ht

    @staticmethod
    @input_guard
    def backward(ctx, do, dht=None):
        x, B_re, B_im, C_re, C_im, a_diag, dt, d_skip, initial_state = ctx.saved_tensors
        discretization = ctx.discretization

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            B_re = B_re.detach().requires_grad_(True)
            B_im = B_im.detach().requires_grad_(True)
            C_re = C_re.detach().requires_grad_(True)
            C_im = C_im.detach().requires_grad_(True)
            a_diag = a_diag.detach().requires_grad_(True)
            dt = dt.detach().requires_grad_(True)
            d_skip = d_skip.detach().requires_grad_(True)

            o = _linoss_recurrent_torch(
                x, B_re, B_im, C_re, C_im, a_diag, dt, d_skip, initial_state, discretization
            )
            o.backward(do)

        return (
            x.grad, B_re.grad, B_im.grad, C_re.grad, C_im.grad,
            a_diag.grad, dt.grad, d_skip.grad,
            None, None, None,
        )


def _linoss_recurrent_torch(
    x: torch.Tensor,
    B_re: torch.Tensor,
    B_im: torch.Tensor,
    C_re: torch.Tensor,
    C_im: torch.Tensor,
    a_diag: torch.Tensor,
    dt: torch.Tensor,
    d_skip: torch.Tensor,
    initial_state: torch.Tensor | None,
    discretization: str,
) -> torch.Tensor:
    """Pure PyTorch implementation of LinOSS recurrence for backward pass.

    Tracks real and imaginary parts of state separately (M is real, so they decouple).
    """
    Bat, T, H = x.shape
    P = a_diag.shape[0]

    a = torch.relu(a_diag)
    step = torch.sigmoid(dt)

    if discretization == 'IMEX':
        M11 = torch.ones_like(a)
        M12 = -step * a
        M21 = step.clone()
        M22 = 1.0 - step * step * a
    else:
        schur = 1.0 / (1.0 + step * step * a)
        M11 = 1.0 - step * step * a * schur
        M12 = -step * a * schur
        M21 = step * schur
        M22 = schur

    Bu_re = torch.einsum('bth,ph->btp', x, B_re)
    Bu_im = torch.einsum('bth,ph->btp', x, B_im)

    h1_re = x.new_zeros(Bat, P)
    h1_im = x.new_zeros(Bat, P)
    h2_re = x.new_zeros(Bat, P)
    h2_im = x.new_zeros(Bat, P)

    if initial_state is not None:
        h1_re = initial_state[:, 0].to(x.dtype)
        h2_re = initial_state[:, 1].to(x.dtype)

    outputs = []
    for t in range(T):
        bu_re_t = Bu_re[:, t]
        bu_im_t = Bu_im[:, t]

        if discretization == 'IMEX':
            f1_re = bu_re_t * step.unsqueeze(0)
            f1_im = bu_im_t * step.unsqueeze(0)
            f2_re = bu_re_t * (step * step).unsqueeze(0)
            f2_im = bu_im_t * (step * step).unsqueeze(0)
        else:
            f1_re = M11.unsqueeze(0) * bu_re_t * step.unsqueeze(0)
            f1_im = M11.unsqueeze(0) * bu_im_t * step.unsqueeze(0)
            f2_re = M21.unsqueeze(0) * bu_re_t * step.unsqueeze(0)
            f2_im = M21.unsqueeze(0) * bu_im_t * step.unsqueeze(0)

        h1_re_new = M11.unsqueeze(0) * h1_re + M12.unsqueeze(0) * h2_re + f1_re
        h1_im_new = M11.unsqueeze(0) * h1_im + M12.unsqueeze(0) * h2_im + f1_im
        h2_re_new = M21.unsqueeze(0) * h1_re + M22.unsqueeze(0) * h2_re + f2_re
        h2_im_new = M21.unsqueeze(0) * h1_im + M22.unsqueeze(0) * h2_im + f2_im

        h1_re, h1_im = h1_re_new, h1_im_new
        h2_re, h2_im = h2_re_new, h2_im_new

        # y_t = Re(C @ h2) + D * u_t = C_re @ h2_re - C_im @ h2_im + D * u_t
        y_t = (torch.einsum('hp,bp->bh', C_re, h2_re)
               - torch.einsum('hp,bp->bh', C_im, h2_im)
               + d_skip.unsqueeze(0) * x[:, t])
        outputs.append(y_t)

    return torch.stack(outputs, dim=1)


@torch.compiler.disable
def fused_recurrent_linoss(
    x: torch.Tensor,
    B_re: torch.Tensor,
    B_im: torch.Tensor,
    C_re: torch.Tensor,
    C_im: torch.Tensor,
    a_diag: torch.Tensor,
    dt: torch.Tensor,
    d_skip: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    discretization: str = 'IM',
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Fused recurrent implementation of LinOSS (Linear Ordinary State Space).

    LinOSS models a second-order ODE system discretized with either IM or IMEX methods.
    Each state dimension has a 2-component state [position, velocity].

    Args:
        x (torch.Tensor):
            Input sequence of shape `[B, T, H]`.
        B_re (torch.Tensor):
            Real part of input matrix B, shape `[P, H]`.
        B_im (torch.Tensor):
            Imaginary part of input matrix B, shape `[P, H]`.
        C_re (torch.Tensor):
            Real part of output matrix C, shape `[H, P]`.
        C_im (torch.Tensor):
            Imaginary part of output matrix C, shape `[H, P]`.
        a_diag (torch.Tensor):
            Diagonal state matrix (pre-relu), shape `[P]`.
        dt (torch.Tensor):
            Discretization step sizes (pre-sigmoid), shape `[P]`.
        d_skip (torch.Tensor):
            Skip connection weights, shape `[H]`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[B, 2, P]`. Default: `None`.
        output_final_state (bool):
            Whether to output the final state. Default: `False`.
        discretization (str):
            Discretization method, either 'IM' or 'IMEX'. Default: `'IM'`.

    Returns:
        o (torch.Tensor):
            Output of shape `[B, T, H]`.
        final_state (Optional[torch.Tensor]):
            Final state of shape `[B, 2, P]` if `output_final_state=True`.
    """
    if x.requires_grad:
        return FusedRecurrentLinOSSFunction.apply(
            x, B_re, B_im, C_re, C_im, a_diag, dt, d_skip,
            initial_state, output_final_state, discretization,
        )
    else:
        return fused_recurrent_linoss_fwd(
            x, B_re, B_im, C_re, C_im, a_diag, dt, d_skip,
            initial_state, output_final_state, discretization,
        )
