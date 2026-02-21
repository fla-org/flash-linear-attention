from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.compiler.disable
def chunk_linoss(
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
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Chunk-wise parallel implementation of LinOSS (Linear Ordinary State Space).

    Exploits the time-invariant transition matrix M to parallelize across
    sequence chunks. The algorithm:

    1. Split the sequence into chunks and process each independently (assuming h0=0)
    2. Propagate states across chunk boundaries sequentially
    3. Apply corrections using precomputed matrix powers of M

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
            Initial state of shape `[B, 4, P]` with components
            `[h1_re, h1_im, h2_re, h2_im]`. Default: `None`.
        output_final_state (bool):
            Whether to output the final state. Default: `False`.
        discretization (str):
            Discretization method, either 'IM' or 'IMEX'. Default: `'IM'`.
        chunk_size (int):
            Size of each chunk. Default: `64`.

    Returns:
        o (torch.Tensor):
            Output of shape `[B, T, H]`.
        final_state (Optional[torch.Tensor]):
            Final state of shape `[B, 4, P]` if `output_final_state=True`.
    """
    dtype = x.dtype
    Bat, T, H = x.shape
    P = a_diag.shape[0]
    L = chunk_size

    x_f = x.float()
    B_re_f, B_im_f = B_re.float(), B_im.float()
    C_re_f, C_im_f = C_re.float(), C_im.float()

    a = torch.relu(a_diag.float())
    step = torch.sigmoid(dt.float())

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

    Bu_re = torch.einsum('bth,ph->btp', x_f, B_re_f)
    Bu_im = torch.einsum('bth,ph->btp', x_f, B_im_f)

    if discretization == 'IMEX':
        f1_re = Bu_re * step
        f1_im = Bu_im * step
        f2_re = Bu_re * (step * step)
        f2_im = Bu_im * (step * step)
    else:
        f1_re = M11 * Bu_re * step
        f1_im = M11 * Bu_im * step
        f2_re = M21 * Bu_re * step
        f2_im = M21 * Bu_im * step

    pad = (L - T % L) % L
    if pad > 0:
        f1_re = F.pad(f1_re, (0, 0, 0, pad))
        f1_im = F.pad(f1_im, (0, 0, 0, pad))
        f2_re = F.pad(f2_re, (0, 0, 0, pad))
        f2_im = F.pad(f2_im, (0, 0, 0, pad))
    Tp = T + pad
    NC = Tp // L

    f1_re_c = f1_re.view(Bat, NC, L, P)
    f1_im_c = f1_im.view(Bat, NC, L, P)
    f2_re_c = f2_re.view(Bat, NC, L, P)
    f2_im_c = f2_im.view(Bat, NC, L, P)
    BN = Bat * NC
    f1_re_f = f1_re_c.reshape(BN, L, P)
    f1_im_f = f1_im_c.reshape(BN, L, P)
    f2_re_f = f2_re_c.reshape(BN, L, P)
    f2_im_f = f2_im_c.reshape(BN, L, P)

    h1r = x_f.new_zeros(BN, P)
    h1i = x_f.new_zeros(BN, P)
    h2r = x_f.new_zeros(BN, P)
    h2i = x_f.new_zeros(BN, P)

    h2r_seq, h2i_seq = [], []
    for t in range(L):
        h1r, h1i, h2r, h2i = (
            M11 * h1r + M12 * h2r + f1_re_f[:, t],
            M11 * h1i + M12 * h2i + f1_im_f[:, t],
            M21 * h1r + M22 * h2r + f2_re_f[:, t],
            M21 * h1i + M22 * h2i + f2_im_f[:, t],
        )
        h2r_seq.append(h2r)
        h2i_seq.append(h2i)

    h2r_local = torch.stack(h2r_seq, dim=1)   # [BN, L, P]
    h2i_local = torch.stack(h2i_seq, dim=1)

    h1r_fin = h1r.view(Bat, NC, P)
    h1i_fin = h1i.view(Bat, NC, P)
    h2r_fin = h2r.view(Bat, NC, P)
    h2i_fin = h2i.view(Bat, NC, P)

    mp11, mp12 = torch.ones_like(a), torch.zeros_like(a)
    mp21, mp22 = torch.zeros_like(a), torch.ones_like(a)
    mp21_seq, mp22_seq = [], []
    for _ in range(L):
        mp11, mp12, mp21, mp22 = (
            M11 * mp11 + M12 * mp21,
            M11 * mp12 + M12 * mp22,
            M21 * mp11 + M22 * mp21,
            M21 * mp12 + M22 * mp22,
        )
        mp21_seq.append(mp21)
        mp22_seq.append(mp22)

    mp21_all = torch.stack(mp21_seq)  # [L, P]
    mp22_all = torch.stack(mp22_seq)
    mc11, mc12, mc21, mc22 = mp11, mp12, mp21, mp22  # M^L

    s1r = x_f.new_zeros(Bat, P)
    s1i = x_f.new_zeros(Bat, P)
    s2r = x_f.new_zeros(Bat, P)
    s2i = x_f.new_zeros(Bat, P)
    if initial_state is not None:
        s1r = initial_state[:, 0].float()
        s1i = initial_state[:, 1].float()
        s2r = initial_state[:, 2].float()
        s2i = initial_state[:, 3].float()

    s1r_all, s1i_all = [s1r], [s1i]
    s2r_all, s2i_all = [s2r], [s2i]

    for k in range(NC - 1):
        s1r, s1i, s2r, s2i = (
            h1r_fin[:, k] + mc11 * s1r + mc12 * s2r,
            h1i_fin[:, k] + mc11 * s1i + mc12 * s2i,
            h2r_fin[:, k] + mc21 * s1r + mc22 * s2r,
            h2i_fin[:, k] + mc21 * s1i + mc22 * s2i,
        )
        s1r_all.append(s1r)
        s1i_all.append(s1i)
        s2r_all.append(s2r)
        s2i_all.append(s2i)

    s1r_t = torch.stack(s1r_all, dim=1)  # [Bat, NC, P]
    s1i_t = torch.stack(s1i_all, dim=1)
    s2r_t = torch.stack(s2r_all, dim=1)
    s2i_t = torch.stack(s2i_all, dim=1)

    h2r_corr = mp21_all[None, None] * s1r_t[:, :, None] + mp22_all[None, None] * s2r_t[:, :, None]
    h2i_corr = mp21_all[None, None] * s1i_t[:, :, None] + mp22_all[None, None] * s2i_t[:, :, None]

    h2r_act = h2r_local.view(Bat, NC, L, P) + h2r_corr
    h2i_act = h2i_local.view(Bat, NC, L, P) + h2i_corr

    h2r_out = h2r_act.reshape(Bat, Tp, P)[:, :T]
    h2i_out = h2i_act.reshape(Bat, Tp, P)[:, :T]

    o = (
        torch.einsum('hp,btp->bth', C_re_f, h2r_out)
        - torch.einsum('hp,btp->bth', C_im_f, h2i_out)
        + d_skip.float() * x_f
    ).to(dtype)

    final_state = None
    if output_final_state:
        T_rem = T - (NC - 1) * L
        last_s1r, last_s1i = s1r_all[-1], s1i_all[-1]
        last_s2r, last_s2i = s2r_all[-1], s2i_all[-1]

        if T_rem == L and pad == 0:
            fh1r = h1r_fin[:, -1] + mc11 * last_s1r + mc12 * last_s2r
            fh1i = h1i_fin[:, -1] + mc11 * last_s1i + mc12 * last_s2i
            fh2r = h2r_fin[:, -1] + mc21 * last_s1r + mc22 * last_s2r
            fh2i = h2i_fin[:, -1] + mc21 * last_s1i + mc22 * last_s2i
        else:
            r1r, r1i = last_s1r, last_s1i
            r2r, r2i = last_s2r, last_s2i
            for t in range(T_rem):
                r1r, r1i, r2r, r2i = (
                    M11 * r1r + M12 * r2r + f1_re_c[:, -1, t],
                    M11 * r1i + M12 * r2i + f1_im_c[:, -1, t],
                    M21 * r1r + M22 * r2r + f2_re_c[:, -1, t],
                    M21 * r1i + M22 * r2i + f2_im_c[:, -1, t],
                )
            fh1r, fh1i, fh2r, fh2i = r1r, r1i, r2r, r2i
        final_state = torch.stack([
            fh1r.to(dtype), fh1i.to(dtype),
            fh2r.to(dtype), fh2i.to(dtype),
        ], dim=1)

    return o, final_state
