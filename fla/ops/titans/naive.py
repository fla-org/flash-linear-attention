# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

from fla.ops.titans.log_impl import combine_params_log


def cal_n(theta, eta, seq_len):
    n = torch.zeros(*theta.shape, seq_len, dtype=theta.dtype).to(
        theta.device
    )  # [batch_size, num_heads, seq_len, seq_len]

    # 1. deal with diagonal elements
    indices = torch.arange(seq_len, device=theta.device)
    n[..., indices, indices] = theta[..., indices]

    # 2. Create a cumulative product matrix
    # First create a mask to mark the positions where eta needs to be multiplied
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(theta.device)
    # Convert mask to boolean type
    mask = mask.bool()
    # Expand eta to match the target shape
    eta_expanded = eta.unsqueeze(-2).expand(*theta.shape[:-1], seq_len, seq_len)
    # Create a matrix filled with 1s for cumulative product
    cumulative = torch.ones_like(eta_expanded)
    cumulative = torch.where(mask, eta_expanded, cumulative)
    # Calculate the cumulative product
    cumulative_prod = torch.cumprod(cumulative, dim=-1)

    # 3. Calculate non-diagonal elements
    # Create an expanded version of theta
    theta_expanded = theta.unsqueeze(-1).expand(*theta.shape[:-1], seq_len, seq_len)
    # Create a mask to keep only the upper triangular part (excluding the diagonal)
    upper_triangular = torch.triu(torch.ones_like(n), diagonal=1).bool()
    # Combine theta and cumulative product
    n = torch.where(upper_triangular, theta_expanded * cumulative_prod, n)
    return n


def cal_f(beta, seq_len, m):
    a = torch.tril(beta.to(torch.float32).unsqueeze(-1).expand(*beta.shape, seq_len), 0)
    ratio = (m.to(torch.float32) / beta.to(torch.float32)).unsqueeze(-1)
    f = torch.matmul(a, ratio).squeeze(-1)
    return f.to(beta.dtype)


def cal_G(beta, n, seq_len):
    i_indices = torch.arange(seq_len, device=beta.device)
    j_indices = torch.arange(seq_len, device=beta.device)
    k_indices = torch.arange(seq_len, device=beta.device)
    beta_ratio = beta[..., :, None] / beta[..., None, :]  # [..., i, k]

    # create mask
    k_mask = (k_indices[None, None, :] >= j_indices[None, :, None]) & (
        k_indices[None, None, :] <= i_indices[:, None, None]
    )

    # use mask to filter out invalid values
    masked_beta_ratio = beta_ratio[..., :, None, :] * k_mask  # [..., i, j, k]
    masked_n = n[..., None, :, :] * k_mask  # [..., i, j, k]
    # calculate G
    G = torch.sum(masked_beta_ratio * masked_n, dim=-1)  # [..., i, j]
    return G


def combine_params(theta, alpha, eta, seq_len):
    theta = theta.squeeze(-1)
    eta = eta.squeeze(-1)
    alpha = alpha.squeeze(-1)
    beta = torch.cumprod(1 - alpha, dim=-1)  # β_t = ∏(1 - α_t) in titans paper
    beta_T = beta[..., -1]  # β_T
    # Calculate m_i = ∏(k=1 to i) η_k
    m = torch.cumprod(eta, dim=-1)  # [batch_size, num_heads, seq_len]
    m_T = m[..., -1]  # m_T
    # Calculate n_{i,j}
    # We need to calculate ∏(k=j+1 to i) η_k for each i,j pair
    # # this may be optimized
    # n = torch.zeros(*theta.shape, seq_len, dtype = theta.dtype).to(
    #     theta.device)  # [batch_size, num_heads, seq_len, seq_len]
    # for i in range(seq_len):
    #     for j in range(i + 1):
    #         if i == j:
    #             n[..., j, i] = theta[..., j]
    #         else:
    #             # Calculate product of eta from j+1 to i
    #             eta_product = torch.prod(eta[..., j + 1:i + 1], dim = -1)
    #             n[..., j, i] = theta[..., j] * eta_product

    n = cal_n(theta, eta, seq_len)
    n_T = n[..., -1]  # [batch_size, num_heads, seq_len]
    # Calculate f_t = ∑(i=1 to t) (β_t/β_i) m_i
    # f = torch.zeros_like(theta)
    # for t in range(seq_len):
    #     for i in range(t + 1):
    #         f[..., t] += (beta[..., t] / beta[..., i]) * m[..., i]
    f = cal_f(beta, seq_len, m)
    f_T = f[..., -1]  # [batch_size, num_heads, seq_len]
    # Calculate g_j = ∑(i=j to t) (β_t/β_i) n_{i,j}
    # g = torch.zeros_like(theta)  # [batch_size, num_heads, seq_len]
    # for j in range(seq_len):
    #     for i in range(j, seq_len):
    #         g[..., j] += (beta[..., -1] / beta[..., i]) * n[..., j, i]
    # G = torch.zeros(*beta.shape[:-1], seq_len, seq_len, device = beta.device)
    # # Fill in the lower triangular part
    # for i in range(seq_len):  # row
    #     for j in range(i + 1):  # column
    #         # Sum from k=j to i
    #         for k in range(j, i + 1):
    #             G[..., i, j] += (beta[..., i] / beta[..., k]) * n[..., j, k]
    G = cal_G(beta, n, seq_len)
    g = G[:, :, -1, :]  # [batch_size, num_heads, seq_len]
    # g2, G2 = compute_g_and_G(beta, n, seq_len)
    return beta, beta_T, f, f_T, g, G, m_T, n_T


def titans_linear(
    q, k, v, w, b, theta, alpha, eta, eps, chunk_size, initial_state, output_final_state
):
    """
    Implementation of Titans Linear function based on the update rules:
    M_t = (1 - alpha_t) * M_{t-1} + S_t
    S_t = eta_t * S_{t-1} - theta_t * nabla_l(M_{t-1}; x_t)

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        w: Weight tensor
        b: Bias tensor
        theta: Learning rate tensor
        alpha: Momentum decay tensor
        eta: Step size tensor
        eps: Epsilon for numerical stability
        initial_state: Initial state M_0
        output_final_state: Whether to output the final state

    Returns:
        Tuple of (output tensor, final state)
    """
    B, H, T, D = q.shape
    device = q.device
    w = w.reshape(H, 1, D).to(torch.float32)
    b = b.reshape(H, 1, D).to(torch.float32)
    # Initialize states
    if initial_state is None:
        M_prev = torch.zeros(B, H, D, D, device=device)
    else:
        M_prev = initial_state
    M_prev_nabla = M_prev.clone()
    S_prev = torch.zeros_like(M_prev)
    outputs = []

    # Process sequence step by step
    for t in range(T):
        # Get current step inputs
        q_t = q[:, :, t: t + 1, :]  # (batch_size, num_heads, 1, dim)
        k_t = k[:, :, t: t + 1, :]  # (batch_size, num_heads, 1, dim)
        v_t = v[:, :, t: t + 1, :]  # (batch_size, num_heads, 1, dim)
        theta_t = theta[:, :, t: t + 1, :]  # (batch_size, num_heads, 1, dim)
        alpha_t = alpha[:, :, t: t + 1, :]  # (batch_size, num_heads, 1, dim)
        eta_t = eta[:, :, t: t + 1, :]  # (batch_size, num_heads, 1, dim)

        # Compute gradient
        km = k_t @ M_prev_nabla  # (batch_size, num_heads, 1, dim)
        reconstruction_target = v_t - k_t
        mean = km.mean(-1, keepdim=True)
        var = km.var(-1, unbiased=False, keepdim=True).to(torch.float32)
        rstd = torch.sqrt(var + eps).to(torch.float32)
        km_hat = (km - mean) / rstd

        grad = w * km_hat + b - reconstruction_target
        grad = grad * w
        # v_new = (D * grad - grad.sum(-1, keepdim = True) - km_hat * (grad * km_hat).sum(-1, keepdim = True)) / (
        #             rstd * D)
        v_new = D * grad - grad.sum(-1, keepdim=True) / (rstd * D)
        proj_term = km_hat * (grad * km_hat).sum(-1, keepdim=True) / (rstd * D)
        v_new = v_new - proj_term
        # v_new = grad

        # Update S_t
        S_t = eta_t * S_prev - 2 * theta_t * k_t.transpose(-2, -1) @ v_new

        # Update M_t
        M_t = (1 - alpha_t) * M_prev + S_t

        # Store output
        output_t = q_t @ M_t  # (batch_size, num_heads, seq_len, dim)
        mean = output_t.mean(dim=-1, keepdim=True)
        var = output_t.var(dim=-1, unbiased=False, keepdim=True).to(torch.float32)
        rstd = torch.sqrt(var + eps).to(torch.float32)
        output_t = output_t + (output_t - mean) / rstd * w + b
        outputs.append(output_t)

        # Update states for next step
        if (t + 1) % chunk_size == 0:
            M_prev_nabla = M_t.clone()
        M_prev = M_t
        S_prev = S_t

    # Stack outputs along sequence dimension
    output = torch.stack(outputs, dim=-2).squeeze(
        -3
    )  # (batch_size, num_heads, seq_len, dim)

    if output_final_state:
        return output, M_prev
    return output, None


def chunk_titans_linear(
    q, k, v, w, b, theta, alpha, eta, eps, chunk_size, initial_state, output_final_state
):
    B, H, T, D = q.shape
    num_batch = T // chunk_size
    # [num_batch, B, num_heads, mini_batch_size, head_dim]
    _q = q.reshape(B, H, num_batch, chunk_size, D).permute(2, 0, 1, 3, 4)
    _k = k.reshape(B, H, num_batch, chunk_size, D).permute(2, 0, 1, 3, 4)
    _v = v.reshape(B, H, num_batch, chunk_size, D).permute(2, 0, 1, 3, 4)
    # [num_batch, B, num_heads, mini_batch_size, 1]
    _eta = eta.reshape(B, H, num_batch, chunk_size, 1).permute(2, 0, 1, 3, 4)
    _theta = theta.reshape(B, H, num_batch, chunk_size, 1).permute(2, 0, 1, 3, 4)
    _alpha = alpha.reshape(B, H, num_batch, chunk_size, 1).permute(2, 0, 1, 3, 4)
    # [H, 1, D]
    w = w.reshape(H, 1, D).to(torch.float32)
    b = b.reshape(H, 1, D).to(torch.float32)
    # [num_heads, 1, head_dim]
    if initial_state is None:
        M_prev = torch.zeros((B, H, D, D), device=v.device, dtype=v.dtype).to(
            torch.float32
        )
    else:
        M_prev = initial_state

    S_prev = torch.zeros_like(M_prev)

    # [num_batch, B, num_heads, mini_batch_size, head_dim]
    o = torch.empty_like(_v)

    for i in range(num_batch):
        q_i, k_i, v_i, eta_i, theta_i, alpha_i = [
            x[i] for x in [_q, _k, _v, _eta, _theta, _alpha]
        ]

        # beta, beta_T, f, f_T, g, G, m_T, n = combine_params(theta_i, alpha_i, eta_i, chunk_size)
        beta, beta_T, f, f_T, g, G, m_T, n = combine_params_log(
            theta_i, alpha_i, eta_i, chunk_size
        )

        m_T = m_T.unsqueeze(-1).unsqueeze(-1)
        beta_T = beta_T.unsqueeze(-1).unsqueeze(-1)
        f_T = f_T.unsqueeze(-1).unsqueeze(-1)
        g_diag = torch.diag_embed(g).to(q_i.dtype)
        n = torch.diag_embed(n).to(q_i.dtype)
        beta = torch.diag_embed(beta).to(q_i.dtype)
        f = torch.diag_embed(f).to(q_i.dtype)
        km = k_i @ M_prev
        reconstruction_target = v_i - k_i

        mean = km.mean(-1, True)
        var = km.var(-1, unbiased=False, keepdim=True).to(torch.float32)
        rstd = torch.sqrt(var + eps).to(torch.float32)
        km_hat = (km - mean) / rstd

        grad = w * km_hat + b - reconstruction_target
        grad *= w
        v_new = D * grad - grad.sum(-1, keepdim=True) / (rstd * D)
        proj_term = km_hat * (grad * km_hat).sum(-1, keepdim=True) / (rstd * D)
        v_new = v_new - proj_term
        # v_new = (D * grad - grad.sum(-1, True))
        # print(f"Projection term stats: min={torch.abs(beta_T).min()}")

        # v_new = grad

        Attn = torch.tril(q_i @ k_i.transpose(-2, -1)) * G

        # o_i
        output_t = beta @ q_i @ M_prev + f @ q_i @ S_prev - 2 * Attn @ v_new

        M_t = (
            beta_T * M_prev
            + f_T * S_prev
            - 2 * (g_diag @ k_i).transpose(-1, -2) @ v_new
        )
        # cal S_T from S_0
        S_t = m_T * S_prev - 2 * (n @ k_i).transpose(-1, -2) @ v_new
        # layer norm with residuals
        mean = output_t.mean(dim=-1, keepdim=True)
        var = output_t.var(dim=-1, unbiased=False, keepdim=True).to(torch.float32)
        rstd = torch.sqrt(var + eps).to(torch.float32)
        output_t = output_t + (output_t - mean) / rstd * w + b
        o[i] = output_t
        S_prev = S_t
        M_prev = M_t

    # [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
    o = o.permute(1, 2, 0, 3, 4).reshape(B, H, T, D)
    M_prev = M_prev if output_final_state else None
    return o, M_prev


# most of the code is copied from ttt
def chunk_titans_linear_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    theta: torch.Tensor,
    alpha: torch.Tensor,
    eta: torch.Tensor,
    eps: float = 1e-6,
    chunk_size: int = 16,  # chunk size
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    head_first: bool = False,
    use_chunk: bool = True,
):
    assert q.dtype == k.dtype == v.dtype
    assert k.shape[-1] == v.shape[-1], "DK must equal to DV."
    if not head_first:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        eta = eta.transpose(1, 2)
        alpha = alpha.transpose(1, 2)
        theta = theta.transpose(1, 2)
    seq_len = q.shape[-2]
    pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        theta = F.pad(theta, (0, 0, 0, pad_len))
        alpha = F.pad(alpha, (0, 0, 0, pad_len))
        eta = F.pad(eta, (0, 0, 0, pad_len))
        theta[:, :, -1, :] = theta[:, :, -(pad_len + 1), :]
        alpha[:, :, -1, :] = alpha[:, :, -(pad_len + 1), :]
        eta[:, :, -1, :] = eta[:, :, -(pad_len + 1), :]
    assert q.shape[-2] % chunk_size == 0, "Sequence length should be a multiple of BT."
    q, k, v, w, b = map(lambda x: x.to(torch.float32), [q, k, v, w, b])
    if use_chunk:
        o, final_state = chunk_titans_linear(
            q,
            k,
            v,
            w,
            b,
            theta,
            alpha,
            eta,
            eps,
            chunk_size,
            initial_state,
            output_final_state,
        )
    else:
        o, final_state = titans_linear(
            q,
            k,
            v,
            w,
            b,
            theta,
            alpha,
            eta,
            eps,
            chunk_size,
            initial_state,
            output_final_state,
        )
    o = o[:, :, :seq_len, :]
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state
