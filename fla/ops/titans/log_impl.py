import torch


def cal_n_log(log_theta, log_eta, seq_len):
    """
    calculate n_{i,j} in log space
    log(n_{i,j}) = log(θ_j) + sum_{k=j+1}^i log(η_k)
    """
    # create log(n)
    log_n = torch.zeros(*log_theta.shape, seq_len, dtype=log_eta.dtype).to(
        log_eta.device
    )  # [batch_size, num_heads, seq_len, seq_len]
    for i in range(seq_len):
        for j in range(i + 1):
            if i == j:
                log_n[..., j, i] = log_theta[..., j]
            else:
                log_n[..., j, i] = log_theta[..., j] + torch.sum(
                    log_eta[..., j + 1: i + 1], dim=-1
                )

    return log_n


def cal_f_log(log_beta, seq_len, log_m):
    """
    cal_f_log(log_beta, seq_len, log_m) -> f
    log(f_t) = log(sum_{i=1}^t exp(sum_{k=i+1}^t log(1-α_k) + sum_{k=1}^i log(η_k)))
    """
    # create f
    # f = torch.zeros_like(log_beta)
    # for t in range(seq_len):
    #     for i in range(t + 1):
    #         f[..., t] += torch.exp(log_beta[..., t] - log_beta[..., i] + log_m[..., i])
    log_f = torch.zeros_like(log_beta)
    for t in range(seq_len):
        a_i = log_beta[..., t: t + 1] - log_beta[..., : t + 1] + log_m[..., : t + 1]
        log_f[..., t] = torch.logsumexp(a_i, dim=-1)
    f = torch.exp(log_f)

    # this version overflow and even slower
    # t_indices = torch.arange(seq_len, device=log_beta.device)
    # i_indices = torch.arange(seq_len, device=log_beta.device)
    #
    # mask = i_indices.unsqueeze(0) <= t_indices.unsqueeze(1)
    # log_beta_t = log_beta.unsqueeze(-1)  # [..., seq_len, 1]
    # log_beta_i = log_beta.unsqueeze(-2)  # [..., 1, seq_len]
    # log_m_i = log_m.unsqueeze(-2)
    # a_i = log_beta_t - log_beta_i + log_m_i
    # masked_a_i = torch.where(mask, a_i, torch.tensor(-float('inf'), device=a_i.device, dtype=a_i.dtype))
    # log_f = torch.logsumexp(masked_a_i, dim=-1)  # [..., seq_len]
    #
    # f = torch.exp(log_f)
    return f


def cal_G_log(log_beta, log_n, seq_len):
    """
    calculate G_{i,j}
    log(G_{i,j}) = log(sum_{k=j}^i exp(log(β_i/β_k) + log(n_{k,j})))
    """
    # G = torch.zeros(*log_beta.shape[:-1], seq_len, seq_len, device = log_beta.device)
    # # Fill in the lower triangular part
    # for i in range(seq_len):  # row
    #     for j in range(i + 1):  # column
    #         # Sum from k=j to i
    #         for k in range(j, i + 1):
    #             G[..., i, j] += torch.exp(log_beta[..., i] - log_beta[..., k] + log_n[..., j, k])

    log_G = torch.full(
        (*log_beta.shape[:-1], seq_len, seq_len), float("-inf"), device=log_beta.device
    )
    # fill in the lower triangular part
    for i in range(seq_len):  # row
        for j in range(i + 1):  # column
            terms = (
                log_beta[..., i: i + 1]
                - log_beta[..., j: i + 1]
                + log_n[..., j: j + 1, j: i + 1].squeeze(-2)
            )
            # use logsumexp to avoid overflow
            log_G[..., i, j] = torch.logsumexp(terms, dim=-1)

    G = torch.exp(log_G)
    return G


def _combine_params_log(log_theta, log_alpha_complement, log_eta, seq_len):
    """
    Update rule for Titans in log space

    Parameters:
    - log_theta: log(θ)
    - log_alpha_complement: log(1-α)
    - log_eta: log(η)
    - seq_len: sequence length

    Returns:
    - log_beta, beta_T, log_f, f_T, log_g, log_G, m_T, n_T
    """
    # calculate log(β_t) = sum_{k=1}^t log(1-α_k)
    log_beta = torch.cumsum(log_alpha_complement, dim=-1)

    # get β_T
    beta_T = torch.exp(log_beta[..., -1])

    # calculate log(m_i) = sum_{k=1}^i log(η_k)
    log_m = torch.cumsum(log_eta, dim=-1)
    m_T = torch.exp(log_m[..., -1])

    # cal log(n_{i,j})
    log_n = cal_n_log(log_theta, log_eta, seq_len)
    n_T = torch.exp(log_n[..., -1])

    # cal log(f_t)
    f = cal_f_log(log_beta, seq_len, log_m)
    f_T = f[..., -1]

    # cal log(G_{i,j})
    G = cal_G_log(log_beta, log_n, seq_len)
    # get log(g_j) = log(G_{T,j})
    g = G[..., -1, :]

    return log_beta, beta_T, f, f_T, g, G, m_T, n_T


def combine_params_log(theta, alpha, eta, seq_len):
    """
    log space Titians

    Parameters:
    - theta: θ
    - alpha: α
    - eta: η
    - seq_len: sequence length

    Returns:
    - beta, beta_T, f, f_T, g, G, m_T, n_T
    """
    # convert to log space
    log_theta = torch.log(theta.squeeze(-1))
    log_alpha_complement = torch.log(1 - alpha.squeeze(-1))
    log_eta = torch.log(eta.squeeze(-1))

    # combine params in log space
    log_beta, beta_T, f, f_T, g, G, m_T, n_T = _combine_params_log(
        log_theta, log_alpha_complement, log_eta, seq_len
    )

    # convert back to normal space
    beta = torch.exp(log_beta)

    return beta, beta_T, f, f_T, g, G, m_T, n_T
