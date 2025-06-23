import torch


def naive_recurrent_gated_delta_product(q, k, v, g, beta, scale, cu_seqlens,
                                        initial_state=None, output_final_state=False,
                                        num_householder=1):
    q_original_dtype = q.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert k.shape == (B, T*num_householder, H, K)
    assert v.shape == (B, T*num_householder, H, V)
    assert beta.shape == (B, T*num_householder, H)
    if g is not None:
        assert g.shape == (B, T, H)
    q, k, v, beta = map(lambda x: x.float(), (q, k, v, beta))

    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        h = initial_state

    o = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)

    for i in range(T):
        if g is not None:
            h = h * g[:, i, :].exp()[..., None, None]
        # multiple state transition
        for j in range(num_householder):
            k_ij = k[:, i*num_householder+j, :, :]
            v_ij = v[:, i*num_householder+j, :, :]
            beta_ij = beta[:, i*num_householder+j, :]
            h = h + (v_ij - (h * k_ij[..., None]).sum(-2)).unsqueeze(-2) * k_ij[..., None] * beta_ij[..., None, None]
        # memory readout
        q_i = q[:, i, :, :]
        o_i = (h * q_i[..., None]).sum(-2)
        o[:, i] = o_i
    return o.to(q_original_dtype), h
