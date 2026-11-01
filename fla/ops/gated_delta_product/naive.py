# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def naive_recurrent_gated_delta_product(q, k, v, g, beta, scale, cu_seqlens=None,
                                        initial_state=None, output_final_state=False,
                                        num_householder=1):
    if cu_seqlens is not None:
        outputs, final_states = [], []
        offsets = cu_seqlens.tolist()
        for i, (bos, eos) in enumerate(zip(offsets[:-1], offsets[1:], strict=False)):
            o_i, h_i = naive_recurrent_gated_delta_product(
                q=q[:, bos:eos],
                k=k[:, bos*num_householder:eos*num_householder],
                v=v[:, bos*num_householder:eos*num_householder],
                g=None if g is None else g[:, bos:eos],
                beta=beta[:, bos*num_householder:eos*num_householder],
                scale=scale,
                initial_state=None if initial_state is None else initial_state[i:i + 1],
                output_final_state=output_final_state,
                num_householder=num_householder,
            )
            outputs.append(o_i)
            final_states.append(h_i)

        return torch.cat(outputs, dim=1), torch.cat(final_states)

    q_original_dtype = q.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert k.shape == (B, T*num_householder, H, K)
    assert v.shape == (B, T*num_householder, H, V)
    assert beta.shape == (B, T*num_householder, H)
    if g is not None:
        assert g.shape == (B, T, H)
    q, k, v, beta = map(lambda x: x.float(), (q, k, v, beta))
    q = q * scale

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
