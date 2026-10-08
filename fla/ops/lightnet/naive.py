# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def naive_lightnet_gate(
    x: torch.Tensor,
    initial_state: torch.Tensor | None,
    lengths: tuple[int, ...],
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = x.unbind(0) if layout == "dense" else x.squeeze(0).split(lengths)
    keys, gates, final_states = [], [], []
    for i, sequence in enumerate(sequences):
        state = initial_state[i] if initial_state is not None else None
        if sequence.shape[0] == 0:
            keys.append(sequence)
            gates.append(sequence)
            if state is None:
                state = x.new_full((1, *x.shape[2:]), float("-inf"), dtype=torch.float32)
            final_states.append(state)
            continue

        z = sequence.float().logcumsumexp(0)
        if state is not None:
            z = torch.logaddexp(state, z)
            previous = torch.cat([state, z[:-1]], dim=0)
        else:
            previous = torch.cat([z[:1], z[:-1]], dim=0)
        keys.append((sequence.float() - z).exp().to(x.dtype))
        gates.append(torch.nan_to_num(previous - z, nan=0.0, posinf=0.0, neginf=0.0).to(x.dtype))
        final_states.append(z[-1:])

    if layout == "dense":
        k, g = torch.stack(keys), torch.stack(gates)
    else:
        k, g = torch.cat(keys).unsqueeze(0), torch.cat(gates).unsqueeze(0)
    return k, g, torch.stack(final_states)
