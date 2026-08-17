# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch


def sample_lognormal_packed_lengths(
    total_tokens: int,
    num_sequences: int,
    max_length: int,
    min_length: int = 16,
    sigma: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample right-skewed sequence lengths with an exact packed-token budget.

    The log-normal weights are a configurable workload proxy, not a claim about any particular dataset.
    Pass an observed sequence count, length bounds, and sigma from the workload being measured.
    """
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")
    if num_sequences <= 0:
        raise ValueError(f"num_sequences must be positive, got {num_sequences}")
    if not 0 < min_length <= max_length:
        raise ValueError(f"expected 0 < min_length <= max_length, got {min_length=} and {max_length=}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    min_tokens = num_sequences * min_length
    max_tokens = num_sequences * max_length
    if not min_tokens <= total_tokens <= max_tokens:
        raise ValueError(
            f"cannot pack {total_tokens} tokens into {num_sequences} sequences with "
            f"lengths in [{min_length}, {max_length}]"
        )

    weights = torch.empty(num_sequences, dtype=torch.float64).log_normal_(
        mean=0.0,
        std=sigma,
        generator=generator,
    )
    lengths = torch.full((num_sequences,), min_length, dtype=torch.long)
    capacity = torch.full_like(lengths, max_length - min_length)
    remaining = total_tokens - min_tokens

    while remaining:
        active_weights = weights.masked_fill(capacity == 0, 0)
        picks = torch.multinomial(active_weights, remaining, replacement=True, generator=generator)
        requested = torch.bincount(picks, minlength=num_sequences)
        added = torch.minimum(requested, capacity)
        lengths += added
        capacity -= added
        remaining -= int(added.sum())

    return lengths
