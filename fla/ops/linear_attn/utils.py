
import torch


@torch.jit.script
def normalize_output(q: torch.Tensor, k: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    k = k.cumsum(1)
    z = (q * k).sum(-1, keepdim=True)
    return o / (z + 1e-10)
