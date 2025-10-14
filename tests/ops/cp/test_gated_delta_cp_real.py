import pytest
import torch
import torch.distributed as dist
from dtest import DTest


class TestDTest(DTest):
    @pytest.mark.parametrize("n", list(range(1, 4)))
    def test_all_reduce(self, n: int) -> None:
        t = torch.arange(n * self.world_size, device=self.device)
        dist.all_reduce(t)
        self.print_rank(f"{t=}")
