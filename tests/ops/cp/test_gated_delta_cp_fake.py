# -*- coding: utf-8 -*-


import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
from fla.utils import assert_close, device


class TestFakeCP:
    B: int = 1
    T: int = 256
    H: int = 1
    D: int = 64
    scale: float = 1.0
    gate_logit_normalizer: float = 1.0
    mask_p: float = 0.0
    dtype: torch.dtype = torch.bfloat16

    def cp_shard(
        self,
        tensor: torch.Tensor,
        cp_degree: int | None = None,
        shard_dim: int | None = 1,
    ) -> torch.Tensor:
        return tensor.tensor_split(cp_degree or self.cp_degree, dim=shard_dim)

    def cp_unshard(
        self,
        tensor_list: list[torch.Tensor],
        shard_dim: int | None = 1,
    ) -> torch.Tensor:
        return torch.cat(tensor_list, dim=shard_dim)

    @pytest.mark.parametrize("cp_degree", [2, 4, 8])
    def test_chunk_cp_fwd(self, cp_degree: int):
        torch.manual_seed(42)
        q = torch.rand(self.B, self.T, self.H, self.D, dtype=self.dtype)
        k = torch.rand(self.B, self.T, self.H, self.D, dtype=self.dtype)
        v = torch.rand(self.B, self.T, self.H, self.D, dtype=self.dtype)
        beta = torch.rand(self.B, self.T, self.H, dtype=self.dtype).sigmoid()
        g = F.logsigmoid(torch.rand(self.B, self.T, self.H, dtype=torch.float32))
        g = g / self.gate_logit_normalizer
        g = g * (torch.rand_like(g) > self.mask_p)
        h0 = torch.zeros(self.B, self.H, self.D, self.D, dtype=torch.float32)
        q, k, v, beta, g, h0 = map(lambda x: x.to(device), (q, k, v, beta, g, h0))

        g_out, o, A, _ = chunk_gated_delta_rule_fwd(
            q=F.normalize(q, p=2, dim=-1),
            k=F.normalize(k, p=2, dim=-1),
            v=v,
            g=g,
            beta=beta,
            scale=self.scale,
            initial_state=h0,
            output_final_state=True,
        )

        # Recompute the same outputs in a fake CP setup:
        # 1) All inputs are sharded along the seq dim.
        # 2) Sequential ranks pass their final state to the next rank as its initial state.
        q_cp_list = self.cp_shard(q, cp_degree=cp_degree)
        k_cp_list = self.cp_shard(k, cp_degree=cp_degree)
        v_cp_list = self.cp_shard(v, cp_degree=cp_degree)
        g_cp_list = self.cp_shard(g, cp_degree=cp_degree)
        beta_cp_list = self.cp_shard(beta, cp_degree=cp_degree)

        initial_state = h0
        g_out_cp_list = []
        o_cp_list = []
        A_cp_list = []
        for cp_rank in range(cp_degree):
            q, k, v, g, beta = (
                q_cp_list[cp_rank],
                k_cp_list[cp_rank],
                v_cp_list[cp_rank],
                g_cp_list[cp_rank],
                beta_cp_list[cp_rank],
            )
            g_out_cp, o_cp, A_cp, initial_state = chunk_gated_delta_rule_fwd(
                q=F.normalize(q, p=2, dim=-1),
                k=F.normalize(k, p=2, dim=-1),
                v=v,
                g=g,
                beta=beta,
                scale=self.scale,
                initial_state=initial_state,
                output_final_state=True,
            )
            g_out_cp_list.append(g_out_cp)
            o_cp_list.append(o_cp)
            A_cp_list.append(A_cp)

        # Only o is returned from chunk_gated_delta_rule (and final_state, but we don't care about #
        # that).
        assert_close("o", o, self.cp_unshard(o_cp_list), 0.002)
