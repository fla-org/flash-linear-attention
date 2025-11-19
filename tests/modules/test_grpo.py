# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import pytest
import torch

from fla.modules.grpo import fused_grpo_loss, grpo_loss_torch
from fla.utils import IS_NVIDIA_HOPPER, assert_close, device, device_torch_lib


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [16, 1024, 4096])
@pytest.mark.parametrize("V", [32000, 65536, 131072])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("repeat", [100])
def test_fused_grpos(B: int, T: int, V: int, dtype: torch.dtype, inplace: bool, repeat: int):
    device_torch_lib.manual_seed(42)
    for i in range(repeat):
        if not IS_NVIDIA_HOPPER and T == 4096:
            pytest.skip("Skip test for T=4096 on Intel Alchemist")

        def get_random_ref_log_probs(logits, input_ids):
            with torch.inference_mode():
                logits = logits[:, :-1]
                per_token_logps = []
                for logits_row, input_ids_row in zip(logits, input_ids[:, -logits.size(1):], strict=False):
                    log_probs = torch.randn_like(logits_row).log_softmax(dim=-1)
                    token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                    per_token_logps.append(token_log_prob)
                device_torch_lib.empty_cache()
                return torch.stack(per_token_logps)

        logits = torch.randn(B, T + 1, V, device=device, dtype=dtype)
        logits.requires_grad_(True)
        advantages = torch.randn(B, device=device, dtype=torch.float32)
        input_ids = torch.randint(0, V-1, (B, T + 64), device=device)
        ref_logp = get_random_ref_log_probs(logits, input_ids)
        beta = 0.04
        completion_mask = torch.ones(B, T, dtype=torch.int32, device=device)
        completion_mask[::2, T//3: T//2] = 0
        save_kl = True

        gold_logits = logits.detach().clone().float()
        gold_logits.requires_grad_(True)
        gold_ref_logp = ref_logp.clone().float()
        device_torch_lib.empty_cache()
        y1 = fused_grpo_loss(logits, ref_logp, input_ids, advantages, beta, completion_mask, save_kl=save_kl, inplace=inplace)
        y2 = grpo_loss_torch(gold_logits, gold_ref_logp, input_ids, advantages, beta, completion_mask, save_kl)
        if save_kl:
            y1, kl2 = y1
            y2, kl3 = y2
            assert (kl2-kl3).abs().max() < 1e-3
        dy = torch.randn_like(y1) * 10
        y1.backward(dy)
        y2.backward(dy.float())
        assert (y1-y2).abs().max() < 1e-3
        assert_close(" dlogits", gold_logits.grad, logits.grad, 3e-3)
