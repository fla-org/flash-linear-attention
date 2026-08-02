# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.modules.grpo import fused_grpo_loss, grpo_loss_torch, grpo_loss_with_old_logps
from fla.utils import IS_NVIDIA_HOPPER, assert_close, device, device_torch_lib


def grpo_loss_with_old_logps_torch(
    logps: torch.Tensor,
    ref_logps: torch.Tensor,
    old_logps: torch.Tensor,
    pad_mask: torch.Tensor,
    logits_to_keep: int,
    rewards: torch.Tensor,
    beta: float,
    epsilon: float,
) -> torch.Tensor:
    batch_size = logps.shape[0]
    rewards_shaped = rewards.view(-1, batch_size)
    advantages = (rewards_shaped - rewards_shaped.mean(dim=1, keepdim=True)) / (
        rewards_shaped.std(dim=1, keepdim=True) + 1e-8
    )
    advantages = advantages.view(-1, 1)

    log_ratio = logps - old_logps
    importance_weights = torch.exp(log_ratio)
    importance_weights_clipped = torch.clamp(importance_weights, 1 - epsilon, 1 + epsilon)
    per_token_kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
    completion_mask = (torch.arange(logits_to_keep, device=logps.device)[None, :] >= 0) & pad_mask
    per_token_objective = torch.min(
        advantages * importance_weights,
        advantages * importance_weights_clipped,
    ) - beta * per_token_kl
    return -(per_token_objective * completion_mask).sum() / completion_mask.sum()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case", ["unclipped", "clipped", "kl", "masked"])
def test_grpo_loss_with_old_logps(dtype: torch.dtype, case: str):
    device_torch_lib.manual_seed(42)
    batch_size, num_tokens = 4, 5
    logps = (torch.randn(batch_size, num_tokens, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    ref_logps = logps.detach().clone()
    old_logps = logps.detach().clone()
    pad_mask = torch.ones(batch_size, num_tokens, device=device, dtype=torch.bool)
    rewards = torch.tensor([1.5, -0.5, 2.0, -1.0], device=device)
    beta = 0.0
    epsilon = 0.2

    if case == "clipped":
        old_logps = old_logps + torch.tensor(
            [[-0.5], [0.5], [-0.5], [0.5]],
            device=device,
            dtype=dtype,
        )
    elif case == "kl":
        ref_logps = ref_logps + 0.3
        beta = 0.2
    elif case == "masked":
        pad_mask[0, 1:] = False
        pad_mask[2, 3:] = False

    reference_logps = logps.detach().clone().requires_grad_(True)
    expected = grpo_loss_with_old_logps_torch(
        logps=reference_logps,
        ref_logps=ref_logps,
        old_logps=old_logps,
        pad_mask=pad_mask,
        logits_to_keep=num_tokens,
        rewards=rewards,
        beta=beta,
        epsilon=epsilon,
    )
    actual = grpo_loss_with_old_logps(
        logps=logps,
        ref_logps=ref_logps,
        old_logps=old_logps,
        pad_mask=pad_mask,
        logits_to_keep=num_tokens,
        rewards=rewards,
        beta=beta,
        epsilon=epsilon,
    )
    expected.backward()
    actual.backward()

    atol = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
    torch.testing.assert_close(logps.grad, reference_logps.grad, atol=atol, rtol=0)


def test_grpo_loss_with_old_logps_gradient_direction():
    logps = torch.zeros(2, 1, device=device, dtype=torch.float32, requires_grad=True)
    rewards = torch.tensor([1.0, -1.0], device=device)
    loss = grpo_loss_with_old_logps(
        logps=logps,
        ref_logps=torch.zeros_like(logps),
        old_logps=torch.zeros_like(logps),
        pad_mask=torch.ones_like(logps, dtype=torch.bool),
        logits_to_keep=1,
        rewards=rewards,
        beta=0.0,
        epsilon=0.2,
    )
    loss.backward()

    assert logps.grad[0, 0] < 0
    assert logps.grad[1, 0] > 0


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [16, 1024, 4096])
@pytest.mark.parametrize("V", [32000, 65536])
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
            assert (kl2-kl3).abs().max() <= 2e-3
        dy = torch.randn_like(y1) * 10
        y1.backward(dy)
        y2.backward(dy.float())
        assert (y1-y2).abs().max() < 1e-3
        assert_close(" dlogits", gold_logits.grad, logits.grad, 3e-3)
