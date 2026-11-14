# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Real Trainer accumulation of SwiLA's CE and sequence-averaged MI objective."""

import pytest
import torch

from fla.ops.swila.checkpoint_context import get_swila_checkpoint_phase, swila_checkpoint_context_fn
from fla.utils import IS_NVIDIA, device
from tests.models.swila_helpers import compare_model_gradients, model_gradients, model_pair, reference_ce
from tests.ops.swila_helpers import assert_swila_close, assert_swila_grad

pytest.importorskip("accelerate")


class ZeroCE(torch.nn.Module):
    """Keep the CE graph connected while isolating the model's MI backward."""

    ignore_index = -100
    reduction = "mean"

    def forward(self, logits, labels):
        return logits.sum() * 0


def trainer_for(model, directory, configured_steps):
    from transformers import Trainer, TrainingArguments

    arguments = TrainingArguments(
        output_dir=str(directory), per_device_train_batch_size=1,
        gradient_accumulation_steps=configured_steps, max_grad_norm=0.,
        report_to=[], save_strategy="no", logging_strategy="no", eval_strategy="no",
        disable_tqdm=True, dataloader_pin_memory=False,
    )
    trainer = Trainer(model=model, args=arguments)
    assert trainer.accelerator.gradient_accumulation_steps == 1
    return trainer


def accumulated_gradients(trainer, batches, *, mi_only):
    model = trainer.model
    model.criterion = ZeroCE() if mi_only else None
    model.zero_grad(set_to_none=True)
    samples, denominator = trainer.get_batch_samples(iter(batches), len(batches), trainer.args.device)
    trainer.current_gradient_accumulation_steps = len(samples)
    received = []
    hook = model.register_forward_pre_hook(
        lambda module, args, kwargs: received.append(kwargs.get("num_items_in_batch")), with_kwargs=True,
    )
    try:
        losses = [trainer.training_step(model, dict(batch), denominator) for batch in samples]
    finally:
        hook.remove()
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    return gradients, torch.stack(losses).sum(), denominator, received


def reference_gradients(model, batches):
    cross_entropy, auxiliary = [], []
    for batch in batches:
        output = model(batch["input_ids"], use_cache=False)
        cross_entropy.append(reference_ce(output.logits, batch["labels"]))
        auxiliary.append(model.model._total_aux_loss)
    ce, mi = torch.stack(cross_entropy).mean(), torch.stack(auxiliary).mean()
    total = model_gradients(ce + mi, model, retain_graph=True)
    return total, model_gradients(mi, model), ce.detach()


def assert_router_mi(reference, observed):
    for suffix in ("pi_q_proj.out.weight", "pi_k_proj.out.weight"):
        names = [name for name in reference if name.endswith(suffix)]
        assert names, suffix
        for name in names:
            expected = reference[name]
            assert expected is not None and torch.count_nonzero(expected) > 0, name
            assert_swila_grad(f"trainer-mi/{name}", expected, observed[name], require_relative=True, profile=torch.float32)


@pytest.mark.skipif(not IS_NVIDIA, reason="SwiLA Trainer qualification uses real NVIDIA model kernels")
@pytest.mark.parametrize("family,sizes,configured_steps,checkpointed", [
    pytest.param("plain", (1, 1), 2, False, id="plain-equal"),
    pytest.param("temporal_gated", (1, 1), 3, True, id="temporal-gated-remainder-checkpoint"),
    pytest.param("plain", (2, 1), 2, False, id="plain-unequal-sequences-and-labels"),
])
def test_swila_trainer_accumulation_cuda(family, sizes, configured_steps, checkpointed, tmp_path):
    torch.manual_seed(42)
    model, reference = model_pair(family, device=device, use_cache=False)
    tokens = torch.tensor([[1, 3, 7, 9, 4], [1, 8, 2, 6, 9], [1, 4, 8, 3, 2]], device=device)[:sum(sizes)]
    labels = tokens.clone()
    labels[:, 0] = -100
    if sizes[0] != sizes[1]:
        labels[0, 2:] = -100
        labels[1, 3] = -100
    batches, start = [], 0
    for size in sizes:
        batches.append({"input_ids": tokens[start:start + size], "labels": labels[start:start + size], "use_cache": False})
        start += size

    expected_total, expected_mi, expected_ce = reference_gradients(reference, batches)
    if sizes[0] == sizes[1]:
        whole_total, whole_mi, whole_ce = reference_gradients(
            reference, [{"input_ids": tokens, "labels": labels}],
        )
        compare_model_gradients(whole_total, expected_total, profile=torch.float32)
        compare_model_gradients(whole_mi, expected_mi, profile=torch.float32)
        assert_swila_close("whole-batch-ce", whole_ce, expected_ce)

    phases = []
    if checkpointed:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            "use_reentrant": False, "context_fn": swila_checkpoint_context_fn,
        })
        assert model.is_gradient_checkpointing
    hooks = [layer.attn.register_forward_pre_hook(lambda module, args: phases.append(get_swila_checkpoint_phase()))
             for layer in model.model.layers]
    trainer = trainer_for(model, tmp_path / "ordinary", configured_steps)
    try:
        observed_mi, zero_loss, denominator, received = accumulated_gradients(trainer, batches, mi_only=True)
        # Numerical checks precede the protocol assertion so the unfixed code
        # demonstrates the MI gradient defect rather than only a missing flag.
        assert_router_mi(expected_mi, observed_mi)
        compare_model_gradients(expected_mi, observed_mi, profile=torch.float32)
        assert zero_loss.item() == 0.
        observed_total, returned_ce, _, _ = accumulated_gradients(trainer, batches, mi_only=False)
        compare_model_gradients(expected_total, observed_total, profile=torch.float32)
        assert_swila_close("trainer-returned-ce", expected_ce, returned_ce)
        assert trainer.model_accepts_loss_kwargs is False
        assert denominator is None and received == [None] * len(batches)
        assert trainer.current_gradient_accumulation_steps == len(batches)
        if checkpointed:
            assert phases.count("prepass") == phases.count("recompute") == 2 * len(batches)
            assert "normal" not in phases and get_swila_checkpoint_phase() == "normal"
    finally:
        for hook in hooks:
            hook.remove()

    if family == "plain" and sizes == (1, 1):
        # The old automatic denominator path leaves MI unscaled on every call.
        model.accepts_loss_kwargs = True
        mutated = trainer_for(model, tmp_path / "mutated", configured_steps)
        old_mi, _, old_denominator, _ = accumulated_gradients(mutated, batches, mi_only=True)
        doubled = {name: None if value is None else len(batches) * value for name, value in expected_mi.items()}
        compare_model_gradients(doubled, old_mi, profile=torch.float32)
        assert_router_mi(doubled, old_mi)
        assert old_denominator is not None
        with pytest.raises(AssertionError):
            assert_router_mi(expected_mi, old_mi)
