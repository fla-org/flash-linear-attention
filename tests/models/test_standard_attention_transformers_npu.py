# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch_npu  # noqa: F401  # register the NPU torch backend before model construction
from transformers import AutoModelForCausalLM

from fla.layers.attn import Attention
from fla.models import GLAConfig
from fla.utils import IS_NPU, device


def _npu_available():
    try:
        return IS_NPU and torch.npu.is_available()
    except (RuntimeError, OSError):
        return False


pytestmark = pytest.mark.skipif(not _npu_available(), reason="requires an available Ascend NPU")


def _config():
    return GLAConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_heads=4,
        num_kv_heads=2,
        vocab_size=64,
        max_position_embeddings=64,
        attn={"layers": [0], "num_heads": 4, "num_kv_heads": 2},
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        fuse_linear_cross_entropy=False,
        use_cache=True,
    )


def _model():
    model = AutoModelForCausalLM.from_config(_config())
    model = model.to(device=device, dtype=torch.float16).eval()
    assert isinstance(model.model.layers[0].attn, Attention)
    return model


def test_transformers_model_forward_inputs_embeds_loss_and_save_load(tmp_path, monkeypatch):
    torch.manual_seed(11)
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "auto")
    model = _model()
    input_ids = torch.tensor([[1, 5, 7, 9], [2, 6, 8, 10]], device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    with torch.no_grad():
        dense = model(input_ids=input_ids, use_cache=False).logits
        masked = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        embeds = model.get_input_embeddings()(input_ids)
        embedded = model(inputs_embeds=embeds, attention_mask=attention_mask, use_cache=False).logits

    torch.testing.assert_close(dense, masked, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(masked, embedded, rtol=5e-2, atol=5e-2)

    model.train()
    loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())

    model.eval()
    with torch.no_grad():
        model.save_pretrained(tmp_path)
        loaded = AutoModelForCausalLM.from_pretrained(tmp_path, torch_dtype=torch.float16).to(device).eval()
        reloaded = loaded(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(dense, reloaded, rtol=5e-2, atol=5e-2)


def test_transformers_model_cache_and_generate(monkeypatch):
    torch.manual_seed(12)
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "auto")
    model = _model()
    input_ids = torch.tensor([[1, 5, 7, 9, 11]], device=device)
    prefix = input_ids[:, :3]
    prefix_mask = torch.ones_like(prefix, dtype=torch.bool)
    full_mask = torch.ones_like(input_ids, dtype=torch.bool)

    with torch.no_grad():
        full = model(input_ids=input_ids, use_cache=False).logits
        prefill = model(input_ids=prefix, attention_mask=prefix_mask, use_cache=True)
        step = model(
            input_ids=input_ids[:, 3:4],
            attention_mask=full_mask[:, :4],
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
        manual_prefill = model(
            input_ids=prefix,
            attention_mask=prefix_mask,
            use_cache=True,
        )
        manual_next = manual_prefill.logits[:, -1:].argmax(dim=-1)
        manual_tokens = [manual_next]
        manual_cache = manual_prefill.past_key_values
        for _ in range(1, 2):
            manual_step = model(
                input_ids=manual_next,
                attention_mask=torch.ones(
                    (prefix.shape[0], prefix.shape[1] + len(manual_tokens)),
                    dtype=torch.bool,
                    device=device,
                ),
                past_key_values=manual_cache,
                use_cache=True,
            )
            manual_cache = manual_step.past_key_values
            manual_next = manual_step.logits[:, -1:].argmax(dim=-1)
            manual_tokens.append(manual_next)
        manual_generated = torch.cat(manual_tokens, dim=1)
        generated = model.generate(
            prefix,
            attention_mask=prefix_mask,
            max_new_tokens=2,
            do_sample=False,
        )

    torch.testing.assert_close(full[:, 3], step.logits[:, 0], rtol=5e-2, atol=5e-2)
    assert generated.shape == (1, prefix.shape[1] + 2)
    assert torch.equal(generated[:, :prefix.shape[1]], prefix)
    assert torch.equal(generated[:, prefix.shape[1]:], manual_generated)
