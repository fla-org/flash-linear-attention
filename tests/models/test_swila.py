# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Real Transformers loading and CUDA model contracts for the SwiLA port."""

import copy
import json
import math
from dataclasses import fields

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.layers import SwitchingLinearAttention
from fla.models.cache_utils import Cache
from fla.models.swila import SwiLAConfig, SwiLAForCausalLM, SwiLAModel
from fla.modules.swila import capture_swila_diagnostics
from fla.ops.swila.checkpoint_context import get_swila_checkpoint_phase, swila_checkpoint_context_fn
from fla.utils import IS_NVIDIA, device
from tests.layers.swila_helpers import compare_tree, eager_recurrence
from tests.models.swila_helpers import (
    assert_complete_loading,
    assert_loaded_state,
    cloned_state,
    compare_model_gradients,
    historical_config,
    model_config,
    model_gradients,
    model_pair,
    reference_ce,
    trained_state,
)
from tests.ops.swila_helpers import assert_swila_close, assert_swila_grad

requires_cuda = pytest.mark.skipif(not IS_NVIDIA, reason="SwiLA CUDA model qualification requires NVIDIA")


@pytest.mark.parametrize("run", ("0353524c", "908eda67"))
def test_swila_historical_config_roundtrip_cpu(run, tmp_path):
    raw = historical_config(run)
    raw["architectures"] = ["SLAForCausalLM"]
    original = json.dumps(raw)
    (tmp_path / "config.json").write_text(original)
    config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=False, local_files_only=True)
    assert type(config) is SwiLAConfig and config.model_type == "swila"
    assert config.architectures == ["SwiLAForCausalLM"]
    assert config.temporal == raw["temporal"] and config.use_forget_gate
    retired = {
        "b_granularity", "alpha_granularity", "decay_residual_mode", "skip_reset_parameters",
        "pi_input_proj_narrow", "pi_init_orthogonal", "conv_activation", "pi_conv_activation",
        "qk_activation", "pi_activation", "use_input_pi", "init_state_learnable_init",
        "num_deer_iterations", "chunk_size", "full_jacobian", "pred_init_mode", "l_max",
        "gate_init_mode", "gate_min_half_life", "gate_max_half_life", "tie_responsibilities",
    }
    assert not retired.intersection(vars(config))
    assert not retired.intersection(config.to_dict())
    assert (tmp_path / "config.json").read_text() == original
    config.update({"b_granularity": "value", "alpha_granularity": "key", "decay_residual_mode": "post_decay"})
    destination = tmp_path / "canonical"
    config.save_pretrained(destination)
    restored = SwiLAConfig.from_pretrained(destination, b_granularity="value", alpha_granularity="key")
    assert restored.model_type == "swila" and not retired.intersection(restored.to_dict())
    assert restored.to_dict()["temporal"] == raw["temporal"]
    assert type(AutoModel.from_config(restored)) is SwiLAModel
    assert type(AutoModelForCausalLM.from_config(restored)) is SwiLAForCausalLM
    assert type(AutoConfig.for_model("sla", hidden_size=16)) is SwiLAConfig
    assert type(AutoConfig.for_model("swila", hidden_size=16)) is SwiLAConfig


@pytest.mark.parametrize("key,value,error", [
    ("b_granularity", "value_expert", ValueError),
    ("alpha_granularity", "key_expert", ValueError),
    ("decay_residual_mode", "pre_decay", ValueError),
    ("pi_activation", "sigmoid", ValueError),
    ("skip_reset_parameters", True, ValueError),
    ("pi_init_orthogonal", True, ValueError),
    ("mode", "chunk", ValueError),
    ("temporal_transition_mode", "exact_identity", TypeError),
])
def test_swila_config_rejects_conflicting_options_cpu(key, value, error, tmp_path):
    with pytest.raises(error):
        SwiLAConfig(**{key: value})
    config = model_config()
    before = copy.deepcopy(config.to_dict())
    with pytest.raises(error):
        config.update({key: value})
    assert config.to_dict() == before
    with pytest.raises(error):
        SwiLAConfig.from_dict(before, **{key: value})
    config.save_pretrained(tmp_path)
    with pytest.raises(error):
        SwiLAConfig.from_pretrained(tmp_path, **{key: value})


@pytest.mark.parametrize("options,error", [
    ({"return_output_only": True}, ValueError),
    ({"unknown_swila_option": True}, TypeError),
])
def test_swila_config_constructor_rejection_cpu(options, error):
    with pytest.raises(error):
        SwiLAConfig(**options)


def test_swila_downstream_metadata_roundtrip_cpu(tmp_path):
    metadata = {"enabled": True, "options": {"mode": "chunk", "mi_loss_weight": 17.0}}
    original = copy.deepcopy(metadata)
    baseline = model_config()
    baseline_dict = baseline.to_dict()
    config = model_config(downstream_config=metadata)
    actual = config.to_dict()
    assert actual.pop("downstream_config") == original
    assert actual == baseline_dict and metadata == original
    assert "downstream_config" not in baseline_dict
    assert SwiLAConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
    overridden, unused = SwiLAConfig.from_dict(baseline_dict, downstream_config=original, return_unused_kwargs=True)
    assert not unused and overridden.to_dict() == config.to_dict()
    assert "downstream_config" not in baseline_dict

    baseline.save_pretrained(tmp_path)
    loaded = SwiLAConfig.from_pretrained(tmp_path, downstream_config=original, local_files_only=True)
    assert loaded.downstream_config == original
    replacement = {"enabled": False, "annotation": ["training", "evaluation"]}
    loaded.update({"downstream_config": replacement})
    loaded.save_pretrained(tmp_path)
    assert SwiLAConfig.from_pretrained(tmp_path, local_files_only=True).downstream_config == replacement
    with pytest.raises(TypeError, match="unknown_swila_option"):
        SwiLAConfig(downstream_config=metadata, unknown_swila_option=True)
    with pytest.raises(ValueError, match="b_granularity"):
        SwiLAConfig(downstream_config=metadata, b_granularity="value_expert")

    torch.manual_seed(42)
    expected_model = SwiLAForCausalLM(baseline)
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    actual_model = SwiLAForCausalLM(config)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    expected_state = expected_model.state_dict()
    actual_state = actual_model.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for name, value in actual_state.items():
        torch.testing.assert_close(value, expected_state[name], atol=0, rtol=0, msg=name)


@pytest.mark.parametrize("attn", [
    {"layers": [1], "num_heads": 2, "annotation": "preserve"},
    [{"layers": [1], "num_heads": 2, "num_kv_heads": 1, "window_size": 5}],
])
def test_swila_hybrid_normalization_cpu(attn, tmp_path):
    original = copy.deepcopy(attn)
    expected = [{"layers": [1], "num_heads": 2, "num_kv_heads": 2, "qkv_bias": False,
                 "window_size": None, "rope_theta": 10000., "annotation": "preserve"}]
    if isinstance(attn, list):
        expected = [{"layers": [1], "num_heads": 2, "num_kv_heads": 1, "qkv_bias": False,
                     "window_size": 5, "rope_theta": 10000.}]
    config = model_config(num_hidden_layers=2, attn=attn)
    assert attn == original
    assert (config.attn if isinstance(config.attn, list) else [config.attn]) == expected
    config.save_pretrained(tmp_path)
    restored = SwiLAConfig.from_pretrained(tmp_path)
    assert restored.attn == config.attn
    changed = model_config(num_hidden_layers=2)
    changed.update({"attn": attn})
    assert changed.attn == config.attn and attn == original
    model = SwiLAForCausalLM(restored)
    assert isinstance(model.model.layers[0].attn, SwitchingLinearAttention)
    assert model.model.layers[1].is_transformer_attention
    assert not isinstance(model.model.layers[1].attn, SwitchingLinearAttention)


@pytest.mark.parametrize("attn", [
    {"layers": [0, 0], "num_heads": 2},
    {"layers": [2], "num_heads": 2},
    {"layers": [True], "num_heads": 2},
    {"layers": [0], "num_heads": 0},
    [{"layers": [0], "num_heads": 2}, {"layers": [0], "num_heads": 2}],
])
def test_swila_invalid_hybrid_cpu(attn):
    with pytest.raises((ValueError, TypeError)):
        model_config(num_hidden_layers=2, attn=attn)


@pytest.mark.parametrize("attn", (None, []))
def test_swila_empty_hybrid_seed_and_metadata_cpu(attn):
    # Preserve the source's named seed regression, including the next RNG state.
    torch.manual_seed(1729)
    expected = SwiLAForCausalLM(model_config("temporal_gated", attn=None))
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(1729)
    actual = SwiLAForCausalLM(model_config("temporal_gated", attn=attn))
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    expected_parameters = dict(expected.named_parameters())
    assert dict(actual.named_parameters()).keys() == expected_parameters.keys()
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(parameter, expected_parameters[name], atol=0, rtol=0)
        for marker in ("_no_weight_decay", "_is_hf_initialized", "_no_reinit"):
            assert getattr(parameter, marker, None) == getattr(expected_parameters[name], marker, None)


@pytest.mark.parametrize("run,dtype,tied", [
    ("0353524c", torch.float32, False),
    ("0353524c", torch.bfloat16, True),
    ("908eda67", torch.bfloat16, False),
    ("908eda67", torch.float16, True),
])
def test_swila_exact_trained_checkpoint_loading_cpu(run, dtype, tied, tmp_path):
    torch.manual_seed(42)
    raw = historical_config(run) | {"tie_word_embeddings": tied}
    model = SwiLAForCausalLM(SwiLAConfig(**raw))
    assert (model.lm_head.weight is model.model.embeddings.weight) is tied
    expected = trained_state(model)
    model.save_pretrained(tmp_path)
    # Serialized remote-code hints cannot select a historical implementation.
    raw.update(architectures=["SLAForCausalLM"], auto_map={"AutoModelForCausalLM": "obsolete.Model"})
    original = json.dumps(raw)
    (tmp_path / "config.json").write_text(original)
    loaded, info = AutoModelForCausalLM.from_pretrained(
        tmp_path, dtype=dtype, trust_remote_code=False, local_files_only=True, output_loading_info=True,
    )
    assert type(loaded) is SwiLAForCausalLM and loaded.__class__.__module__ == "fla.models.swila.modeling_swila"
    assert_complete_loading(info)
    assert_loaded_state(loaded, expected, dtype)
    assert (loaded.lm_head.weight is loaded.model.embeddings.weight) is tied
    assert (tmp_path / "config.json").read_text() == original
    layer = loaded.model.layers[0].attn
    assert torch.count_nonzero(layer.init_state)
    assert not torch.equal(layer.pi_q_input_proj.weight, layer.pi_k_input_proj.weight)
    assert not torch.equal(expected["model.layers.0.attn.A_log"], expected["model.layers.0.attn.A_log"].bfloat16().float())
    exported = tmp_path / "exported"
    loaded.save_pretrained(exported)
    metadata = json.loads((exported / "config.json").read_text())
    assert metadata["model_type"] == "swila" and metadata["architectures"] == ["SwiLAForCausalLM"]
    assert not {"b_granularity", "alpha_granularity", "decay_residual_mode"}.intersection(metadata)
    reloaded, info = SwiLAForCausalLM.from_pretrained(exported, dtype=dtype, output_loading_info=True)
    assert_complete_loading(info)
    assert_loaded_state(reloaded, expected, dtype)


@pytest.mark.parametrize("defect", ("missing", "unexpected", "mismatched"))
def test_swila_incomplete_checkpoint_is_detected_cpu(defect, tmp_path):
    model = SwiLAForCausalLM(model_config("temporal_gated"))
    expected = trained_state(model)
    model.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    name = "model.layers.0.attn.pi_q_input_proj.weight"
    if defect == "missing":
        del state[name]
    elif defect == "unexpected":
        state["unexpected.weight"] = torch.ones(2)
    else:
        state[name] = state[name][:-1]
    save_file(state, path, metadata={"format": "pt"})
    if defect == "mismatched":
        with pytest.raises((ValueError, RuntimeError), match="(?i)size|shape|mismatch"):
            SwiLAForCausalLM.from_pretrained(tmp_path, output_loading_info=True)
        return
    loaded, info = SwiLAForCausalLM.from_pretrained(tmp_path, output_loading_info=True)
    with pytest.raises(AssertionError, match=f"{defect}_keys"):
        assert_complete_loading(info)
    for key, value in loaded.state_dict().items():
        if key != name:
            torch.testing.assert_close(value, expected[key], atol=0, rtol=0, msg=key)


@pytest.mark.parametrize("historical", (False, True))
def test_swila_meta_materialization_and_profile_cpu(historical):
    profile = dict(architecture_profile="historical_8dc05f93", b_low_rank_proj_type="swiglu", pi_align_qk_init=False)
    config = model_config(**(profile if historical else {}))
    with torch.device("meta"):
        model = SwiLAForCausalLM(config)
    assert all(parameter.is_meta for parameter in model.parameters())
    with pytest.raises(RuntimeError, match="unmaterialized meta"):
        model.post_init()
    # Match Flame's explicit meta -> cleared HF markers -> materialization lifecycle.
    model.apply(lambda module: setattr(module, "_is_hf_initialized", False))
    model.to_empty(device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
    torch.manual_seed(42)
    model.post_init()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    layer = model.model.layers[0].attn
    assert torch.count_nonzero(layer.init_state) == 0
    terminal = layer.b_proj.out if historical else layer.b_proj.up
    assert terminal.weight.shape[0] == (32 if historical else 8)


def test_swila_partial_meta_rejected_cpu():
    model = SwiLAForCausalLM(model_config())
    model.model.layers[0].attn.q_proj.weight = torch.nn.Parameter(torch.empty(16, 16, device="meta"))
    with pytest.raises(RuntimeError, match="unmaterialized meta"):
        model.post_init()


def test_swila_historical_profile_override_atomicity_cpu(tmp_path):
    profile = dict(architecture_profile="historical_8dc05f93", b_low_rank_proj_type="swiglu", pi_align_qk_init=False)
    config = model_config(**profile)
    config.save_pretrained(tmp_path)
    assert SwiLAConfig.from_pretrained(tmp_path).architecture_profile == "historical_8dc05f93"
    expected = copy.deepcopy(config.to_dict())
    for update in ({"temporal": True}, {"b_granularity": "value"}, {"b_low_rank_proj_type": "linear"}):
        with pytest.raises(ValueError):
            config.update(update)
        assert config.to_dict() == expected
        with pytest.raises(ValueError):
            SwiLAConfig.from_dict(expected, **update)


def test_swila_hf_metadata_aliases_cpu(tmp_path):
    config = model_config()
    config.save_pretrained(tmp_path)
    loaded = AutoConfig.from_pretrained(tmp_path, num_labels=3, local_files_only=True)
    assert type(loaded) is SwiLAConfig and loaded.num_labels == 3
    assert loaded.name_or_path == str(tmp_path)
    loaded.save_pretrained(tmp_path / "exported")
    restored = AutoConfig.from_pretrained(tmp_path / "exported", local_files_only=True)
    assert restored.num_labels == 3 and restored.id2label == loaded.id2label


def test_swila_legacy_gate_noise_bias_repair_cpu(tmp_path):
    model = SwiLAForCausalLM(model_config("temporal_gated", inject_gate_noise=True))
    expected = trained_state(model)
    model.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    missing = {f"model.layers.0.attn.W_noise_gate_{side}.bias" for side in ("q", "kv")}
    for name in missing:
        del state[name]
    save_file(state, path, metadata={"format": "pt"})
    loaded, info = SwiLAForCausalLM.from_pretrained(tmp_path, output_loading_info=True)
    # HF reports which values were absent, even though SwiLA fills only these
    # compatibility biases. Strict complete-checkpoint callers still see it.
    assert set(info["missing_keys"]) == missing
    assert not any(info.get(key) for key in ("unexpected_keys", "mismatched_keys", "error_msgs"))
    for name, value in loaded.state_dict().items():
        if name in missing:
            target = torch.full_like(value, math.log(math.expm1(model.config.gate_noise_init_std)))
        else:
            target = expected[name]
        torch.testing.assert_close(value, target, rtol=0, atol=0, msg=name)
    for side in ("q", "kv"):
        assert getattr(loaded.model.layers[0].attn, f"W_noise_gate_{side}").bias._no_weight_decay


@requires_cuda
@pytest.mark.parametrize("family,dtype", [
    ("plain", torch.float32), ("gated", torch.bfloat16),
    ("temporal", torch.float32), ("temporal_gated", torch.bfloat16),
])
def test_swila_model_ce_mi_reference_cuda(family, dtype):
    torch.manual_seed(42)
    actual, reference = model_pair(family, dtype=dtype, device=device, num_hidden_layers=2)
    tokens = torch.tensor([[1, 3, 7, 9, 4, 2, 5], [1, 8, 2, 6, 9, 3, 7]], device=device)
    labels = tokens.clone()
    labels[0, 3] = -100
    layer_losses = []
    handles = [layer.register_forward_hook(lambda module, args, output: layer_losses.append(output[3]))
               for layer in reference.model.layers]
    observed = actual(tokens, labels=labels, use_cache=False, output_attentions=True, output_hidden_states=True)
    with eager_recurrence():
        expected = reference(tokens, labels=labels, use_cache=False, output_attentions=True, output_hidden_states=True)
    for handle in handles:
        handle.remove()
    assert len(layer_losses) == 2 and all(value.requires_grad for value in layer_losses)
    auxiliary = sum(layer_losses)
    compare_tree(auxiliary, reference.model._total_aux_loss, prefix="explicit-layer-mi-sum", profile=dtype)
    compare_tree(expected.logits, observed.logits, prefix="model-logits", profile=dtype)
    compare_tree(expected.hidden_states, observed.hidden_states, prefix="model-hidden", profile=dtype)
    compare_tree(expected.attentions, observed.attentions, prefix="model-responsibilities", profile=dtype)
    ce = reference_ce(expected.logits, labels).to(observed.loss.dtype)
    assert_swila_close("reported-ce", ce, observed.loss, profile=dtype)

    # An independent CE+sum(layer-MI) objective checks the model aggregation,
    # rather than comparing two executions of the same loss-assembly method.
    expected_gradients = model_gradients(ce + auxiliary, reference, retain_graph=True)
    actual_gradients = model_gradients(observed.loss, actual, retain_graph=True)
    compare_model_gradients(expected_gradients, actual_gradients, profile=dtype)
    scale = tokens.numel() * 2 * 4 / actual.config.mi_loss_weight
    expected_mi = model_gradients(scale * auxiliary, reference)
    actual_mi = model_gradients(scale * actual.model._total_aux_loss, actual)
    compare_model_gradients(expected_mi, actual_mi, profile=dtype)
    for name in ("model.embeddings.weight", "model.layers.0.attn.pi_q_input_proj.weight",
                 "model.layers.0.attn.pi_k_input_proj.weight"):
        assert expected_mi[name] is not None and torch.count_nonzero(expected_mi[name])
        assert_swila_grad(name, expected_mi[name], actual_mi[name], profile=dtype, require_relative=True)
        with pytest.raises(AssertionError):
            assert_swila_grad(name, expected_mi[name], torch.zeros_like(expected_mi[name]),
                              profile=dtype, require_relative=True)


@requires_cuda
@pytest.mark.parametrize("family,hybrid", [
    ("plain", False), ("gated", False), ("temporal", False),
    ("temporal_gated", False), ("temporal_gated", True),
])
def test_swila_model_generation_cache_cuda(family, hybrid):
    torch.manual_seed(42)
    options = dict(mi_loss_weight=0.)
    if hybrid:
        options.update(num_hidden_layers=2, attn={"layers": [1], "num_heads": 2, "num_kv_heads": 1})
    model = SwiLAForCausalLM(model_config(family, **options)).to(device=device, dtype=torch.bfloat16).eval()
    tokens = torch.tensor([[1, 7, 11, 19, 3]], device=device)
    calls = []

    def record_call(module, args, kwargs):
        cache = kwargs.get("past_key_values")
        calls.append((kwargs["input_ids"].shape[1], cache.get_seq_length(0) if cache is not None else 0))

    handle = model.register_forward_pre_hook(record_call, with_kwargs=True)
    with torch.no_grad():
        generated = model.generate(
            tokens, max_new_tokens=3, min_new_tokens=3, do_sample=False,
            use_cache=True, return_dict_in_generate=True, output_logits=True,
        )
    handle.remove()
    assert calls == [(5, 0), (1, 5), (1, 6)]
    sequence = tokens
    with torch.no_grad():
        for logits in generated.logits:
            full = model(sequence, use_cache=False).logits[:, -1]
            # Transformers stores generated raw logits in FP32 after a lossless
            # cast; the BF16 model's numerical comparison policy still applies.
            assert full.dtype == torch.bfloat16 and logits.dtype == torch.float32
            assert_swila_close("cached-next-token", full.float(), logits, profile=torch.bfloat16)
            sequence = torch.cat((sequence, full.argmax(-1, keepdim=True)), dim=1)
    assert torch.equal(sequence, generated.sequences)
    assert isinstance(generated.past_key_values, Cache)
    for index, block in enumerate(model.model.layers):
        assert generated.past_key_values.get_seq_length(index) == 7
        state = generated.past_key_values[index]
        if block.is_transformer_attention:
            assert len(state["attn_state"]) == 2
            assert state["attn_state"][0].shape[1] == 7
        else:
            assert len(state["conv_state"]) == 5 and all(value is not None for value in state["conv_state"])
            if block.attn.temporal:
                assert isinstance(state["recurrent_state"], tuple) and len(state["recurrent_state"]) == 3
    # Supplied caches advance even when callers suppress returning the cache.
    cache = generated.past_key_values
    with torch.no_grad():
        full = model(sequence, use_cache=False).logits[:, -1:]
        observed = model(sequence[:, -1:], past_key_values=cache, use_cache=False).logits
    assert_swila_close("supplied-cache-with-return-disabled", full, observed, profile=torch.bfloat16)
    assert cache.get_seq_length(0) == 8


@requires_cuda
def test_swila_model_left_padding_cuda():
    torch.manual_seed(42)
    model = SwiLAForCausalLM(model_config("temporal_gated", mi_loss_weight=0.)).to(
        device=device, dtype=torch.bfloat16,
    ).eval()
    tokens = torch.tensor([[0, 1, 7, 9], [0, 0, 1, 3]], device=device)
    mask = tokens.ne(0).long()
    next_tokens = torch.tensor([[5], [8]], device=device)
    with torch.no_grad():
        padded = model(tokens, attention_mask=mask, use_cache=True)
        decoded = model(next_tokens, attention_mask=torch.cat((mask, torch.ones_like(next_tokens)), 1),
                        past_key_values=padded.past_key_values, use_cache=True)
        for batch, start in enumerate((1, 2)):
            unpadded = model(tokens[batch:batch + 1, start:], use_cache=True)
            compare_tree(unpadded.logits, padded.logits[batch:batch + 1, start:], profile=torch.bfloat16)
            expected = model(next_tokens[batch:batch + 1], past_key_values=unpadded.past_key_values, use_cache=True)
            compare_tree(expected.logits, decoded.logits[batch:batch + 1], profile=torch.bfloat16)
            actual_state = cloned_state(decoded.past_key_values, 0)
            expected_state = cloned_state(expected.past_key_values, 0)
            for name in ("recurrent_state", "conv_state"):
                def slice_batch(value):
                    if isinstance(value, torch.Tensor):
                        return value[batch:batch + 1]
                    return tuple(slice_batch(item) for item in value)
                compare_tree(expected_state[name], slice_batch(actual_state[name]), prefix=name, profile=torch.bfloat16)
            assert expected.past_key_values.get_seq_length(0) == 5 - start
    assert decoded.past_key_values.get_seq_length(0) == 5


@requires_cuda
@pytest.mark.parametrize("kind", ("custom", "fused", "fused_linear"))
def test_swila_model_loss_interfaces_cuda(kind):
    torch.manual_seed(42)
    options = dict(fuse_cross_entropy=kind == "fused", fuse_linear_cross_entropy=kind == "fused_linear",
                   use_l2warp=True)
    model, reference = model_pair("temporal_gated", device=device, **options)
    reference.config.fuse_cross_entropy = reference.config.fuse_linear_cross_entropy = False
    reference.config.use_l2warp = False
    if kind == "custom":
        model.criterion = torch.nn.CrossEntropyLoss(label_smoothing=.1)
    tokens = torch.tensor([[1, 3, 7, 9, 4], [1, 8, 2, 6, 9]], device=device)
    labels = tokens.clone()
    labels[0, 2] = -100
    shifted = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], -100)), 1)
    denominator = torch.tensor(23, device=device)
    observed = model(tokens, labels=labels, logits_to_keep=1, use_cache=False)
    expected = reference(tokens, use_cache=False)
    ce = torch.nn.functional.cross_entropy(
        expected.logits.reshape(-1, 41), shifted.reshape(-1), label_smoothing=.1 if kind == "custom" else 0.,
    )
    assert_swila_close("default-loss", ce, observed.loss)
    assert (observed.logits is None) is (kind == "fused_linear")
    if observed.logits is not None:
        assert observed.logits.shape == (2, 5, 41)
        compare_tree(expected.logits, observed.logits)
    # Independent max-logit penalty has the same VJP as l2_warp while leaving
    # the returned loss value unchanged. It is not a call to the shared helper.
    maxima = expected.logits.max(-1).values
    penalty = .5e-4 / tokens.numel() * maxima.square().sum()
    objective = ce + reference.model._total_aux_loss + penalty
    compare_model_gradients(model_gradients(objective, reference), model_gradients(observed.loss, model),
                            profile=torch.float32)

    observed_tuple = model(tokens, shift_labels=shifted, num_items_in_batch=denominator, logits_to_keep=2,
                           use_cache=False, return_dict=False)
    expected = reference(tokens, use_cache=False)
    ce_sum = torch.nn.functional.cross_entropy(
        expected.logits.reshape(-1, 41), shifted.reshape(-1), reduction="sum",
        label_smoothing=.1 if kind == "custom" else 0.,
    ) / denominator
    assert_swila_close("explicit-shift-and-denominator", ce_sum, observed_tuple[0])
    assert (observed_tuple[1] is None) is (kind == "fused_linear")
    # The shared FLA helper divides the wrapped CE (including its penalty VJP)
    # by num_items_in_batch, but never rescales SwiLA's independently reduced MI.
    penalty = .5e-4 / tokens.numel() * expected.logits.max(-1).values.square().sum() / denominator
    objective = ce_sum + reference.model._total_aux_loss + penalty
    compare_model_gradients(model_gradients(objective, reference), model_gradients(observed_tuple[0], model),
                            profile=torch.float32)
    with torch.no_grad():
        full = model(tokens, use_cache=False).logits
        for selection in (None, 0, 2, torch.tensor([0, 3], device=device)):
            selected = model(tokens, logits_to_keep=selection, use_cache=False, return_dict=False)[0]
            target = full if selection is None or isinstance(selection, int) and selection == 0 else (
                full[:, -selection:] if isinstance(selection, int) else full[:, selection]
            )
            compare_tree(target, selected, prefix="logits-selection")


@requires_cuda
@pytest.mark.parametrize("family", ("plain", "temporal_gated"))
def test_swila_hf_checkpoint_mi_cuda(family):
    torch.manual_seed(42)
    noisy = family == "temporal_gated"
    model, expected_model = model_pair(
        family, device=device, num_hidden_layers=2, inject_noise=noisy, inject_gate_noise=noisy,
    )
    tokens = torch.tensor([[1, 7, 2, 9, 3, 6]], device=device)
    normal_phases = []
    handles = [layer.register_forward_pre_hook(lambda module, args: normal_phases.append(get_swila_checkpoint_phase()))
               for layer in expected_model.model.layers]
    torch.manual_seed(42)
    expected = expected_model(tokens, labels=tokens, use_cache=False)
    expected_auxiliary = expected_model.model._total_aux_loss
    expected_gradients = model_gradients(expected.loss, expected_model)
    expected_cpu_rng = torch.random.get_rng_state().clone()
    expected_cuda_rng = torch.cuda.get_rng_state(device).clone()
    for handle in handles:
        handle.remove()
    assert normal_phases == ["normal", "normal"]

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        "use_reentrant": False, "context_fn": swila_checkpoint_context_fn,
    })
    assert model.is_gradient_checkpointing
    phases = []
    # Observe inside SwiLABlock so Transformers' checkpoint wrapper has already
    # entered the correct context before the attention call is made.
    handles = [layer.attn.register_forward_pre_hook(lambda module, args: phases.append(get_swila_checkpoint_phase()))
               for layer in model.model.layers]
    torch.manual_seed(42)
    with capture_swila_diagnostics(model, retain_tensors=True) as capture:
        actual = model(tokens, labels=tokens, use_cache=False)
        saved_records = dict(capture.records)
        assert set(saved_records) == {layer.attn for layer in model.model.layers}
        observed_auxiliary = model.model._total_aux_loss
        actual_gradients = model_gradients(actual.loss, model)
        assert torch.equal(torch.random.get_rng_state(), expected_cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(device), expected_cuda_rng)
        assert set(capture.records) == set(saved_records)
        assert all(capture.records[layer] is record for layer, record in saved_records.items())
        for record in capture.records.values():
            for field in fields(record):
                value = getattr(record, field.name)
                if isinstance(value, torch.Tensor):
                    assert not value.requires_grad and value.grad_fn is None
    for handle in handles:
        handle.remove()
    assert phases.count("prepass") == 2 and phases.count("recompute") == 2
    assert "normal" not in phases and get_swila_checkpoint_phase() == "normal"
    assert not capture.hook_handles
    compare_tree(expected.logits, actual.logits, prefix="checkpoint-logits")
    compare_tree(expected.loss, actual.loss, prefix="checkpoint-loss")
    compare_tree(expected_auxiliary, observed_auxiliary, prefix="checkpoint-mi")
    compare_model_gradients(expected_gradients, actual_gradients, profile=torch.float32)
    model.gradient_checkpointing_disable()
    assert not model.is_gradient_checkpointing


@requires_cuda
def test_swila_model_packed_cache_cuda():
    torch.manual_seed(42)
    model = SwiLAForCausalLM(model_config("temporal_gated", mi_loss_weight=0.)).to(device).eval()
    tokens = torch.tensor([[1, 7, 3, 1, 9, 2, 8]], device=device)
    boundaries = torch.tensor([0, 3, 7], device=device, dtype=torch.int32)
    with torch.no_grad():
        packed = model(tokens, cu_seqlens=boundaries, use_cache=True)
        for index, (start, end) in enumerate(((0, 3), (3, 7))):
            independent = model(tokens[:, start:end], use_cache=True)
            compare_tree(independent.logits, packed.logits[:, start:end], prefix="packed-logits")
            for name in ("recurrent_state", "conv_state"):
                expected = independent.past_key_values[0][name]
                actual = packed.past_key_values[0][name]
                compare_tree(expected, tuple(value[index:index + 1] for value in actual), prefix=name)
