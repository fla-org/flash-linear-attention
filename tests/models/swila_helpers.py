# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""Small real-HF fixtures; recurrence adapters leave FLA model components intact."""

import copy

import torch
import torch.nn.functional as F

from fla.layers import SwitchingLinearAttention
from fla.models.swila import SwiLAConfig, SwiLAForCausalLM
from tests.layers.swila_helpers import install_nonuniform_router_fixture
from tests.ops.swila_helpers import assert_swila_grad


def model_config(family="plain", **overrides):
    options = dict(
        hidden_size=16, num_hidden_layers=1, num_heads=2, num_experts=4,
        expand_k=1., expand_v=.5, intermediate_size=32, vocab_size=41,
        pi_proj_rank=4, b_proj_rank=4, alpha_proj_rank=4,
        use_short_conv=True, use_pi_conv=True, conv_size=3,
        init_state_mode="learnable", temporal="temporal" in family,
        use_forget_gate="gated" in family, mi_loss_weight=.0005,
        fuse_norm=True, fuse_swiglu=True, fuse_cross_entropy=False,
        fuse_linear_cross_entropy=False, use_cache=True,
        pad_token_id=0, bos_token_id=1, eos_token_id=None,
        max_position_embeddings=32,
    )
    return SwiLAConfig(**(options | overrides))


def historical_config(run):
    """Original gated run options, reduced only in model dimensions for CPU loading.

    The source configs are public 4b43bb5c tests/models/fixtures/0353524c_config.json
    and 908eda67_config.json. Historical metadata is intentionally not pre-cleaned.
    """
    assert run in {"0353524c", "908eda67"}
    temporal = run == "908eda67"
    options = dict(
        model_type="sla", hidden_size=16, num_hidden_layers=1, num_heads=2,
        hidden_ratio=4, intermediate_size=32, hidden_act="swish", vocab_size=41,
        tie_word_embeddings=False, mode="fused_recurrent", expand_k=1., expand_v=.5,
        num_experts=4, num_deer_iterations=1, chunk_size=64, full_jacobian=False,
        scale=None, use_gate=False, return_output_only=False, tie_responsibilities=False,
        skip_reset_parameters=False, use_forget_gate=True, alpha_proj_rank=4,
        qk_activation="silu", qk_norm="l2", use_short_conv=True, conv_size=4,
        conv_bias=False, conv_activation="silu", l_max=4096, pi_activation="softmax",
        use_input_pi=True, use_pi_conv=True, pi_proj_rank=4, pi_input_proj_narrow=True,
        b_proj_rank=4, alpha_granularity="key", b_granularity="value",
        low_rank_proj_type="swiglu", pi_low_rank_proj_type="swiglu",
        b_low_rank_proj_type="linear", alpha_low_rank_proj_type="linear",
        pi_conv_activation="silu", pi_init_orthogonal=False, temporal=temporal,
        gate_init_bias=-2.4 if temporal else -1., inject_noise=False, inject_gate_noise=False,
        mi_loss_weight=.00025 if temporal else .0005, pred_init_mode="recurrent",
        init_state_mode="learnable", init_state_learnable_init="zeros", initializer_range=.02,
        norm_eps=1e-6, fuse_cross_entropy=True, fuse_norm=False, fuse_swiglu=True,
        fuse_linear_cross_entropy=False, use_l2warp=False, use_cache=True,
        bos_token_id=1, eos_token_id=2, max_position_embeddings=4096,
    )
    if temporal:
        options.update(gate_init_mode="bias", gate_min_half_life=2., gate_max_half_life=32., gate_noise_init_std=.02)
    return options


def trained_state(model):
    """Every parameter differs from initialization, including non-BF16-exact decay."""
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.linspace(
                (index + 1) * .00123456789, (index + 2) * .00123456789,
                parameter.numel(), device=parameter.device, dtype=torch.float32,
            )
            parameter.copy_(values.reshape_as(parameter))
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_complete_loading(info):
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(name):
            raise AssertionError(f"Incomplete SwiLA checkpoint: {name}={info[name]}")


def assert_loaded_state(model, expected, dtype=torch.float32):
    observed = model.state_dict()
    assert observed.keys() == expected.keys()
    for name, value in observed.items():
        target_dtype = torch.float32 if name.rsplit(".", 1)[-1] in {"A_log", "dt_bias"} else dtype
        assert value.dtype == target_dtype, name
        torch.testing.assert_close(value, expected[name].to(target_dtype), rtol=0, atol=0, msg=name)
    for layer in model.modules():
        if isinstance(layer, SwitchingLinearAttention) and layer.use_forget_gate:
            assert layer.A_log._no_weight_decay and layer.dt_bias._no_weight_decay


def model_pair(family="plain", *, dtype=torch.float32, device="cuda", **options):
    model = SwiLAForCausalLM(model_config(family, **options)).to(device=device, dtype=dtype).train()
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, SwitchingLinearAttention):
                layer.init_state.copy_(.05 * torch.randn_like(layer.init_state))
                install_nonuniform_router_fixture(layer)
    reference = SwiLAForCausalLM(copy.deepcopy(model.config)).to(device=device, dtype=dtype).train()
    reference.load_state_dict(model.state_dict())
    return model, reference


def model_gradients(loss, model, *, retain_graph=False):
    parameters = dict(model.named_parameters())
    values = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True, retain_graph=retain_graph)
    return dict(zip(parameters, values, strict=True))


def compare_model_gradients(reference, actual, *, profile):
    assert reference.keys() == actual.keys()
    for name, expected in reference.items():
        assert_swila_grad(name, expected, actual[name], connected=expected is not None, profile=profile)


def reference_ce(logits, labels=None, *, shift_labels=None, denominator=None):
    if shift_labels is None:
        shift_labels = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], -100)), dim=1)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), shift_labels.reshape(-1),
        reduction="mean" if denominator is None else "sum",
    )
    return loss if denominator is None else loss / denominator


def cloned_state(cache, index):
    def clone(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        return value

    return clone(cache[index])
