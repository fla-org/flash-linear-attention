# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab

"""SwiLA construction, eager-reference gradients and real FLA cache contracts."""

import inspect
from dataclasses import fields

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from fla.layers import SwitchingLinearAttention
from fla.models.cache_utils import Cache
from fla.modules.swila import diagnostics, initialization_checks
from fla.modules.swila.diagnostics import capture_swila_diagnostics
from fla.modules.swila.initialization_checks import check_swila_initialization
from fla.ops.swila.checkpoint_context import swila_checkpoint_context_fn
from fla.utils import IS_NVIDIA, device
from tests.layers.swila_helpers import (
    compare_layer_gradients,
    compare_tree,
    eager_recurrence,
    install_nonuniform_router_fixture,
    layer_gradients,
    layer_objective,
    make_gpu_layer,
    make_layer,
    recurrent_cache,
)
from tests.ops.swila_helpers import assert_swila_close, assert_swila_grad

requires_cuda = pytest.mark.skipif(not IS_NVIDIA, reason="SwiLA layer CUDA tests require an NVIDIA GPU")
pytestmark = pytest.mark.filterwarnings("ignore:ShortConvolution is crucial:UserWarning")


@pytest.mark.parametrize("options", [
    pytest.param({}, id="plain"),
    pytest.param({"temporal": True}, id="temporal"),
    pytest.param({"use_forget_gate": True}, id="gated"),
    pytest.param({"temporal": True, "use_forget_gate": True, "inject_noise": True, "inject_gate_noise": True},
                 id="temporal-gated-noise"),
    pytest.param({"pi_proj_rank": 0, "pi_align_qk_init": False, "init_state_mode": "zeros"}, id="full-rank-zero-state"),
])
def test_swila_initialization_cpu(options):
    torch.manual_seed(42)
    layer = make_layer(**options)
    assert check_swila_initialization(layer) is None
    assert layer.key_dim == layer.value_dim == 16
    assert layer.head_k_dim == layer.head_v_dim == 8
    assert "b_granularity" not in inspect.signature(SwitchingLinearAttention).parameters
    assert not hasattr(layer, "_calibrated_reset_record")
    if layer.init_state is not None:
        assert torch.count_nonzero(layer.init_state) == 0


def test_swila_initialization_check_is_optional_and_stateless_cpu(monkeypatch):
    torch.manual_seed(42)
    layer = make_layer("temporal_gated", inject_noise=True, inject_gate_noise=True)
    parameters = dict(layer.named_parameters())
    for value in parameters.values():
        value.grad = torch.ones_like(value)
    snapshot = {name: (value.detach().clone(), value._version, value.grad.clone(), dict(vars(value)))
                for name, value in parameters.items()}
    rng = torch.random.get_rng_state().clone()
    check_swila_initialization(layer)
    check_swila_initialization(layer)
    assert torch.equal(torch.random.get_rng_state(), rng)
    for name, value in parameters.items():
        original, version, grad, metadata = snapshot[name]
        torch.testing.assert_close(value, original, rtol=0, atol=0)
        torch.testing.assert_close(value.grad, grad, rtol=0, atol=0)
        assert value._version == version and vars(value) == metadata

    def unexpected(*args, **kwargs):
        raise AssertionError("optional initialization check called implicitly")

    monkeypatch.setattr(initialization_checks, "check_swila_initialization", unexpected)
    layer = make_layer("temporal_gated")
    layer.reset_parameters()
    layer.load_state_dict(layer.state_dict())
    layer.to(dtype=torch.float64)


@pytest.mark.parametrize("assign", [False, True])
def test_swila_load_preserves_trained_values_and_metadata_cpu(assign):
    torch.manual_seed(42)
    layer = make_layer("temporal_gated", inject_noise=True, inject_gate_noise=True)
    state = {name: value.clone() for name, value in layer.state_dict().items()}
    for name in ("init_state", "A_log", "dt_bias", "gate_kv_proj.bias", "gate_q_proj.bias"):
        state[name] = torch.linspace(.1234567, .8765432, state[name].numel()).reshape_as(state[name])
    loaded = layer.load_state_dict(state, strict=True, assign=assign)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    protected = {name: getattr(layer, name).detach().clone() for name in ("A_log", "dt_bias")}
    layer.bfloat16()
    for name in protected:
        value = getattr(layer, name)
        assert value.dtype == torch.float32 and value._no_weight_decay is True
        torch.testing.assert_close(value, protected[name], rtol=0, atol=0)
    for module in (layer.gate_q_proj, layer.gate_kv_proj, layer.W_noise_gate_q, layer.W_noise_gate_kv):
        assert module.bias._no_weight_decay is True


def test_swila_legacy_missing_gate_noise_bias_load_cpu():
    torch.manual_seed(42)
    layer = make_layer("temporal", inject_gate_noise=True, gate_noise_init_std=.02)
    state = {name: value.clone() for name, value in layer.state_dict().items()}
    for name in ("W_noise_gate_kv.bias", "W_noise_gate_q.bias"):
        del state[name]
    with torch.no_grad():
        state["W_noise_gate_kv.weight"].fill_(.125)
        layer.W_noise_gate_kv.bias.fill_(7.)
        layer.W_noise_gate_q.bias.fill_(-7.)
    loaded = layer.load_state_dict(state, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    for projection in (layer.W_noise_gate_kv, layer.W_noise_gate_q):
        torch.testing.assert_close(F.softplus(projection.bias), torch.full_like(projection.bias, .02), rtol=1e-6, atol=0)
        assert projection.bias._no_weight_decay is True
    torch.testing.assert_close(layer.W_noise_gate_kv.weight, state["W_noise_gate_kv.weight"], rtol=0, atol=0)


@pytest.mark.parametrize(("options", "error", "message"), [
    ({"mode": "chunk"}, ValueError, "mode must be"),
    ({"num_experts": 9}, AssertionError, "num_experts <= 8"),
    ({"pi_proj_rank": 3}, ValueError, "divisible"),
    ({"b_granularity": "value"}, TypeError, "b_granularity"),
    ({"architecture_profile": "historical_8dc05f93", "temporal": True}, ValueError, "requires plain"),
])
def test_swila_invalid_options_preserve_rng_cpu(options, error, message):
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(error, match=message):
        make_layer(**options)
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize(("name", "value", "message"), [
    ("gate_q_proj.bias", .25, "temporal-gate bias"),
    ("init_state", .25, "not exactly zero"),
    ("A_log", float("nan"), "nonfinite"),
])
def test_swila_initialization_detects_corruption_cpu(name, value, message):
    layer = make_layer("temporal_gated")
    with torch.no_grad():
        dict(layer.named_parameters())[name].fill_(value)
    with pytest.raises(RuntimeError, match=message):
        check_swila_initialization(layer)


def test_swila_historical_expert_beta_equations_cpu():
    torch.manual_seed(1729)
    layer = make_layer(
        architecture_profile="historical_8dc05f93", b_low_rank_proj_type="swiglu", pi_align_qk_init=False,
        d_model=8, num_experts=4, b_proj_rank=3,
    ).double()
    hidden = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
    actual = layer.compute_beta(hidden, compact=True)
    weights = (layer.b_proj.gate.weight, layer.b_proj.feat.weight, layer.b_proj.out.weight)
    expected = F.linear(F.silu(F.linear(hidden, weights[0])) * F.linear(hidden, weights[1]), weights[2])
    expected = expected.sigmoid().reshape(2, 7, 2, 4, 4)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    assert not torch.equal(actual[..., 0], actual[..., 1])
    torch.testing.assert_close(actual, layer.compute_beta(hidden, compact=False), rtol=0, atol=0)
    cotangent = torch.arange(actual.numel(), dtype=torch.float64).sin().view_as(actual)
    got = torch.autograd.grad(actual, (hidden, *weights), cotangent, retain_graph=True)
    wanted = torch.autograd.grad(expected, (hidden, *weights), cotangent)
    for actual_grad, expected_grad in zip(got, wanted, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-12)


def test_swila_shared_beta_broadcast_gradient_cpu():
    torch.manual_seed(42)
    layer = make_layer().double()
    hidden = torch.randn(1, 5, 16, dtype=torch.float64, requires_grad=True)
    compact = layer.compute_beta(hidden, compact=True)
    dense = layer.compute_beta(hidden)
    torch.testing.assert_close(dense, compact.unsqueeze(-1).expand_as(dense), rtol=0, atol=0)
    cotangent = torch.randn_like(dense)
    got = torch.autograd.grad(dense, (hidden, *layer.b_proj.parameters()), cotangent, retain_graph=True)
    wanted = torch.autograd.grad(compact, (hidden, *layer.b_proj.parameters()), cotangent.sum(-1))
    for actual, expected in zip(got, wanted, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_swila_optional_capture_scope_cpu():
    layer, other = make_layer(), make_layer("temporal")
    model = torch.nn.ModuleList([layer, other])
    fields_before = {id(module): dict(vars(module)) for module in model.modules()}
    with capture_swila_diagnostics(layer) as capture:
        assert diagnostics.active_swila_capture(layer) is capture
        assert diagnostics.find_active_swila_diagnostics(model) is capture
        with pytest.raises(RuntimeError, match="overlapping"), capture_swila_diagnostics(model):
            pass
        assert diagnostics.active_swila_capture(other) is None
    assert diagnostics.active_swila_capture(layer) is None
    assert not capture.hook_handles
    with pytest.raises(ValueError, match="capture exception"), capture_swila_diagnostics(layer):
        raise ValueError("capture exception")
    assert diagnostics.active_swila_capture(layer) is None
    for module in model.modules():
        before = fields_before[id(module)]
        assert vars(module).keys() == before.keys()
        assert all(vars(module)[name] is value for name, value in before.items())


@requires_cuda
@pytest.mark.parametrize(("family", "dtype", "conv", "prior_conv", "output_gate"), [
    pytest.param("plain", torch.float32, False, False, False, id="plain-fp32"),
    pytest.param("gated", torch.bfloat16, True, False, False, id="gated-bf16-conv"),
    pytest.param("temporal", torch.float32, True, True, False, id="temporal-fp32-prior-conv"),
    pytest.param("temporal_gated", torch.bfloat16, True, True, True, id="temporal-gated-bf16-output-gate"),
])
def test_swila_layer_eager_reference_cuda(family, dtype, conv, prior_conv, output_gate):
    torch.manual_seed(42)
    options = dict(use_short_conv=conv, use_pi_conv=prior_conv, use_gate=output_gate, mi_loss_weight=.0005)
    actual_layer = make_gpu_layer(family, **options).to(device=device, dtype=dtype).train()
    with torch.no_grad():
        actual_layer.init_state.copy_(.05 * torch.randn_like(actual_layer.init_state))
    expected_layer = make_gpu_layer(family, **options).to(device=device, dtype=dtype).train()
    expected_layer.load_state_dict(actual_layer.state_dict())
    data = .4 * torch.randn(2, 7, 16, device=device, dtype=dtype)
    hidden, ref_hidden = data.clone().requires_grad_(), data.clone().requires_grad_()
    cotangent = torch.randn_like(data)
    actual, attention, _, auxiliary = actual_layer(hidden, output_attentions=True)
    with eager_recurrence():
        expected, ref_attention, _, ref_auxiliary = expected_layer(ref_hidden, output_attentions=True)
    compare_tree(expected, actual, prefix="output")
    compare_tree(ref_attention, attention, prefix="responsibilities", profile=dtype)
    compare_tree(ref_auxiliary, auxiliary, prefix="mi", profile=dtype)
    assert auxiliary.requires_grad
    # Remove the configured loss weight and token/value mean reduction with a
    # fixed, shape-derived cotangent; tiny MI weights must not hide dropped VJPs.
    mi_scale = data.shape[0] * data.shape[1] * actual_layer.num_heads * actual_layer.head_v_dim
    mi_scale /= actual_layer.mi_loss_weight
    mi_expected = layer_gradients(mi_scale * ref_auxiliary, expected_layer, ref_hidden, retain_graph=True)
    mi_actual = layer_gradients(mi_scale * auxiliary, actual_layer, hidden, retain_graph=True)
    disconnected = ("q_proj.", "q_conv1d.", "g_proj.", "o_norm.", "o_proj.")
    for name in mi_expected:
        assert_swila_grad(
            f"initialized-mi-only/{name}", mi_expected[name], mi_actual[name],
            connected=not name.startswith(disconnected), profile=dtype,
        )
    wanted = layer_gradients(layer_objective(expected, ref_attention, ref_auxiliary, cotangent), expected_layer, ref_hidden)
    got = layer_gradients(layer_objective(actual, attention, auxiliary, cotangent), actual_layer, hidden)
    # FP32 stored decay parameters can receive BF16 projection/VJP rounding in
    # a complete BF16 layer. Isolated gate tests retain their strict FP32 row.
    compare_layer_gradients(wanted, got, profile=dtype)

    # MI is stationary for token-constant router probabilities. Preserve the
    # initialized case above, then require relative accuracy on a fixed trained
    # router fixture with time-varying, nonuniform probabilities.
    install_nonuniform_router_fixture(actual_layer)
    expected_layer.load_state_dict(actual_layer.state_dict())
    hidden, ref_hidden = data.clone().requires_grad_(), data.clone().requires_grad_()
    _, attention, _, auxiliary = actual_layer(hidden, output_attentions=True)
    with eager_recurrence():
        _, ref_attention, _, ref_auxiliary = expected_layer(ref_hidden, output_attentions=True)
    compare_tree(ref_attention, attention, prefix="trained-responsibilities", profile=dtype)
    compare_tree(ref_auxiliary, auxiliary, prefix="trained-mi", profile=dtype)
    mi_expected = layer_gradients(mi_scale * ref_auxiliary, expected_layer, ref_hidden)
    mi_actual = layer_gradients(mi_scale * auxiliary, actual_layer, hidden)
    relative_required = {"hidden", "pi_q_input_proj.weight", "pi_k_input_proj.weight"}
    for name in mi_expected:
        if name in relative_required:
            assert mi_expected[name] is not None and torch.count_nonzero(mi_expected[name]) > 0
        assert_swila_grad(
            f"trained-mi-only/{name}", mi_expected[name], mi_actual[name],
            connected=not name.startswith(disconnected), profile=dtype, require_relative=name in relative_required,
        )
    # A missing MI contribution must fail even though the configured weight is small.
    for name in relative_required:
        with pytest.raises(AssertionError):
            assert_swila_grad(
                f"dropped-mi/{name}", mi_expected[name], torch.zeros_like(mi_expected[name]),
                connected=True, profile=dtype, require_relative=True,
            )


@requires_cuda
@pytest.mark.parametrize(("family", "conv", "prior_conv"), [
    ("plain", False, False), ("gated", True, False), ("temporal", True, True), ("temporal_gated", True, True),
])
def test_swila_real_cache_prefill_decode_cuda(family, conv, prior_conv):
    torch.manual_seed(3121)
    layer = make_gpu_layer(family, use_short_conv=conv, use_pi_conv=prior_conv).to(device).eval()
    data = .4 * torch.randn(2, 5, 16, device=device)
    cache, uninterrupted = Cache(), Cache()
    with torch.no_grad():
        expected, _, _, _ = layer(data, past_key_values=uninterrupted, use_cache=False)
        outputs = []
        for start, end in ((0, 3), (3, 4), (4, 5)):
            output, _, returned, _ = layer(data[:, start:end], past_key_values=cache, use_cache=False)
            assert returned is cache
            outputs.append(output)
    compare_tree(expected, torch.cat(outputs, 1), prefix="cached-output")
    assert cache.get_seq_length(0) == uninterrupted.get_seq_length(0) == 5
    compare_tree(uninterrupted[0], cache[0], prefix="cache")
    if conv:
        assert len(cache[0]["conv_state"]) == 5
        assert all(value is not None for value in cache[0]["conv_state"][:3])
        assert all((value is not None) == prior_conv for value in cache[0]["conv_state"][3:])
    original = tuple(value.clone() for value in recurrent_cache(cache))
    layer_one = make_gpu_layer(family, layer_idx=1, use_short_conv=conv, use_pi_conv=prior_conv).to(device).eval()
    with torch.no_grad():
        layer_one(data[:, :2], past_key_values=cache, use_cache=True)
    assert cache.get_seq_length(0) == 5 and cache.get_seq_length(1) == 2
    for before, after in zip(original, recurrent_cache(cache), strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


@requires_cuda
def test_swila_left_padded_prefill_decode_cuda():
    torch.manual_seed(42)
    layer = make_gpu_layer("temporal_gated", use_short_conv=True, use_pi_conv=True).to(device).eval()
    prompt = .4 * torch.randn(2, 4, 16, device=device)
    token = .4 * torch.randn(2, 1, 16, device=device)
    mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]], device=device, dtype=torch.bool)
    growing_mask = torch.cat((mask, torch.ones(2, 1, device=device, dtype=torch.bool)), 1)
    cache = Cache()
    independent_caches = []
    with torch.no_grad():
        prefill, _, returned, _ = layer(prompt, attention_mask=mask, past_key_values=cache, use_cache=False)
        assert returned is cache and cache.get_seq_length(0) == 4
        decoded, _, returned, _ = layer(token, attention_mask=growing_mask, past_key_values=cache, use_cache=False)
        assert returned is cache and cache.get_seq_length(0) == 5
        for index, length in enumerate((3, 2)):
            independent = Cache()
            full = torch.cat((prompt[index:index + 1, -length:], token[index:index + 1]), 1)
            expected, _, _, _ = layer(full, past_key_values=independent, use_cache=False)
            compare_tree(expected[:, :-1], prefill[index:index + 1, -length:], prefix=f"unpadded-prefill-{index}")
            compare_tree(expected[:, -1:], decoded[index:index + 1], prefix=f"unpadded-decode-{index}")
            assert independent.get_seq_length(0) == length + 1
            independent_caches.append(independent)
    assert torch.count_nonzero(prefill[~mask]) == 0
    for index, actual in enumerate(recurrent_cache(cache)):
        expected = torch.cat([recurrent_cache(independent)[index] for independent in independent_caches], 0)
        compare_tree(expected, actual, prefix=f"unpadded-recurrent-state-{index}")
    assert len(cache[0]["conv_state"]) == 5
    for index, actual in enumerate(cache[0]["conv_state"]):
        expected = torch.cat([independent[0]["conv_state"][index] for independent in independent_caches], 0)
        compare_tree(expected, actual, prefix=f"unpadded-conv-state-{index}")


@requires_cuda
def test_swila_differentiable_cache_parameter_vjps_cuda():
    torch.manual_seed(42)
    layer = make_gpu_layer("temporal_gated", use_short_conv=True, use_pi_conv=True).to(device).train()
    with torch.no_grad():
        layer.init_state.copy_(.05 * torch.randn_like(layer.init_state))
    data = .4 * torch.randn(2, 7, 16, device=device)
    full_input, split_input = data.clone().requires_grad_(), data.clone().requires_grad_()
    full_cache, split_cache = Cache(), Cache()
    full, _, _, _ = layer(full_input, past_key_values=full_cache, use_cache=False)
    prefix, _, _, _ = layer(split_input[:, :3], past_key_values=split_cache, use_cache=False)
    prefix_states = (*recurrent_cache(split_cache), *split_cache[0]["conv_state"])
    assert all(value.requires_grad and value.grad_fn is not None for value in prefix_states)
    suffix, _, _, _ = layer(split_input[:, 3:], past_key_values=split_cache, use_cache=False)
    split = torch.cat((prefix, suffix), 1)
    compare_tree(full, split, prefix="differentiable-cache-output")
    compare_tree(full_cache[0], split_cache[0], prefix="differentiable-cache-state")
    assert full_cache.get_seq_length(0) == split_cache.get_seq_length(0) == 7
    cotangent = torch.randn_like(full)
    boundary_gradients = torch.autograd.grad((suffix * cotangent[:, 3:]).sum(), prefix_states, retain_graph=True)
    assert all(torch.isfinite(value).all() and torch.count_nonzero(value) > 0 for value in boundary_gradients)

    def objective(output, cache):
        loss = (output * cotangent).sum()
        for state in (*recurrent_cache(cache), *cache[0]["conv_state"]):
            weights = torch.arange(state.numel(), device=state.device, dtype=torch.float32).cos().reshape(state.shape)
            loss = loss + .1 * (state * weights).sum()
        return loss

    expected = layer_gradients(objective(full, full_cache), layer, full_input)
    actual = layer_gradients(objective(split, split_cache), layer, split_input)
    compare_layer_gradients(expected, actual)


@requires_cuda
@pytest.mark.parametrize("family", ("plain", "temporal_gated"))
def test_swila_mask_precedence_packed_mi_cuda(family):
    torch.manual_seed(3181)
    layer = make_gpu_layer(family, use_short_conv=True, use_pi_conv=True, mi_loss_weight=.0005).to(device).train()
    mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]], device=device, dtype=torch.bool)
    hidden = (.4 * torch.randn(2, 4, 16, device=device)).requires_grad_()
    packed = hidden.detach()[mask].unsqueeze(0).requires_grad_()
    output, attention, _, auxiliary = layer(
        hidden, attention_mask=mask, cu_seqlens=torch.tensor([0, 8], device=device, dtype=torch.int64),
        output_attentions=True,
    )
    packed_output, packed_attention, _, packed_auxiliary = layer(
        packed, cu_seqlens=torch.tensor([0, 3, 5], device=device, dtype=torch.int32), output_attentions=True,
    )
    compare_tree(packed_output.squeeze(0), output[mask], prefix="masked-output")
    compare_tree(packed_auxiliary, auxiliary, prefix="masked-mi")
    assert torch.count_nonzero(output[~mask]) == 0
    padded_attentions = attention.values() if isinstance(attention, dict) else (attention,)
    packed_attentions = packed_attention.values() if isinstance(packed_attention, dict) else (packed_attention,)
    for padded, flat in zip(padded_attentions, packed_attentions, strict=True):
        compare_tree(flat.squeeze(0), padded[mask], prefix="masked-responsibilities")
        assert torch.count_nonzero(padded[~mask]) == 0
    for padded_loss, flat_loss in ((output.square().sum(), packed_output.square().sum()), (auxiliary, packed_auxiliary)):
        got = torch.autograd.grad(padded_loss, hidden, retain_graph=True)[0]
        wanted = torch.autograd.grad(flat_loss, packed, retain_graph=True)[0]
        assert_swila_close("masked-dhidden", wanted.squeeze(0), got[mask], gradient=True)
        assert torch.count_nonzero(got[~mask]) == 0


@requires_cuda
@pytest.mark.parametrize("family", ("plain", "temporal_gated"))
def test_swila_packed_cache_independent_sequences_cuda(family):
    torch.manual_seed(42)
    layer = make_gpu_layer(family, use_short_conv=True, use_pi_conv=True).to(device).eval()
    hidden = (.4 * torch.randn(1, 8, 16, device=device)).requires_grad_()
    cu = torch.tensor([0, 3, 8], device=device, dtype=torch.int32)
    packed_cache = Cache()
    output, _, _, _ = layer(hidden, cu_seqlens=cu, past_key_values=packed_cache, use_cache=False)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_outputs, independent_caches = [], []
    with eager_recurrence():
        for start, end in ((0, 3), (3, 8)):
            cache = Cache()
            value, _, _, _ = layer(reference_hidden[:, start:end], past_key_values=cache, use_cache=False)
            reference_outputs.append(value)
            independent_caches.append(cache)
    expected = torch.cat(reference_outputs, 1)
    compare_tree(expected, output, prefix="independent-packed-output")
    for index, state in enumerate(recurrent_cache(packed_cache)):
        target = torch.cat([recurrent_cache(cache)[index] for cache in independent_caches], 0)
        compare_tree(target, state, prefix=f"independent-packed-state-{index}")
    cotangent = torch.randn_like(output)
    wanted = torch.autograd.grad((expected * cotangent).sum(), reference_hidden)[0]
    actual = torch.autograd.grad((output * cotangent).sum(), hidden, retain_graph=True)[0]
    assert_swila_close("independent-packed-dhidden", wanted, actual, gradient=True)
    boundary = torch.autograd.grad(output[:, :3].sum(), hidden)[0]
    assert torch.count_nonzero(boundary[:, 3:]) == 0
    cache = Cache()
    with torch.no_grad():
        prefix, _, _, _ = layer(torch.cat((hidden[:, :1], hidden[:, 3:5]), 1),
                                cu_seqlens=torch.tensor([0, 1, 3], device=device), past_key_values=cache)
        suffix, _, _, _ = layer(torch.cat((hidden[:, 1:3], hidden[:, 5:]), 1),
                                cu_seqlens=torch.tensor([0, 2, 5], device=device), past_key_values=cache)
    restored = torch.cat((prefix[:, :1], suffix[:, :2], prefix[:, 1:], suffix[:, 2:]), 1)
    compare_tree(output.detach(), restored, prefix="unequal-packed-cache-output")
    compare_tree(packed_cache[0], cache[0], prefix="unequal-packed-cache-state")


@requires_cuda
@pytest.mark.parametrize("family", ("plain", "temporal_gated"))
def test_swila_mi_capture_checkpoint_invariance_cuda(family):
    torch.manual_seed(42)
    layer = make_gpu_layer(family, mi_loss_weight=.0005).to(device).train()
    data = .4 * torch.randn(1, 5, 16, device=device)
    reference_input = data.clone().requires_grad_()
    expected_output, _, _, expected_mi = layer(reference_input)
    expected_loss = expected_output.square().sum() + expected_mi
    expected_gradients = layer_gradients(expected_loss, layer, reference_input)
    hidden = data.clone().requires_grad_()

    def execute(values):
        output, _, _, loss = layer(values)
        return output, loss

    with capture_swila_diagnostics(layer, grad_norms=True, retain_tensors=True) as capture:
        output, loss = checkpoint(execute, hidden, use_reentrant=False, context_fn=swila_checkpoint_context_fn)
        record = capture.records[layer]
        actual_gradients = layer_gradients(output.square().sum() + loss, layer, hidden)
        assert capture.records[layer] is record
        assert capture.grad_norms[layer].output is not None
        for field in fields(record):
            value = getattr(record, field.name)
            if isinstance(value, torch.Tensor):
                assert value.grad_fn is None and not value.requires_grad
    compare_tree(expected_output, output, prefix="checkpoint-output")
    compare_tree(expected_mi, loss, prefix="checkpoint-mi")
    compare_layer_gradients(expected_gradients, actual_gradients)
    assert not capture.hook_handles and diagnostics.active_swila_capture(layer) is None
