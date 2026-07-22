# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import json
from copy import deepcopy

import pytest
import torch

import fla.layers.attn as attn_module
from fla.layers.attn import Attention
from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.gla import GatedLinearAttention
from fla.layers.mamba import Mamba
from fla.models import (
    ABCConfig,
    CombaConfig,
    DeltaFormerConfig,
    DeltaNetConfig,
    GatedDeltaNetConfig,
    GatedDeltaProductConfig,
    GLAConfig,
    GSAConfig,
    HGRN2Config,
    HGRNConfig,
    KDAConfig,
    LightNetConfig,
    LinearAttentionConfig,
    MesaNetConfig,
    MomConfig,
    RavenConfig,
    RetNetConfig,
    RodimusConfig,
    RWKV6Config,
    RWKV7Config,
    SambaConfig,
)
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetBlock
from fla.models.gla.modeling_gla import GLABlock, GLAForCausalLM
from fla.models.hybrid import get_hybrid_attention_spec, normalize_hybrid_attention_config
from fla.models.samba.modeling_samba import SambaBlock
from fla.utils import assert_close, device, device_platform, device_torch_lib

CONFIG_CLASSES = (
    ABCConfig,
    CombaConfig,
    DeltaNetConfig,
    DeltaFormerConfig,
    GatedDeltaNetConfig,
    GatedDeltaProductConfig,
    GLAConfig,
    GSAConfig,
    HGRNConfig,
    HGRN2Config,
    KDAConfig,
    LightNetConfig,
    LinearAttentionConfig,
    MesaNetConfig,
    MomConfig,
    RavenConfig,
    RetNetConfig,
    RodimusConfig,
    RWKV6Config,
    RWKV7Config,
    SambaConfig,
)

SAMBA_CANONICAL_ATTN = {
    'layers': [1, 3, 5, 7, 9, 11, 13, 15, 17],
    'num_heads': 18,
    'num_kv_heads': 18,
    'qkv_bias': False,
    'window_size': 2048,
    'rope_theta': 10000.,
}


def test_normalize_none():
    assert normalize_hybrid_attention_config(None, num_hidden_layers=4) is None
    assert normalize_hybrid_attention_config(None, num_hidden_layers='unused') is None


def test_normalize_dictionary_defaults_and_unknown_keys():
    source = {
        'layers': [3, 1],
        'num_heads': 8,
        'extension': {'mode': 'custom'},
    }
    original = deepcopy(source)

    normalized = normalize_hybrid_attention_config(source, num_hidden_layers=4)

    assert isinstance(normalized, dict)
    assert normalized == {
        'layers': [3, 1],
        'num_heads': 8,
        'num_kv_heads': 8,
        'qkv_bias': False,
        'window_size': None,
        'rope_theta': 10000.,
        'extension': {'mode': 'custom'},
    }
    assert source == original
    assert normalized is not source
    assert normalized['layers'] is not source['layers']
    json.dumps(normalized)


def test_normalize_list_defaults_each_specification():
    source = [
        {'layers': [0, 2], 'num_heads': 8, 'num_kv_heads': 2, 'qkv_bias': True, 'window_size': 128},
        {'layers': [3], 'num_heads': 4, 'num_kv_heads': None, 'rope_theta': 500000.},
    ]
    original = deepcopy(source)

    normalized = normalize_hybrid_attention_config(source, num_hidden_layers=4)

    assert isinstance(normalized, list)
    assert normalized[0] == {
        'layers': [0, 2],
        'num_heads': 8,
        'num_kv_heads': 2,
        'qkv_bias': True,
        'window_size': 128,
        'rope_theta': 10000.,
    }
    assert normalized[1] == {
        'layers': [3],
        'num_heads': 4,
        'num_kv_heads': 4,
        'qkv_bias': False,
        'window_size': None,
        'rope_theta': 500000.,
    }
    assert source == original
    assert normalized is not source


@pytest.mark.parametrize(
    ('attn', 'expected'),
    [
        pytest.param([], [], id='empty-plan'),
        pytest.param({'layers': [], 'num_heads': 2}, {
            'layers': [],
            'num_heads': 2,
            'num_kv_heads': 2,
            'qkv_bias': False,
            'window_size': None,
            'rope_theta': 10000.,
        }, id='empty-layers'),
    ],
)
def test_normalize_empty_forms(attn, expected):
    assert normalize_hybrid_attention_config(attn, num_hidden_layers=4) == expected


@pytest.mark.parametrize(
    ('attn', 'match'),
    [
        pytest.param(1, r"attn.*dictionary.*1", id='invalid-outer'),
        pytest.param([{'layers': [0], 'num_heads': 2}, 'invalid'], r"index 1.*dictionary.*'invalid'", id='list-item'),
        pytest.param({'num_heads': 2}, r"layers.*missing", id='missing-layers'),
        pytest.param({'layers': [0]}, r"num_heads.*missing", id='missing-num-heads'),
        pytest.param({'layers': 0, 'num_heads': 2}, r"layers.*list or tuple.*0", id='layers-container'),
        pytest.param({'layers': [-1], 'num_heads': 2}, r"layers.*-1", id='negative-layer'),
        pytest.param({'layers': [4], 'num_heads': 2}, r"layers.*4", id='out-of-range-layer'),
        pytest.param({'layers': [1, 1], 'num_heads': 2}, r"duplicate layer 1", id='duplicate-layer'),
        pytest.param([
            {'layers': [1], 'num_heads': 2},
            {'layers': [1], 'num_heads': 2},
        ], r"index 1.*conflicting layer 1", id='overlapping-specs'),
        pytest.param({'layers': [True], 'num_heads': 2}, r"layers.*True", id='boolean-layer'),
        pytest.param({'layers': [1.], 'num_heads': 2}, r"layers.*1.0", id='non-integer-layer'),
    ],
)
def test_normalize_invalid_structure(attn, match):
    with pytest.raises(ValueError, match=match):
        normalize_hybrid_attention_config(attn, num_hidden_layers=4)


@pytest.mark.parametrize('value', [0, -1, True, 1.5, None, '8'])
def test_normalize_invalid_num_heads(value):
    with pytest.raises(ValueError, match=rf"num_heads.*{value!r}"):
        normalize_hybrid_attention_config({'layers': [0], 'num_heads': value}, num_hidden_layers=1)


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '2'])
def test_normalize_invalid_num_kv_heads(value):
    with pytest.raises(ValueError, match=rf"num_kv_heads.*{value!r}"):
        normalize_hybrid_attention_config(
            {'layers': [0], 'num_heads': 4, 'num_kv_heads': value},
            num_hidden_layers=1,
        )


@pytest.mark.parametrize('value', [0, 1, None, 'false'])
def test_normalize_invalid_qkv_bias(value):
    with pytest.raises(ValueError, match=rf"qkv_bias.*{value!r}"):
        normalize_hybrid_attention_config({'layers': [0], 'num_heads': 2, 'qkv_bias': value}, num_hidden_layers=1)


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_normalize_invalid_window_size(value):
    with pytest.raises(ValueError, match=rf"window_size.*{value!r}"):
        normalize_hybrid_attention_config({'layers': [0], 'num_heads': 2, 'window_size': value}, num_hidden_layers=1)


@pytest.mark.parametrize(
    'value',
    [0, -1., True, None, '10000', float('nan'), float('inf'), pytest.param(10**1000, id='overflowing-integer')],
)
def test_normalize_invalid_rope_theta(value):
    with pytest.raises(ValueError, match=rf"rope_theta.*{value!r}"):
        normalize_hybrid_attention_config({'layers': [0], 'num_heads': 2, 'rope_theta': value}, num_hidden_layers=1)


def test_list_error_identifies_specification_index_and_value():
    with pytest.raises(ValueError, match=r"index 1.*window_size.*0"):
        normalize_hybrid_attention_config(
            [
                {'layers': [0], 'num_heads': 2},
                {'layers': [1], 'num_heads': 2, 'window_size': 0},
            ],
            num_hidden_layers=2,
        )


def test_lookup_heterogeneous_plan():
    normalized = normalize_hybrid_attention_config(
        [
            {'layers': [4, 1], 'num_heads': 8, 'window_size': 128},
            {'layers': [5], 'num_heads': 4, 'window_size': None},
        ],
        num_hidden_layers=6,
    )

    assert get_hybrid_attention_spec(normalized, layer_idx=1) is normalized[0]
    assert get_hybrid_attention_spec(normalized, layer_idx=4) is normalized[0]
    assert get_hybrid_attention_spec(normalized, layer_idx=5) is normalized[1]
    assert get_hybrid_attention_spec(normalized, layer_idx=0) is None


def test_lookup_dictionary_and_one_item_list_are_equivalent():
    spec = {'layers': [1, 3], 'num_heads': 4, 'num_kv_heads': 2}
    dictionary = normalize_hybrid_attention_config(spec, num_hidden_layers=4)
    plan = normalize_hybrid_attention_config([spec], num_hidden_layers=4)

    for layer_idx in range(4):
        assert get_hybrid_attention_spec(dictionary, layer_idx=layer_idx) == get_hybrid_attention_spec(
            plan,
            layer_idx=layer_idx,
        )


def test_lookup_list_order_does_not_change_disjoint_assignments():
    first = normalize_hybrid_attention_config(
        [{'layers': [0], 'num_heads': 2}, {'layers': [2], 'num_heads': 4}],
        num_hidden_layers=3,
    )
    second = normalize_hybrid_attention_config(
        [{'layers': [2], 'num_heads': 4}, {'layers': [0], 'num_heads': 2}],
        num_hidden_layers=3,
    )

    for layer_idx in range(3):
        assert get_hybrid_attention_spec(first, layer_idx=layer_idx) == get_hybrid_attention_spec(
            second,
            layer_idx=layer_idx,
        )


@pytest.mark.parametrize('config_class', CONFIG_CLASSES, ids=lambda config_class: config_class.model_type)
def test_config_accepts_none(config_class):
    assert config_class(num_hidden_layers=3, attn=None).attn is None


@pytest.mark.parametrize('config_class', CONFIG_CLASSES, ids=lambda config_class: config_class.model_type)
@pytest.mark.parametrize('outer_type', ['dictionary', 'list'])
def test_config_normalization_and_round_trip(config_class, outer_type, tmp_path):
    source_spec = {'layers': [1], 'num_heads': 2, 'extension': 'preserved'}
    source = source_spec if outer_type == 'dictionary' else [source_spec]
    original = deepcopy(source)

    config = config_class(num_hidden_layers=3, attn=source)
    serialized = config.to_dict()['attn']

    assert source == original
    assert isinstance(config.attn, dict if outer_type == 'dictionary' else list)
    assert isinstance(serialized, dict if outer_type == 'dictionary' else list)
    spec = config.attn if outer_type == 'dictionary' else config.attn[0]
    assert spec['num_kv_heads'] == 2
    assert spec['qkv_bias'] is False
    assert spec['window_size'] is None
    assert spec['rope_theta'] == 10000.
    assert spec['extension'] == 'preserved'

    config.save_pretrained(tmp_path)
    reloaded = config_class.from_pretrained(tmp_path)
    assert isinstance(reloaded.attn, dict if outer_type == 'dictionary' else list)
    assert reloaded.attn == config.attn


@pytest.mark.parametrize('config_class', CONFIG_CLASSES, ids=lambda config_class: config_class.model_type)
@pytest.mark.parametrize('outer_type', ['dictionary', 'list'])
def test_config_assignment_normalizes(config_class, outer_type):
    source_spec = {'layers': [1], 'num_heads': 2, 'extension': 'preserved'}
    source = source_spec if outer_type == 'dictionary' else [source_spec]
    original = deepcopy(source)
    config = config_class(num_hidden_layers=3, attn=None)

    config.attn = source

    spec = config.attn if outer_type == 'dictionary' else config.attn[0]
    assert source == original
    assert spec['num_kv_heads'] == 2
    assert spec['qkv_bias'] is False
    assert spec['window_size'] is None
    assert spec['rope_theta'] == 10000.
    assert spec['extension'] == 'preserved'


@pytest.mark.parametrize('config_class', CONFIG_CLASSES, ids=lambda config_class: config_class.model_type)
@pytest.mark.parametrize('outer_type', ['dictionary', 'list'])
def test_config_from_dict_override_normalizes(config_class, outer_type):
    source_spec = {'layers': [1], 'num_heads': 2}
    source = source_spec if outer_type == 'dictionary' else [source_spec]

    config = config_class.from_dict(
        {'num_hidden_layers': 3, 'attn': None},
        attn=source,
    )

    spec = config.attn if outer_type == 'dictionary' else config.attn[0]
    assert spec['num_kv_heads'] == 2
    assert spec['qkv_bias'] is False
    assert spec['window_size'] is None
    assert spec['rope_theta'] == 10000.


def test_config_from_pretrained_override_normalizes(tmp_path):
    GLAConfig(num_hidden_layers=3, attn=None).save_pretrained(tmp_path)

    config = GLAConfig.from_pretrained(
        tmp_path,
        attn=[{'layers': [1], 'num_heads': 2}],
    )

    assert config.attn == [{
        'layers': [1],
        'num_heads': 2,
        'num_kv_heads': 2,
        'qkv_bias': False,
        'window_size': None,
        'rope_theta': 10000.,
    }]


def test_config_assignment_rejects_invalid_plan():
    config = GLAConfig(num_hidden_layers=2, attn=None)

    with pytest.raises(ValueError, match=r"layers.*2"):
        config.attn = {'layers': [2], 'num_heads': 2}


def test_samba_canonical_default_adapts_to_model_depth():
    shallow = SambaConfig(num_hidden_layers=2)
    deep = SambaConfig(num_hidden_layers=20)

    assert shallow.attn['layers'] == [1]
    assert deep.attn['layers'] == [1, 3, 5, 7, 9, 11, 13, 15, 17]
    assert 19 not in deep.attn['layers']


def test_samba_legacy_default_load_adapts_to_model_depth(tmp_path):
    legacy_config = SambaConfig().to_dict()
    legacy_config['num_hidden_layers'] = 2
    (tmp_path / 'config.json').write_text(json.dumps(legacy_config))

    reloaded = SambaConfig.from_pretrained(tmp_path)

    assert reloaded.attn['layers'] == [1]


@pytest.mark.parametrize('outer_type', ['dictionary', 'list'])
def test_samba_explicit_canonical_out_of_range_plan_is_rejected(outer_type):
    attn = SAMBA_CANONICAL_ATTN if outer_type == 'dictionary' else [SAMBA_CANONICAL_ATTN]

    with pytest.raises(ValueError, match=r"layers.*3"):
        SambaConfig(num_hidden_layers=2, attn=attn)


@pytest.mark.parametrize('outer_type', ['dictionary', 'list'])
def test_samba_from_dict_explicit_canonical_out_of_range_plan_is_rejected(outer_type):
    attn = SAMBA_CANONICAL_ATTN if outer_type == 'dictionary' else [SAMBA_CANONICAL_ATTN]

    with pytest.raises(ValueError, match=r"layers.*3"):
        SambaConfig.from_dict({'num_hidden_layers': 2, 'attn': attn})


def test_rodimus_qk_norm_defaults_each_specification():
    config = RodimusConfig(num_hidden_layers=2, attn=None)
    config.attn = [
        {'layers': [0], 'num_heads': 2},
        {'layers': [1], 'num_heads': 2, 'qk_norm': True},
    ]

    assert config.attn[0]['qk_norm'] is False
    assert config.attn[1]['qk_norm'] is True


@pytest.fixture
def allow_attention_construction(monkeypatch):
    monkeypatch.setattr(attn_module, 'flash_attn_func', object())


REPRESENTATIVE_BLOCKS = [
    pytest.param(
        GLAConfig,
        GLABlock,
        'attn',
        GatedLinearAttention,
        {'num_heads': 4, 'expand_v': 1., 'hidden_ratio': 2},
        id='gla',
    ),
    pytest.param(
        GatedDeltaNetConfig,
        GatedDeltaNetBlock,
        'attn',
        GatedDeltaNet,
        {'num_heads': 4, 'head_dim': 8, 'expand_v': 1., 'hidden_ratio': 2},
        id='gated-deltanet',
    ),
    pytest.param(
        SambaConfig,
        SambaBlock,
        'mixer',
        Mamba,
        {'state_size': 4, 'expand': 2, 'hidden_ratio': 2},
        id='samba',
    ),
]


@pytest.mark.parametrize(
    ('config_class', 'block_class', 'mixer_attr', 'native_class', 'config_kwargs'),
    REPRESENTATIVE_BLOCKS,
)
def test_heterogeneous_model_block_construction(
    allow_attention_construction,
    config_class,
    block_class,
    mixer_attr,
    native_class,
    config_kwargs,
):
    config = config_class(
        hidden_size=32,
        num_hidden_layers=6,
        attn=[
            {
                'layers': [1, 4],
                'num_heads': 4,
                'num_kv_heads': 2,
                'qkv_bias': True,
                'window_size': 8,
                'rope_theta': 20000.,
            },
            {
                'layers': [5],
                'num_heads': 8,
                'num_kv_heads': 4,
                'qkv_bias': False,
                'window_size': None,
                'rope_theta': 40000.,
            },
        ],
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        vocab_size=64,
        **config_kwargs,
    )
    blocks = [block_class(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]

    for layer_idx in (0, 2, 3):
        assert isinstance(getattr(blocks[layer_idx], mixer_attr), native_class)
    for layer_idx in (1, 4, 5):
        assert isinstance(getattr(blocks[layer_idx], mixer_attr), Attention)

    local_attn = getattr(blocks[1], mixer_attr)
    assert local_attn.num_heads == 4
    assert local_attn.num_kv_heads == 2
    assert local_attn.qkv_bias is True
    assert local_attn.window_size == 8
    assert local_attn.rope_theta == 20000.

    full_attn = getattr(blocks[5], mixer_attr)
    assert full_attn.num_heads == 8
    assert full_attn.num_kv_heads == 4
    assert full_attn.qkv_bias is False
    assert full_attn.window_size is None
    assert full_attn.rope_theta == 40000.


@pytest.mark.parametrize(
    ('config_class', 'block_class', 'mixer_attr', 'native_class', 'config_kwargs'),
    REPRESENTATIVE_BLOCKS,
)
def test_old_dictionary_model_block_construction(
    allow_attention_construction,
    config_class,
    block_class,
    mixer_attr,
    native_class,
    config_kwargs,
):
    config = config_class(
        hidden_size=32,
        num_hidden_layers=2,
        attn={
            'layers': [1],
            'num_heads': 4,
            'num_kv_heads': 2,
            'qkv_bias': True,
            'window_size': 8,
            'rope_theta': 20000.,
        },
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        vocab_size=64,
        **config_kwargs,
    )

    native_block = block_class(config, 0)
    attention_block = block_class(config, 1)
    attention = getattr(attention_block, mixer_attr)

    assert isinstance(getattr(native_block, mixer_attr), native_class)
    assert isinstance(attention, Attention)
    assert attention.num_heads == 4
    assert attention.num_kv_heads == 2
    assert attention.qkv_bias is True
    assert attention.window_size == 8
    assert attention.rope_theta == 20000.


def _tiny_gla_config(attn):
    return GLAConfig(
        hidden_size=32,
        num_hidden_layers=3,
        num_heads=4,
        hidden_ratio=2,
        attn=attn,
        max_position_embeddings=32,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        vocab_size=64,
    )


def _has_bf16_flash_attention():
    return (
        attn_module.flash_attn_func is not None
        and device_platform in ('cuda', 'hip')
        and device_torch_lib.is_bf16_supported()
    )


def test_dictionary_and_one_item_list_architecture_equivalence(allow_attention_construction, tmp_path):
    spec = {
        'layers': [1],
        'num_heads': 4,
        'num_kv_heads': 2,
        'qkv_bias': True,
        'window_size': 16,
        'rope_theta': 20000.,
    }
    dictionary_model = GLAForCausalLM(_tiny_gla_config(spec))
    list_model = GLAForCausalLM(_tiny_gla_config([spec]))

    dictionary_state = dictionary_model.state_dict()
    list_state = list_model.state_dict()
    assert dictionary_state.keys() == list_state.keys()
    assert {key: value.shape for key, value in dictionary_state.items()} == {
        key: value.shape for key, value in list_state.items()
    }
    assert [type(layer.attn) for layer in dictionary_model.model.layers] == [
        type(layer.attn) for layer in list_model.model.layers
    ]
    dictionary_attention = dictionary_model.model.layers[1].attn
    list_attention = list_model.model.layers[1].attn
    for attribute in ('num_heads', 'num_kv_heads', 'qkv_bias', 'window_size', 'rope_theta'):
        assert getattr(dictionary_attention, attribute) == getattr(list_attention, attribute)
    list_model.load_state_dict(dictionary_state, strict=True)

    for name, config in (('dictionary', dictionary_model.config), ('list', list_model.config)):
        output_dir = tmp_path / name
        config.save_pretrained(output_dir)
        reloaded_config = GLAConfig.from_pretrained(output_dir)
        reloaded_model = GLAForCausalLM(reloaded_config)
        assert reloaded_model.state_dict().keys() == dictionary_state.keys()
        assert isinstance(reloaded_config.attn, dict if name == 'dictionary' else list)


@pytest.mark.skipif(
    not _has_bf16_flash_attention(),
    reason="numerical hybrid equivalence requires FlashAttention and BF16 on a CUDA or ROCm device",
)
def test_dictionary_and_one_item_list_numerical_equivalence():
    spec = {
        'layers': [1],
        'num_heads': 4,
        'num_kv_heads': 2,
        'qkv_bias': True,
        'window_size': None,
        'rope_theta': 10000.,
    }
    common = {
        'hidden_size': 32,
        'num_hidden_layers': 2,
        'num_heads': 4,
        'hidden_ratio': 2,
        'max_position_embeddings': 16,
        'fuse_norm': False,
        'fuse_swiglu': False,
        'fuse_cross_entropy': False,
        'vocab_size': 64,
    }
    torch.manual_seed(42)
    dictionary_model = GLAForCausalLM(GLAConfig(attn=spec, **common)).to(torch.bfloat16).to(device)
    list_model = GLAForCausalLM(GLAConfig(attn=[spec], **common)).to(torch.bfloat16).to(device)
    list_model.load_state_dict(dictionary_model.state_dict(), strict=True)
    dictionary_model.train()
    list_model.train()

    input_ids = torch.randint(0, 64, (1, 8), device=device)
    dictionary_logits = dictionary_model(input_ids, use_cache=False).logits
    list_logits = list_model(input_ids, use_cache=False).logits
    assert_close('logits', dictionary_logits, list_logits, 1e-5)

    dictionary_logits.float().square().mean().backward()
    list_logits.float().square().mean().backward()
    dictionary_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in dictionary_model.named_parameters()
        if parameter.grad is not None
    }
    list_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in list_model.named_parameters()
        if parameter.grad is not None
    }
    assert dictionary_grads.keys() == list_grads.keys()
    for name in dictionary_grads:
        assert_close(f'gradient {name}', dictionary_grads[name], list_grads[name], 1e-5)


@pytest.mark.skipif(
    not _has_bf16_flash_attention(),
    reason="heterogeneous plan smoke requires FlashAttention and BF16 on a CUDA or ROCm device",
)
def test_heterogeneous_plan_numerical_smoke():
    """A plan with two different specs must run end to end on GPU.

    Covers the actual heterogeneous case (SWA + native mixer + full attention
    in one model) that the dictionary/list equivalence tests cannot reach.
    """
    attn = [
        {'layers': [0], 'num_heads': 4, 'num_kv_heads': 2, 'window_size': 4},
        {'layers': [2], 'num_heads': 4, 'num_kv_heads': 4, 'qkv_bias': True, 'window_size': None},
    ]
    torch.manual_seed(42)
    model = GLAForCausalLM(_tiny_gla_config(attn)).to(torch.bfloat16).to(device)
    model.train()

    input_ids = torch.randint(0, 64, (2, 16), device=device)
    logits = model(input_ids, use_cache=False).logits
    assert logits.shape == (2, 16, 64)
    assert torch.isfinite(logits).all()

    logits.float().square().mean().backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), f'non-finite gradient: {name}'
