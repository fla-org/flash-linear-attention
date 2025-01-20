# -*- coding: utf-8 -*-

import os
import re

import torch

import fla  # noqa
from fla.models.rwkv7 import RWKV7Config, RWKV7ForCausalLM


def convert(
    rwkv7_dir: str,
    output_dir: str
):
    blinkdl_weights = torch.load(rwkv7_dir, weights_only=True)
    config = RWKV7Config()
    config.vocab_size = blinkdl_weights['emb.weight'].shape[0]  # 50304
    config.hidden_size = blinkdl_weights['blocks.0.ffn.key.weight'].shape[1]  # 768
    config.hidden_ratio = blinkdl_weights['blocks.0.ffn.key.weight'].shape[0] / \
        blinkdl_weights['blocks.0.ffn.key.weight'].shape[1]  # 4.0
    config.intermediate_size = blinkdl_weights['blocks.0.ffn.key.weight'].shape[0]
    config.num_hidden_layers = 0
    while f'blocks.{config.num_hidden_layers}.ffn.key.weight' in blinkdl_weights:
        config.num_hidden_layers += 1
    # 12
    config.decay_low_rank_dim = blinkdl_weights['blocks.0.att.w1'].shape[1]  # 64
    config.gate_low_rank_dim = blinkdl_weights['blocks.0.att.g1'].shape[1]  # 128
    config.a_low_rank_dim = blinkdl_weights['blocks.0.att.a1'].shape[1]  # 64
    try:
        config.v_low_rank_dim = blinkdl_weights['blocks.1.att.v1'].shape[1]  # 32
    except Exception:
        config.v_low_rank_dim = 32

    model = RWKV7ForCausalLM._from_config(config)
    print(model)
    model_dict = model.state_dict()
    model_names = [n for n in model_dict]

    unused_weights = ['model.layers.0.attn.v_lora.lora.0.weight',
                      'model.layers.0.attn.v_lora.lora.2.weight', 'model.layers.0.attn.v_lora.lora.2.bias']
    possible_absent_weights = ['model.layers.0.pre_norm.weight', 'model.layers.0.pre_norm.bias']

    def translate_into_fla(blink_name):
        transposed = False
        emb_head = {
            'emb.weight': 'model.embeddings.weight',
            'ln_out.weight': 'model.norm.weight',
            'ln_out.bias': 'model.norm.bias',
            'head.weight': 'lm_head.weight'
        }
        proj = {
            'receptance': 'r_proj',
            'key': 'k_proj',
            'value': 'v_proj',
            'ln_x': 'g_norm',
            'output': 'o_proj',
        }
        if blink_name in emb_head:
            return emb_head[blink_name], False
        name_compo = blink_name.split('.')
        assert name_compo[0] == 'blocks'
        name_compo[0] = 'model.layers'
        assert int(name_compo[1]) in range(config.num_hidden_layers)
        name_compo[2] = {
            'att': 'attn',
            'ffn': 'ffn',
            'ln0': 'pre_norm',
            'ln1': 'attn_norm',
            'ln2': 'ffn_norm'
        }[name_compo[2]]
        if re.match("[wvag][012]", name_compo[3]):
            typ, num = name_compo[3]
            name_compo[3] = f'{typ}_lora.lora.' + {
                '0': '2.bias',
                '1': '0.weight',
                '2': '2.weight',
            }[num]
            transposed |= (num in ['1', '2'])
        elif name_compo[2] == 'attn' and name_compo[3] in proj:
            name_compo[3] = proj[name_compo[3]]
        return '.'.join(name_compo), transposed

    for blink_name in blinkdl_weights:
        fla_name, transposed = translate_into_fla(blink_name)
        print(f"{blink_name:32} -> {fla_name:50}, {transposed}")
        weight = blinkdl_weights[blink_name] if not transposed else blinkdl_weights[blink_name].T
        if re.match('.*[wva]0', blink_name):
            weight.squeeze_()
        assert model_dict[fla_name].shape == weight.shape
        model_dict[fla_name].data.copy_(weight)
        model_names.remove(fla_name)

    print("unused parameters: ", model_names)
    for n in model_names:
        if not (n in unused_weights or n in possible_absent_weights):
            raise KeyError(n)
    os.makedirs(output_dir, exist_ok=True)

    from safetensors.torch import save_file
    save_file(model.state_dict(), os.path.join(output_dir, 'model.safetensors'))
    model.config.save_pretrained(output_dir)


convert("/home/zhangping/zrc/RWKV-x070-Pile-168M-20241120-ctx4096.pth", "../rwkv-pile-fla")
convert("/home/zhangping/zrc/RWKV-x070-Pile-168M-20241120-ctx4096.pth", "../rwkv-pile-fla")
