# -*- coding: utf-8 -*-

# scripts for converting pretrained hf model weights to fla style

import argparse
import os
import re

import torch
from transformers import AutoModelForCausalLM

import fla  # noqa
from fla.models.rwkv7 import RWKV7Config


def convert(
    rwkv7: str,
    output: str,
    precision: str = 'float32'
):
    weights = torch.load(rwkv7, weights_only=True)
    config = RWKV7Config()
    config.vocab_size = weights['emb.weight'].shape[0]  # 50304
    config.hidden_size = weights['blocks.0.ffn.key.weight'].shape[1]  # 768
    config.hidden_ratio = weights['blocks.0.ffn.key.weight'].shape[0] / weights['blocks.0.ffn.key.weight'].shape[1]  # 4.0
    config.intermediate_size = weights['blocks.0.ffn.key.weight'].shape[0]
    config.num_hidden_layers = 0
    while f'blocks.{config.num_hidden_layers}.ffn.key.weight' in weights:
        config.num_hidden_layers += 1
    # 12
    config.value_dim = [config.hidden_size] * config.num_hidden_layers
    config.decay_low_rank_dim = weights['blocks.0.att.w1'].shape[1]  # 64
    config.gate_low_rank_dim = weights['blocks.0.att.g1'].shape[1]  # 128
    config.a_low_rank_dim = weights['blocks.0.att.a1'].shape[1]  # 64
    try:
        config.v_low_rank_dim = weights['blocks.1.att.v1'].shape[1]  # 32
    except KeyError:
        config.v_low_rank_dim = 32
    config.torch_dtype = precision

    print(f"Creating model with config:\n{config}")
    model = AutoModelForCausalLM.from_config(config)

    if precision in ['bf16', 'bfloat16']:
        model = model.to(torch.bfloat16)
    if precision in ['fp16', 'float16']:
        model = model.to(torch.float16)
    if precision in ['fp64', 'double', 'float64']:
        model = model.to(torch.double)

    print(model)
    model_dict = model.state_dict()
    model_names = [n for n in model_dict]

    # these parameters may be present in pth file but are never used:
    unused_names = ['blocks.0.att.v0', 'blocks.0.att.v1', 'blocks.0.att.v2']
    # these parameters may or may not be present in pth file:
    possible_absent_weights = [
        'model.layers.0.pre_norm.weight', 'model.layers.0.pre_norm.bias'
    ]
    # other parameters may raise a KeyError

    def translate_into_fla(name):
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
        if name in unused_names:
            return '', False
        if name in emb_head:
            return emb_head[name], False
        name_compo = name.split('.')
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

    for name in weights:
        fla_name, transposed = translate_into_fla(name)
        print(f'{name:32} -> {fla_name:42}, {transposed}')
        if not fla_name:
            print('redundant parameters in source weight: ', name, '\n')
            continue
        weight = weights[name]
        # print shape information
        shape1 = list(weight.shape)
        shape2 = list(model_dict[fla_name].shape)
        print(f'{str(shape1):32}    {str(shape2)}\n')

        if transposed:
            weight.t_()
        if shape1 == [1, 1, config.hidden_size]:
            weight.squeeze_()

        assert model_dict[fla_name].shape == weight.shape
        model_dict[fla_name].data.copy_(weight)
        model_names.remove(fla_name)

    print("uninitialized parameters: ", model_names)
    for n in model_names:
        if n not in possible_absent_weights:
            raise KeyError(n)

    os.makedirs(output, exist_ok=True)

    model.save_pretrained(output, max_shard_size="1000GB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert RWKV7')
    parser.add_argument('--rwkv7', type=str, help='Path to the input model')
    parser.add_argument('--output', type=str, help='Directory to save model')
    parser.add_argument('--precision', type=str, default='float32')
    args = parser.parse_args()
    convert(args.rwkv7, args.output, precision=args.precision)
