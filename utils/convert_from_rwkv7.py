# -*- coding: utf-8 -*-

import os
import re

import torch

import fla  # noqa
from fla.models.rwkv7 import RWKV7Config, RWKV7ForCausalLM
import argparse


def convert(
    rwkv7_pth: str,
    output_dir: str,
    precision: str = 'float32',
):
    blink_weights = torch.load(rwkv7_pth, weights_only=True)
    config = RWKV7Config()
    config.vocab_size = blink_weights['emb.weight'].shape[0]  # 50304
    config.hidden_size = blink_weights['blocks.0.ffn.key.weight'].shape[1]  # 768
    config.hidden_ratio = blink_weights['blocks.0.ffn.key.weight'].shape[0] / \
        blink_weights['blocks.0.ffn.key.weight'].shape[1]  # 4.0
    config.intermediate_size = blink_weights['blocks.0.ffn.key.weight'].shape[0]
    config.num_hidden_layers = 0
    while f'blocks.{config.num_hidden_layers}.ffn.key.weight' in blink_weights:
        config.num_hidden_layers += 1
    # 12
    config.decay_low_rank_dim = blink_weights['blocks.0.att.w1'].shape[1]  # 64
    config.gate_low_rank_dim = blink_weights['blocks.0.att.g1'].shape[1]  # 128
    config.a_low_rank_dim = blink_weights['blocks.0.att.a1'].shape[1]  # 64
    try:
        config.v_low_rank_dim = blink_weights['blocks.1.att.v1'].shape[1]  # 32
    except KeyError:
        config.v_low_rank_dim = 32
    config.torch_dtype = precision

    model = RWKV7ForCausalLM._from_config(config)
    print(model)
    model_dict = model.state_dict()
    model_names = [n for n in model_dict]

    # these parameters may be present in pth file but are never used:
    unused_names = ['blocks.0.attn.v0', 'blocks.0.attn.v1', 'blocks.0.attn.v2']
    # these parameters may or may not be present in pth file:
    possible_absent_weights = [
        'model.layers.0.pre_norm.weight', 'model.layers.0.pre_norm.bias'
    ]
    # other parameters may raise a KeyError

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
        if blink_name in unused_names:
            return '', False
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
        if name_compo[2] == 'attn' and re.match("x_[rwkvag]", name_compo[3]):
            name_compo[3] = 'x_x'
        elif re.match("[wvag][012]", name_compo[3]):
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

    for blink_name in blink_weights:
        fla_name, transposed = translate_into_fla(blink_name)
        print(f'{blink_name:32} -> {fla_name:42}, {transposed}')
        if not fla_name:
            print('redundant parameters in source weight: ', blink_name, '\n')
            continue
        weight = blink_weights[blink_name]
        # print shape information
        shape1 = list(weight.shape)
        shape2 = list(model_dict[fla_name].shape)
        print(f'{str(shape1):32}    {str(shape2)}\n')

        if transposed:
            weight.t_()
        if shape1 == [1, 1, config.hidden_size]:
            weight.squeeze_()

        # fix: fusing x_[rwkvag] to x_x
        if fla_name.endswith('attn.x_x'):
            model_dict[fla_name].data['rwkvag'.find(blink_name[-1])].copy_(weight)
            if fla_name in model_names:
                model_names.remove(fla_name)
        else:
            assert model_dict[fla_name].shape == weight.shape
            model_dict[fla_name].data.copy_(weight)
            model_names.remove(fla_name)

    print("uninitialized parameters: ", model_names)
    for n in model_names:
        if n not in possible_absent_weights:
            raise KeyError(n)

    os.makedirs(output_dir, exist_ok=True)

    from safetensors.torch import save_file
    save_file(model.state_dict(), os.path.join(output_dir, 'model.safetensors'))
    model.config.save_pretrained(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert RWKV7')
    parser.add_argument('rwkv7_pth', type=str, help='Path to the input model')
    parser.add_argument('output_dir', type=str, help='Directory to save model')
    parser.add_argument('--precision', type=str, default='float32')
    args = parser.parse_args()
    convert(args.rwkv7_pth, args.output_dir, precision=args.precision)
