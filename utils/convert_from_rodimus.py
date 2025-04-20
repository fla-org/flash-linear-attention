from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import warnings

import json
import os
import sys
import shutil

sys.path.append('./')


import fla  # noqa


def hasvalue(d, k, default):
    return d[k] if k in d else default


def replace(d, old_k, new_k):
    assert old_k in d, f'{old_k} not in weights'
    if old_k == new_k:
        return
    print(f'Replace {old_k} with {new_k} ...')
    d[new_k] = d[old_k]
    d.pop(old_k)


def chunk(d, old_k, new_k1, new_k2):
    assert old_k in d, f'{old_k} not in weights'
    weight = d[old_k]
    new_w1, new_2 = torch.chunk(weight, 2)
    print(f'Split {old_k} into {new_k1} and {new_k2} ...')
    d[new_k1] = new_w1
    d[new_k2] = new_2
    d.pop(old_k)


def convert_config(ckpt_dir, out_dir):
    config_file_path = os.path.join(ckpt_dir, 'config.json')
    assert os.path.exists(config_file_path)
    with open(config_file_path, 'r') as f:
        config = json.load(f)

    new_config = {}
    new_config['block_type'] = config['block_type']
    new_config['hidden_size'] = config['d_model']
    new_config['num_hidden_layers'] = config['n_layer']
    new_config['attn_mode'] = config['mixer_cfg']['mode']
    new_config['residual_in_fp32'] = config['residual_in_fp32']
    new_config['expand_ratio'] = config['mixer_cfg']['mem_size']
    new_config['input_gate_low_rank'] = config['mixer_cfg']['input_gate_low_rank']
    new_config['use_short_conv'] = hasvalue(config['mixer_cfg'], 'use_short_conv', True)
    new_config['conv_size'] = hasvalue(config['mixer_cfg'], 'conv_size', 4)
    new_config['hidden_ratio'] = config['attn_cfg']['ffn_expand_ratio']
    new_config['max_position_embeddings'] = config['max_position_embeddings']
    new_config['norm_eps'] = config['norm_epsilon']
    new_config['k_norm_eps'] = hasvalue(config['mixer_cfg'], 'normalize_epsilon', None)

    new_config['ska_attn'] = {
        'window_size': config['attn_cfg']['window_size'] // 2,
        'num_heads': config['attn_cfg']['num_heads'],
        'rope_theta': hasvalue(config['attn_cfg'], 'rotary_emb_base', 10000.0),
    }

    new_config['use_cache'] = True
    new_config['pad_token_id'] = hasvalue(config, 'pad_token_id', None)
    new_config['bos_token_id'] = hasvalue(config, 'bos_token_id', 126080)
    new_config['eos_token_id'] = hasvalue(config, 'eos_token_id', 126081)
    new_config['tie_word_embeddings'] = hasvalue(config, 'tie_word_embeddings', True)
    new_config['initializer_range'] = hasvalue(config, 'initializer_range', True)
    new_config['fuse_norm'] = True
    new_config['fuse_swiglu'] = True
    new_config['fuse_cross_entropy'] = True
    new_config['vocab_size'] = config['vocab_size']
    new_config['torch_dtype'] = hasvalue(config, 'torch_dtype', 'float16')

    print(f'New config: \n{new_config}')

    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(new_config, f, indent=4)


def convert_tokenizer(ckpt_dir, out_dir):
    shutil.copyfile(os.path.join(ckpt_dir, 'special_tokens_map.json'), os.path.join(out_dir, 'special_tokens_map.json'))
    shutil.copyfile(os.path.join(ckpt_dir, 'tokenizer_config.json'), os.path.join(out_dir, 'tokenizer_config.json'))
    shutil.copyfile(os.path.join(ckpt_dir, 'tokenizer.json'), os.path.join(out_dir, 'tokenizer.json'))


def convert_weights(ckpt_dir, out_dir):
    weights_path = os.path.join(ckpt_dir, 'pytorch_model.bin')
    assert os.path.exists(weights_path)
    weights = torch.load(weights_path, mmap=True)

    replace(weights, 'model.norm_f.weight', 'model.norm.weight')

    with open(os.path.join(ckpt_dir, 'config.json'), 'r') as f:
        config = json.load(f)
        num_hidden_layers = config['n_layer']
        block_type = config['block_type']

    for i in range(num_hidden_layers):
        # rodimus attention
        replace(weights, f'model.layers.{i}.mixer_norm.weight', f'model.layers.{i}.mixer_norm.weight')
        replace(weights, f'model.layers.{i}.mixer.act_norm.weight', f'model.layers.{i}.mixer.activation_norm.weight')
        chunk(weights, f'model.layers.{i}.mixer.fc.weight', f'model.layers.{i}.mixer.up_proj.weight', f'model.layers.{i}.mixer.gate_proj.weight')
        replace(weights, f'model.layers.{i}.mixer.out_proj.weight', f'model.layers.{i}.mixer.down_proj.weight')
        replace(weights, f'model.layers.{i}.mixer.inner_mixer.short_conv.conv1d.weight', f'model.layers.{i}.mixer.short_conv.weight')
        replace(weights, f'model.layers.{i}.mixer.inner_mixer.short_conv.conv1d.bias', f'model.layers.{i}.mixer.short_conv.bias')
        replace(weights, f'model.layers.{i}.mixer.inner_mixer.residual_weight', f'model.layers.{i}.mixer.residual_weight')
        chunk(weights, f'model.layers.{i}.mixer.inner_mixer.in_proj.weight',
              f'model.layers.{i}.mixer.q_proj.weight', f'model.layers.{i}.mixer.k_proj.weight')
        replace(weights, f'model.layers.{i}.mixer.inner_mixer.ch_gate_proj.0.weight', f'model.layers.{i}.mixer.i_gate_proj.0.weight')
        replace(weights, f'model.layers.{i}.mixer.inner_mixer.ch_gate_proj.1.weight', f'model.layers.{i}.mixer.i_gate_proj.1.weight')
        replace(weights, f'model.layers.{i}.mixer.inner_mixer.ch_gate_proj.1.bias', f'model.layers.{i}.mixer.i_gate_proj.1.bias', )
        chunk(weights, f'model.layers.{i}.mixer.inner_mixer.mem_gate_proj.weight',
              f'model.layers.{i}.mixer.g_gate_proj.weight', f'model.layers.{i}.mixer.tau_gate_proj.weight')
        chunk(weights, f'model.layers.{i}.mixer.inner_mixer.mem_gate_proj.bias',
              f'model.layers.{i}.mixer.g_gate_proj.bias', f'model.layers.{i}.mixer.tau_gate_proj.bias')

        if block_type == 'rodimus_plus':
            # ska
            replace(weights, f'model.layers.{i}.attn_norm.weight', f'model.layers.{i}.ska_attn_norm.weight')
            replace(weights, f'model.layers.{i}.attn.q_proj.weight', f'model.layers.{i}.ska_attn.q_proj.weight')
            replace(weights, f'model.layers.{i}.attn.k_proj.weight', f'model.layers.{i}.ska_attn.k_proj.weight')
            replace(weights, f'model.layers.{i}.attn.v_proj.weight', f'model.layers.{i}.ska_attn.v_proj.weight')
            replace(weights, f'model.layers.{i}.attn.out_proj.weight', f'model.layers.{i}.ska_attn.o_proj.weight')

            # mlp
            replace(weights, f'model.layers.{i}.ffn_norm.weight', f'model.layers.{i}.mlp_norm.weight')
            chunk(weights, f'model.layers.{i}.ffn.fc.weight', f'model.layers.{i}.mlp.up_proj.weight', f'model.layers.{i}.mlp.gate_proj.weight')
            replace(weights, f'model.layers.{i}.ffn.out_proj.weight', f'model.layers.{i}.mlp.down_proj.weight')

    out_path = os.path.join(out_dir, 'pytorch_model.bin')
    torch.save(weights, out_path)
    print(f'Save model in {out_path}')


def convert(ckpt_dir, out_dir, ):
    os.makedirs(out_dir, exist_ok=True)

    print('Converting config ...')
    convert_config(ckpt_dir, out_dir)
    print('Converting tokenizer ...')
    convert_tokenizer(ckpt_dir, out_dir)
    print('Converting weights ...')
    convert_weights(ckpt_dir, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", help='Path to the input model')
    parser.add_argument("--out_dir", help='Directory to save model')
    args = parser.parse_args()
    convert(args.ckpt_dir, args.out_dir)
