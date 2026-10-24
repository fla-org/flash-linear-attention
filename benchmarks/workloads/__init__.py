# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

WORKLOADS = {
    'operator_latency': ('Operator forward/backward latency; select variants with --op', 'GPU / NPU'),
    'activations_latency': ('Activation provider latency', 'GPU'),
    'conv_latency': ('Convolution provider latency', 'GPU / NPU'),
    'cross_entropy_latency': ('Cross-entropy provider latency', 'GPU'),
    'l2norm_latency': ('L2-normalization provider latency', 'GPU'),
    'layernorm_latency': ('Layer/RMS-normalization provider latency', 'GPU'),
    'tokenshift_latency': ('Token-shift provider latency', 'GPU'),
    'training': ('Complete model training throughput; select a model with --name', 'CUDA'),
    'generation': ('Prompt processing and generation throughput; load a model with --path', 'CUDA'),
    'kda_graph': ('KDA eager execution versus graph replay', 'Ascend NPU'),
    'operator_cp': ('GDN/KDA operator context-parallel latency; select with --op', 'CUDA, torchrun'),
    'layer_cp': ('GDN/KDA layer context-parallel latency; select with --layer', 'CUDA, torchrun'),
    'conv_cp': ('Context-parallel convolution latency', 'CUDA, torchrun'),
    'kda_cp_tp': ('KDA context/tensor-parallel configurations', 'CUDA, torchrun'),
}
