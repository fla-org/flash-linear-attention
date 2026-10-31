# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Record or compare NPU eager outputs and gradients across two source checkouts."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--record', type=Path)
    mode.add_argument('--compare', type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    os.environ['FLA_CI_ENV'] = '0'
    os.environ['FLA_FLASH_QLA'] = '0'
    os.environ['FLA_DISABLE_BACKEND_DISPATCH'] = '0'

    import torch
    import torch.nn.functional as F
    import torch_npu  # noqa: F401

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.utils import IS_NPU

    if not IS_NPU:
        raise RuntimeError('This comparison requires Ascend NPU')
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '-C', str(repo), 'status', '--short'], text=True)
    metadata = dict(commit=commit, status=status, torch=str(torch.__version__), torch_npu=str(torch_npu.__version__))
    if args.record:
        cases = []
        generator = torch.Generator().manual_seed(1155)
        for dtype in (torch.float16, torch.bfloat16):
            for dim in (64, 128):
                for packed in (False, True):
                    shape = (1, 193, 2, dim)
                    q, k = [F.normalize(torch.randn(shape, generator=generator), dim=-1).to(dtype) for _ in range(2)]
                    v = torch.randn(shape, generator=generator).to(dtype)
                    g = F.logsigmoid(torch.randn(shape[:3], generator=generator)) / 8
                    beta = torch.rand(shape[:3], generator=generator).to(dtype)
                    h0 = torch.randn(4 if packed else 1, 2, dim, dim, generator=generator)
                    cases.append(dict(inputs=[q, k, v, g, beta, h0], packed=packed))
        snapshot = dict(metadata=metadata, cases=cases)
    else:
        snapshot = torch.load(args.compare, map_location='cpu', weights_only=True)
        print(f"Baseline: {snapshot['metadata']}")
    print(f'Current: {metadata}')

    for index, case in enumerate(snapshot['cases']):
        leaves = [tensor.to('npu').requires_grad_() for tensor in case['inputs']]
        q, k, v, g, beta, h0 = leaves
        cu = torch.tensor([0, 1, 64, 129, 193], device='npu') if case['packed'] else None
        # omit use_graph so this script also runs on the unmodified baseline.
        o, ht = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
        )
        (o.float().square().sum() + ht.square().sum()).backward()
        names = ('o', 'ht', 'dq', 'dk', 'dv', 'dg', 'dbeta', 'dh0')
        tensors = (o, ht, *(tensor.grad for tensor in leaves))
        results = {}
        for name, tensor in zip(names, tensors):
            if tensor is None or not torch.isfinite(tensor).all():
                raise AssertionError(f'case {index}: missing or non-finite {name}')
            results[name] = tensor.detach().cpu()
        if args.record:
            case['results'] = results
        else:
            for name, result in results.items():
                torch.testing.assert_close(result, case['results'][name], rtol=0, atol=0, msg=f'case {index}: {name}')
        print(f'case {index}: {q.dtype}, D={q.shape[-1]}, packed={case["packed"]}: passed')
    if args.record:
        # do not silently replace a previously collected baseline.
        with args.record.open('xb') as output:
            torch.save(snapshot, output)


if __name__ == '__main__':
    main()
