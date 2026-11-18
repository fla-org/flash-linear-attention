# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
"""Compare B300 PTX and FlashQLA in separate processes on the exported shapes."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.gated_delta_rule.backends.qla_ptx import WORKLOADS, QLAPTXBackend


def timing(function):
    for _ in range(5):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [function() for _ in range(3)]
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(20):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 3)
    assert outputs
    return {"median_us": statistics.median(samples), "samples_us": samples}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ptx", "flash-qla"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    result = {
        "backend": options.backend,
        "gpu": torch.cuda.get_device_name(),
        "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "rows": [],
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    for w in WORKLOADS:
        for has_state in (False, True):
            torch.manual_seed(42)
            b, t, hq, hv = (w[k] for k in ("B", "T", "Hq", "Hv"))
            q, k = [F.normalize(torch.randn(b, t, hq, 128, device="cuda"), dim=-1).bfloat16() for _ in range(2)]
            v = torch.randn(b, t, hv, 128, device="cuda", dtype=torch.bfloat16)
            a = torch.rand(hv, device="cuda") * 16
            a[0], a[-1] = 0, 16
            g = -a * F.softplus(torch.randn(b, t, hv, device="cuda") * 0.5 + 1)
            beta = torch.randn(b, t, hv, device="cuda").sigmoid()
            cpu_cu = torch.tensor(w["cu"], dtype=torch.int64) if "cu" in w else None
            n = b if cpu_cu is None else len(cpu_cu) - 1
            state = torch.randn(n, hv, 128, 128, device="cuda") if has_state else None
            args = dict(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                output_final_state=True,
                cu_seqlens=cpu_cu.cuda() if cpu_cu is not None else None,
            )
            if options.backend == "ptx":
                args["cu_seqlens_cpu"] = cpu_cu
                backend = QLAPTXBackend()
                assert backend.is_available()
                accepted, reason = backend.chunk_gated_delta_rule_verifier(**args)
                assert accepted, reason
                function = chunk_gated_delta_rule
            else:
                from flash_qla import chunk_gated_delta_rule as flash_qla

                function = flash_qla
                args.update(auto_cp=True, enable_fwd_cp_cache=False)
            row = {"workload": w["name"], "initial_state": has_state, **timing(lambda: function(**args))}
            result["rows"].append(row)
            options.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
